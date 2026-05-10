//! Headless test harness: build a `ps_core::App` with the same factory set
//! as the binary, render one frame to an offscreen `Rgba8Unorm` target
//! sized to a small HDR framebuffer, read back the pixels.
//!
//! Used by `tests/integration.rs`. The windowed binary in `main.rs` does
//! not depend on this module.

use anyhow::{Context, Result};
use glam::{Vec3, Vec4};
use ps_backdrop::BackdropFactory;
use ps_core::{
    App, AppBuilder, Config, FrameUniforms, GpuContext, HdrFramebufferImpl, PrepareContext,
    RenderContext,
};
use ps_ground::GroundFactory;
use ps_postprocess::{Tonemap, TonemapMode};
use ps_tint::TintFactory;

use crate::main_helpers::{build_stub_bind_group, encode_frame_clear};

/// Per-test GPU resources owned outside [`HeadlessApp`] so they can be
/// constructed once and reused across `render_one_frame()` calls.
pub struct TestSetup {
    /// HDR offscreen target.
    pub hdr: HdrFramebufferImpl,
    /// Final 8-bit unorm output. Sized to the HDR target.
    pub output: wgpu::Texture,
    /// View on `output`.
    pub output_view: wgpu::TextureView,
    /// Staging buffer for readback.
    pub staging: wgpu::Buffer,
    /// Tone-map pass writing into `output_view`.
    pub tonemap: Tonemap,
    /// Stub bind group for `RenderContext::frame_bind_group` /
    /// `world_bind_group` (Phase 1 placeholder).
    pub stub_bind_group: wgpu::BindGroup,
    /// Pixel size.
    pub size: (u32, u32),
    /// Padded bytes-per-row (must satisfy COPY_BYTES_PER_ROW_ALIGNMENT).
    pub padded_bytes_per_row: u32,
}

impl TestSetup {
    /// Build everything sized to `(w, h)`. The output is `Rgba8Unorm` (NOT
    /// sRGB) so test code reads exactly what the tone-mapper wrote, no GPU
    /// gamma encode in the way.
    pub fn new(gpu: &GpuContext, _config: &Config, (w, h): (u32, u32)) -> Self {
        let hdr = HdrFramebufferImpl::new(gpu, (w, h));
        let output = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let bytes_per_row = w * 4;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = bytes_per_row.div_ceil(alignment) * alignment;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-staging"),
            size: (padded_bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let tonemap = Tonemap::new(&gpu.device, &hdr, wgpu::TextureFormat::Rgba8Unorm);
        let stub_bind_group = build_stub_bind_group(&gpu.device);

        Self {
            hdr,
            output,
            output_view,
            staging,
            tonemap,
            stub_bind_group,
            size: (w, h),
            padded_bytes_per_row,
        }
    }
}

/// Headless equivalent of the windowed `RunState`.
pub struct HeadlessApp {
    setup: TestSetup,
    app: App,
    /// Cached EV / tone-map mode picked from the config.
    ev100: f32,
    tonemap_mode: TonemapMode,
    /// Most-recent config (kept so resize / reconfigure don't lose values).
    last_config: Config,
}

impl HeadlessApp {
    /// Build with the same factory set the binary registers.
    pub fn new(gpu: &GpuContext, config: &Config, setup: TestSetup) -> Result<Self> {
        let app = AppBuilder::new()
            .with_factory(Box::new(BackdropFactory))
            .with_factory(Box::new(GroundFactory))
            .with_factory(Box::new(TintFactory))
            .build(config, gpu)
            .context("AppBuilder::build")?;
        Ok(Self {
            setup,
            app,
            ev100: config.render.ev100,
            tonemap_mode: TonemapMode::from_config(&config.render.tone_mapper),
            last_config: config.clone(),
        })
    }

    /// Apply a new config in place (mirrors the hot-reload path).
    pub fn reconfigure(&mut self, gpu: &GpuContext, config: &Config) -> Result<()> {
        self.app
            .reconfigure(config, gpu)
            .context("App::reconfigure")?;
        self.ev100 = config.render.ev100;
        self.tonemap_mode = TonemapMode::from_config(&config.render.tone_mapper);
        self.last_config = config.clone();
        Ok(())
    }

    /// Render one frame using the supplied `GpuContext`. Returns raw
    /// `Rgba8Unorm` pixels (top-left origin), tightly packed.
    pub fn render_one_frame(&mut self, gpu: &GpuContext) -> Vec<u8> {
        let (w, h) = self.setup.size;
        let aspect = w as f32 / h as f32;
        let camera = ps_core::camera::FlyCamera::default();
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let frame_uniforms = FrameUniforms {
            view,
            proj,
            view_proj: proj * view,
            camera_position_world: Vec3::ZERO,
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: 0.0,
            simulated_seconds: 0.0,
            frame_index: 0,
            ev100: self.ev100,
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-frame"),
            });

        // Pre-frame clear.
        encode_frame_clear(
            &mut encoder,
            &self.setup.hdr,
            !self.last_config.render.subsystems.backdrop,
            self.last_config.render.clear_color,
        );

        let world = ps_core::WorldState;
        let weather = ps_core::WeatherState;
        let mut prepare_ctx = PrepareContext {
            device: &gpu.device,
            queue: &gpu.queue,
            world: &world,
            weather: &weather,
            frame_uniforms: &frame_uniforms,
            atmosphere_luts: None,
            dt_seconds: 1.0 / 60.0,
        };
        let render_ctx = RenderContext {
            device: &gpu.device,
            queue: &gpu.queue,
            framebuffer: &self.setup.hdr,
            frame_bind_group: &self.setup.stub_bind_group,
            world_bind_group: &self.setup.stub_bind_group,
            luts_bind_group: None,
            frame_uniforms: &frame_uniforms,
        };
        self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);

        self.setup.tonemap.render(
            &mut encoder,
            &gpu.queue,
            &self.setup.output_view,
            self.ev100,
            self.tonemap_mode,
        );

        // Copy output → staging.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.setup.output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.setup.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.setup.padded_bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        // Map + read.
        let slice = self.setup.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        // Drive the device until the map completes.
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv()
            .expect("map recv")
            .expect("map success");

        let data = slice.get_mapped_range();
        let bytes_per_row = w * 4;
        let mut out = Vec::with_capacity((bytes_per_row * h) as usize);
        for row in 0..h {
            let start = (row * self.setup.padded_bytes_per_row) as usize;
            let end = start + bytes_per_row as usize;
            out.extend_from_slice(&data[start..end]);
        }
        drop(data);
        self.setup.staging.unmap();
        out
    }
}

