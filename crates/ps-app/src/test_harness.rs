//! Headless test harness: build a `ps_core::App` with the same factory set
//! as the binary, render one frame to an offscreen `Rgba8Unorm` target
//! sized to a small HDR framebuffer, read back the pixels.
//!
//! Used by `tests/integration.rs`. The windowed binary in `main.rs` does
//! not depend on this module.

use anyhow::{Context, Result};
use glam::Vec4;
use ps_atmosphere::AtmosphereFactory;
use ps_aurora::AuroraFactory;
use ps_backdrop::BackdropFactory;
use ps_bloom::BloomFactory;
use ps_godrays::GodraysFactory;
use ps_lightning::{LightningFactory, LightningPublish};
use ps_clouds::CloudsFactory;
use ps_core::{
    App, AppBuilder, Config, FrameUniforms, GpuContext, HdrFramebufferImpl, PrepareContext,
    RenderContext,
};
use ps_ground::GroundFactory;
use ps_postprocess::{Tonemap, TonemapMode};
use ps_precip::PrecipFactory;
use ps_tint::TintFactory;
use ps_water::WaterFactory;
use ps_windsock::WindsockFactory;
use std::sync::Arc;

use crate::main_helpers::encode_frame_clear;

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
    /// Tone-map pass writing into `output_view`. Shared with the
    /// in-graph `TonemapSubsystem` via Arc.
    pub tonemap: Arc<Tonemap>,
    /// Phase 9.2 auto-exposure compute pass. Shared with the in-graph
    /// `TonemapSubsystem` via Arc.
    pub auto_exposure: Arc<ps_postprocess::AutoExposure>,
    /// Phase 4 §4.2 — bind groups 0 (FrameUniforms) and 1 (WorldUniforms).
    pub bindings: ps_core::FrameWorldBindings,
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
        let tonemap = Arc::new(Tonemap::new(
            &gpu.device,
            &hdr,
            wgpu::TextureFormat::Rgba8Unorm,
        ));
        let auto_exposure = Arc::new(ps_postprocess::AutoExposure::new(&gpu.device, &hdr));
        let bindings = ps_core::FrameWorldBindings::new(&gpu.device);

        Self {
            hdr,
            output,
            output_view,
            staging,
            tonemap,
            auto_exposure,
            bindings,
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
    /// Phase 5: LUTs handle published by `AtmosphereFactory`. `None`
    /// when atmosphere is disabled in config.
    atmosphere_luts: Option<Arc<ps_core::AtmosphereLuts>>,
    /// Phase 12.3: snapshot publish handle from `LightningFactory`.
    /// `None` when lightning is disabled. Read between frames to
    /// splice cloud illumination into FrameUniforms.
    lightning_publish: Option<LightningPublish>,
    /// Phase 9.1: side-channel handle to the in-graph TonemapSubsystem.
    tonemap_handle: ps_postprocess::TonemapHandle,
}

impl HeadlessApp {
    /// Build with the same factory set the binary registers.
    pub fn new(gpu: &GpuContext, config: &Config, setup: TestSetup) -> Result<Self> {
        let (atmosphere_factory, luts_cell) = AtmosphereFactory::new();
        let (lightning_factory, lightning_cell) = LightningFactory::new();
        let clouds_factory = CloudsFactory::with_atmosphere_luts(luts_cell.clone());
        let (tonemap_factory, tonemap_handle) = ps_postprocess::TonemapFactory::new();
        tonemap_handle.inject(
            setup.tonemap.clone(),
            setup.auto_exposure.clone(),
            ps_postprocess::TonemapState {
                ev100: config.render.ev100,
                mode: TonemapMode::from_config(&config.render.tone_mapper),
                auto_exposure_enabled: config.debug.auto_exposure,
            },
        );
        let app = AppBuilder::new()
            .with_factory(Box::new(BackdropFactory))
            .with_factory(Box::new(atmosphere_factory))
            .with_factory(Box::new(GroundFactory))
            .with_factory(Box::new(WaterFactory))
            .with_factory(Box::new(clouds_factory))
            .with_factory(Box::new(PrecipFactory))
            .with_factory(Box::new(TintFactory))
            .with_factory(Box::new(GodraysFactory))
            .with_factory(Box::new(lightning_factory))
            .with_factory(Box::new(AuroraFactory))
            .with_factory(Box::new(BloomFactory))
            .with_factory(Box::new(WindsockFactory))
            .with_factory(Box::new(tonemap_factory))
            .build(config, gpu)
            .context("AppBuilder::build")?;
        let atmosphere_luts = luts_cell
            .lock()
            .map_err(|e| anyhow::anyhow!("luts cell poisoned: {e}"))?
            .clone();
        let lightning_publish = lightning_cell
            .lock()
            .map_err(|e| anyhow::anyhow!("lightning cell poisoned: {e}"))?
            .clone();
        Ok(Self {
            setup,
            app,
            ev100: config.render.ev100,
            tonemap_mode: TonemapMode::from_config(&config.render.tone_mapper),
            last_config: config.clone(),
            atmosphere_luts,
            lightning_publish,
            tonemap_handle,
        })
    }

    /// Read access to the atmosphere LUTs for diagnostic readback in
    /// integration tests. Returns `None` when atmosphere is disabled.
    pub fn atmosphere_luts_for_diag(&self) -> Option<&Arc<ps_core::AtmosphereLuts>> {
        self.atmosphere_luts.as_ref()
    }

    /// Test-only accessor for the inner `ps_core::App`. Used by Phase 9
    /// tests that introspect registered passes.
    pub fn app_for_test(&self) -> &App {
        &self.app
    }

    /// Test-only accessor for the canonical group-0 bind group.
    pub fn frame_bind_group_for_test(&self) -> &wgpu::BindGroup {
        &self.setup.bindings.frame_bind_group
    }

    /// Test-only accessor for the canonical group-1 bind group.
    pub fn world_bind_group_for_test(&self) -> &wgpu::BindGroup {
        &self.setup.bindings.world_bind_group
    }

    /// Test/CLI helper: read back the live HDR target as `f16` RGBA.
    /// Returns the pixels in row-major top-left origin order, length =
    /// `w * h * 4`.
    pub fn read_hdr_for_test(&self, gpu: &GpuContext) -> Vec<half::f16> {
        let (w, h) = self.setup.size;
        let bytes_per_pixel = 8u32;
        let unpadded = w * bytes_per_pixel;
        let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read-hdr-staging"),
            size: (aligned * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read-hdr-copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.setup.hdr.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
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
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("recv").expect("map");
        let bytes = slice.get_mapped_range().to_vec();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let row = &bytes[(y * aligned) as usize..(y * aligned + unpadded) as usize];
            for chunk in row.chunks_exact(2) {
                out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        out
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

    /// Test-only: override the EV100 used by the next render.
    pub fn set_ev100(&mut self, ev100: f32) {
        self.ev100 = ev100;
    }

    /// Render one frame using the supplied `GpuContext`. Returns raw
    /// `Rgba8Unorm` pixels (top-left origin), tightly packed.
    pub fn render_one_frame(&mut self, gpu: &GpuContext) -> Vec<u8> {
        self.render_one_frame_with(gpu, ps_core::camera::FlyCamera::default())
    }

    /// Variant of [`Self::render_one_frame`] that lets the caller supply
    /// a specific camera (useful for tests that want to look at a
    /// particular region of the sky).
    pub fn render_one_frame_with(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
    ) -> Vec<u8> {
        self.render_one_frame_with_surface(gpu, camera, None)
    }

    /// Variant of [`Self::render_one_frame_with`] that lets the caller
    /// inject a custom `SurfaceParams` (otherwise the stub default is
    /// used). Useful for Phase 7 wet-surface / snow tests.
    pub fn render_one_frame_with_surface(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
        surface_override: Option<ps_core::SurfaceParams>,
    ) -> Vec<u8> {
        self.render_one_frame_with_surface_and_mask(gpu, camera, surface_override, None)
    }

    /// Full variant: also overrides the top-down cloud-density mask. The
    /// stub defaults the mask to 1.0 (fully covered); pass `Some(0.0)` to
    /// simulate "no clouds present" for Phase 8 cloud-occlusion tests.
    pub fn render_one_frame_with_surface_and_mask(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
        surface_override: Option<ps_core::SurfaceParams>,
        cloud_mask_override: Option<f32>,
    ) -> Vec<u8> {
        self.render_one_frame_full(
            gpu,
            camera,
            surface_override,
            cloud_mask_override,
            None,
        )
    }

    /// Phase 11.3 — render one frame using a real synthesised
    /// `WeatherState` from a `Scene`. This is the path the headless
    /// `ps-app render` subcommand and the golden-image regression
    /// harness use, because they need the scene's cloud layers + wind
    /// field + density mask to actually reach the GPU (the simpler
    /// stub-based renderers bypass synthesis entirely).
    pub fn render_one_frame_with_scene(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
        scene: &ps_core::Scene,
        world: ps_core::WorldState,
    ) -> Vec<u8> {
        let (w, h) = self.setup.size;
        let aspect = w as f32 / h as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        // Build the live WeatherState from the scene + world.
        let mut weather = ps_synthesis::synthesise(scene, &self.last_config, &world, gpu)
            .expect("synthesise WeatherState");
        weather.sun_direction = world.sun_direction_world;
        weather.sun_illuminance = glam::Vec3::splat(world.toa_illuminance_lux);

        let mut frame_uniforms = FrameUniforms {
            camera_position_world: Vec4::new(
                camera.position.x,
                camera.position.y.max(0.1),
                camera.position.z,
                0.0,
            ),
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: 0.0,
            simulated_seconds: 0.0,
            frame_index: 0,
            ev100: self.ev100,
            ..FrameUniforms::default()
        };
        frame_uniforms.set_matrices(view, proj);
        frame_uniforms.set_sun(
            world.sun_direction_world,
            self.last_config
                .render
                .atmosphere
                .sun_angular_radius_deg
                .to_radians(),
            glam::Vec3::splat(world.toa_illuminance_lux),
            world.toa_illuminance_lux,
        );
        // Phase 12.3 — splice prior-frame lightning snapshot. For
        // single-frame renders (the golden suite) the snapshot is
        // zero, so no cloud-illumination drift here.
        if let Some(publish) = &self.lightning_publish {
            let snap = *publish.lock().expect("lightning publish lock");
            frame_uniforms.lightning_illuminance = snap.illuminance;
            frame_uniforms.lightning_origin_world = snap.origin_world;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test-frame"),
            });
        crate::main_helpers::encode_frame_clear(
            &mut encoder,
            &self.setup.hdr,
            !self.last_config.render.subsystems.backdrop,
            self.last_config.render.clear_color,
        );
        self.setup
            .bindings
            .write(&gpu.queue, &frame_uniforms, &weather.atmosphere);
        let luts_ref = self.atmosphere_luts.as_deref();
        let luts_bind_group = luts_ref.map(|l| &l.bind_group);
        // Tonemap state per current ev100/mode.
        self.tonemap_handle
            .set_state(ps_postprocess::TonemapState {
                ev100: self.ev100,
                mode: self.tonemap_mode,
                auto_exposure_enabled: false,
            });
        let mut prepare_ctx = PrepareContext {
            device: &gpu.device,
            queue: &gpu.queue,
            world: &world,
            weather: &weather,
            frame_uniforms: &frame_uniforms,
            atmosphere_luts: luts_ref,
            dt_seconds: 1.0 / 60.0,
        };
        let render_ctx = RenderContext {
            device: &gpu.device,
            queue: &gpu.queue,
            framebuffer: &self.setup.hdr,
            frame_bind_group: &self.setup.bindings.frame_bind_group,
            world_bind_group: &self.setup.bindings.world_bind_group,
            luts_bind_group,
            frame_uniforms: &frame_uniforms,
            weather: &weather,
            tonemap_target: Some(&self.setup.output_view),
            tonemap_target_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);

        // Copy output → staging (same as render_one_frame_full).
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

        let slice = self.setup.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("recv").expect("map");
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

    /// Phase 13.7 — render one frame of an animation sequence with
    /// an explicit `simulated_seconds` value and an optional cached
    /// `WeatherState`. When `cached_weather` is `Some`, the call
    /// **does not** re-run synthesis (the caller is responsible for
    /// having created the cached state with `ps_synthesis::synthesise`
    /// against the same `scene` + `config`). When `None`, this method
    /// synthesises fresh each frame — slower, but advances any
    /// state-dependent synthesised quantities (e.g. randomly seeded
    /// per-frame elements).
    ///
    /// In both cases `world.sun_direction` / `toa_illuminance` are
    /// updated each call so the wall clock animates correctly.
    pub fn render_animation_frame(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
        scene: &ps_core::Scene,
        world: ps_core::WorldState,
        simulated_seconds: f32,
        frame_index: u32,
        cached_weather: Option<&mut ps_core::WeatherState>,
    ) -> Vec<u8> {
        let (w, h) = self.setup.size;
        let aspect = w as f32 / h as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);

        // Either reuse the cached WeatherState (synthesise-once) or
        // build a fresh one (default — picks up per-frame changes in
        // scene state). For the cached path we still refresh the sun
        // direction / illuminance so the wall clock animates.
        let mut owned: Option<ps_core::WeatherState> = None;
        let weather: &mut ps_core::WeatherState = if let Some(w) = cached_weather {
            w.sun_direction = world.sun_direction_world;
            w.sun_illuminance = glam::Vec3::splat(world.toa_illuminance_lux);
            w
        } else {
            let mut fresh = ps_synthesis::synthesise(scene, &self.last_config, &world, gpu)
                .expect("synthesise WeatherState");
            fresh.sun_direction = world.sun_direction_world;
            fresh.sun_illuminance = glam::Vec3::splat(world.toa_illuminance_lux);
            owned = Some(fresh);
            owned.as_mut().expect("just-set")
        };

        let mut frame_uniforms = FrameUniforms {
            camera_position_world: Vec4::new(
                camera.position.x,
                camera.position.y.max(0.1),
                camera.position.z,
                0.0,
            ),
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: simulated_seconds,
            simulated_seconds,
            frame_index,
            ev100: self.ev100,
            ..FrameUniforms::default()
        };
        frame_uniforms.set_matrices(view, proj);
        frame_uniforms.set_sun(
            world.sun_direction_world,
            self.last_config
                .render
                .atmosphere
                .sun_angular_radius_deg
                .to_radians(),
            glam::Vec3::splat(world.toa_illuminance_lux),
            world.toa_illuminance_lux,
        );
        if let Some(publish) = &self.lightning_publish {
            let snap = *publish.lock().expect("lightning publish lock");
            frame_uniforms.lightning_illuminance = snap.illuminance;
            frame_uniforms.lightning_origin_world = snap.origin_world;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("animation-frame"),
            });
        crate::main_helpers::encode_frame_clear(
            &mut encoder,
            &self.setup.hdr,
            !self.last_config.render.subsystems.backdrop,
            self.last_config.render.clear_color,
        );
        self.setup
            .bindings
            .write(&gpu.queue, &frame_uniforms, &weather.atmosphere);
        let luts_ref = self.atmosphere_luts.as_deref();
        let luts_bind_group = luts_ref.map(|l| &l.bind_group);
        self.tonemap_handle
            .set_state(ps_postprocess::TonemapState {
                ev100: self.ev100,
                mode: self.tonemap_mode,
                auto_exposure_enabled: false,
            });
        {
            let mut prepare_ctx = PrepareContext {
                device: &gpu.device,
                queue: &gpu.queue,
                world: &world,
                weather,
                frame_uniforms: &frame_uniforms,
                atmosphere_luts: luts_ref,
                dt_seconds: 1.0 / 60.0,
            };
            let render_ctx = RenderContext {
                device: &gpu.device,
                queue: &gpu.queue,
                framebuffer: &self.setup.hdr,
                frame_bind_group: &self.setup.bindings.frame_bind_group,
                world_bind_group: &self.setup.bindings.world_bind_group,
                luts_bind_group,
                frame_uniforms: &frame_uniforms,
                weather,
                tonemap_target: Some(&self.setup.output_view),
                tonemap_target_format: wgpu::TextureFormat::Rgba8Unorm,
            };
            self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);
        }

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
        let slice = self.setup.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("recv").expect("map");
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
        // If we allocated a fresh WeatherState, dropping `owned` here
        // frees the underlying GPU resources before the next frame
        // re-allocates them. The cached path leaves the caller's
        // WeatherState intact.
        drop(owned);
        out
    }

    /// Full-fat variant for tests that need to override the world state
    /// (e.g. twilight tests that move the sun below the horizon).
    pub fn render_one_frame_full(
        &mut self,
        gpu: &GpuContext,
        camera: ps_core::camera::FlyCamera,
        surface_override: Option<ps_core::SurfaceParams>,
        cloud_mask_override: Option<f32>,
        world_override: Option<ps_core::WorldState>,
    ) -> Vec<u8> {
        let (w, h) = self.setup.size;
        let aspect = w as f32 / h as f32;
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        // World state for tests defaults to J2000.0 noon at the equator —
        // gives a sun overhead, useful for sky tests.
        let world = world_override.unwrap_or_default();
        let mut weather = ps_core::WeatherState::stub_for_tests(gpu);
        // Sync the stubbed weather to the (possibly overridden) world's
        // sun direction so the atmosphere LUT bake reflects the test
        // scenario.
        weather.sun_direction = world.sun_direction_world;
        weather.sun_illuminance = glam::Vec3::splat(world.toa_illuminance_lux);
        if let Some(s) = surface_override {
            weather.surface = s;
        }
        if let Some(mask_value) = cloud_mask_override {
            let v = (mask_value.clamp(0.0, 1.0) * 255.0).round() as u8;
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &weather.textures.top_down_density_mask,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                std::slice::from_ref(&v),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let mut frame_uniforms = FrameUniforms {
            // Use the supplied camera position so tests can place the
            // camera arbitrarily (the sky-view LUT depends on radius).
            camera_position_world: Vec4::new(
                camera.position.x,
                camera.position.y.max(0.1),
                camera.position.z,
                0.0,
            ),
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: 0.0,
            simulated_seconds: 0.0,
            frame_index: 0,
            ev100: self.ev100,
            ..FrameUniforms::default()
        };
        frame_uniforms.set_matrices(view, proj);
        frame_uniforms.set_sun(
            world.sun_direction_world,
            self.last_config
                .render
                .atmosphere
                .sun_angular_radius_deg
                .to_radians(),
            glam::Vec3::splat(world.toa_illuminance_lux),
            world.toa_illuminance_lux,
        );
        // Phase 12.3 — splice prior-frame lightning snapshot. For
        // single-frame renders (the golden suite) the snapshot is
        // zero, so no cloud-illumination drift here.
        if let Some(publish) = &self.lightning_publish {
            let snap = *publish.lock().expect("lightning publish lock");
            frame_uniforms.lightning_illuminance = snap.illuminance;
            frame_uniforms.lightning_origin_world = snap.origin_world;
        }

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
        // Phase 5 diagnostic (test path): eprintln since the test framework
        // doesn't initialise a tracing subscriber.
        eprintln!(
            "[phase5_check] camera_pos={:?} sun_dir={:?} sun_illum={:?} atmo_planet_r={}",
            frame_uniforms.camera_position_world,
            frame_uniforms.sun_direction,
            frame_uniforms.sun_illuminance,
            weather.atmosphere.planet_radius_m,
        );
        // Phase 4: upload bind groups 0 (FrameUniforms) and 1 (WorldUniforms).
        self.setup
            .bindings
            .write(&gpu.queue, &frame_uniforms, &weather.atmosphere);
        let luts_ref = self.atmosphere_luts.as_deref();
        let luts_bind_group = luts_ref.map(|l| &l.bind_group);
        let mut prepare_ctx = PrepareContext {
            device: &gpu.device,
            queue: &gpu.queue,
            world: &world,
            weather: &weather,
            frame_uniforms: &frame_uniforms,
            atmosphere_luts: luts_ref,
            dt_seconds: 1.0 / 60.0,
        };
        let render_ctx = RenderContext {
            device: &gpu.device,
            queue: &gpu.queue,
            framebuffer: &self.setup.hdr,
            frame_bind_group: &self.setup.bindings.frame_bind_group,
            world_bind_group: &self.setup.bindings.world_bind_group,
            luts_bind_group,
            frame_uniforms: &frame_uniforms,
            weather: &weather,
            tonemap_target: Some(&self.setup.output_view),
            tonemap_target_format: wgpu::TextureFormat::Rgba8Unorm,
        };
        // Phase 9.1: push per-frame state into the in-graph tonemap
        // subsystem; the tonemap pass runs at PassStage::ToneMap inside
        // App::frame writing to render_ctx.tonemap_target (the offscreen
        // output view).
        self.tonemap_handle
            .set_state(ps_postprocess::TonemapState {
                ev100: self.ev100,
                mode: self.tonemap_mode,
                auto_exposure_enabled: self.last_config.debug.auto_exposure,
            });
        self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);

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
        // Phase 9.2: drain any pending auto-exposure read-back.
        self.tonemap_handle.drain_auto_exposure(gpu);

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
        rx.recv().expect("map recv").expect("map success");

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
