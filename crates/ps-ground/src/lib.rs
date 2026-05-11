//! Phase 7 ground subsystem: PBR ground with wet surface and snow.
//!
//! Replaces the Phase 0 procedural checker. The fragment shader
//! implements:
//! - GGX/Smith specular + Lambertian diffuse over a 5 m Voronoi-tiled
//!   3-entry palette (plan §7.1).
//! - Lagarde 2013 wet surface chain (plan §7.2): darkened albedo,
//!   reduced roughness, optional thin water layer for puddles above
//!   `surface.puddle_start`.
//! - Snow layer gated by `temp_c < 0.5 && snow_depth_m > 0` (plan §7.3).
//! - Aerial perspective applied in-shader from the AP LUT (plan §7.4).
//!
//! Bind groups:
//! - 0 — `FrameUniforms` (engine-wide).
//! - 1 — `WorldUniforms` (engine-wide).
//! - 2 — ground-owned `SurfaceParams` uniform (this crate).
//! - 3 — atmosphere LUTs (Phase 5).
//!
//! `SurfaceParams` is uploaded each frame in `prepare()` from
//! `PrepareContext::weather.surface`.

#![deny(missing_docs)]

use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout, Config,
    GpuContext, HdrFramebuffer, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory, SurfaceParams,
};

/// Baked shader source — used as the fallback when no runtime
/// override is registered (the default for headless tests and
/// production builds without `[debug] shader_hot_reload`).
const SHADER_BAKED: &str = include_str!("../../../shaders/ground/pbr.wgsl");
/// Path of the shader file relative to `shaders/`. The hot-reload
/// loader uses this to find the live source on disk.
const SHADER_REL: &str = "ground/pbr.wgsl";
const QUAD_HALF_EXTENT_M: f32 = 100_000.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// Stable subsystem name (matches `[render.subsystems].ground`).
pub const NAME: &str = "ground";

/// Build the bind-group layout for the ground subsystem's group 2: a
/// single `SurfaceParams` uniform.
fn surface_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ground-surface-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<SurfaceParams>() as u64),
            },
            count: None,
        }],
    })
}

/// Procedural ground plane. Constructed via [`GroundSubsystem`]; kept
/// public so tests / host code can build one directly.
pub struct PbrGround {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    surface_buf: wgpu::Buffer,
    surface_bg: wgpu::BindGroup,
}

impl PbrGround {
    /// Build the pipeline, vertex buffer, and surface uniform buffer +
    /// bind group.
    pub fn new(device: &wgpu::Device) -> Self {
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_LUT_SAMPLING_WGSL,
            &live_src,
        ]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground/pbr.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let surface_layout = surface_bind_group_layout(device);
        let lut_layout = atmosphere_lut_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ground-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&surface_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ground-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HdrFramebuffer::COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let h = QUAD_HALF_EXTENT_M;
        let vertices: [Vertex; 6] = [
            Vertex { position: [-h, 0.0, -h] },
            Vertex { position: [h, 0.0, -h] },
            Vertex { position: [h, 0.0, h] },
            Vertex { position: [-h, 0.0, -h] },
            Vertex { position: [h, 0.0, h] },
            Vertex { position: [-h, 0.0, h] },
        ];
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-vb"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        vertex_buf.unmap();

        let surface_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-surface-ub"),
            size: std::mem::size_of::<SurfaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ground-surface-bg"),
            layout: &surface_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: surface_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            vertex_buf,
            surface_buf,
            surface_bg,
        }
    }
}

/// `RenderSubsystem` wrapper around [`PbrGround`].
pub struct GroundSubsystem {
    enabled: bool,
    inner: Arc<PbrGround>,
    /// Most-recent SurfaceParams (uploaded each frame in `prepare()`).
    surface: Mutex<SurfaceParams>,
    /// `true` when `[render.subsystems].wet_surface` is on. When off the
    /// ground shader still runs but `prepare()` zeros out the wetness +
    /// snow inputs so the dry BRDF path is taken.
    wet_surface_enabled: bool,
}

impl GroundSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        Self {
            enabled: true,
            inner: Arc::new(PbrGround::new(&gpu.device)),
            surface: Mutex::new(SurfaceParams::default()),
            wet_surface_enabled: config.render.subsystems.wet_surface,
        }
    }
}

impl RenderSubsystem for GroundSubsystem {
    fn name(&self) -> &'static str {
        "ground"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let mut surface = ctx.weather.surface;
        // The wet_surface master toggle (Phase 7.2 / 7.3) gates only the
        // wetness + puddle features — snow is a distinct ground material
        // and stays driven by the scene's snow_depth_m + temperature_c.
        if !self.wet_surface_enabled {
            surface.ground_wetness = 0.0;
            surface.puddle_coverage = 0.0;
        }
        ctx.queue
            .write_buffer(&self.inner.surface_buf, 0, bytes_of(&surface));
        *self.surface.lock().expect("ground: surface lock") = surface;
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let inner = self.inner.clone();
        vec![RegisteredPass {
            name: "ground-pbr",
            stage: PassStage::Opaque,
            run: Box::new(move |encoder, ctx| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ground-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ctx.framebuffer.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &ctx.framebuffer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let Some(luts_bg) = ctx.luts_bind_group else {
                    // Atmosphere disabled — skip; the shader needs the LUTs.
                    return;
                };
                pass.set_pipeline(&inner.pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                pass.set_bind_group(1, ctx.world_bind_group, &[]);
                pass.set_bind_group(2, &inner.surface_bg, &[]);
                pass.set_bind_group(3, luts_bg, &[]);
                pass.set_vertex_buffer(0, inner.vertex_buf.slice(..));
                pass.draw(0..6, 0..1);
            }),
        }]
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`.
pub struct GroundFactory;

impl SubsystemFactory for GroundFactory {
    fn name(&self) -> &'static str {
        "ground"
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(GroundSubsystem::new(config, gpu)))
    }
}
