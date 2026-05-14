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

use bytemuck::{bytes_of, Pod, Zeroable};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout,
    BindGroupCache, Config, GpuContext, HdrFramebuffer, PassDescriptor, PassId, PassStage,
    PrepareContext, RenderContext, RenderSubsystem, SubsystemFactory, SurfaceParams,
};

const PASS_GROUND: PassId = 0;

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

/// Build the bind-group layout for the ground subsystem's group 2:
///
///   binding 0: SurfaceParams uniform
///   binding 1: top-down cloud density mask (Phase 12.6 — overcast
///              diffuse modulation). The mask is sampled in the
///              ground shader at the surface point's XZ to determine
///              how much cloud is overhead.
///   binding 2: linear-clamp sampler for the mask.
fn surface_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ground-surface-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<SurfaceParams>() as u64,
                    ),
                },
                count: None,
            },
            // Phase 12.6 — top-down cloud density mask.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: true,
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Procedural ground plane. Constructed via [`GroundSubsystem`]; kept
/// public so tests / host code can build one directly.
///
/// The group-2 bind group is rebuilt each frame in `prepare()` rather
/// than cached, because it references the top-down density mask
/// texture view from the live WeatherState (which can be replaced
/// when synthesis re-runs for a hot-reload).
pub struct PbrGround {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    surface_buf: wgpu::Buffer,
    /// Layout used to rebuild the group-2 bind group each frame.
    surface_layout: wgpu::BindGroupLayout,
    /// Cached sampler for the density-mask texture binding.
    density_mask_sampler: wgpu::Sampler,
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

        // Phase 12.6 — sampler for the top-down density mask. Linear
        // filtering smooths cloud-edge transitions across grid cells;
        // clamp-to-edge means surface points outside the 32 km mask
        // extent get the boundary value (a v1 stationary-camera
        // limitation).
        let density_mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ground-density-mask-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            vertex_buf,
            surface_buf,
            surface_layout,
            density_mask_sampler,
        }
    }
}

/// `RenderSubsystem` wrapper around [`PbrGround`].
pub struct GroundSubsystem {
    inner: PbrGround,
    /// Most-recent SurfaceParams (uploaded each frame in `prepare()`).
    surface: SurfaceParams,
    /// `true` when `[render.subsystems].wet_surface` is on. When off the
    /// ground shader still runs but `prepare()` zeros out the wetness +
    /// snow inputs so the dry BRDF path is taken.
    wet_surface_enabled: bool,
    /// Live group-2 bind group published by `prepare()` for `dispatch_pass`.
    /// Built via a revision-keyed cache so the wgpu hub touch only
    /// happens when the underlying density mask view changes (synthesis
    /// rerun), not every frame.
    live_surface_bg: Option<std::sync::Arc<wgpu::BindGroup>>,
    /// Revision-keyed cache for the surface bind group.
    surface_bg_cache: BindGroupCache<u64>,
}

impl GroundSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        Self {
            inner: PbrGround::new(&gpu.device),
            surface: SurfaceParams::default(),
            wet_surface_enabled: config.render.subsystems.wet_surface,
            live_surface_bg: None,
            surface_bg_cache: BindGroupCache::new(),
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
        self.surface = surface;

        // Phase 12.6 — publish the group-2 bind group for `dispatch_pass`.
        // The entries reference: `surface_buf` (stable handle; per-frame
        // writes don't invalidate the bind group), `mask_view` (only
        // changes when synthesis re-runs and bumps `weather.revision`),
        // and a sampler (stable). Keying the cache on `weather.revision`
        // alone is sufficient.
        let inner = &self.inner;
        let bg = self
            .surface_bg_cache
            .get_or_build(ctx.weather.revision, || {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ground-surface-bg"),
                    layout: &inner.surface_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: inner.surface_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &ctx.weather.textures.top_down_density_mask_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &inner.density_mask_sampler,
                            ),
                        },
                    ],
                })
            });
        self.live_surface_bg = Some(bg);
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![PassDescriptor {
            name: "ground-pbr",
            stage: PassStage::Opaque,
            id: PASS_GROUND,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        // Phase 12.6 — read the live group-2 bind group built in
        // `prepare()` against this frame's density mask.
        let Some(surface_bg) = self.live_surface_bg.as_ref() else {
            // prepare() hasn't run yet (first frame in some headless
            // test paths); skip cleanly.
            return;
        };
        let Some(luts_bg) = ctx.luts_bind_group else {
            // Atmosphere disabled — skip; the shader needs the LUTs.
            return;
        };
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
        pass.set_pipeline(&self.inner.pipeline);
        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
        pass.set_bind_group(1, ctx.world_bind_group, &[]);
        pass.set_bind_group(2, surface_bg.as_ref(), &[]);
        pass.set_bind_group(3, luts_bg, &[]);
        pass.set_vertex_buffer(0, self.inner.vertex_buf.slice(..));
        pass.draw(0..6, 0..1);
    }

    /// Phase 19.A — refresh runtime-tunable flags so the UI subsystem
    /// checkboxes actually take effect mid-session.
    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        self.wet_surface_enabled = config.render.subsystems.wet_surface;
        Ok(())
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
