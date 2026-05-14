//! PedalSky precipitation subsystem (Phase 8: Schneider/Hillaire-style
//! near rain particles + far-rain screen-space streaks + snow + cloud
//! occlusion).
//!
//! Pass schedule:
//! - `Compute` — particle advance (pos += v·dt, respawn outside cylinder).
//! - `Translucent` — instanced quad render of near particles.
//! - `Translucent` — three layered far-rain streak fullscreen quads.
//!
//! All passes early-return when:
//! - intensity_mm_per_h == 0 (no precipitation), or
//! - the scene has zero clouds (overcast_field is uniformly 0 →
//!   shader-side `cloud_mask * intensity` collapses to 0).
//!
//! `[render.subsystems].precipitation` is the master toggle — when off
//! the factory returns the no-op subsystem variant to skip allocation.

#![deny(missing_docs)]

pub mod uniforms;

use std::sync::Arc;

use bytemuck::{bytes_of, Pod, Zeroable};
use ps_core::{
    frame_bind_group_layout, Config, GpuContext, HdrFramebuffer, PassDescriptor, PassId,
    PassStage, PrepareContext, RenderContext, RenderSubsystem, SubsystemFactory,
};

const PASS_ADVANCE: PassId = 0;
const PASS_NEAR: PassId = 1;
const PASS_FAR: PassId = 2;

pub use uniforms::{FarRainLayerGpu, PrecipUniformsGpu};

/// Stable subsystem name (matches `[render.subsystems].precipitation`).
pub const NAME: &str = "precipitation";

const ADVANCE_BAKED: &str =
    include_str!("../../../shaders/precip/particle_advance.comp.wgsl");
const ADVANCE_REL: &str = "precip/particle_advance.comp.wgsl";
const RENDER_BAKED: &str = include_str!("../../../shaders/precip/particle_render.wgsl");
const RENDER_REL: &str = "precip/particle_render.wgsl";
const FAR_RAIN_BAKED: &str = include_str!("../../../shaders/precip/far_rain.wgsl");
const FAR_RAIN_REL: &str = "precip/far_rain.wgsl";

const SPAWN_RADIUS_M: f32 = 50.0;
const SPAWN_TOP_M: f32 = 30.0;
const RAIN_FALL_MPS: f32 = 6.0;
const SNOW_FALL_MPS: f32 = 1.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct Particle {
    position: [f32; 3],
    age: f32,
    velocity: [f32; 3],
    kind: u32,
}

/// One particle pool (rain or snow). Owns the storage buffer + the
/// uniform buffer + the draw-indirect args buffer (whose `instance_count`
/// the compute shader atomically increments per live particle). Both
/// compute and render bind groups are rebuilt every frame in `prepare()`
/// because they include the live wind_field / density-mask views from
/// WeatherState.
struct ParticlePool {
    particle_buf: wgpu::Buffer,
    uniforms_buf: wgpu::Buffer,
    /// 16 B buffer holding `(vertex_count=6, instance_count=atomic,
    /// first_vertex=0, first_instance=0)`. INDIRECT|STORAGE|COPY_DST.
    /// Reset to `[6, 0, 0, 0]` by the host each frame before the
    /// compute pass; render uses draw_indirect at offset 0.
    draw_args_buf: wgpu::Buffer,
    count: u32,
    kind: u32,
    fall_mps: f32,
}

/// Far-rain layer (per-layer uniform). The bind group itself is rebuilt
/// every frame against the live density-mask view.
struct FarRainLayer {
    layer_buf: wgpu::Buffer,
    #[allow(dead_code)]
    depth_m: f32,
}

/// Phase 8 precipitation subsystem.
pub struct PrecipSubsystem {
    /// Two pools: one for rain, one for snow. Both are always allocated;
    /// only the kind matching the current `PrecipKind` is dispatched.
    /// `Arc` is retained because pool buffers are read by the render
    /// bind groups; reconfigure swaps the Arc when `near_particle_count`
    /// changes.
    rain_pool: Arc<ParticlePool>,
    snow_pool: Arc<ParticlePool>,

    /// Pre-built far-rain layers. Reconfigure replaces the Vec when
    /// `far_layers` changes.
    far_layers: Vec<Arc<FarRainLayer>>,

    /// Compute pipeline (shared between rain + snow pools).
    advance_pipeline: wgpu::ComputePipeline,
    /// Particle render pipeline.
    render_pipeline: wgpu::RenderPipeline,
    /// Far-rain render pipeline.
    far_pipeline: wgpu::RenderPipeline,

    /// Most-recent kind from WeatherState (0 none, 1 rain, 2 snow, 3 sleet).
    state: PrecipState,

    /// Layouts + sampler used to rebuild bind groups in `prepare()`.
    bg_builder: BindGroupBuilder,

    /// Live compute bind group for the active pool (rebuilt each frame
    /// against the current wind_field view).
    live_compute_bg: Option<wgpu::BindGroup>,

    /// Live render bind group (rebuilt each frame against the current
    /// density mask view).
    live_render_bg: Option<wgpu::BindGroup>,

    /// Live far-rain bind groups (one per layer).
    live_far_bgs: Vec<wgpu::BindGroup>,

    /// Plan §Cross-Cutting/Determinism — user-supplied seed XOR'd into
    /// the per-particle respawn jitter. Sourced from
    /// `config.debug.seed` (truncated to 32 bits).
    user_seed: u32,
}

#[derive(Default)]
struct PrecipState {
    kind: u32,
    intensity_mm_per_h: f32,
}

impl PrecipSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let queue = &gpu.queue;

        let count = config.render.precip.near_particle_count.max(64);

        // Compute bind layout: storage particle buffer + uniform +
        // wind_field 3D texture + sampler. The 3D texture is rebound
        // each frame because its view is owned by WeatherState.
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precip-compute-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Render bind layout: storage particle (read-only) + uniform +
        // top-down density mask + sampler.
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precip-render-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Far-rain bind layout: PrecipUniforms + density mask + sampler +
        // LayerUniform.
        let far_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("precip-far-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Compute pipeline (one, shared).
        let advance_src = ps_core::shaders::load_shader(ADVANCE_REL, ADVANCE_BAKED);
        let advance_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("precip::advance"),
            source: wgpu::ShaderSource::Wgsl(advance_src.into()),
        });
        let advance_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("precip::advance-pl"),
            bind_group_layouts: &[Some(&compute_layout)],
            immediate_size: 0,
        });
        let advance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("precip::advance-pipeline"),
            layout: Some(&advance_pl_layout),
            module: &advance_module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Particle render pipeline.
        let render_main = ps_core::shaders::load_shader(RENDER_REL, RENDER_BAKED);
        let render_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &render_main,
        ]);
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("precip::render"),
            source: wgpu::ShaderSource::Wgsl(render_src.into()),
        });
        let frame_layout = frame_bind_group_layout(device);
        let render_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("precip::render-pl"),
            bind_group_layouts: &[Some(&frame_layout), Some(&render_layout)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("precip::render-pipeline"),
                layout: Some(&render_pl_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HdrFramebuffer::COLOR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
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
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Far rain pipeline.
        let far_main = ps_core::shaders::load_shader(FAR_RAIN_REL, FAR_RAIN_BAKED);
        let far_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &far_main,
        ]);
        let far_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("precip::far"),
            source: wgpu::ShaderSource::Wgsl(far_src.into()),
        });
        let far_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("precip::far-pl"),
            bind_group_layouts: &[Some(&frame_layout), Some(&far_layout)],
            immediate_size: 0,
        });
        let far_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("precip::far-pipeline"),
                layout: Some(&far_pl_layout),
                vertex: wgpu::VertexState {
                    module: &far_module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &far_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HdrFramebuffer::COLOR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
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
                    depth_write_enabled: Some(false),
                    // Reverse-Z: draw in front of farther geometry only.
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Sampler shared by render + far passes.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("precip-density-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Per-pool resources. The bind groups need a top-down density
        // texture view at construction time; we use a 1x1 placeholder that
        // gets rebuilt every frame in `prepare()`.
        let placeholder_mask = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("precip-mask-placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_view =
            placeholder_mask.create_view(&wgpu::TextureViewDescriptor::default());

        let rain_pool = make_pool(device, queue, count, 0, RAIN_FALL_MPS);
        let snow_pool = make_pool(device, queue, count, 1, SNOW_FALL_MPS);

        let far_layers = make_far_layers(device, queue, config.render.precip.far_layers);
        let _ = placeholder_view;

        // Wind sampler (linear, clamp-to-edge) for the compute shader.
        let wind_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("precip-wind-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Stash the layouts + samplers on the subsystem so prepare() can
        // rebuild bind groups against the live density mask + wind field
        // views from WeatherState.
        let bg_builder = BindGroupBuilder {
            compute_layout,
            render_layout,
            far_layout,
            sampler,
            wind_sampler,
        };

        Self {
            rain_pool,
            snow_pool,
            far_layers,
            advance_pipeline,
            render_pipeline,
            far_pipeline,
            state: PrecipState::default(),
            bg_builder,
            live_compute_bg: None,
            live_render_bg: None,
            live_far_bgs: Vec::new(),
            user_seed: config.debug.seed as u32,
        }
    }
}

struct BindGroupBuilder {
    compute_layout: wgpu::BindGroupLayout,
    render_layout: wgpu::BindGroupLayout,
    far_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Sampler used by the compute shader's wind_field lookup. Linear
    /// filtering so wind varies smoothly across the texture.
    wind_sampler: wgpu::Sampler,
}

impl PrecipSubsystem {
    /// Build the render bind group for `pool` with `mask_view` plumbed in.
    fn build_render_bg(
        device: &wgpu::Device,
        builder: &BindGroupBuilder,
        pool: &ParticlePool,
        mask_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precip-render-bg-live"),
            layout: &builder.render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&builder.sampler),
                },
            ],
        })
    }

    /// Build the compute bind group for `pool` with `wind_view` plumbed in.
    fn build_compute_bg(
        device: &wgpu::Device,
        builder: &BindGroupBuilder,
        pool: &ParticlePool,
        wind_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precip-compute-bg-live"),
            layout: &builder.compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pool.particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool.uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(wind_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&builder.wind_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pool.draw_args_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn build_far_bg(
        device: &wgpu::Device,
        builder: &BindGroupBuilder,
        active_pool: &ParticlePool,
        layer: &FarRainLayer,
        mask_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("precip-far-bg-live"),
            layout: &builder.far_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: active_pool.uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&builder.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer.layer_buf.as_entire_binding(),
                },
            ],
        })
    }
}

impl RenderSubsystem for PrecipSubsystem {
    fn name(&self) -> &'static str {
        "precipitation"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let intensity = ctx.weather.surface.precip_intensity_mm_per_h.max(0.0);
        let kind = ctx.weather.surface.precip_kind as u32;
        self.state = PrecipState {
            kind,
            intensity_mm_per_h: intensity,
        };

        // Wind sample at the camera. For Phase 8 we use the surface wind
        // (10 m AGL) — the wind_field 3D texture sampling at altitude
        // would require either a CPU-side trilinear lookup or another
        // bind group; Phase 3's surface scalar is correct for near-camera
        // particles.
        let surface = ctx.weather.surface;
        let wind_dir_rad = surface.wind_dir_deg.to_radians();
        // Meteorological convention: wind_dir_deg is the direction the
        // wind comes from, so the velocity vector points opposite.
        let wind_vec = [
            -surface.wind_speed_mps * wind_dir_rad.sin(),
            0.0,
            -surface.wind_speed_mps * wind_dir_rad.cos(),
            0.0,
        ];
        let camera = [
            ctx.frame_uniforms.camera_position_world.x,
            ctx.frame_uniforms.camera_position_world.y,
            ctx.frame_uniforms.camera_position_world.z,
            0.0,
        ];

        for (pool, active_kind) in [(&self.rain_pool, 0u32), (&self.snow_pool, 1u32)] {
            // Only feed live intensity to the matching pool; the other
            // gets zero so its particles are still updated (cheap) but
            // not rendered.
            let live = match (kind, active_kind) {
                (1, 0) | (3, 0) => intensity, // rain or sleet → rain
                (2, 1) | (3, 1) => intensity, // snow or sleet → snow
                _ => 0.0,
            };
            let u = PrecipUniformsGpu {
                camera_position: camera,
                wind_velocity: wind_vec,
                intensity_mm_per_h: live,
                dt_seconds: ctx.dt_seconds,
                simulated_seconds: ctx.frame_uniforms.simulated_seconds,
                kind: pool.kind,
                particle_count: pool.count,
                spawn_radius_m: SPAWN_RADIUS_M,
                spawn_top_m: SPAWN_TOP_M,
                fall_speed_mps: pool.fall_mps,
                user_seed: self.user_seed,
                _pad: [0.0; 3],
            };
            ctx.queue.write_buffer(&pool.uniforms_buf, 0, bytes_of(&u));
        }

        // Rebuild bind groups against the live texture views from
        // WeatherState (density mask for render + far, wind_field for
        // compute). Both views may have been replaced by hot-reload.
        let mask_view = &ctx.weather.textures.overcast_field_view;
        let wind_view = &ctx.weather.textures.wind_field_view;
        let active_pool: &ParticlePool = match kind {
            2 => &self.snow_pool,
            _ => &self.rain_pool,
        };
        // Reset the draw-indirect args for the active pool. Vertex count
        // is fixed at 6 (the quad); instance_count starts at 0 and the
        // compute shader atomically increments it per live particle.
        // first_vertex and first_instance are 0.
        let reset_args: [u32; 4] = [6, 0, 0, 0];
        ctx.queue.write_buffer(
            &active_pool.draw_args_buf,
            0,
            bytemuck::cast_slice(&reset_args),
        );
        self.live_compute_bg = Some(Self::build_compute_bg(
            ctx.device,
            &self.bg_builder,
            active_pool,
            wind_view,
        ));
        self.live_render_bg = Some(Self::build_render_bg(
            ctx.device,
            &self.bg_builder,
            active_pool,
            mask_view,
        ));
        self.live_far_bgs = self
            .far_layers
            .iter()
            .map(|layer| {
                Self::build_far_bg(ctx.device, &self.bg_builder, active_pool, layer, mask_view)
            })
            .collect();
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![
            PassDescriptor {
                name: "precip::advance",
                stage: PassStage::Compute,
                id: PASS_ADVANCE,
            },
            PassDescriptor {
                name: "precip::near",
                stage: PassStage::Translucent,
                id: PASS_NEAR,
            },
            PassDescriptor {
                name: "precip::far",
                stage: PassStage::Translucent,
                id: PASS_FAR,
            },
        ]
    }

    fn dispatch_pass(
        &mut self,
        id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        // Common skip — both renderable passes are no-ops at zero
        // intensity.
        if self.state.intensity_mm_per_h <= 0.0 {
            return;
        }
        let active_pool: &ParticlePool = match self.state.kind {
            2 => &self.snow_pool,
            _ => &self.rain_pool,
        };
        match id {
            PASS_ADVANCE => {
                let Some(bg) = self.live_compute_bg.as_ref() else {
                    return;
                };
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("precip::advance"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.advance_pipeline);
                pass.set_bind_group(0, bg, &[]);
                let groups = active_pool.count.div_ceil(64);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            PASS_NEAR => {
                let Some(bg) = self.live_render_bg.as_ref() else {
                    return;
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("precip::near"),
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
                pass.set_pipeline(&self.render_pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                pass.set_bind_group(1, bg, &[]);
                // draw_indirect reads (vertex_count, instance_count,
                // first_vertex, first_instance) from the args buffer
                // the compute shader just populated atomically (plan
                // §8.1).
                pass.draw_indirect(&active_pool.draw_args_buf, 0);
            }
            PASS_FAR => {
                if self.live_far_bgs.is_empty() {
                    return;
                }
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("precip::far"),
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
                pass.set_pipeline(&self.far_pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                for bg in &self.live_far_bgs {
                    pass.set_bind_group(1, bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            _ => panic!("precip: unknown pass id {id}"),
        }
    }

    fn reconfigure(&mut self, config: &Config, gpu: &GpuContext) -> anyhow::Result<()> {
        // Plan §10.3: when near_particle_count or far_layers changes,
        // rebuild the corresponding GPU resources so the new sliders
        // take effect on the next frame's prepare/render.
        let new_count = config.render.precip.near_particle_count.max(64);
        if new_count != self.rain_pool.count {
            self.rain_pool = make_pool(&gpu.device, &gpu.queue, new_count, 0, RAIN_FALL_MPS);
            self.snow_pool = make_pool(&gpu.device, &gpu.queue, new_count, 1, SNOW_FALL_MPS);
        }
        let cur_far = self.far_layers.len() as u32;
        if config.render.precip.far_layers != cur_far {
            self.far_layers =
                make_far_layers(&gpu.device, &gpu.queue, config.render.precip.far_layers);
        }
        Ok(())
    }
}

/// Factory wired by `AppBuilder`.
pub struct PrecipFactory;

impl SubsystemFactory for PrecipFactory {
    fn name(&self) -> &'static str {
        "precipitation"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(PrecipSubsystem::new(config, gpu)))
    }
}

impl PrecipSubsystem {
    /// Test helper: returns the rain particle pool's count.
    pub fn rain_particle_count(&self) -> u32 {
        self.rain_pool.count
    }
}

fn make_pool(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    count: u32,
    kind: u32,
    fall_mps: f32,
) -> Arc<ParticlePool> {
    // Allocate the storage buffer with all particles flagged as
    // "fresh" (age < 0) so the first compute dispatch respawns them
    // inside the cylinder.
    let mut initial = vec![
        Particle {
            position: [0.0; 3],
            age: -1.0,
            velocity: [0.0; 3],
            kind,
        };
        count as usize
    ];
    for (i, p) in initial.iter_mut().enumerate() {
        p.age = -1.0 - i as f32 * 1e-6;
    }
    let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("precip-particle-buf"),
        size: std::mem::size_of_val(initial.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&particle_buf, 0, bytemuck::cast_slice(&initial));
    let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("precip-uniforms-buf"),
        size: std::mem::size_of::<PrecipUniformsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let draw_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("precip-draw-args"),
        size: 16,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Arc::new(ParticlePool {
        particle_buf,
        uniforms_buf,
        draw_args_buf,
        count,
        kind,
        fall_mps,
    })
}

fn make_far_layers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: u32,
) -> Vec<Arc<FarRainLayer>> {
    // Plan §8.2 specifies depths 50/200/1000 m for 3 layers. For other
    // counts we interpolate logarithmically between 50 m and 1000 m so
    // the spec defaults are recovered exactly when n=3.
    let n = n.max(1);
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        // log-spaced depth.
        let depth_m = 50.0_f32 * (1000.0_f32 / 50.0_f32).powf(t);
        // Density falls with depth; alpha similarly.
        let density = 60.0 * (25.0 / 60.0_f32).powf(t);
        let length_px = 120.0 * (50.0 / 120.0_f32).powf(t);
        let alpha = 1.0 * (0.4_f32 / 1.0).powf(t);
        let layer = FarRainLayerGpu {
            depth_m,
            streak_density: density,
            streak_length_px: length_px,
            intensity_scale: alpha,
        };
        let layer_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("precip-far-layer-buf"),
            size: std::mem::size_of::<FarRainLayerGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&layer_buf, 0, bytes_of(&layer));
        out.push(Arc::new(FarRainLayer { layer_buf, depth_m }));
    }
    out
}
