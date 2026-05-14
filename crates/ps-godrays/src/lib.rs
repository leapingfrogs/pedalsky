//! Phase 12.4 — screen-space crepuscular rays / godrays.
//!
//! Tatarchuk-style radial blur of the HDR target toward the sun's
//! screen position, additively blended back into the HDR target
//! before tone-mapping.
//!
//! Pipeline: two passes, both at `PassStage::PostProcess`:
//!
//!   1. `radial`    — reads the HDR target (sampled via a scratch
//!                    copy), runs N radial samples toward the sun
//!                    NDC, writes the accumulation to a half-res
//!                    scratch RT.
//!   2. `composite` — reads the half-res RT and blends additively
//!                    into the HDR target.
//!
//! The pass is a no-op when the sun is below the horizon or off
//! screen — those cases write zero into the scratch and the
//! composite blend then adds nothing.

#![deny(missing_docs)]

use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use ps_core::{
    Config, GpuContext, HdrFramebuffer, PassDescriptor, PassId, PassStage, PrepareContext,
    RenderContext, RenderSubsystem, SubsystemFactory,
};

const PASS_RADIAL: PassId = 0;
const PASS_COMPOSITE: PassId = 1;

const RADIAL_BAKED: &str = include_str!("../../../shaders/godrays/radial.wgsl");
const RADIAL_REL: &str = "godrays/radial.wgsl";
const COMPOSITE_BAKED: &str = include_str!("../../../shaders/godrays/composite.wgsl");
const COMPOSITE_REL: &str = "godrays/composite.wgsl";

/// Stable subsystem name (matches `[render.subsystems].godrays`).
pub const NAME: &str = "godrays";

/// Half-res scratch for the radial pass: divides the HDR width
/// and height by this factor. 2 = 1/4 the area, plenty smooth.
const HALF_RES_FACTOR: u32 = 2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GodraysParamsGpu {
    sun_ndc: [f32; 4],
    tunables: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GodraysCompositeParamsGpu {
    config: [f32; 4],
}

/// Phase 12.4 godrays subsystem. Holds two pipelines + a
/// dynamically-sized half-res scratch RT.
pub struct GodraysSubsystem {
    radial_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    radial_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    radial_params: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    scratch: Option<ScratchState>,
    tuning: TuningSnapshot,
}

struct ScratchState {
    /// Held to keep the texture alive while `hdr_copy_view` is in use.
    #[allow(dead_code)]
    hdr_copy: wgpu::Texture,
    hdr_copy_view: wgpu::TextureView,
    /// Held to keep the texture alive while `rays_rt_view` is in use.
    #[allow(dead_code)]
    rays_rt: wgpu::Texture,
    rays_rt_view: wgpu::TextureView,
    full_size: (u32, u32),
}

#[derive(Default, Clone, Copy)]
struct TuningSnapshot {
    samples: u32,
    decay: f32,
    intensity: f32,
    bright_threshold: f32,
}

impl GodraysSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        // --- Radial pipeline ---
        let radial_src = ps_core::shaders::load_shader(RADIAL_REL, RADIAL_BAKED);
        let composed_radial = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &radial_src,
        ]);
        let radial_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godrays/radial.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed_radial.into()),
        });

        let frame_layout = ps_core::frame_bind_group_layout(device);
        let radial_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("godrays-radial-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let radial_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("godrays-radial-pl"),
                bind_group_layouts: &[Some(&frame_layout), Some(&radial_layout)],
                immediate_size: 0,
            });
        let radial_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("godrays-radial-rp"),
                layout: Some(&radial_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &radial_module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &radial_module,
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
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // --- Composite pipeline (additive blend) ---
        let composite_src = ps_core::shaders::load_shader(COMPOSITE_REL, COMPOSITE_BAKED);
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godrays/composite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composite_src.into()),
        });
        let composite_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("godrays-composite-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("godrays-composite-pl"),
                bind_group_layouts: &[Some(&composite_layout)],
                immediate_size: 0,
            });
        let composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("godrays-composite-rp"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HdrFramebuffer::COLOR_FORMAT,
                        // Additive blend: dst = dst + src.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("godrays-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let radial_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("godrays-radial-params"),
            size: std::mem::size_of::<GodraysParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("godrays-composite-params"),
            size: std::mem::size_of::<GodraysCompositeParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let g = &config.render.godrays;
        Self {
            radial_pipeline,
            composite_pipeline,
            radial_layout,
            composite_layout,
            sampler,
            radial_params,
            composite_params,
            scratch: None,
            tuning: TuningSnapshot {
                samples: g.samples,
                decay: g.decay,
                intensity: g.intensity,
                bright_threshold: g.bright_threshold,
            },
        }
    }
}

impl RenderSubsystem for GodraysSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // Project sun direction onto the screen. Sun is infinitely
        // far away, so we project the *direction vector* through the
        // view+proj rather than a point at finite world distance —
        // this avoids reverse-Z infinite-far perspective sending the
        // homogeneous w to zero. Direction transform has w=0.
        let sun_world = ctx.frame_uniforms.sun_direction;
        let mut sun_on_screen = 0.0_f32;
        let mut sun_ndc = [0.0_f32, 0.0_f32];
        if sun_world.y > 0.0 {
            // view_proj * vec4(dir, 0) gives clip-space coords whose
            // .xy/.w is the screen position the direction projects to
            // (treating the sun as at infinity).
            let dir_world = Vec4::new(sun_world.x, sun_world.y, sun_world.z, 0.0);
            let dir_clip = ctx.frame_uniforms.view_proj * dir_world;
            // For a direction transform, dir_clip.w is the projected
            // depth and is positive when the direction points into
            // the camera's forward half-space.
            if dir_clip.w > 1.0e-6 {
                let ndc_x = dir_clip.x / dir_clip.w;
                let ndc_y = dir_clip.y / dir_clip.w;
                // Allow some over-scan: sun just outside the screen
                // still casts inward rays toward the visible region.
                if ndc_x.abs() <= 2.0 && ndc_y.abs() <= 2.0 {
                    sun_on_screen = 1.0;
                    sun_ndc = [ndc_x, ndc_y];
                }
            }
        }

        let tuning = self.tuning;
        let radial = GodraysParamsGpu {
            sun_ndc: [sun_ndc[0], sun_ndc[1], sun_on_screen, 0.0],
            tunables: [
                tuning.samples as f32,
                tuning.decay,
                tuning.intensity,
                tuning.bright_threshold,
            ],
        };
        ctx.queue
            .write_buffer(&self.radial_params, 0, bytemuck::bytes_of(&radial));

        let composite = GodraysCompositeParamsGpu {
            config: [tuning.intensity, sun_on_screen, 0.0, 0.0],
        };
        ctx.queue.write_buffer(
            &self.composite_params,
            0,
            bytemuck::bytes_of(&composite),
        );
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![
            PassDescriptor {
                name: "godrays-radial",
                stage: PassStage::PostProcess,
                id: PASS_RADIAL,
            },
            PassDescriptor {
                name: "godrays-composite",
                stage: PassStage::PostProcess,
                id: PASS_COMPOSITE,
            },
        ]
    }

    fn dispatch_pass(
        &mut self,
        id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        match id {
            PASS_RADIAL => {
                let device = ctx.device;
                let hdr = ctx.framebuffer;
                let full_size = hdr.size;
                let half_size = (
                    (full_size.0 / HALF_RES_FACTOR).max(1),
                    (full_size.1 / HALF_RES_FACTOR).max(1),
                );
                let needs_alloc = self.scratch.as_ref().is_none_or(|s| s.full_size != full_size);
                if needs_alloc {
                    let hdr_copy = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("godrays-hdr-copy"),
                        size: wgpu::Extent3d {
                            width: full_size.0,
                            height: full_size.1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: HdrFramebuffer::COLOR_FORMAT,
                        usage: wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let hdr_copy_view =
                        hdr_copy.create_view(&wgpu::TextureViewDescriptor::default());
                    let rays_rt = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("godrays-rays-rt"),
                        size: wgpu::Extent3d {
                            width: half_size.0,
                            height: half_size.1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: HdrFramebuffer::COLOR_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let rays_rt_view =
                        rays_rt.create_view(&wgpu::TextureViewDescriptor::default());
                    self.scratch = Some(ScratchState {
                        hdr_copy,
                        hdr_copy_view,
                        rays_rt,
                        rays_rt_view,
                        full_size,
                    });
                }
                let scratch = self.scratch.as_ref().expect("just allocated");

                // Copy HDR → hdr_copy so the radial pass can sample
                // without a read+write conflict on HDR.
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &hdr.color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &scratch.hdr_copy,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: full_size.0,
                        height: full_size.1,
                        depth_or_array_layers: 1,
                    },
                );

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("godrays-radial-bg"),
                    layout: &self.radial_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&scratch.hdr_copy_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.radial_params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("godrays-radial"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &scratch.rays_rt_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.radial_pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                pass.set_bind_group(1, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            PASS_COMPOSITE => {
                let device = ctx.device;
                let hdr = ctx.framebuffer;
                let Some(scratch) = self.scratch.as_ref() else {
                    return;
                };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("godrays-composite-bg"),
                    layout: &self.composite_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&scratch.rays_rt_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.composite_params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("godrays-composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &hdr.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.composite_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            _ => panic!("godrays: unknown pass id {id}"),
        }
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        let g = &config.render.godrays;
        self.tuning = TuningSnapshot {
            samples: g.samples,
            decay: g.decay,
            intensity: g.intensity,
            bright_threshold: g.bright_threshold,
        };
        Ok(())
    }
}

/// Factory wired by `AppBuilder`.
pub struct GodraysFactory;

impl SubsystemFactory for GodraysFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.godrays
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(GodraysSubsystem::new(config, gpu)))
    }
}
