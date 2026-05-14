//! Phase 13.3 — HDR bloom.
//!
//! Standard bright-pass + Gaussian-pyramid + additive-composite chain.
//! Pipeline:
//!
//!   1. **Bright pass** — fullscreen read of the HDR target;
//!      writes a half-res scratch where pixels above the soft
//!      knee threshold contribute. Pre-multiplied so the
//!      downstream blur runs on already-isolated highlights.
//!   2. **Downsample chain** — 5-tap Gaussian downsample from
//!      half → quarter → eighth res. Each level is its own RT.
//!   3. **Upsample chain** — bilinear-add upsample from eighth →
//!      quarter → half. Each step bakes the smaller blurred level
//!      into the next-larger one with additive blend.
//!   4. **Composite** — final half-res result is bilinear-sampled
//!      and additively composited into the HDR target with the
//!      tunable intensity scalar.
//!
//! All passes are at `PassStage::PostProcess` and run before the
//! tonemap so the bloom contribution lands in HDR space.

#![deny(missing_docs)]

use bytemuck::{Pod, Zeroable};
use ps_core::{
    Config, GpuContext, HdrFramebuffer, PassDescriptor, PassId, PassStage, PrepareContext,
    RenderContext, RenderSubsystem, SubsystemFactory,
};
use tracing::debug;

const PASS_BLOOM: PassId = 0;

const BRIGHT_BAKED: &str = include_str!("../../../shaders/bloom/bright_pass.wgsl");
const BRIGHT_REL: &str = "bloom/bright_pass.wgsl";
const DOWN_BAKED: &str = include_str!("../../../shaders/bloom/downsample.wgsl");
const DOWN_REL: &str = "bloom/downsample.wgsl";
const UP_BAKED: &str = include_str!("../../../shaders/bloom/upsample.wgsl");
const UP_REL: &str = "bloom/upsample.wgsl";
const COMPOSITE_BAKED: &str = include_str!("../../../shaders/bloom/composite.wgsl");
const COMPOSITE_REL: &str = "bloom/composite.wgsl";

/// Stable subsystem name (matches `[render.subsystems].bloom`).
pub const NAME: &str = "bloom";

/// Number of pyramid levels (excluding the full-res HDR source).
/// 3 = half / quarter / eighth resolution. Pyramid is built level-0
/// (largest) → level-(N-1) (smallest) on the way down, then
/// upsampled in reverse.
const PYRAMID_LEVELS: u32 = 3;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct BrightParamsGpu {
    /// `x` = linear luminance threshold (`exp2(threshold_ev100)`),
    /// `y` = soft-knee falloff width (linear),
    /// `z` = inverse src texture width,
    /// `w` = inverse src texture height.
    config: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct DownParamsGpu {
    /// `xy` = inverse source texture dimensions; `zw` unused.
    config: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct UpParamsGpu {
    /// `xy` = inverse smaller-level source dimensions; `z` = additive
    /// scale for this upsample step (1.0 by default); `w` unused.
    config: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct CompositeParamsGpu {
    /// `x` = intensity scalar; `yzw` unused.
    config: [f32; 4],
}

/// Tuning snapshot taken at construction / reconfigure.
#[derive(Clone, Copy, Debug)]
struct TuningSnapshot {
    threshold_ev100: f32,
    intensity: f32,
    knee_ev: f32,
}

impl TuningSnapshot {
    fn from_config(config: &Config) -> Self {
        let b = &config.render.bloom;
        Self {
            threshold_ev100: b.threshold_ev100,
            intensity: b.intensity,
            knee_ev: b.knee_ev,
        }
    }
}

/// Per-frame GPU resources that depend on framebuffer size. Reallocated
/// on resize.
struct ScratchState {
    /// Bright-pass output (half-res).
    levels: Vec<LevelTextures>,
    full_size: (u32, u32),
}

struct LevelTextures {
    #[allow(dead_code)]
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// Phase 13.3 bloom subsystem.
pub struct BloomSubsystem {
    bright_pipeline: wgpu::RenderPipeline,
    down_pipeline: wgpu::RenderPipeline,
    up_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    bright_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    up_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bright_params: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    /// One per pyramid downsample step (level 1..N-1).
    down_params: Vec<wgpu::Buffer>,
    /// One per pyramid upsample step (level N-1..0).
    up_params: Vec<wgpu::Buffer>,
    /// HDR copy used as a sample source for the bright pass without
    /// a read+write conflict.
    hdr_copy: Option<HdrCopy>,
    /// Per-size pyramid scratch.
    scratch: Option<ScratchState>,
    tuning: TuningSnapshot,
}

struct HdrCopy {
    #[allow(dead_code)]
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    full_size: (u32, u32),
}

impl BloomSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Common single-texture-sampler-uniform layout used by
        // bright / downsample / composite. Upsample also takes two
        // textures (smaller blur + larger pyramid level) so it has
        // its own layout.
        let make_simple_layout = |label: &'static str| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
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
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
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
            })
        };
        let bright_layout = make_simple_layout("bloom-bright-bgl");
        let down_layout = make_simple_layout("bloom-down-bgl");
        let composite_layout = make_simple_layout("bloom-composite-bgl");

        // Upsample binds two source textures (the smaller pyramid
        // level + the next-larger level it adds into).
        let up_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom-up-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(
                        wgpu::SamplerBindingType::Filtering,
                    ),
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

        // --- Pipelines.
        let make_pipeline = |label: &'static str,
                             rel: &'static str,
                             baked: &'static str,
                             layout: &wgpu::BindGroupLayout,
                             additive: bool| {
            let src = ps_core::shaders::load_shader(rel, baked);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                },
            );
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HdrFramebuffer::COLOR_FORMAT,
                        blend: if additive {
                            Some(wgpu::BlendState {
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
                            })
                        } else {
                            None
                        },
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
            })
        };
        let bright_pipeline =
            make_pipeline("bloom-bright-rp", BRIGHT_REL, BRIGHT_BAKED, &bright_layout, false);
        let down_pipeline =
            make_pipeline("bloom-down-rp", DOWN_REL, DOWN_BAKED, &down_layout, false);
        // Upsample writes additively into the next-larger blurred
        // level, accumulating the pyramid as it climbs.
        let up_pipeline =
            make_pipeline("bloom-up-rp", UP_REL, UP_BAKED, &up_layout, true);
        let composite_pipeline = make_pipeline(
            "bloom-composite-rp",
            COMPOSITE_REL,
            COMPOSITE_BAKED,
            &composite_layout,
            true,
        );

        // Uniform buffers — one bright + one composite + N down + N up.
        let bright_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom-bright-params"),
            size: std::mem::size_of::<BrightParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom-composite-params"),
            size: std::mem::size_of::<CompositeParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let down_params: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bloom-down-params"),
                    size: std::mem::size_of::<DownParamsGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let up_params: Vec<wgpu::Buffer> = (0..PYRAMID_LEVELS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bloom-up-params"),
                    size: std::mem::size_of::<UpParamsGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();

        debug!(target: "ps_bloom", "subsystem ready");
        Self {
            bright_pipeline,
            down_pipeline,
            up_pipeline,
            composite_pipeline,
            bright_layout,
            down_layout,
            up_layout,
            composite_layout,
            sampler,
            bright_params,
            composite_params,
            down_params,
            up_params,
            hdr_copy: None,
            scratch: None,
            tuning: TuningSnapshot::from_config(config),
        }
    }
}

/// Build the per-size pyramid scratch + the HDR-copy texture. Sized
/// against the current HDR target; reallocated when the target
/// resizes.
fn allocate_pyramid(
    device: &wgpu::Device,
    full_size: (u32, u32),
) -> (HdrCopy, ScratchState) {
    let hdr_copy_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom-hdr-copy"),
        size: wgpu::Extent3d {
            width: full_size.0,
            height: full_size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HdrFramebuffer::COLOR_FORMAT,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let hdr_copy_view = hdr_copy_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut levels = Vec::with_capacity(PYRAMID_LEVELS as usize);
    let mut size = (
        (full_size.0 / 2).max(1),
        (full_size.1 / 2).max(1),
    );
    for i in 0..PYRAMID_LEVELS {
        let label = match i {
            0 => "bloom-pyramid-half",
            1 => "bloom-pyramid-quarter",
            _ => "bloom-pyramid-eighth",
        };
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
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
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        levels.push(LevelTextures { tex, view, size });
        size = ((size.0 / 2).max(1), (size.1 / 2).max(1));
    }

    (
        HdrCopy {
            tex: hdr_copy_tex,
            view: hdr_copy_view,
            full_size,
        },
        ScratchState { levels, full_size },
    )
}

impl RenderSubsystem for BloomSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let tuning = self.tuning;

        // Bright pass: convert ev100 stops into a linear luminance
        // threshold. ev100 stops are log2 multiples of the scene's
        // exposure target; threshold_ev100 = 2 means "4× the
        // average exposure target".
        let scene_ev = ctx.frame_uniforms.ev100;
        let threshold_lin = (scene_ev + tuning.threshold_ev100).exp2();
        let knee_lin = (tuning.knee_ev * std::f32::consts::LN_2).max(1e-6); // ~1 ev = ln(2)
        let bright = BrightParamsGpu {
            config: [threshold_lin, knee_lin, 0.0, 0.0],
        };
        ctx.queue
            .write_buffer(&self.bright_params, 0, bytemuck::bytes_of(&bright));

        // Composite intensity is the raw scalar.
        let composite = CompositeParamsGpu {
            config: [tuning.intensity, 0.0, 0.0, 0.0],
        };
        ctx.queue.write_buffer(
            &self.composite_params,
            0,
            bytemuck::bytes_of(&composite),
        );
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        // The whole pyramid runs as a single registered pass — the
        // per-frame allocation/dispatch is too coupled to split
        // across multiple pass entries cleanly, and a single
        // PostProcess pass keeps the GPU timestamp output clean.
        vec![PassDescriptor {
            name: "bloom",
            stage: PassStage::PostProcess,
            id: PASS_BLOOM,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        let device = ctx.device;
        let queue = ctx.queue;
        let hdr = ctx.framebuffer;
        let full_size = hdr.size;

        // (Re)allocate scratch + hdr-copy if size changed.
        let needs_alloc = match (&self.hdr_copy, &self.scratch) {
            (Some(c), Some(s)) => c.full_size != full_size || s.full_size != full_size,
            _ => true,
        };
        if needs_alloc {
            let (c, s) = allocate_pyramid(device, full_size);
            self.hdr_copy = Some(c);
            self.scratch = Some(s);
        }
        let copy_state = self.hdr_copy.as_ref().expect("just allocated");
        let scratch_state = self.scratch.as_ref().expect("just allocated");

        // Snapshot HDR → hdr_copy so the bright pass can sample
        // without a read+write conflict.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &hdr.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &copy_state.tex,
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

        // --- Pass 1: bright pass into level-0 (half res).
        {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom-bright-bg"),
                layout: &self.bright_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&copy_state.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.bright_params.as_entire_binding(),
                    },
                ],
            });
            let level0 = &scratch_state.levels[0];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-bright"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &level0.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.bright_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Pass 2..N: downsample chain.
        for i in 1..(PYRAMID_LEVELS as usize) {
            let src = &scratch_state.levels[i - 1];
            let dst = &scratch_state.levels[i];
            let down = DownParamsGpu {
                config: [1.0 / src.size.0 as f32, 1.0 / src.size.1 as f32, 0.0, 0.0],
            };
            queue.write_buffer(&self.down_params[i], 0, bytemuck::bytes_of(&down));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom-down-bg"),
                layout: &self.down_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.down_params[i].as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-down"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Pass N+1..2N-1: upsample chain.
        for i in (1..(PYRAMID_LEVELS as usize)).rev() {
            let src = &scratch_state.levels[i];
            let dst = &scratch_state.levels[i - 1];
            let up = UpParamsGpu {
                config: [1.0 / src.size.0 as f32, 1.0 / src.size.1 as f32, 1.0, 0.0],
            };
            queue.write_buffer(&self.up_params[i], 0, bytemuck::bytes_of(&up));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom-up-bg"),
                layout: &self.up_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.up_params[i].as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-up"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load — the larger level already holds its own
                        // blur; we add the upsampled smaller level on top.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.up_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- Final composite: sample level-0 (the fully accumulated
        // pyramid) and additively blend into the HDR target.
        {
            let level0 = &scratch_state.levels[0];
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom-composite-bg"),
                layout: &self.composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&level0.view),
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
                label: Some("bloom-composite"),
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
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        self.tuning = TuningSnapshot::from_config(config);
        Ok(())
    }
}

/// Factory wired by `AppBuilder`.
pub struct BloomFactory;

impl SubsystemFactory for BloomFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.bloom
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(BloomSubsystem::new(config, gpu)))
    }
}
