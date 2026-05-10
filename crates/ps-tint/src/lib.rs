//! Phase 1 demo subsystem: fullscreen RGB multiply at `PassStage::PostProcess`.
//!
//! All work happens inside the registered pass closure: copy HDR → scratch
//! (so wgpu doesn't see the same texture as both sample source and render
//! attachment), update the multiplier uniform, build the bind group, draw.
//! `prepare()` is a no-op — Phase 4 will widen `PrepareContext` to expose an
//! encoder and the framebuffer, at which point the copy/uniform-write
//! moves there.
//!
//! The scratch texture is owned by the subsystem and re-allocated when the
//! HDR target's size changes.

#![deny(missing_docs)]

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use ps_core::{
    Config, GpuContext, HdrFramebuffer, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory,
};

const SHADER_SRC: &str = include_str!("../../../shaders/tint/multiply.wgsl");

/// Stable subsystem name (matches `[render.subsystems].tint`).
pub const NAME: &str = "tint";

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct TintUniformsGpu {
    multiplier: [f32; 4],
}

/// Phase 1 demo subsystem.
pub struct TintSubsystem {
    enabled: bool,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    /// Scratch texture sized to match the HDR target. Rebuilt when size changes.
    scratch: Arc<Mutex<Option<ScratchState>>>,
    multiplier: Arc<Mutex<[f32; 3]>>,
}

struct ScratchState {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

impl TintSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tint/multiply.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tint-bgl"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tint-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tint-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
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
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tint-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tint-uniforms"),
            size: std::mem::size_of::<TintUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let [r, g, b] = config.render.tint.multiplier;
        Self {
            enabled: true,
            pipeline,
            bgl,
            sampler,
            uniforms,
            scratch: Arc::new(Mutex::new(None)),
            multiplier: Arc::new(Mutex::new([r, g, b])),
        }
    }
}

impl RenderSubsystem for TintSubsystem {
    fn name(&self) -> &'static str {
        "tint"
    }

    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {
        // No-op: all per-frame work happens inside the pass closure where
        // an encoder is available. Phase 4's widened PrepareContext lets
        // us move that work here.
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        // wgpu objects are internally Arc-shared; cloning them into the closure
        // does not deep-copy GPU state.
        let pipeline = self.pipeline.clone();
        let bgl = self.bgl.clone();
        let sampler = self.sampler.clone();
        let uniforms = self.uniforms.clone();
        let scratch = self.scratch.clone();
        let multiplier = self.multiplier.clone();

        vec![RegisteredPass {
            name: "tint-multiply",
            stage: PassStage::PostProcess,
            run: Box::new(move |encoder, ctx| {
                let device = ctx.device;
                let queue = ctx.queue;
                let hdr = ctx.framebuffer;
                let size = hdr.size;

                // (Re)allocate scratch if size changed.
                let mut scratch_guard = scratch.lock().expect("tint scratch lock poisoned");
                let needs_alloc = scratch_guard.as_ref().is_none_or(|s| s.size != size);
                if needs_alloc {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("tint-scratch"),
                        size: wgpu::Extent3d {
                            width: size.0,
                            height: size.1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: HdrFramebuffer::COLOR_FORMAT,
                        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                    *scratch_guard = Some(ScratchState {
                        texture,
                        view,
                        size,
                    });
                }
                let scratch = scratch_guard.as_ref().expect("just allocated");

                // HDR → scratch.
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &hdr.color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &scratch.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                );

                // Update uniform.
                let [r, g, b] = *multiplier.lock().expect("tint multiplier lock poisoned");
                queue.write_buffer(
                    &uniforms,
                    0,
                    bytemuck::bytes_of(&TintUniformsGpu {
                        multiplier: [r, g, b, 1.0],
                    }),
                );

                // Build bind group + draw.
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tint-bg"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&scratch.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniforms.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tint-multiply"),
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
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }),
        }]
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        let [r, g, b] = config.render.tint.multiplier;
        *self
            .multiplier
            .lock()
            .expect("tint multiplier lock poisoned") = [r, g, b];
        Ok(())
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`.
pub struct TintFactory;

impl SubsystemFactory for TintFactory {
    fn name(&self) -> &'static str {
        "tint"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(TintSubsystem::new(config, gpu)))
    }
}
