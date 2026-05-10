//! Phase 0 / Phase 1 ground subsystem: a 200×200 km quad in the XZ plane
//! with a procedural checker shader.
//!
//! Renders into the HDR target with reverse-Z depth-test enabled at
//! `PassStage::Opaque`. Phase 7 replaces this with real PBR + wet surface;
//! this crate exists to give the camera something to fly over while
//! atmosphere/clouds come online.
//!
//! The ground pass `Load`s the colour target — the host (ps-app) is
//! responsible for clearing the HDR colour and depth before any subsystem
//! draws. The Phase 1 BackdropSubsystem provides the colour clear via its
//! own SkyBackdrop pass; when it's disabled, ps-app falls back to clearing
//! to `[render] clear_color`.

#![deny(missing_docs)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ps_core::{
    Config, FrameUniformsGpu, GpuContext, HdrFramebuffer, PassStage, PrepareContext,
    RegisteredPass, RenderSubsystem, SubsystemFactory,
};

const SHADER_SRC: &str = include_str!("../../../shaders/ground/checker.wgsl");
const QUAD_HALF_EXTENT_M: f32 = 100_000.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// Procedural checker ground plane. Constructed via [`GroundSubsystem`];
/// kept public so tests / host code can build one directly.
pub struct CheckerGround {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    /// Frame uniforms buffer. Written each `prepare()` call from
    /// `ctx.frame_uniforms`.
    pub frame_uniform_buf: wgpu::Buffer,
}

/// Stable subsystem name (matches `[render.subsystems].ground`).
pub const NAME: &str = "ground";

impl CheckerGround {
    /// Build the pipeline + vertex buffer + uniform buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground/checker.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ground-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ground-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
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
            Vertex {
                position: [-h, 0.0, -h],
            },
            Vertex {
                position: [h, 0.0, -h],
            },
            Vertex {
                position: [h, 0.0, h],
            },
            Vertex {
                position: [-h, 0.0, -h],
            },
            Vertex {
                position: [h, 0.0, h],
            },
            Vertex {
                position: [-h, 0.0, h],
            },
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

        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-frame-uniforms"),
            size: std::mem::size_of::<FrameUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ground-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group,
            vertex_buf,
            frame_uniform_buf,
        }
    }
}

/// `RenderSubsystem` wrapper around [`CheckerGround`].
///
/// Phase 7 replaces this with a PBR + wet-surface implementation.
pub struct GroundSubsystem {
    enabled: bool,
    inner: Arc<CheckerGround>,
}

impl GroundSubsystem {
    /// Construct.
    pub fn new(gpu: &GpuContext) -> Self {
        Self {
            enabled: true,
            inner: Arc::new(CheckerGround::new(&gpu.device)),
        }
    }
}

impl RenderSubsystem for GroundSubsystem {
    fn name(&self) -> &'static str {
        "ground"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // Upload the latest frame uniforms.
        ctx.queue.write_buffer(
            &self.inner.frame_uniform_buf,
            0,
            bytemuck::bytes_of(&FrameUniformsGpu::from_cpu(ctx.frame_uniforms)),
        );
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let inner = self.inner.clone();
        vec![RegisteredPass {
            name: "ground-checker",
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
                pass.set_pipeline(&inner.pipeline);
                pass.set_bind_group(0, &inner.bind_group, &[]);
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
        _config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(GroundSubsystem::new(gpu)))
    }
}
