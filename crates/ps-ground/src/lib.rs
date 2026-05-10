//! Phase 0 ground subsystem: a 200×200 km quad in the XZ plane with a
//! procedural checker shader.
//!
//! Renders into the HDR target with reverse-Z depth-test enabled. Phase 7
//! replaces this with a real PBR ground + wet surface; this crate exists
//! purely to give the Phase 0 camera something to fly over.

#![deny(missing_docs)]

use bytemuck::{Pod, Zeroable};
use ps_core::{FrameUniforms, FrameUniformsGpu, HdrFramebuffer};

const SHADER_SRC: &str = include_str!("../../../shaders/ground/checker.wgsl");
const QUAD_HALF_EXTENT_M: f32 = 100_000.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

/// Procedural checker ground plane.
pub struct CheckerGround {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    frame_uniform_buf: wgpu::Buffer,
}

impl CheckerGround {
    /// Build the pipeline. Reads the HDR target's color format and the depth
    /// format from the framebuffer.
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
                // Reverse-Z: a fragment passes if its depth is GREATER than
                // what's already in the buffer (closer to 1.0 = closer to
                // the camera).
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
            Vertex { position: [ h, 0.0, -h] },
            Vertex { position: [ h, 0.0,  h] },
            Vertex { position: [-h, 0.0, -h] },
            Vertex { position: [ h, 0.0,  h] },
            Vertex { position: [-h, 0.0,  h] },
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
            bind_group_layout,
            bind_group,
            vertex_buf,
            frame_uniform_buf,
        }
    }

    /// Encode the ground pass.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        hdr: &HdrFramebuffer,
        frame_uniforms: &FrameUniforms,
    ) {
        queue.write_buffer(
            &self.frame_uniform_buf,
            0,
            bytemuck::bytes_of(&FrameUniformsGpu::from_cpu(frame_uniforms)),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ground-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &hdr.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear so we don't depend on a backdrop subsystem
                    // existing yet. Phase 1's BackdropSubsystem (deferred)
                    // will own the clear when it lands.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.03,
                        b: 0.05,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &hdr.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Reverse-Z: clear to 0.0 (= far plane).
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..6, 0..1);
        let _ = &self.bind_group_layout;
    }
}
