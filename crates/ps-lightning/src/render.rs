//! Bolt rendering — billboarded additive quads, depth test off, in
//! the Translucent stage. v1 intentionally renders bolts as
//! always-on-top emissive geometry per the scope-doc punt; cloud
//! volumes don't occlude them.
//!
//! Layout: one quad per `BoltSegment` issued via instanced draws. The
//! per-instance buffer holds (a, b, thickness, emission_intensity).
//! The vertex shader builds a view-aligned billboard around the
//! a→b axis; the fragment shader writes the trunk's emissive HDR
//! colour.

use std::sync::Mutex;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use ps_core::{HdrFramebuffer, RenderContext};

use crate::strike::{pulse_envelope, ActiveStrike};
use crate::STRIKE_LIFETIME_S;

/// One per-instance vertex-shader input (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct InstanceData {
    a: [f32; 3],
    thickness: f32,
    b: [f32; 3],
    emission: f32,
}

const SHADER_REL: &str = "lightning/bolt.wgsl";
const SHADER_BAKED: &str = include_str!("../../../shaders/lightning/bolt.wgsl");

/// Holds the GPU resources for the bolt pass. The instance buffer is
/// re-uploaded each frame in `upload_active_strikes`.
pub struct BoltRender {
    pipeline: wgpu::RenderPipeline,
    instance_buf: wgpu::Buffer,
    instance_capacity: u32,
    /// Number of instances written by the most-recent
    /// `upload_active_strikes` call. Drives the draw count.
    instance_count: Mutex<u32>,
}

impl BoltRender {
    /// Construct. `max_active_strikes` × per-bolt-segments-cap →
    /// instance buffer size. We cap per-bolt at 256 for a comfortable
    /// upper bound (trunk 32 + 3 forks × ~8 = 56; 256 is generous).
    pub fn new(device: &wgpu::Device, max_active_strikes: u32) -> Self {
        let per_bolt_max = 256_u32;
        let instance_capacity = max_active_strikes * per_bolt_max;

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lightning-bolt-instances"),
            size: (instance_capacity as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &live_src,
        ]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lightning/bolt.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = ps_core::frame_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lightning-bolt-pl"),
            bind_group_layouts: &[Some(&frame_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lightning-bolt-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4, // a + thickness
                        1 => Float32x4, // b + emission
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HdrFramebuffer::COLOR_FORMAT,
                    // Additive HDR blend — emissive bolts always
                    // brighten the destination, never darken.
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
            // Depth test off per scope-doc v1 punt — bolts are
            // always-on-top emissive geometry.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            instance_buf,
            instance_capacity,
            instance_count: Mutex::new(0),
        }
    }

    /// Walk the active strikes, expand each into per-segment instance
    /// records weighted by the strike's pulse envelope, and upload to
    /// the GPU buffer.
    pub fn upload_active_strikes(
        &self,
        queue: &wgpu::Queue,
        active: &[ActiveStrike],
        bolt_peak_emission: f32,
    ) {
        let mut data: Vec<InstanceData> = Vec::with_capacity(64);
        for s in active {
            let env = pulse_envelope(s.age_s, STRIKE_LIFETIME_S);
            if env <= 0.0 {
                continue;
            }
            for seg in &s.bolt.segments {
                if data.len() as u32 >= self.instance_capacity {
                    break;
                }
                let emission = bolt_peak_emission * env * seg.emission_scale;
                data.push(InstanceData {
                    a: seg.a.to_array(),
                    thickness: seg.thickness,
                    b: seg.b.to_array(),
                    emission,
                });
            }
        }
        let count = data.len() as u32;
        if count > 0 {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&data));
        }
        *self
            .instance_count
            .lock()
            .expect("lightning instance count lock") = count;
    }

    /// Draw all currently-loaded bolt segments. No-op when no
    /// instances are uploaded.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext<'_>) {
        let count = *self
            .instance_count
            .lock()
            .expect("lightning instance count lock");
        if count == 0 {
            // Skip the pass entirely when no bolts are live — saves
            // an unnecessary render-pass open/close.
            let _ = ctx;
            let _ = Vec3::ZERO;
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lightning-bolts"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
        pass.set_vertex_buffer(0, self.instance_buf.slice(..));
        // 6 vertices per quad (two triangles), `count` instances.
        pass.draw(0..6, 0..count);
    }
}
