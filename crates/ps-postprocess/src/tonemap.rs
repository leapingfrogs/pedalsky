//! ACES Filmic / Passthrough fullscreen tone-map pass.
//!
//! Reads the HDR target, applies EV100 exposure, runs the chosen curve, and
//! writes a `*Srgb` swapchain texture. The GPU encodes linear→sRGB on store,
//! so the shader output is linear.

use std::sync::Mutex;

use bytemuck::{Pod, Zeroable};
use ps_core::HdrFramebuffer;
pub use ps_core::TonemapMode;

const SHADER_BAKED: &str = include_str!("../../../shaders/postprocess/tonemap.wgsl");
const SHADER_REL: &str = "postprocess/tonemap.wgsl";

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct TonemapUniformsGpu {
    ev100: f32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Fullscreen tone-map pass.
///
/// Construct once at app startup. Call [`Self::resize`] when the HDR target
/// changes; call [`Self::render`] each frame to write the swapchain.
pub struct Tonemap {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    /// Bound to the HDR target. Behind a `Mutex` so resize-rebuild
    /// (which the host triggers via [`Self::rebuild_bindings`]) can run
    /// through a shared `Arc<Tonemap>` reference without exclusive
    /// ownership.
    bind_group: Mutex<wgpu::BindGroup>,
    output_format: wgpu::TextureFormat,
}

impl Tonemap {
    /// Build the pipeline. `output_format` is the swapchain (or test target)
    /// format the tone-mapper writes into; should be sRGB-suffixed for
    /// correct gamma encoding.
    pub fn new(
        device: &wgpu::Device,
        hdr: &HdrFramebuffer,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemap.wgsl"),
            source: wgpu::ShaderSource::Wgsl(live_src.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap-bgl"),
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
            label: Some("tonemap-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap-rp"),
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
                    format: output_format,
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
            label: Some("tonemap-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap-uniforms"),
            size: std::mem::size_of::<TonemapUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group =
            Mutex::new(build_bind_group(device, &bind_group_layout, hdr, &sampler, &uniforms));

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniforms,
            bind_group,
            output_format,
        }
    }

    /// The output format the tone-mapper writes into.
    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Rebind to a new HDR target after resize. Takes `&self` so the
    /// host can call this through a shared `Arc<Tonemap>` reference.
    pub fn rebuild_bindings(&self, device: &wgpu::Device, hdr: &HdrFramebuffer) {
        let bg = build_bind_group(
            device,
            &self.bind_group_layout,
            hdr,
            &self.sampler,
            &self.uniforms,
        );
        *self.bind_group.lock().expect("tonemap bind group lock") = bg;
    }

    /// Encode the tone-map pass into `encoder`, reading the HDR target and
    /// writing into `target_view`.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target_view: &wgpu::TextureView,
        ev100: f32,
        mode: TonemapMode,
    ) {
        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&TonemapUniformsGpu {
                ev100,
                mode: mode.as_u32(),
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let bg = self.bind_group.lock().expect("tonemap bind group lock");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    hdr: &HdrFramebuffer,
    sampler: &wgpu::Sampler,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr.color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms.as_entire_binding(),
            },
        ],
    })
}
