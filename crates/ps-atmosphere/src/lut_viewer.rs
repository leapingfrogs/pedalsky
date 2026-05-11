//! Phase 10.2 — fullscreen LUT viewer.
//!
//! Single-pass post-tonemap overlay that draws one of the four
//! atmosphere LUTs stretched to fill the swapchain. Selected by
//! `mode` (1-4); a depth-slice slider controls the AP mode. Used
//! exclusively for debugging atmosphere bakes.

use bytemuck::{Pod, Zeroable};
use ps_core::{atmosphere_lut_bind_group_layout, AtmosphereLuts, GpuContext};

const SHADER_BAKED: &str = include_str!("../../../shaders/atmosphere/lut_viewer.wgsl");
const SHADER_REL: &str = "atmosphere/lut_viewer.wgsl";

/// Per-frame uniform passed to the viewer shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct LutViewerUniforms {
    /// 0 = off (host should skip the draw), 1 = transmittance,
    /// 2 = multi-scatter, 3 = sky-view, 4 = aerial-perspective.
    pub mode: u32,
    /// Padding (std140 alignment).
    pub _pad: u32,
    /// Depth slice ∈ [0, 1] for AP mode.
    pub depth_slice: f32,
    /// Output multiplier so dim per-unit-illuminance LUTs become
    /// visible.
    pub scale: f32,
}

impl Default for LutViewerUniforms {
    fn default() -> Self {
        Self {
            mode: 0,
            _pad: 0,
            depth_slice: 0.5,
            scale: 1.0,
        }
    }
}

/// Owns the viewer pipeline + bind group.
pub struct LutViewer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    /// Output format (swapchain).
    pub output_format: wgpu::TextureFormat,
}

impl LutViewer {
    /// Build the pipeline.
    pub fn new(gpu: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lut_viewer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(live_src.into()),
        });
        // Group 0 reuses the canonical atmosphere LUT bind layout
        // (transmittance/MS/skyview/AP/sampler at bindings 0..=4) and
        // adds the viewer uniform at binding 5. To stay compatible
        // with the layout, we declare a *new* layout that includes the
        // extra binding.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lut-viewer-bgl"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("lut-viewer-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lut-viewer-rp"),
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
            label: Some("lut-viewer-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lut-viewer-uniforms"),
            size: std::mem::size_of::<LutViewerUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Suppress unused warning until callers reach for the canonical
        // layout for compatibility checks.
        let _ = atmosphere_lut_bind_group_layout;
        Self {
            pipeline,
            layout,
            sampler,
            uniforms,
            output_format,
        }
    }

    /// Render the selected LUT to `target_view`.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        target_view: &wgpu::TextureView,
        luts: &AtmosphereLuts,
        uniforms: &LutViewerUniforms,
    ) {
        if uniforms.mode == 0 {
            return;
        }
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lut-viewer-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&luts.transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&luts.multiscatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&luts.skyview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&luts.aerial_perspective_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.uniforms.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lut-viewer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
