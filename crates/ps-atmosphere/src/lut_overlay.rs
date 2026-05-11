//! Phase 5 debug visualiser — tiles the four atmosphere LUTs on top of the
//! swapchain so a human can eyeball whether the bakes are producing
//! meaningful data without reading back pixels in tests.
//!
//! Triggered from ps-app via `--lut-overlay` (CLI) or
//! `[debug].atmosphere_lut_overlay = true` (config).

use bytemuck::{Pod, Zeroable};
use ps_core::{AtmosphereLuts, GpuContext};

const SHADER_BAKED: &str = include_str!("../../../shaders/atmosphere/lut_overlay.wgsl");
const SHADER_REL: &str = "atmosphere/lut_overlay.wgsl";

/// Per-frame uniforms uploaded to the overlay shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct LutOverlayUniforms {
    /// `(tile_w, tile_h, gap, _pad)` in pixels.
    pub tile_layout: [f32; 4],
    /// Visualisation scales: `.r = transmittance, .g = multi-scatter,
    /// .b = sky-view, .a = aerial-perspective`.
    pub scales: [f32; 4],
}

impl Default for LutOverlayUniforms {
    fn default() -> Self {
        Self {
            // 256-wide tiles match the transmittance LUT's native width.
            // Sky-view is 192x108 so a 256-square tile gives it room.
            // Multi-scatter and AP are 32x32 — happy at small sizes.
            tile_layout: [256.0, 128.0, 8.0, 0.0],
            // Multipliers tuned so that physical values become roughly
            // visible at EV-0 passthrough. Transmittance is already in
            // [0, 1], no scaling. Multi-scatter is ~1e-3 per unit-illum
            // so we boost. Sky-view and AP are in cd/m²·sr; scale down
            // to fit the 0..1 visual range.
            scales: [1.0, 1000.0, 1.0 / 5000.0, 1.0 / 5000.0],
        }
    }
}

/// Owns the overlay pipeline + bind group layout + uniform buffer.
pub struct LutOverlay {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    /// Output format the overlay is configured for (swapchain format).
    pub output_format: wgpu::TextureFormat,
}

impl LutOverlay {
    /// Build the pipeline. `output_format` is the swapchain format the
    /// overlay writes into.
    pub fn new(gpu: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atmosphere::lut_overlay.wgsl"),
            source: wgpu::ShaderSource::Wgsl(live_src.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmosphere::lut_overlay-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<LutOverlayUniforms>() as u64,
                        ),
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
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atmosphere::lut_overlay-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("atmosphere::lut_overlay-rp"),
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
            label: Some("atmosphere::lut_overlay-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmosphere::lut_overlay-uniforms"),
            size: std::mem::size_of::<LutOverlayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniforms,
            output_format,
        }
    }

    /// Encode the overlay pass. `target_view` must have the format the
    /// overlay was constructed with (typically the swapchain).
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        target_view: &wgpu::TextureView,
        luts: &AtmosphereLuts,
        uniforms: &LutOverlayUniforms,
    ) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere::lut_overlay-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&luts.transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&luts.multiscatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&luts.skyview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&luts.aerial_perspective_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("atmosphere::lut_overlay"),
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
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
