//! Phase 6 cloud render pipelines.
//!
//! Two render pipelines are needed:
//! - `march` — fragment shader that does the volumetric raymarch and
//!   writes premultiplied luminance to the cloud RT.
//! - `composite` — premultiplied-alpha blit of the cloud RT over the HDR
//!   target. `One, OneMinusSrcAlpha` blend.

use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout,
    HdrFramebuffer,
};

const CLOUD_UNIFORMS_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_uniforms.wgsl");
const CLOUD_UNIFORMS_REL: &str = "clouds/cloud_uniforms.wgsl";
const CLOUD_MARCH_BAKED: &str = include_str!("../../../shaders/clouds/cloud_march.wgsl");
const CLOUD_MARCH_REL: &str = "clouds/cloud_march.wgsl";
const CLOUD_COMPOSITE_BAKED: &str =
    include_str!("../../../shaders/clouds/cloud_composite.wgsl");
const CLOUD_COMPOSITE_REL: &str = "clouds/cloud_composite.wgsl";

/// Group-0 bind layout for the composite pass: cloud RT view + sampler.
pub fn composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clouds-composite-bgl"),
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
        ],
    })
}

/// Cloud render pipelines + the bind-group layout owned by the
/// composite pass (the march pipeline reuses `CloudNoise::layout`
/// passed in at construction).
pub struct CloudPipelines {
    /// Fragment raymarch pipeline.
    pub march: wgpu::RenderPipeline,
    /// Premultiplied-alpha blit pipeline (with blend state).
    pub composite: wgpu::RenderPipeline,
    /// Composite pass group-0 layout (cloud RT + sampler).
    pub composite_layout: wgpu::BindGroupLayout,
}

impl CloudPipelines {
    /// Build both pipelines. `cloud_data_layout` matches
    /// `CloudNoise::layout` and bundles noise textures + samplers + cloud
    /// uniforms into a single group-2 binding.
    pub fn new(device: &wgpu::Device, cloud_data_layout: &wgpu::BindGroupLayout) -> Self {
        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let lut_layout = atmosphere_lut_bind_group_layout(device);
        let composite_layout = composite_bind_group_layout(device);

        // March pipeline.
        let cloud_uniforms_src =
            ps_core::shaders::load_shader(CLOUD_UNIFORMS_REL, CLOUD_UNIFORMS_BAKED);
        let cloud_march_src =
            ps_core::shaders::load_shader(CLOUD_MARCH_REL, CLOUD_MARCH_BAKED);
        let march_src = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_MATH_WGSL,
            &cloud_uniforms_src,
            &cloud_march_src,
        ]);
        let march_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-march-shader"),
            source: wgpu::ShaderSource::Wgsl(march_src.into()),
        });
        let march_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-march-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(cloud_data_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let march = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-march"),
            layout: Some(&march_pl),
            vertex: wgpu::VertexState {
                module: &march_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &march_module,
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

        // Composite pipeline (premultiplied alpha blend).
        let cloud_composite_src =
            ps_core::shaders::load_shader(CLOUD_COMPOSITE_REL, CLOUD_COMPOSITE_BAKED);
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds-composite-shader"),
            source: wgpu::ShaderSource::Wgsl(cloud_composite_src.into()),
        });
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds-composite-pl"),
            bind_group_layouts: &[Some(&composite_layout)],
            immediate_size: 0,
        });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds-composite"),
            layout: Some(&composite_pl),
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
                ..Default::default()
            },
            // The composite pass writes over the HDR target which has a depth
            // attachment; setting depth_stencil = None means the render pass
            // descriptor must omit the depth attachment too. The pass we
            // build in lib.rs does exactly that.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            march,
            composite,
            composite_layout,
        }
    }
}
