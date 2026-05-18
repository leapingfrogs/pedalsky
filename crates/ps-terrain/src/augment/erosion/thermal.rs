//! Stage 1.3 thermal erosion compute pipelines.

const SHADER_SRC: &str = include_str!("../../../../../shaders/terrain/thermal.wgsl");

pub(super) struct ThermalBindings {
    pub uniforms: wgpu::Buffer,
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline_outflow: wgpu::ComputePipeline,
    pub pipeline_apply: wgpu::ComputePipeline,
}

impl ThermalBindings {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain/thermal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("erosion-thermal-uniforms"),
            size: std::mem::size_of::<super::params::ThermalUniformGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion-thermal-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<super::params::ThermalUniformGpu>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_2d(1, wgpu::TextureFormat::R32Float),
                storage_2d(2, wgpu::TextureFormat::Rgba32Float),
                storage_2d(3, wgpu::TextureFormat::Rgba32Float),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("erosion-thermal-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mkpipe = |name: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            uniforms,
            bgl,
            pipeline_outflow: mkpipe("erosion-thermal-outflow", "compute_outflow"),
            pipeline_apply: mkpipe("erosion-thermal-apply", "apply_outflow"),
        }
    }
}

fn storage_2d(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::ReadWrite,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
