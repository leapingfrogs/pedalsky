//! Stage 1.2 hydraulic erosion compute pipelines.
//!
//! Builds the four `wgpu::ComputePipeline`s and the bind-group layout
//! that the runtime dispatches each iteration.

const SHADER_SRC: &str = include_str!("../../../../../shaders/terrain/hydraulic.wgsl");

/// Bound resources for the hydraulic passes. Order matches the WGSL
/// `group(0) binding(N)` declarations.
pub(super) struct HydraulicBindings {
    pub uniforms: wgpu::Buffer,
    pub bgl: wgpu::BindGroupLayout,
    pub pipeline_add_water_and_flux: wgpu::ComputePipeline,
    pub pipeline_update_water_and_velocity: wgpu::ComputePipeline,
    pub pipeline_erosion_deposition: wgpu::ComputePipeline,
    pub pipeline_advect_sediment: wgpu::ComputePipeline,
}

impl HydraulicBindings {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain/hydraulic.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("erosion-hydraulic-uniforms"),
            size: std::mem::size_of::<super::params::HydraulicUniformGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion-hydraulic-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            super::params::HydraulicUniformGpu,
                        >() as u64),
                    },
                    count: None,
                },
                storage_2d(1, wgpu::TextureFormat::R32Float),
                storage_2d(2, wgpu::TextureFormat::R32Float),
                storage_2d(3, wgpu::TextureFormat::R32Float),
                storage_2d(4, wgpu::TextureFormat::Rgba32Float),
                // Velocity: split write-only storage (binding 5) and
                // sampled read view (binding 9) of the same texture.
                // Metal has no read_write tier for rg32float, so we
                // can't use ReadWrite here. See shader header.
                storage_2d_write(5, wgpu::TextureFormat::Rg32Float),
                storage_2d(6, wgpu::TextureFormat::R32Float),
                storage_2d(7, wgpu::TextureFormat::Rgba32Float),
                storage_2d(8, wgpu::TextureFormat::R32Float),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("erosion-hydraulic-pl"),
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
            pipeline_add_water_and_flux: mkpipe(
                "erosion-hydraulic-add-water-and-flux",
                "add_water_and_flux",
            ),
            pipeline_update_water_and_velocity: mkpipe(
                "erosion-hydraulic-update-water-and-velocity",
                "update_water_and_velocity",
            ),
            pipeline_erosion_deposition: mkpipe(
                "erosion-hydraulic-erosion-deposition",
                "erosion_deposition",
            ),
            pipeline_advect_sediment: mkpipe(
                "erosion-hydraulic-advect-sediment",
                "advect_sediment",
            ),
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

fn storage_2d_write(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
