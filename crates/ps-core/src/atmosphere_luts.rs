//! Atmosphere look-up tables (Phase 5).
//!
//! Plan §5.2 specifies four Hillaire-2020 LUTs:
//! - **Transmittance** — 256×64 `Rgba16Float`.
//! - **Multi-scattering** — 32×32 `Rgba16Float`.
//! - **Sky-view** — 192×108 `Rgba16Float`.
//! - **Aerial perspective** — 32×32×32 (3D) `Rgba16Float`.
//!
//! This module owns the GPU textures + the canonical group-3 bind-group
//! layout. The compute shaders that *populate* the LUTs live in
//! `ps-atmosphere`.

use crate::gpu::GpuContext;

/// Transmittance LUT pixel size.
pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
/// Multi-scattering LUT pixel size.
pub const MULTISCATTER_SIZE: (u32, u32) = (32, 32);
/// Sky-view LUT pixel size.
pub const SKYVIEW_SIZE: (u32, u32) = (192, 108);
/// Aerial-perspective LUT 3D dimensions (X × Y × Z).
pub const AP_SIZE: (u32, u32, u32) = (32, 32, 32);

/// LUT colour format used by all four maps.
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bundle of the four atmosphere LUT textures + their default views, plus
/// a shared filtering sampler and the canonical group-3 bind group + layout.
pub struct AtmosphereLuts {
    /// Transmittance LUT (256×64).
    pub transmittance: wgpu::Texture,
    /// Default view for the transmittance LUT.
    pub transmittance_view: wgpu::TextureView,

    /// Multi-scattering LUT (32×32).
    pub multiscatter: wgpu::Texture,
    /// Default view for the multi-scattering LUT.
    pub multiscatter_view: wgpu::TextureView,

    /// Sky-view LUT (192×108).
    pub skyview: wgpu::Texture,
    /// Default view for the sky-view LUT.
    pub skyview_view: wgpu::TextureView,

    /// Aerial-perspective LUT (32×32×32).
    pub aerial_perspective: wgpu::Texture,
    /// Default view for the aerial-perspective LUT.
    pub aerial_perspective_view: wgpu::TextureView,

    /// Linear-clamp sampler used by every shader that samples these LUTs.
    pub sampler: wgpu::Sampler,

    /// Group-3 bind group layout (4 textures + 1 sampler).
    pub layout: wgpu::BindGroupLayout,
    /// Group-3 bind group attaching the four views and the sampler.
    pub bind_group: wgpu::BindGroup,
}

impl AtmosphereLuts {
    /// Allocate the four textures + sampler + bind group.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let make_2d = |label: &'static str, (w, h): (u32, u32)| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let make_3d = |label: &'static str, (w, h, d): (u32, u32, u32)| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: d,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };

        let (transmittance, transmittance_view) =
            make_2d("atmosphere-transmittance", TRANSMITTANCE_SIZE);
        let (multiscatter, multiscatter_view) =
            make_2d("atmosphere-multiscatter", MULTISCATTER_SIZE);
        let (skyview, skyview_view) = make_2d("atmosphere-skyview", SKYVIEW_SIZE);
        let (aerial_perspective, aerial_perspective_view) = make_3d("atmosphere-ap", AP_SIZE);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere-luts-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = atmosphere_lut_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere-luts-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&multiscatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&skyview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&aerial_perspective_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            transmittance,
            transmittance_view,
            multiscatter,
            multiscatter_view,
            skyview,
            skyview_view,
            aerial_perspective,
            aerial_perspective_view,
            sampler,
            layout,
            bind_group,
        }
    }
}

/// Build a "transmittance-only" sampling bind-group layout for shaders
/// that only need the transmittance LUT (currently the multi-scatter
/// bake). Bindings: 0=transmittance, 4=sampler. Matches the
/// canonical layout's bindings 0 and 4 so the shader can be authored
/// against `@group(3) @binding(0)` / `@binding(4)` consistently.
pub fn atmosphere_transmittance_only_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ps-core::group3-atmosphere-transmittance-only"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Build the corresponding bind group for the transmittance-only layout.
pub fn atmosphere_transmittance_only_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    transmittance_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ps-core::group3-atmosphere-transmittance-only-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(transmittance_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Build a "transmittance + multi-scatter" sampling layout — used by
/// shaders that read both static LUTs while writing one of the dynamic
/// ones (sky-view, AP). Bindings: 0=transmittance, 1=multi-scatter,
/// 4=sampler.
pub fn atmosphere_static_only_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ps-core::group3-atmosphere-static-only"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Build the corresponding bind group for the static-only layout.
pub fn atmosphere_static_only_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    transmittance_view: &wgpu::TextureView,
    multiscatter_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ps-core::group3-atmosphere-static-only-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(transmittance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(multiscatter_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Canonical bind-group layout for the atmosphere LUT bind group (group 3).
///
/// Bindings:
/// 0 — transmittance LUT (texture_2d<f32>)
/// 1 — multi-scattering LUT (texture_2d<f32>)
/// 2 — sky-view LUT (texture_2d<f32>)
/// 3 — aerial-perspective LUT (texture_3d<f32>)
/// 4 — shared filtering sampler.
pub fn atmosphere_lut_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_2d = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let texture_3d = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ps-core::group3-atmosphere-luts"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: texture_3d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}
