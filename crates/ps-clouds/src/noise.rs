//! Phase 6.1 — Schneider/Hillaire cloud noise volumes.
//!
//! Three GPU compute bakes (base shape, detail, curl) plus a CPU-baked
//! 2D blue-noise tile used for spatial march-start jitter. All bakes are
//! deterministic for fixed (seed, dims), so re-running a clean build
//! produces byte-identical textures.
//!
//! Sizes match plan §6.1:
//! - base shape:   128³ Rgba8Unorm
//! - detail:        32³ Rgba8Unorm
//! - curl:         128² Rg8Unorm  (2D)
//! - blue noise:    64² R8Unorm   (2D)
//!
//! The bake currently runs at construction time. Disk caching keyed by
//! content hash is reserved for follow-up work (the bake takes < 50 ms
//! on a desktop GPU, so the warm-cold gap is small).

use ps_core::GpuContext;

const BASE_NOISE_SHADER: &str = include_str!("../../../shaders/clouds/noise_base.comp.wgsl");
const DETAIL_NOISE_SHADER: &str = include_str!("../../../shaders/clouds/noise_detail.comp.wgsl");
const CURL_NOISE_SHADER: &str = include_str!("../../../shaders/clouds/noise_curl.comp.wgsl");

/// Base shape volume edge in voxels.
pub const BASE_SIZE: u32 = 128;
/// Detail volume edge in voxels.
pub const DETAIL_SIZE: u32 = 32;
/// Curl tile edge in texels.
pub const CURL_SIZE: u32 = 128;
/// Blue-noise tile edge in texels.
pub const BLUE_NOISE_SIZE: u32 = 64;

/// All four noise textures plus a shared linear-repeat sampler.
pub struct CloudNoise {
    /// 128³ Rgba8Unorm base shape (R = Perlin–Worley, GBA = Worley FBM 2/8/14).
    pub base: wgpu::Texture,
    /// View on `base`.
    pub base_view: wgpu::TextureView,
    /// 32³ Rgba8Unorm detail (RGB = Worley FBM 2/8/16, A spare).
    pub detail: wgpu::Texture,
    /// View on `detail`.
    pub detail_view: wgpu::TextureView,
    /// 128² Rg8Unorm 2D curl tile.
    pub curl: wgpu::Texture,
    /// View on `curl`.
    pub curl_view: wgpu::TextureView,
    /// 64² R8Unorm blue-noise tile (CPU-generated void-and-cluster).
    pub blue_noise: wgpu::Texture,
    /// View on `blue_noise`.
    pub blue_noise_view: wgpu::TextureView,
    /// Repeat sampler used by the cloud march for the 3D + 2D noise lookups.
    pub sampler: wgpu::Sampler,
    /// Nearest sampler used by `textureLoad` for blue-noise (kept for symmetry).
    pub nearest_sampler: wgpu::Sampler,
    /// Group-2 bind-group layout (textures + samplers + uniforms). Owned
    /// here so the same layout is reused by [`CloudsSubsystem`] when it
    /// builds the bind group with the cloud uniforms.
    pub layout: wgpu::BindGroupLayout,
}

impl CloudNoise {
    /// Build the four textures, run the three GPU bakes, upload the CPU
    /// blue-noise tile, and assemble the bind group.
    pub fn bake(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let queue = &gpu.queue;

        let base = make_3d(device, "clouds-noise-base", BASE_SIZE, wgpu::TextureFormat::Rgba8Unorm);
        let base_view = base.create_view(&wgpu::TextureViewDescriptor::default());
        let detail = make_3d(
            device,
            "clouds-noise-detail",
            DETAIL_SIZE,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let detail_view = detail.create_view(&wgpu::TextureViewDescriptor::default());
        let curl = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("clouds-noise-curl"),
            size: wgpu::Extent3d {
                width: CURL_SIZE,
                height: CURL_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let curl_view = curl.create_view(&wgpu::TextureViewDescriptor::default());

        // Blue noise: CPU-generated, then uploaded.
        let blue_noise_pixels = generate_blue_noise(BLUE_NOISE_SIZE);
        let blue_noise = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("clouds-noise-blue"),
            size: wgpu::Extent3d {
                width: BLUE_NOISE_SIZE,
                height: BLUE_NOISE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &blue_noise,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &blue_noise_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BLUE_NOISE_SIZE),
                rows_per_image: Some(BLUE_NOISE_SIZE),
            },
            wgpu::Extent3d {
                width: BLUE_NOISE_SIZE,
                height: BLUE_NOISE_SIZE,
                depth_or_array_layers: 1,
            },
        );
        let blue_noise_view = blue_noise.create_view(&wgpu::TextureViewDescriptor::default());

        // Bake passes.
        bake_3d(
            device,
            queue,
            "clouds::noise-base-bake",
            BASE_NOISE_SHADER,
            &base_view,
            wgpu::TextureFormat::Rgba8Unorm,
            BASE_SIZE,
        );
        bake_3d(
            device,
            queue,
            "clouds::noise-detail-bake",
            DETAIL_NOISE_SHADER,
            &detail_view,
            wgpu::TextureFormat::Rgba8Unorm,
            DETAIL_SIZE,
        );
        bake_2d(
            device,
            queue,
            "clouds::noise-curl-bake",
            CURL_NOISE_SHADER,
            &curl_view,
            wgpu::TextureFormat::Rg8Unorm,
            CURL_SIZE,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clouds-noise-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clouds-noise-nearest-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = noise_bind_group_layout(device);

        Self {
            base,
            base_view,
            detail,
            detail_view,
            curl,
            curl_view,
            blue_noise,
            blue_noise_view,
            sampler,
            nearest_sampler,
            layout,
        }
    }
}

fn make_3d(
    device: &wgpu::Device,
    label: &'static str,
    size: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn bake_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    shader_src: &str,
    target: &wgpu::TextureView,
    format: wgpu::TextureFormat,
    size: u32,
) {
    let storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        }],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&storage_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &storage_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(target),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = size.div_ceil(4);
        pass.dispatch_workgroups(groups, groups, groups);
    }
    queue.submit([encoder.finish()]);
}

fn bake_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    shader_src: &str,
    target: &wgpu::TextureView,
    format: wgpu::TextureFormat,
    size: u32,
) {
    let storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&storage_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &storage_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(target),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = size.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    queue.submit([encoder.finish()]);
}

/// Group-2 bind-group layout used by the cloud march pipeline.
///
/// Bindings:
/// 0 — base shape volume (texture_3d<f32>)
/// 1 — detail volume    (texture_3d<f32>)
/// 2 — curl tile        (texture_2d<f32>)
/// 3 — blue noise tile  (texture_2d<f32>, sampled via textureLoad)
/// 4 — linear-repeat sampler
/// 5 — nearest-repeat sampler
///
/// Bindings 6 and 7 (CloudParams uniform + CloudLayerGpu storage) are
/// declared by the march pipeline's overlay layout in
/// [`crate::pipeline`]; tests that read back the noise textures use only
/// the 0..=5 prefix layout returned by [`noise_only_bind_group_layout`].
pub fn noise_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture_3d = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D3,
        multisampled: false,
    };
    let texture_2d = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let texture_2d_unfiltered = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clouds-data-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_3d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_3d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d_unfiltered,
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: texture_2d,
                count: None,
            },
        ],
    })
}

/// CPU void-and-cluster blue noise generator (Christensen & Kensler).
///
/// Iteratively places samples to maximise minimum spacing, then ranks
/// them by insertion order. Output bytes are quantised ranks scaled to
/// [0, 255]. Slow (O(N² log N) for N = size²), but `size = 64` →
/// ~16 ms on a desktop CPU, run once per process.
fn generate_blue_noise(size: u32) -> Vec<u8> {
    let n = (size * size) as usize;
    let mut taken = vec![false; n];
    let mut ranks = vec![0u32; n];

    // Energy field: gaussian splat around each placed sample. Each new
    // sample is the cell with the lowest energy; placing it raises the
    // local energy.
    let mut energy = vec![0f32; n];
    let sigma = 1.9_f32;
    let radius = 6_i32;

    let xy = |i: usize| ((i as i32) % size as i32, (i as i32) / size as i32);
    let idx = |x: i32, y: i32| -> usize {
        let mx = x.rem_euclid(size as i32);
        let my = y.rem_euclid(size as i32);
        (my as usize) * size as usize + mx as usize
    };

    // Seed: a single deterministic placement, then iterative replacement
    // until the configuration is "rotated to lowest-energy". For
    // simplicity we just iterate placements in best-then-worst-then-best
    // order.
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    let splat = |energy: &mut [f32], cx: i32, cy: i32, sign: f32| {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let r2 = (dx * dx + dy * dy) as f32;
                let w = (-r2 * inv_two_sigma2).exp() * sign;
                let i = idx(cx + dx, cy + dy);
                energy[i] += w;
            }
        }
    };

    for placement in 0..n {
        // Find the cell with the lowest energy among unplaced cells.
        let mut best = 0usize;
        let mut best_e = f32::INFINITY;
        for (i, &e) in energy.iter().enumerate() {
            if !taken[i] && e < best_e {
                best_e = e;
                best = i;
            }
        }
        taken[best] = true;
        ranks[best] = placement as u32;
        let (cx, cy) = xy(best);
        splat(&mut energy, cx, cy, 1.0);
    }

    // Quantise rank → 0..=255.
    let max_rank = (n - 1) as f32;
    ranks
        .iter()
        .map(|&r| ((r as f32 / max_rank) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_noise_distribution_is_uniform() {
        let pixels = generate_blue_noise(BLUE_NOISE_SIZE);
        // The histogram should be roughly flat: every rank ~equally likely.
        let mut hist = [0u32; 16];
        for &p in &pixels {
            hist[(p as usize) >> 4] += 1;
        }
        let avg = pixels.len() as f32 / 16.0;
        for (i, &h) in hist.iter().enumerate() {
            let dev = (h as f32 - avg).abs() / avg;
            assert!(dev < 0.5, "blue-noise bucket {i} = {h}, avg {avg}, dev {dev}");
        }
    }
}
