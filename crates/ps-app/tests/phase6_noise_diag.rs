//! Phase 6.1 noise volume diagnostic.
//!
//! Bakes the cloud noise textures via `CloudsSubsystem::new` and reads
//! back a handful of texels from each volume to confirm:
//! - Output is non-zero across all four bands.
//! - Values stay in [0, 1] (the bakes write Rgba8Unorm).
//! - There is genuine spatial variation (not a constant value).

use std::sync::OnceLock;

use ps_clouds::CloudsSubsystem;
use ps_core::{Config, GpuContext};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping noise diag — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

#[test]
fn base_noise_volume_has_variation() {
    let Some(gpu) = gpu() else { return };
    let config = Config::default();
    let subsys = CloudsSubsystem::new(&config, gpu);
    let noise = subsys.noise();

    // Read back the entire base volume (128³ × 4 bytes = 8 MiB).
    let pixels = readback_3d_rgba8(gpu, &noise.base, 128);

    // Sanity: histogram each channel; ensure non-degenerate distribution.
    for (channel, name) in ["r", "g", "b", "a"].iter().enumerate() {
        let mut sum: u64 = 0;
        let mut min = u8::MAX;
        let mut max = u8::MIN;
        for chunk in pixels.chunks_exact(4) {
            let v = chunk[channel];
            sum += v as u64;
            min = min.min(v);
            max = max.max(v);
        }
        let avg = sum as f32 / (pixels.len() / 4) as f32;
        eprintln!("base.{name}: min={min} max={max} avg={avg:.1}");
        assert!(max > min + 16, "base.{name}: not enough variation (min={min} max={max})");
        assert!(avg > 8.0 && avg < 248.0, "base.{name}: degenerate avg={avg}");
    }
}

#[test]
fn detail_noise_volume_has_variation() {
    let Some(gpu) = gpu() else { return };
    let config = Config::default();
    let subsys = CloudsSubsystem::new(&config, gpu);
    let noise = subsys.noise();

    let pixels = readback_3d_rgba8(gpu, &noise.detail, 32);
    for (channel, name) in ["r", "g", "b"].iter().enumerate() {
        let mut sum: u64 = 0;
        let mut min = u8::MAX;
        let mut max = u8::MIN;
        for chunk in pixels.chunks_exact(4) {
            let v = chunk[channel];
            sum += v as u64;
            min = min.min(v);
            max = max.max(v);
        }
        let avg = sum as f32 / (pixels.len() / 4) as f32;
        eprintln!("detail.{name}: min={min} max={max} avg={avg:.1}");
        assert!(max > min + 16, "detail.{name}: not enough variation");
    }
}

#[test]
fn curl_tile_has_variation() {
    let Some(gpu) = gpu() else { return };
    let config = Config::default();
    let subsys = CloudsSubsystem::new(&config, gpu);
    let noise = subsys.noise();
    let pixels = readback_2d_rg8(gpu, &noise.curl, 128);
    let mut min_r = u8::MAX;
    let mut max_r = u8::MIN;
    let mut min_g = u8::MAX;
    let mut max_g = u8::MIN;
    for chunk in pixels.chunks_exact(2) {
        min_r = min_r.min(chunk[0]);
        max_r = max_r.max(chunk[0]);
        min_g = min_g.min(chunk[1]);
        max_g = max_g.max(chunk[1]);
    }
    eprintln!("curl: r=[{min_r},{max_r}] g=[{min_g},{max_g}]");
    assert!(max_r > min_r + 32);
    assert!(max_g > min_g + 32);
}

#[test]
fn blue_noise_tile_is_uniform() {
    let Some(gpu) = gpu() else { return };
    let config = Config::default();
    let subsys = CloudsSubsystem::new(&config, gpu);
    let noise = subsys.noise();
    let pixels = readback_2d_r8(gpu, &noise.blue_noise, 64);
    // 16-bin histogram should be roughly flat (void-and-cluster gives
    // an even rank distribution).
    let mut hist = [0u32; 16];
    for &p in &pixels {
        hist[(p as usize) >> 4] += 1;
    }
    let avg = pixels.len() as f32 / 16.0;
    for (i, &h) in hist.iter().enumerate() {
        let dev = (h as f32 - avg).abs() / avg;
        eprintln!("blue_noise bucket {i:2}: count={h:4}, dev={dev:.2}");
        assert!(dev < 0.5, "bucket {i} dev={dev}");
    }
}

fn readback_3d_rgba8(gpu: &GpuContext, texture: &wgpu::Texture, size: u32) -> Vec<u8> {
    let bytes_per_pixel = 4u32;
    let unpadded = size * bytes_per_pixel;
    let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise-diag-staging-3d"),
        size: (aligned * size * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise-diag-copy-3d"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("recv").expect("map");
    let bytes = slice.get_mapped_range().to_vec();
    drop(staging);

    let mut out = Vec::with_capacity((size * size * size * bytes_per_pixel) as usize);
    for z in 0..size {
        for y in 0..size {
            let row_start = ((z * size + y) * aligned) as usize;
            out.extend_from_slice(&bytes[row_start..row_start + (size * bytes_per_pixel) as usize]);
        }
    }
    out
}

fn readback_2d_rg8(gpu: &GpuContext, texture: &wgpu::Texture, size: u32) -> Vec<u8> {
    readback_2d_n(gpu, texture, size, 2)
}

fn readback_2d_r8(gpu: &GpuContext, texture: &wgpu::Texture, size: u32) -> Vec<u8> {
    readback_2d_n(gpu, texture, size, 1)
}

fn readback_2d_n(gpu: &GpuContext, texture: &wgpu::Texture, size: u32, bpp: u32) -> Vec<u8> {
    let unpadded = size * bpp;
    let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise-diag-staging-2d"),
        size: (aligned * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise-diag-copy-2d"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("recv").expect("map");
    let bytes = slice.get_mapped_range().to_vec();

    let mut out = Vec::with_capacity((size * size * bpp) as usize);
    for y in 0..size {
        let row_start = (y * aligned) as usize;
        out.extend_from_slice(&bytes[row_start..row_start + (size * bpp) as usize]);
    }
    out
}
