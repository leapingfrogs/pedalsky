//! Phase 3 §3.3 — `headless_dump` CLI subcommand.
//!
//! Loads `pedalsky.toml` + the configured scene, synthesises a
//! [`WeatherState`] using a *headless* `GpuContext`, reads the textures
//! back, and writes:
//!
//! - `weather_map.png` — 2×2 grid: top-left R (greyscale = coverage),
//!   top-right G (per-cloud-type palette stripe; v1 stays black since
//!   per-pixel cloud type is deferred), bottom-left B (greyscale = base
//!   offset), bottom-right A (blue gradient = precipitation/wetness).
//! - `wind_field_xz_slices.png` — 3 horizontal slices stacked vertically
//!   (low / mid / upper altitude). Each slice is 32×32 with the (u, v, w)
//!   channels visualised by mapping speed to brightness and direction
//!   to hue.
//! - `top_down_density.png` — direct R8 → greyscale.
//! - `weather_dump.json` — full `WeatherState` scalars plus min/max/mean
//!   summaries of every texture channel.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use half::f16;
use image::{ImageBuffer, Rgb};
use ps_core::{Config, GpuContext, Scene, WeatherState, WorldState};
use serde::Serialize;
use tracing::info;

const WEATHER_MAP_SIZE: u32 = 128;
const WIND_FIELD_SIZE: (u32, u32, u32) = (32, 16, 32); // (x, y, z)
const TOP_DOWN_SIZE: u32 = 128;

/// Parse `--headless-dump <dir>` from `args`. Returns `None` when the
/// flag is absent, `Some(path)` when present and well-formed.
pub fn parse_args(args: &[String]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--headless-dump" {
            return it.next().map(PathBuf::from);
        }
        if let Some(rest) = a.strip_prefix("--headless-dump=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

/// Run the dump: synthesise then write all four files into `out_dir`.
pub fn run(workspace_root: &Path, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let config_path = workspace_root.join("pedalsky.toml");
    let config = Config::load(&config_path)?;
    config.validate_with_base(config_path.parent())?;
    let scene_path = if config.paths.weather.is_absolute() {
        config.paths.weather.clone()
    } else {
        workspace_root.join(&config.paths.weather)
    };
    let scene = Scene::load(&scene_path)?;
    scene.validate()?;

    let gpu = ps_core::gpu::init_headless().context("init_headless")?;
    let world = build_world(&config);
    let weather = ps_synthesis::synthesise(&scene, &config, &world, &gpu).context("synthesise")?;

    info!(
        target: "ps_app::headless_dump",
        out = %out_dir.display(),
        cloud_layers = weather.cloud_layer_count,
        haze_per_m = weather.haze_extinction_per_m.x,
        "synthesised; writing dump"
    );

    let weather_map_pixels =
        read_back_2d_rgba16f(&gpu, &weather.textures.weather_map, WEATHER_MAP_SIZE)?;
    let wind_field_pixels =
        read_back_3d_rgba16f(&gpu, &weather.textures.wind_field, WIND_FIELD_SIZE)?;
    let top_down_pixels =
        read_back_2d_r8(&gpu, &weather.textures.top_down_density_mask, TOP_DOWN_SIZE)?;

    write_weather_map_png(&out_dir.join("weather_map.png"), &weather_map_pixels)?;
    write_wind_field_png(
        &out_dir.join("wind_field_xz_slices.png"),
        &wind_field_pixels,
    )?;
    write_top_down_png(&out_dir.join("top_down_density.png"), &top_down_pixels)?;
    write_weather_dump_json(
        &out_dir.join("weather_dump.json"),
        &weather,
        &weather_map_pixels,
        &wind_field_pixels,
        &top_down_pixels,
    )?;
    info!(target: "ps_app::headless_dump", "dump complete");
    Ok(())
}

fn build_world(config: &Config) -> WorldState {
    let initial_utc = crate::config_initial_utc(config);
    WorldState::new(
        initial_utc,
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    )
}

// --- Texture readback ------------------------------------------------------

fn read_back_2d_rgba16f(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    size: u32,
) -> Result<Vec<[f32; 4]>> {
    let bytes_per_pixel = 8u32;
    let unpadded = size * bytes_per_pixel;
    let aligned = align_up(unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hd-staging-2d-rgba16f"),
        size: (aligned * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hd-2d-readback"),
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
    let bytes = map_and_read(gpu, &staging)?;
    let mut out = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let row_start = (y * aligned) as usize;
        for x in 0..size {
            let off = row_start + (x * bytes_per_pixel) as usize;
            let r = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
            let g = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
            let b = f16::from_le_bytes([bytes[off + 4], bytes[off + 5]]).to_f32();
            let a = f16::from_le_bytes([bytes[off + 6], bytes[off + 7]]).to_f32();
            out.push([r, g, b, a]);
        }
    }
    Ok(out)
}

fn read_back_3d_rgba16f(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    (sx, sy, sz): (u32, u32, u32),
) -> Result<Vec<[f32; 4]>> {
    let bytes_per_pixel = 8u32;
    let unpadded = sx * bytes_per_pixel;
    let aligned = align_up(unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hd-staging-3d-rgba16f"),
        size: (aligned * sy * sz) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hd-3d-readback"),
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
                rows_per_image: Some(sy),
            },
        },
        wgpu::Extent3d {
            width: sx,
            height: sy,
            depth_or_array_layers: sz,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    let bytes = map_and_read(gpu, &staging)?;
    let mut out = Vec::with_capacity((sx * sy * sz) as usize);
    for z in 0..sz {
        for y in 0..sy {
            let row_start = ((z * sy + y) * aligned) as usize;
            for x in 0..sx {
                let off = row_start + (x * bytes_per_pixel) as usize;
                let r = f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
                let g = f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32();
                let b = f16::from_le_bytes([bytes[off + 4], bytes[off + 5]]).to_f32();
                let a = f16::from_le_bytes([bytes[off + 6], bytes[off + 7]]).to_f32();
                out.push([r, g, b, a]);
            }
        }
    }
    Ok(out)
}

fn read_back_2d_r8(gpu: &GpuContext, texture: &wgpu::Texture, size: u32) -> Result<Vec<u8>> {
    let unpadded = size;
    let aligned = align_up(unpadded, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hd-staging-r8"),
        size: (aligned * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hd-r8-readback"),
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
    let bytes = map_and_read(gpu, &staging)?;
    let mut out = Vec::with_capacity((size * size) as usize);
    for y in 0..size {
        let row_start = (y * aligned) as usize;
        out.extend_from_slice(&bytes[row_start..row_start + size as usize]);
    }
    Ok(out)
}

fn map_and_read(gpu: &GpuContext, buf: &wgpu::Buffer) -> Result<Vec<u8>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow::anyhow!("device poll: {e:?}"))?;
    rx.recv()
        .map_err(|e| anyhow::anyhow!("map recv: {e}"))?
        .map_err(|e| anyhow::anyhow!("map failed: {e:?}"))?;
    let view = slice.get_mapped_range();
    let bytes = view.to_vec();
    drop(view);
    buf.unmap();
    Ok(bytes)
}

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

// --- PNG writers -----------------------------------------------------------

fn write_weather_map_png(path: &Path, pixels: &[[f32; 4]]) -> Result<()> {
    let s = WEATHER_MAP_SIZE;
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(s * 2, s * 2);
    for y in 0..s {
        for x in 0..s {
            let p = pixels[(y * s + x) as usize];
            // Top-left: R as greyscale (coverage).
            put(&mut img, x, y, grey(p[0]));
            // Top-right: G as red stripe (cloud type — currently 0).
            put(&mut img, x + s, y, [(p[1] * 255.0) as u8, 0, 0]);
            // Bottom-left: B remap [-1, 1] → [0, 255] (cloud-base bias).
            let b_norm = ((p[2] + 1.0) * 0.5).clamp(0.0, 1.0);
            put(&mut img, x, y + s, grey(b_norm));
            // Bottom-right: A as blue gradient (precip / wetness).
            let a = p[3].clamp(0.0, 1.0);
            put(
                &mut img,
                x + s,
                y + s,
                [(a * 60.0) as u8, (a * 80.0) as u8, (a * 200.0 + 55.0) as u8],
            );
        }
    }
    img.save(path).context("save weather_map.png")?;
    Ok(())
}

fn write_wind_field_png(path: &Path, pixels: &[[f32; 4]]) -> Result<()> {
    let (sx, sy, sz) = WIND_FIELD_SIZE;
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(sx, sz * 3);
    let slices = [1u32, sy / 2, sy - 1]; // low, mid, upper
    for (slice_i, &y) in slices.iter().enumerate() {
        for z in 0..sz {
            for x in 0..sx {
                let idx = ((z * sy + y) * sx + x) as usize;
                let p = pixels[idx];
                let speed = (p[0] * p[0] + p[1] * p[1]).sqrt();
                let bright = (speed / 30.0).clamp(0.0, 1.0); // 30 m/s ceiling
                let dir = p[1].atan2(p[0]);
                let hue = (dir + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                let rgb = hsv_to_rgb(hue, 0.6, bright);
                let yy = slice_i as u32 * sz + z;
                put(
                    &mut img,
                    x,
                    yy,
                    [
                        (rgb[0] * 255.0) as u8,
                        (rgb[1] * 255.0) as u8,
                        (rgb[2] * 255.0) as u8,
                    ],
                );
            }
        }
    }
    img.save(path).context("save wind_field_xz_slices.png")?;
    Ok(())
}

fn write_top_down_png(path: &Path, pixels: &[u8]) -> Result<()> {
    let s = TOP_DOWN_SIZE;
    let img = ImageBuffer::<image::Luma<u8>, Vec<u8>>::from_raw(s, s, pixels.to_vec())
        .ok_or_else(|| anyhow::anyhow!("top-down density buffer wrong size"))?;
    img.save(path).context("save top_down_density.png")?;
    Ok(())
}

fn put(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x: u32, y: u32, rgb: [u8; 3]) {
    img.put_pixel(x, y, Rgb(rgb));
}

fn grey(v: f32) -> [u8; 3] {
    let g = (v.clamp(0.0, 1.0) * 255.0) as u8;
    [g, g, g]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as i32 % 6;
    let f = h * 6.0 - h.floor() * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// --- JSON writer -----------------------------------------------------------

#[derive(Serialize)]
struct ChannelStats {
    min: f32,
    max: f32,
    mean: f32,
}

#[derive(Serialize)]
struct WeatherDump {
    cloud_layer_count: u32,
    haze_extinction_per_m: [f32; 3],
    sun_direction: [f32; 3],
    sun_illuminance: [f32; 3],
    surface: SurfaceJson,
    cloud_layers: Vec<CloudLayerJson>,
    weather_map: TextureSummary4,
    wind_field: TextureSummary4,
    top_down_density: TextureSummary1,
}

#[derive(Serialize)]
struct SurfaceJson {
    visibility_m: f32,
    temperature_c: f32,
    wind_dir_deg: f32,
    wind_speed_mps: f32,
    ground_wetness: f32,
}

#[derive(Serialize)]
struct CloudLayerJson {
    cloud_type: u32,
    base_m: f32,
    top_m: f32,
    coverage: f32,
}

#[derive(Serialize)]
struct TextureSummary4 {
    r: ChannelStats,
    g: ChannelStats,
    b: ChannelStats,
    a: ChannelStats,
}

#[derive(Serialize)]
struct TextureSummary1 {
    value: ChannelStats,
}

fn write_weather_dump_json(
    path: &Path,
    weather: &WeatherState,
    weather_map: &[[f32; 4]],
    wind_field: &[[f32; 4]],
    top_down: &[u8],
) -> Result<()> {
    fn stats4(pixels: &[[f32; 4]]) -> TextureSummary4 {
        let chan = |c: usize| -> ChannelStats {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sum = 0.0_f64;
            for p in pixels {
                let v = p[c];
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
            }
            let mean = (sum / pixels.len() as f64) as f32;
            ChannelStats { min, max, mean }
        };
        TextureSummary4 {
            r: chan(0),
            g: chan(1),
            b: chan(2),
            a: chan(3),
        }
    }
    fn stats1(pixels: &[u8]) -> TextureSummary1 {
        let mut min = u8::MAX;
        let mut max = 0u8;
        let mut sum = 0u64;
        for &v in pixels {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as u64;
        }
        let mean = sum as f32 / pixels.len() as f32 / 255.0;
        TextureSummary1 {
            value: ChannelStats {
                min: min as f32 / 255.0,
                max: max as f32 / 255.0,
                mean,
            },
        }
    }

    let dump = WeatherDump {
        cloud_layer_count: weather.cloud_layer_count,
        haze_extinction_per_m: [
            weather.haze_extinction_per_m.x,
            weather.haze_extinction_per_m.y,
            weather.haze_extinction_per_m.z,
        ],
        sun_direction: [
            weather.sun_direction.x,
            weather.sun_direction.y,
            weather.sun_direction.z,
        ],
        sun_illuminance: [
            weather.sun_illuminance.x,
            weather.sun_illuminance.y,
            weather.sun_illuminance.z,
        ],
        surface: SurfaceJson {
            visibility_m: weather.surface.visibility_m,
            temperature_c: weather.surface.temperature_c,
            wind_dir_deg: weather.surface.wind_dir_deg,
            wind_speed_mps: weather.surface.wind_speed_mps,
            ground_wetness: weather.surface.ground_wetness,
        },
        cloud_layers: weather
            .cloud_layers
            .iter()
            .map(|l| CloudLayerJson {
                cloud_type: l.cloud_type,
                base_m: l.base_m,
                top_m: l.top_m,
                coverage: l.coverage,
            })
            .collect(),
        weather_map: stats4(weather_map),
        wind_field: stats4(wind_field),
        top_down_density: stats1(top_down),
    };
    let json = serde_json::to_string_pretty(&dump).context("serialize WeatherDump")?;
    std::fs::write(path, json).context("write weather_dump.json")?;
    Ok(())
}
