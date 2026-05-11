//! Phase 3 §3.2.3 — 128×128 RGBA16Float weather-map texture.
//!
//! Channels:
//! - **R** = coverage in [0, 1] (union of gridded NWP coverage + point
//!   observations; for v1 we only support uniform per-layer coverage and
//!   the optional `coverage_grid`).
//! - **G** = reserved (per-pixel cloud type lives in a separate
//!   R8Uint texture, see [`CloudTypeGrid`]).
//! - **B** = relative cloud-base offset in [-1, 1] of the layer thickness.
//! - **A** = local precipitation intensity / ground-wetness scalar.
//!
//! Anchored to world origin; spatial extent = 32 km. Built CPU-side via
//! `half::f16` and uploaded via `queue.write_texture`.
//!
//! Phase 12.1 — when the scene has `[clouds.coverage_grid]` configured
//! the R channel is loaded from `data_path` (row-major little-endian
//! f32) and resampled onto the 128×128 grid via bilinear interp.
//! When `type_path` is also set, the per-pixel cloud type lives in a
//! companion R8Uint texture (see [`CloudTypeGrid`]) sampled with
//! nearest filtering — interpolating type indices is meaningless.

use half::f16;
use ps_core::{Scene, Surface};
use tracing::warn;

/// Map dimensions (square).
pub const SIZE: u32 = 128;
/// Spatial extent in metres (square, centred on world origin).
pub const EXTENT_M: f32 = 32_000.0;

/// CPU-side weather map: row-major `f16` quadruples (R, G, B, A) of
/// length `SIZE * SIZE * 4`.
#[derive(Debug, Clone)]
pub struct WeatherMap {
    /// Tightly packed `f16` RGBA pixels, top-left origin.
    pub pixels: Vec<f16>,
    /// Width / height in pixels.
    pub size: u32,
    /// Spatial extent in metres.
    pub extent_m: f32,
}

impl WeatherMap {
    /// Synthesise the weather map from the parsed scene + surface.
    pub fn synthesise(scene: &Scene, surface: &Surface) -> Self {
        let mut pixels = vec![f16::from_f32(0.0); (SIZE * SIZE * 4) as usize];

        // The R channel is a *spatial gate* in [0, 1]: 1 means "this
        // region has clouds", 0 means "clear". For scenes without a
        // coverage_grid the gate is uniformly 1 wherever any cloud
        // layer is configured, so the per-layer coverage value (already
        // remapped onto the visible band by `synthesise_cloud_layers`)
        // controls density entirely. With a grid the gate carries the
        // per-pixel spatial pattern.
        let scene_has_layers = !scene.clouds.layers.is_empty();
        let base_coverage_gate: f32 = if scene_has_layers { 1.0 } else { 0.0 };

        // Phase 12.1 — load the gridded coverage once. None when the
        // scene has no grid OR when the file failed to load (the
        // loader logs the reason); per-pixel sampling falls back to
        // the scalar gate below.
        let coverage_grid = load_coverage_grid(scene);

        // Precipitation intensity normalised to [0, 1] using a 50 mm/h
        // ceiling (heaviest expected rain in the v1 scene library). The
        // shaders re-scale by intensity_mm_per_h; the alpha channel here
        // is a per-pixel multiplier, currently spatially uniform until
        // we add gridded precip in v2.
        let precip_alpha = (scene.precipitation.intensity_mm_per_h / 50.0).clamp(0.0, 1.0);
        let wetness_alpha = surface.wetness.ground_wetness.clamp(0.0, 1.0);
        // The plan §3.2.3 description bundles "precip intensity / ground
        // wetness scalar" into a single channel; take the larger.
        let alpha = precip_alpha.max(wetness_alpha);

        // Cloud-base offset noise: deterministic 2D value-noise sampled
        // per-pixel, scaled so the offset stays in [-1, 1]. We use a
        // tiny in-place hash so synthesis is reproducible without a
        // dependency on a separate noise crate.
        for y in 0..SIZE {
            for x in 0..SIZE {
                let idx = ((y * SIZE + x) * 4) as usize;
                let (gx, gy) = pixel_to_world((x, y));
                let coverage = match coverage_grid.as_ref() {
                    Some(g) => sample_gridded_coverage(g, gx, gy),
                    None => base_coverage_gate,
                };
                let bias = value_noise_2d(x as f32 * 0.05, y as f32 * 0.05) * 2.0 - 1.0;
                pixels[idx] = f16::from_f32(coverage);
                pixels[idx + 1] = f16::from_f32(0.0); // reserved
                pixels[idx + 2] = f16::from_f32(bias);
                pixels[idx + 3] = f16::from_f32(alpha);
            }
        }

        Self {
            pixels,
            size: SIZE,
            extent_m: EXTENT_M,
        }
    }

    /// Allocate a wgpu texture and upload `self`. Returns the texture and
    /// its default view.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("weather-map"),
            size: wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size * 8), // 4 channels × f16
                rows_per_image: Some(self.size),
            },
            wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

/// Map pixel index to world (x, z) in metres, anchored at origin.
fn pixel_to_world((x, y): (u32, u32)) -> (f32, f32) {
    let half = EXTENT_M * 0.5;
    let fx = (x as f32 + 0.5) / SIZE as f32; // [0, 1]
    let fy = (y as f32 + 0.5) / SIZE as f32;
    (fx * EXTENT_M - half, fy * EXTENT_M - half)
}

/// Cached, loaded-from-disk coverage grid + spatial metadata. Built
/// once per `WeatherMap::synthesise` call; sampled per-pixel via
/// bilinear interpolation.
struct LoadedCoverageGrid {
    /// Row-major f32 values in [0, 1] — one per source-grid pixel.
    data: Vec<f32>,
    /// Source grid dimensions (width, height).
    src_w: u32,
    src_h: u32,
    /// Spatial extent in metres. The grid covers
    /// `[-extent/2, +extent/2]` on each axis, centred on world origin.
    extent_m: f32,
}

impl LoadedCoverageGrid {
    /// Bilinear-sample at a source-grid UV in [0, 1]^2.
    fn sample(&self, u: f32, v: f32) -> f32 {
        let fx = u.clamp(0.0, 1.0) * (self.src_w as f32 - 1.0);
        let fy = v.clamp(0.0, 1.0) * (self.src_h as f32 - 1.0);
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.src_w - 1);
        let y1 = (y0 + 1).min(self.src_h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let idx = |x: u32, y: u32| (y * self.src_w + x) as usize;
        let v00 = self.data[idx(x0, y0)];
        let v10 = self.data[idx(x1, y0)];
        let v01 = self.data[idx(x0, y1)];
        let v11 = self.data[idx(x1, y1)];
        let bottom = v00 + (v10 - v00) * tx;
        let top = v01 + (v11 - v01) * tx;
        bottom + (top - bottom) * ty
    }
}

/// Try to load the coverage grid for a scene. Returns `Ok(None)` when
/// the scene has no `coverage_grid` block (the common case);
/// `Ok(Some(_))` on successful load; `Err(_)` on file-IO or
/// size-mismatch errors. Errors are logged at `warn` level and the
/// caller falls back to per-layer scalar coverage.
fn load_coverage_grid(scene: &Scene) -> Option<LoadedCoverageGrid> {
    let grid = scene.clouds.coverage_grid.as_ref()?;
    let [src_w, src_h] = grid.size;
    let expected_bytes = (src_w as usize) * (src_h as usize) * 4;
    let bytes = match std::fs::read(&grid.data_path) {
        Ok(b) => b,
        Err(e) => {
            warn!(
                target: "ps_synthesis::weather_map",
                path = %grid.data_path.display(),
                error = %e,
                "coverage grid read failed; falling back to scalar coverage",
            );
            return None;
        }
    };
    if bytes.len() != expected_bytes {
        warn!(
            target: "ps_synthesis::weather_map",
            path = %grid.data_path.display(),
            got = bytes.len(),
            want = expected_bytes,
            "coverage grid size mismatch; falling back to scalar coverage",
        );
        return None;
    }
    let mut data = Vec::with_capacity((src_w as usize) * (src_h as usize));
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(LoadedCoverageGrid {
        data,
        src_w,
        src_h,
        extent_m: grid.extent_m,
    })
}

/// Look up gridded coverage at world (x, z) when a grid is loaded.
/// Maps world XZ → grid UV using the grid's own extent (which may
/// differ from [`EXTENT_M`] — the weather map resamples onto its own
/// 128×128). Returns the bilinear-interpolated coverage in [0, 1],
/// clamp-to-edge outside the grid's extent.
fn sample_gridded_coverage(grid: &LoadedCoverageGrid, x: f32, y: f32) -> f32 {
    let half = grid.extent_m * 0.5;
    let u = (x + half) / grid.extent_m;
    let v = (y + half) / grid.extent_m;
    grid.sample(u, v).clamp(0.0, 1.0)
}

/// Tiny deterministic 2D value-noise so synthesis is reproducible without
/// pulling in an extra dependency. Output in [0, 1].
fn value_noise_2d(x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let h00 = hash2(xi as i32, yi as i32);
    let h10 = hash2(xi as i32 + 1, yi as i32);
    let h01 = hash2(xi as i32, yi as i32 + 1);
    let h11 = hash2(xi as i32 + 1, yi as i32 + 1);
    let sx = xf * xf * (3.0 - 2.0 * xf);
    let sy = yf * yf * (3.0 - 2.0 * yf);
    let bottom = h00 + (h10 - h00) * sx;
    let top = h01 + (h11 - h01) * sx;
    bottom + (top - bottom) * sy
}

fn hash2(x: i32, y: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(2654435761);
    h ^= (y as u32).wrapping_mul(2246822519);
    h ^= h >> 16;
    h = h.wrapping_mul(2654435761);
    h ^= h >> 13;
    (h as f32 / u32::MAX as f32).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::{Clouds, Lightning, Precipitation, Scene, Surface};

    fn empty_scene() -> Scene {
        Scene {
            schema_version: 1,
            surface: Surface::default(),
            clouds: Clouds::default(),
            precipitation: Precipitation::default(),
            lightning: Lightning::default(),
        }
    }

    #[test]
    fn weather_map_dimensions_and_extent() {
        let scene = empty_scene();
        let wm = WeatherMap::synthesise(&scene, &Surface::default());
        assert_eq!(wm.size, 128);
        assert_eq!(wm.extent_m, 32_000.0);
        assert_eq!(wm.pixels.len(), (128 * 128 * 4) as usize);
    }

    #[test]
    fn coverage_channel_is_full_when_layers_present() {
        let mut scene = empty_scene();
        scene.clouds.layers.push(ps_core::CloudLayer {
            cloud_type: ps_core::CloudType::Cumulus,
            base_m: 1500.0,
            top_m: 2300.0,
            coverage: 0.6,
            density_scale: 1.0,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
        });
        let wm = WeatherMap::synthesise(&scene, &Surface::default());
        // R is now a *spatial gate* (plan §3.2.3 followup #57): with no
        // coverage_grid configured the gate is uniformly 1.0 wherever
        // any cloud layer exists. Per-layer coverage controls density
        // entirely via synthesise_cloud_layers' visible-band remap.
        for &i in &[0_usize, 4 * 64 * 128 + 4 * 64, wm.pixels.len() - 4] {
            let r = wm.pixels[i].to_f32();
            assert!((r - 1.0).abs() < 1e-3, "R={r} (expected 1.0)");
        }
    }

    #[test]
    fn coverage_channel_is_zero_when_no_layers() {
        let scene = empty_scene();
        let wm = WeatherMap::synthesise(&scene, &Surface::default());
        for &i in &[0_usize, 4 * 64 * 128 + 4 * 64, wm.pixels.len() - 4] {
            let r = wm.pixels[i].to_f32();
            assert_eq!(r, 0.0, "R={r} (expected 0)");
        }
    }

    #[test]
    fn pixel_to_world_centres_at_origin() {
        let (cx, cy) = pixel_to_world((63, 63));
        // Pixel (63,63) is just left/below world origin.
        assert!(cx.abs() < EXTENT_M / SIZE as f32);
        assert!(cy.abs() < EXTENT_M / SIZE as f32);
    }
}
