//! Phase 3 §3.2.4 — 32×32×16 RGBA16Float wind-field volume.
//!
//! Channels: (u, v, w, turbulence). The Y axis maps `[0, top_m]` to
//! `v ∈ [0, 1]` (texture-space Y). Synthesised from the surface wind in
//! the parsed scene using one of two horizontal profiles:
//!
//! - **Real winds aloft** (Phase 14.B) — when
//!   `scene.surface.winds_aloft` is non-empty (typical for live
//!   Open-Meteo feeds), the (u, v) channels are interpolated directly
//!   between the ingested pressure-level samples. The 10 m surface
//!   wind anchors the bottom of the profile; the lowest ingested
//!   sample (usually 850 hPa ≈ 1.5 km) anchors the next point. The
//!   power-law is used only as a fallback below the lowest ingested
//!   level to fill the boundary layer where Open-Meteo's coarse
//!   sampling under-samples shear.
//! - **Procedural fallback** — when `winds_aloft` is empty (synthetic
//!   scenes / offline renders), the horizontal profile uses:
//!   * **1/7 power law** (Hellmann's exponent α = 0.143, neutral
//!     stability): `u(z) = u_surf · (z / z_ref)^α`, with `z_ref = 10 m`.
//!   * **Ekman spiral** up to 30° clockwise (Northern Hemisphere) by
//!     `top_m`.
//!
//! Either way:
//!
//! - **w** = 0 plus a small Gaussian thermal under any low-base cumulus
//!   layer (peak ≈ 0.5 m/s, σ ≈ 1500 m horizontally, peaks at base − 200 m).
//!   Open-Meteo doesn't expose vertical velocity (omega), so this is
//!   procedural in both branches.
//! - **turbulence** rises near cloud bases and inside CB layers.
//!
//! Anchored at world origin; horizontal extent = 32 km.

use half::f16;
use ps_core::{CloudType, Scene};

/// X-axis pixel count.
pub const SIZE_X: u32 = 32;
/// Z-axis pixel count.
pub const SIZE_Z: u32 = 32;
/// Y-axis pixel count.
pub const SIZE_Y: u32 = 16;

/// Horizontal extent in metres (square, centred on world origin).
pub const EXTENT_M: f32 = 32_000.0;
/// Top altitude AGL in metres (texture Y axis spans [0, top_m]).
pub const TOP_M: f32 = 12_000.0;

/// 10-metre reference height for the 1/7 power-law (Hellmann α).
const Z_REF_M: f32 = 10.0;
/// Hellmann's exponent (neutral stability).
const ALPHA: f32 = 0.143;
/// Maximum Ekman veer angle (radians) at `TOP_M`. 30° clockwise (NH).
const EKMAN_MAX_RAD: f32 = 30.0 * std::f32::consts::PI / 180.0;

/// CPU-side wind field, ready for upload.
#[derive(Debug, Clone)]
pub struct WindField {
    /// Tightly packed `f16` RGBA voxels in (x, y, z) order with a Y-major
    /// memory layout (`pixels[((z * SIZE_Y + y) * SIZE_X + x) * 4 + c]`).
    pub pixels: Vec<f16>,
    /// X / Y / Z dimensions.
    pub size: (u32, u32, u32),
    /// Horizontal extent in metres.
    pub extent_m: f32,
    /// Y-axis top altitude in metres.
    pub top_m: f32,
}

impl WindField {
    /// Synthesise from the parsed scene.
    pub fn synthesise(scene: &Scene) -> Self {
        let surface_speed = scene.surface.wind_speed_mps;
        // "wind_dir_deg" is the meteorological direction the wind blows
        // FROM. Convert to the world-space direction the wind blows TO:
        // toward = wind_dir + 180°. The vector below targets PedalSky's
        // (+X east, +Z south) horizontal plane.
        let dir_to_rad = (scene.surface.wind_dir_deg + 180.0).to_radians();

        // Phase 14.B — if Open-Meteo gave us winds aloft, build a
        // (u, v) anchor table that the per-voxel sampler interpolates
        // across. The surface 10 m wind always sits at the bottom of
        // the table so the lowest voxel reads identically to before;
        // ingested samples extend the profile to whatever the API
        // gave us. The table is sorted by altitude.
        let aloft = AloftProfile::build(scene, surface_speed, dir_to_rad);

        let mut pixels = vec![f16::from_f32(0.0); (SIZE_X * SIZE_Y * SIZE_Z * 4) as usize];

        for z_i in 0..SIZE_Z {
            for y_i in 0..SIZE_Y {
                for x_i in 0..SIZE_X {
                    let alt_m = (y_i as f32 + 0.5) / SIZE_Y as f32 * TOP_M;
                    let world_x = pixel_to_world_axis(x_i, SIZE_X);
                    let world_z = pixel_to_world_axis(z_i, SIZE_Z);

                    let (u, v_h) = aloft.sample(surface_speed, dir_to_rad, alt_m);
                    let w = vertical_wind(scene, world_x, world_z, alt_m);
                    let turb = turbulence(scene, alt_m);

                    let idx = ((z_i * SIZE_Y * SIZE_X + y_i * SIZE_X + x_i) * 4) as usize;
                    pixels[idx] = f16::from_f32(u);
                    pixels[idx + 1] = f16::from_f32(v_h);
                    pixels[idx + 2] = f16::from_f32(w);
                    pixels[idx + 3] = f16::from_f32(turb);
                }
            }
        }

        Self {
            pixels,
            size: (SIZE_X, SIZE_Y, SIZE_Z),
            extent_m: EXTENT_M,
            top_m: TOP_M,
        }
    }

    /// Allocate + upload a 3D RGBA16Float texture.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let (sx, sy, sz) = self.size;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wind-field"),
            size: wgpu::Extent3d {
                width: sx,
                height: sy,
                depth_or_array_layers: sz,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
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
                bytes_per_row: Some(sx * 8),
                rows_per_image: Some(sy),
            },
            wgpu::Extent3d {
                width: sx,
                height: sy,
                depth_or_array_layers: sz,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

/// Power-law profile + Ekman veer. Returns `(u_x_east, w_z_south)` where
/// the channels carry world-space horizontal components. Used as the
/// fallback when `scene.surface.winds_aloft` is empty.
fn horizontal_wind(surface_speed: f32, dir_to_rad: f32, alt_m: f32) -> (f32, f32) {
    let speed = surface_speed * (alt_m.max(Z_REF_M) / Z_REF_M).powf(ALPHA);
    // Veer fraction: 0 at surface, 1 at TOP_M, smooth in between.
    let veer = (alt_m / TOP_M).clamp(0.0, 1.0).powf(0.7);
    let veered_rad = dir_to_rad + veer * EKMAN_MAX_RAD;
    // PedalSky world: +X east, +Z south. Meteorological direction
    // measured clockwise from north → x = sin, z = cos (south = +Z).
    // Wind blowing TOWARD direction (sin, -cos) is:
    let dir_x = veered_rad.sin();
    let dir_z = -veered_rad.cos();
    (speed * dir_x, speed * dir_z)
}

/// Convert a meteorological direction-from (degrees) + speed to a
/// world-space (u_east, v_south) m/s pair, matching the conventions
/// in [`horizontal_wind`].
fn dir_speed_to_uv(speed_mps: f32, dir_from_deg: f32) -> (f32, f32) {
    let dir_to_rad = (dir_from_deg + 180.0).to_radians();
    (speed_mps * dir_to_rad.sin(), -speed_mps * dir_to_rad.cos())
}

/// Interpolation anchors for the horizontal wind profile. When real
/// winds aloft are present, this carries one entry at the 10 m
/// surface anchor plus one per ingested pressure-level sample, sorted
/// by ascending altitude. Interpolating (u, v) directly handles
/// shortest-angle wrap implicitly — a 350° → 010° transition lerps
/// through 000° rather than dragging back through 180° the way raw
/// degrees would.
///
/// When no winds aloft are present, the anchors vec is empty and
/// `sample` defers to the procedural [`horizontal_wind`].
struct AloftProfile {
    anchors: Vec<AloftAnchor>,
}

struct AloftAnchor {
    alt_m: f32,
    u: f32,
    v: f32,
}

impl AloftProfile {
    fn build(scene: &Scene, surface_speed: f32, dir_to_rad: f32) -> Self {
        if scene.surface.winds_aloft.is_empty() {
            return Self { anchors: Vec::new() };
        }
        let mut anchors: Vec<AloftAnchor> = Vec::with_capacity(scene.surface.winds_aloft.len() + 1);

        // Anchor 0: the 10 m surface wind. Encode the same (u, v) the
        // fallback would produce at z = 10 m so the lowest voxel is
        // continuous with the boundary-layer profile.
        let (u_surf, v_surf) = {
            // At z = Z_REF_M the power law multiplier is 1; reuse
            // the conversion to keep the conventions identical.
            let dir_x = dir_to_rad.sin();
            let dir_z = -dir_to_rad.cos();
            (surface_speed * dir_x, surface_speed * dir_z)
        };
        anchors.push(AloftAnchor {
            alt_m: Z_REF_M,
            u: u_surf,
            v: v_surf,
        });

        // Append the ingested samples in their stored order
        // (ascending altitude). Skip any that aren't strictly above
        // the previous anchor — otherwise the bracket search below
        // can land on a zero-width segment.
        let mut last_alt = Z_REF_M;
        for s in &scene.surface.winds_aloft {
            if !(s.altitude_m > last_alt + 1.0) {
                continue;
            }
            let (u, v) = dir_speed_to_uv(s.speed_mps, s.dir_deg);
            anchors.push(AloftAnchor {
                alt_m: s.altitude_m,
                u,
                v,
            });
            last_alt = s.altitude_m;
        }

        Self { anchors }
    }

    /// Sample the profile at `alt_m`. When real-aloft anchors are
    /// present, linearly interpolate (u, v) between the two bracketing
    /// anchors; clamp at the highest anchor for `alt_m > top_anchor`.
    /// Below the surface anchor (rare — Z_REF_M = 10 m), clamp to the
    /// surface anchor. When the anchors vec is empty, defer to the
    /// procedural power-law + Ekman fallback.
    fn sample(&self, surface_speed: f32, dir_to_rad: f32, alt_m: f32) -> (f32, f32) {
        if self.anchors.is_empty() {
            return horizontal_wind(surface_speed, dir_to_rad, alt_m);
        }
        // Below the lowest anchor: clamp.
        if alt_m <= self.anchors[0].alt_m {
            return (self.anchors[0].u, self.anchors[0].v);
        }
        // Above the highest anchor: clamp. Open-Meteo's top wind
        // sample (300 hPa ≈ 9 km) sits below TOP_M (12 km); freezing
        // the wind there is a sensible default — the upper
        // troposphere doesn't shift further much in 3 km.
        let top = self.anchors.last().unwrap();
        if alt_m >= top.alt_m {
            return (top.u, top.v);
        }
        // Bracket. The vec is short (≤ 5 entries) so a linear walk
        // beats a binary search by a constant factor.
        for pair in self.anchors.windows(2) {
            let lo = &pair[0];
            let hi = &pair[1];
            if alt_m >= lo.alt_m && alt_m <= hi.alt_m {
                let t = (alt_m - lo.alt_m) / (hi.alt_m - lo.alt_m);
                let u = lo.u + t * (hi.u - lo.u);
                let v = lo.v + t * (hi.v - lo.v);
                return (u, v);
            }
        }
        // Unreachable given the clamps above; play it safe.
        (top.u, top.v)
    }
}


/// Small Gaussian thermal column under any low-base cumulus layer.
fn vertical_wind(scene: &Scene, world_x: f32, world_z: f32, alt_m: f32) -> f32 {
    let mut w = 0.0_f32;
    for layer in &scene.clouds.layers {
        if !matches!(
            layer.cloud_type,
            CloudType::Cumulus | CloudType::Cumulonimbus
        ) {
            continue;
        }
        // Thermal peak just below cloud base.
        let peak_alt = (layer.base_m - 200.0).max(50.0);
        let alt_falloff = f32::exp(-((alt_m - peak_alt).powi(2)) / (2.0 * 800.0_f32.powi(2)));
        // Spatial bumps: deterministic checker of thermals on a 3 km grid.
        let cell_x = (world_x / 3000.0).round();
        let cell_z = (world_z / 3000.0).round();
        let active = ((cell_x as i32 + cell_z as i32) & 1) == 0;
        if !active {
            continue;
        }
        let dx = world_x - cell_x * 3000.0;
        let dz = world_z - cell_z * 3000.0;
        let r2 = dx * dx + dz * dz;
        let horiz_falloff = f32::exp(-r2 / (2.0 * 1500.0_f32.powi(2)));
        let amp = match layer.cloud_type {
            CloudType::Cumulonimbus => 2.5,
            _ => 0.5,
        };
        w += amp * alt_falloff * horiz_falloff;
    }
    w
}

/// Per-pixel turbulence factor in [0, 1+].
fn turbulence(scene: &Scene, alt_m: f32) -> f32 {
    let mut turb = 0.05_f32; // baseline boundary-layer turbulence
    for layer in &scene.clouds.layers {
        // Ramp up below cloud base, peak at base, taper inside.
        if alt_m >= layer.base_m - 600.0 && alt_m <= layer.top_m + 200.0 {
            let dist_to_base = (alt_m - layer.base_m).abs();
            let factor = (-dist_to_base / 400.0).exp().clamp(0.0, 1.0);
            let amp = match layer.cloud_type {
                CloudType::Cumulonimbus => 1.0,
                CloudType::Cumulus => 0.4,
                _ => 0.2,
            };
            turb = turb.max(amp * factor);
        }
    }
    turb.min(1.5)
}

fn pixel_to_world_axis(i: u32, n: u32) -> f32 {
    let f = (i as f32 + 0.5) / n as f32;
    f * EXTENT_M - EXTENT_M * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_scene() -> Scene {
        Scene {
            schema_version: 1,
            surface: ps_core::Surface {
                wind_speed_mps: 5.0,
                wind_dir_deg: 240.0,
                ..Default::default()
            },
            clouds: ps_core::Clouds::default(),
            precipitation: ps_core::Precipitation::default(),
            lightning: ps_core::Lightning::default(),
            aurora: ps_core::Aurora::default(),
            water: None,
        }
    }

    #[test]
    fn wind_field_dimensions() {
        let wf = WindField::synthesise(&empty_scene());
        assert_eq!(wf.size, (SIZE_X, SIZE_Y, SIZE_Z));
        assert_eq!(wf.pixels.len(), (SIZE_X * SIZE_Y * SIZE_Z * 4) as usize);
    }

    /// 1/7 power law: at z = 100 m, speed should be (100/10)^0.143 ≈ 1.39
    /// times the surface (10 m) value.
    #[test]
    fn power_law_aloft_amplifies_surface_wind() {
        let scene = empty_scene();
        let wf = WindField::synthesise(&scene);
        // Sample y=0 (≈375 m) should be > surface speed; near-Z_REF should
        // be ~ surface speed.
        let surface_speed = scene.surface.wind_speed_mps;
        // Voxel at lowest Y row, mid X/Z.
        // y_i = 0, so the "0 * SIZE_X" row offset drops out — kept implicit
        // for readability. Voxel = (mid X, lowest Y, mid Z).
        let idx_low = ((SIZE_Z / 2 * SIZE_Y * SIZE_X + SIZE_X / 2) * 4) as usize;
        let u_low = wf.pixels[idx_low].to_f32();
        let v_low = wf.pixels[idx_low + 1].to_f32();
        let speed_low = (u_low * u_low + v_low * v_low).sqrt();
        // 375 m altitude → (375/10)^0.143 ≈ 1.66 × 5 = 8.3 m/s
        assert!(
            (7.0..=10.0).contains(&speed_low),
            "low-altitude speed = {speed_low} (expected ~8.3 from {surface_speed} surface)"
        );
        // Voxel at highest Y row should be even faster.
        let idx_high =
            ((SIZE_Z / 2 * SIZE_Y * SIZE_X + (SIZE_Y - 1) * SIZE_X + SIZE_X / 2) * 4) as usize;
        let u_high = wf.pixels[idx_high].to_f32();
        let v_high = wf.pixels[idx_high + 1].to_f32();
        let speed_high = (u_high * u_high + v_high * v_high).sqrt();
        assert!(
            speed_high > speed_low,
            "speed should increase with altitude: low={speed_low}, high={speed_high}"
        );
    }

    /// Phase 14.B — when winds aloft are present, the synthesised
    /// (u, v) at the highest ingested sample's altitude matches the
    /// sample's value exactly. The interpolation must round-trip the
    /// ingested data when sampled on-anchor.
    #[test]
    fn aloft_samples_round_trip_at_anchor_altitude() {
        let mut scene = empty_scene();
        // Two synthetic samples — 850 hPa @ 1500 m blowing east
        // (270° meteorological = wind FROM west = blowing east),
        // 500 hPa @ 5500 m blowing south (000° meteorological).
        scene.surface.winds_aloft = vec![
            ps_core::WindAloftSample {
                pressure_hpa: 850,
                altitude_m: 1500.0,
                speed_mps: 10.0,
                dir_deg: 270.0,
            },
            ps_core::WindAloftSample {
                pressure_hpa: 500,
                altitude_m: 5500.0,
                speed_mps: 30.0,
                dir_deg: 0.0,
            },
        ];

        // Build the profile directly and sample it at the anchor
        // altitudes. The wind-field synthesis goes through this same
        // code path.
        let dir_to_rad = (scene.surface.wind_dir_deg + 180.0).to_radians();
        let profile = AloftProfile::build(&scene, scene.surface.wind_speed_mps, dir_to_rad);
        // 850 hPa anchor: dir_FROM = 270° → dir_TO = 090° → (u, v) = (+10, 0)
        // (i.e. wind blows east, since +X is east in PedalSky).
        let (u_850, v_850) = profile.sample(scene.surface.wind_speed_mps, dir_to_rad, 1500.0);
        assert!(
            (u_850 - 10.0).abs() < 0.001,
            "u at 850 hPa anchor: {u_850} (expected 10.0)"
        );
        assert!(v_850.abs() < 0.001, "v at 850 hPa anchor: {v_850}");

        // 500 hPa anchor: dir_FROM = 000° → dir_TO = 180° (south).
        // In PedalSky +Z is south, so (u, v) = (0, +30).
        let (u_500, v_500) = profile.sample(scene.surface.wind_speed_mps, dir_to_rad, 5500.0);
        assert!(u_500.abs() < 0.001, "u at 500 hPa anchor: {u_500}");
        assert!(
            (v_500 - 30.0).abs() < 0.001,
            "v at 500 hPa anchor: {v_500} (expected 30.0)"
        );
    }

    /// Phase 14.B — interpolation between two anchors lies on the
    /// line between their (u, v) vectors. Direction interpolation
    /// must go through shortest-angle wrap (the (u, v) lerp gives
    /// this for free).
    #[test]
    fn aloft_interpolation_lerps_between_anchors() {
        let mut scene = empty_scene();
        // Anchors at 1000 m and 3000 m blowing east and west
        // respectively at the same speed. The midpoint should have
        // zero net horizontal wind.
        scene.surface.winds_aloft = vec![
            ps_core::WindAloftSample {
                pressure_hpa: 900,
                altitude_m: 1000.0,
                speed_mps: 10.0,
                dir_deg: 270.0, // FROM west → TO east → +X
            },
            ps_core::WindAloftSample {
                pressure_hpa: 700,
                altitude_m: 3000.0,
                speed_mps: 10.0,
                dir_deg: 90.0, // FROM east → TO west → -X
            },
        ];
        let dir_to_rad = (scene.surface.wind_dir_deg + 180.0).to_radians();
        let profile = AloftProfile::build(&scene, scene.surface.wind_speed_mps, dir_to_rad);
        let (u_mid, v_mid) = profile.sample(scene.surface.wind_speed_mps, dir_to_rad, 2000.0);
        assert!(
            u_mid.abs() < 0.1,
            "midpoint u between east and west should cancel: {u_mid}"
        );
        assert!(
            v_mid.abs() < 0.1,
            "midpoint v should be near zero: {v_mid}"
        );
    }

    /// Phase 14.B — the empty-aloft path is unchanged from the
    /// pre-14 power-law + Ekman behaviour. Regression guard.
    #[test]
    fn empty_aloft_uses_power_law_fallback() {
        let scene = empty_scene();
        assert!(scene.surface.winds_aloft.is_empty());
        let wf = WindField::synthesise(&scene);
        // Same expectation as `power_law_aloft_amplifies_surface_wind`:
        // y=0 voxel @ ~375 m → speed in [7, 10] m/s.
        let idx_low = ((SIZE_Z / 2 * SIZE_Y * SIZE_X + SIZE_X / 2) * 4) as usize;
        let u_low = wf.pixels[idx_low].to_f32();
        let v_low = wf.pixels[idx_low + 1].to_f32();
        let speed_low = (u_low * u_low + v_low * v_low).sqrt();
        assert!(
            (7.0..=10.0).contains(&speed_low),
            "fallback path produced wrong speed: {speed_low}"
        );
    }

    #[test]
    fn turbulence_increases_under_cumulus() {
        let mut scene = empty_scene();
        scene.clouds.layers.push(ps_core::CloudLayer {
            cloud_type: CloudType::Cumulus,
            base_m: 1500.0,
            top_m: 2300.0,
            coverage: 0.6,
            density_scale: None,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
            anvil_bias: None,
            droplet_diameter_um: None,
        });
        let t_clear = turbulence(&empty_scene(), 1500.0);
        let t_cumulus = turbulence(&scene, 1500.0);
        assert!(
            t_cumulus > t_clear + 0.1,
            "cumulus base should raise turbulence (clear={t_clear}, cu={t_cumulus})"
        );
    }
}
