//! Section 1.1 — bicubic (Catmull-Rom) upsampling.
//!
//! Pure-CPU upsample from the source DEM grid to the working
//! resolution. Doesn't add real detail; just provides a denser grid
//! for the erosion stages to operate on. Catmull-Rom (a = -0.5) is
//! the standard choice — sharp without ringing.

use crate::tile::HeightmapTile;

/// Bicubic (Catmull-Rom) upsample. Returns a new tile at the requested
/// `(target_w, target_h)` covering the same geographic extent. If the
/// source is already at or above the target resolution, returns the
/// input unchanged.
pub fn bicubic_upsample(src: HeightmapTile, target_w: u32, target_h: u32) -> HeightmapTile {
    if src.width >= target_w && src.height >= target_h {
        return src;
    }

    let sw = src.width as usize;
    let sh = src.height as usize;
    let tw = target_w as usize;
    let th = target_h as usize;
    let mut out = Vec::with_capacity(tw * th);

    // Map output pixel centre to source pixel coordinates. The
    // outer-pixel centres of source and target both align with the
    // tile's geographic edges, so the mapping is a uniform scale.
    let sx_per_tx = (sw as f32 - 1.0) / (tw as f32 - 1.0).max(1.0);
    let sy_per_ty = (sh as f32 - 1.0) / (th as f32 - 1.0).max(1.0);

    for ty in 0..th {
        let sy_f = ty as f32 * sy_per_ty;
        let sy0 = sy_f.floor() as i32;
        let dy = sy_f - sy0 as f32;
        for tx in 0..tw {
            let sx_f = tx as f32 * sx_per_tx;
            let sx0 = sx_f.floor() as i32;
            let dx = sx_f - sx0 as f32;
            out.push(sample_catmull_rom(&src.heights_m, sw, sh, sx0, sy0, dx, dy));
        }
    }

    HeightmapTile {
        heights_m: out,
        width: target_w,
        height: target_h,
        // GSD scales with the new pixel density.
        gsd_m_centre: src.gsd_m_centre * (sw as f32 - 1.0) / (tw as f32 - 1.0).max(1.0),
        extent_deg: src.extent_deg,
        source: src.source,
    }
}

/// Sample a 4×4 Catmull-Rom interpolation around grid point
/// `(sx0+dx, sy0+dy)` with dx, dy ∈ [0, 1].
fn sample_catmull_rom(
    h: &[f32],
    w: usize,
    height: usize,
    sx0: i32,
    sy0: i32,
    dx: f32,
    dy: f32,
) -> f32 {
    let mut rows = [0.0_f32; 4];
    for j in -1..=2 {
        let sy = (sy0 + j).clamp(0, height as i32 - 1) as usize;
        let mut cols = [0.0_f32; 4];
        for i in -1..=2 {
            let sx = (sx0 + i).clamp(0, w as i32 - 1) as usize;
            cols[(i + 1) as usize] = h[sy * w + sx];
        }
        rows[(j + 1) as usize] = cubic_hermite(cols[0], cols[1], cols[2], cols[3], dx);
    }
    cubic_hermite(rows[0], rows[1], rows[2], rows[3], dy)
}

/// Catmull-Rom cubic Hermite interpolation: `t ∈ [0, 1]` interpolates
/// between `p1` and `p2`, with `p0` and `p3` providing tangents.
fn cubic_hermite(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;
    ((a * t + b) * t + c) * t + d
}

/// Result of figuring out what working resolution to upsample to,
/// given the source GSD + the target metres-per-pixel.
pub fn working_resolution(src: &HeightmapTile, target_resolution_m: f32) -> (u32, u32) {
    // Source pixel count must scale by (source_gsd_m / target_resolution_m).
    let scale = src.gsd_m_centre / target_resolution_m.max(0.01);
    let new_w = ((src.width as f32) * scale).round().max(2.0) as u32;
    let new_h = ((src.height as f32) * scale).round().max(2.0) as u32;
    (new_w, new_h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::GeoExtent;

    fn tile(w: u32, h: u32, heights: Vec<f32>) -> HeightmapTile {
        HeightmapTile {
            heights_m: heights,
            width: w,
            height: h,
            extent_deg: GeoExtent {
                west: 0.0,
                east: 1.0,
                south: 0.0,
                north: 1.0,
            },
            source: "test",
            gsd_m_centre: 30.0,
        }
    }

    #[test]
    fn upsample_doubles_dimensions() {
        let src = tile(4, 4, vec![1.0; 16]);
        let up = bicubic_upsample(src, 8, 8);
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 8);
        assert_eq!(up.heights_m.len(), 64);
    }

    #[test]
    fn flat_input_stays_flat() {
        let src = tile(4, 4, vec![3.0; 16]);
        let up = bicubic_upsample(src, 8, 8);
        for v in &up.heights_m {
            assert!((v - 3.0).abs() < 1e-4, "got {v}");
        }
    }

    #[test]
    fn gsd_scales_inversely_with_resolution() {
        let src = tile(4, 4, vec![0.0; 16]);
        let up = bicubic_upsample(src, 8, 8);
        // 4 source pixels span the same extent as 8 target pixels =>
        // pixel pitch halves.
        // src: (4-1) px spans 3*30=90 m; target: (8-1) px spans the
        // same 90 m => 90/7 m per pixel.
        let expected = 30.0 * 3.0 / 7.0;
        assert!(
            (up.gsd_m_centre - expected).abs() < 0.1,
            "got {}, expected {}",
            up.gsd_m_centre,
            expected
        );
    }

    #[test]
    fn upsample_preserves_corner_values() {
        let src = tile(
            3,
            3,
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        );
        let up = bicubic_upsample(src, 9, 9);
        assert!((up.heights_m[0] - 10.0).abs() < 1e-3);
        assert!((up.heights_m[8] - 30.0).abs() < 1e-3);
        assert!((up.heights_m[72] - 70.0).abs() < 1e-3);
        assert!((up.heights_m[80] - 90.0).abs() < 1e-3);
    }

    #[test]
    fn no_op_when_already_at_target() {
        let src = tile(8, 8, vec![1.0; 64]);
        let up = bicubic_upsample(src, 4, 4);
        // Source is already larger than target → returned unchanged.
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 8);
    }

    #[test]
    fn working_resolution_doubles_for_half_target() {
        let src = tile(100, 100, vec![0.0; 10_000]);
        // src is 30m/px, target is 15m/px → 2× density.
        let (nw, nh) = working_resolution(&src, 15.0);
        assert_eq!(nw, 200);
        assert_eq!(nh, 200);
    }
}
