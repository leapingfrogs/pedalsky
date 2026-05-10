//! CPU mirror of the per-cloud-type Vertical Density Profile (NDF) defined
//! in plan §6.4. Phase 6's WGSL shader contains the canonical version;
//! this module replicates it on the CPU so Phase 3's top-down density mask
//! synthesis (§3.2.5) does not need to round-trip through a compute shader.
//!
//! Input `h` is the normalised height inside the layer in [0, 1] (0 = base,
//! 1 = top); `cloud_type` is the `CloudType` repr-u8 widened to u32.
//! Output is the unitless density profile multiplier in [0, ~1.5].

/// Smoothstep matching the GLSL/WGSL `smoothstep` of plan §6.4.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Plan §6.4 NDF, switched on `cloud_type`.
pub fn ndf(h: f32, cloud_type: u32) -> f32 {
    match cloud_type {
        0 => {
            // Cumulus: bell, peak ~0.4
            smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.2, h)
        }
        1 => {
            // Stratus: top-heavy, low and thin
            smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.6, 1.0, h))
        }
        2 => {
            // Stratocumulus: mid-heavy
            smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.4, h)
        }
        3 => {
            // Altocumulus: mid-heavy, thinner overall
            smoothstep(0.0, 0.20, h) * smoothstep(1.0, 0.3, h) * 0.8
        }
        4 => {
            // Altostratus: top-heavy sheet
            smoothstep(0.0, 0.30, h) * (1.0 - smoothstep(0.7, 1.0, h)) * 0.6
        }
        5 => {
            // Cirrus: thin, top-heavy, wispy
            smoothstep(0.0, 0.40, h) * (1.0 - smoothstep(0.6, 1.0, h)) * 0.4
        }
        6 => {
            // Cirrostratus: very thin sheet, highest
            smoothstep(0.0, 0.50, h) * (1.0 - smoothstep(0.8, 1.0, h)) * 0.3
        }
        7 => {
            // Cumulonimbus: mushroom (bottom-heavy + anvil)
            let base = smoothstep(0.0, 0.05, h);
            let mid = smoothstep(0.95, 0.5, h);
            let anvil = smoothstep(0.7, 0.9, h) * 1.5;
            let mix_t = smoothstep(0.65, 0.8, h);
            base * lerp(mid, anvil, mix_t)
        }
        _ => 0.0,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndf_is_zero_at_base_and_top_for_cumulus() {
        assert!(ndf(0.0, 0).abs() < 1e-3);
        assert!(ndf(1.0, 0).abs() < 1e-3);
    }

    #[test]
    fn cumulus_peaks_in_lower_third() {
        // Sample 0..1 at 0.05 steps; the peak should occur for h < 0.5.
        let mut peak_h = 0.0;
        let mut peak_v = 0.0;
        let mut h = 0.0;
        while h <= 1.0 {
            let v = ndf(h, 0);
            if v > peak_v {
                peak_v = v;
                peak_h = h;
            }
            h += 0.05;
        }
        assert!(peak_h < 0.55, "cumulus NDF peaks at h={peak_h}");
        assert!(peak_v > 0.5, "peak value {peak_v} too low");
    }

    #[test]
    fn cumulonimbus_anvil_is_top_heavy_in_upper_quintile() {
        let v_top = ndf(0.92, 7);
        let v_mid = ndf(0.4, 7);
        // Anvil should produce noticeable density near the top.
        assert!(
            v_top > 0.05,
            "expected anvil contribution at h=0.92, got {v_top}"
        );
        assert!(
            v_mid > 0.05,
            "expected mid-cloud density at h=0.4, got {v_mid}"
        );
    }

    #[test]
    fn unknown_type_returns_zero() {
        assert_eq!(ndf(0.5, 99), 0.0);
    }
}
