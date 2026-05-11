//! Phase 6 cloud march uniform parameters.
//!
//! Mirrored on the GPU side by `cloud_uniforms.wgsl::CloudParams`.

use bytemuck::{Pod, Zeroable};

/// Maximum number of cloud layers handled per frame. Matches the WGSL
/// `MAX_CLOUD_LAYERS` constant in `cloud_uniforms.wgsl`.
pub const MAX_CLOUD_LAYERS: u32 = 8;

/// CPU-side mirror of the WGSL `CloudParams` struct.
///
/// 112 bytes (7 × vec4 blocks); std140 boundary respected. WGSL's
/// `vec3<f32>` has *alignment* 16 but *size* 12, which means a
/// following scalar packs into the trailing 4 bytes of the same
/// 16-byte chunk. The Rust layout below mirrors that packing
/// exactly: each `[f32; 3]` is followed by a single `f32` field
/// living in those trailing 4 bytes (used for the next "real"
/// member, not pad). Verified by the naga std140 linter in
/// `crates/ps-app/tests/wgsl_layout.rs`.
///
/// Layout (each block is 16 B):
///
///   0  sigma_s.r/g/b + _pad_after_sigma_s
///   1  sigma_a.r/g/b + g_forward
///   2  g_backward, g_blend, detail_strength, curl_strength
///   3  powder_strength, multi_scatter_a, multi_scatter_b, multi_scatter_c
///   4  ambient_strength, base_scale_m, detail_scale_m, weather_scale_m
///   5  light_steps, cloud_steps, multi_scatter_octaves, cloud_layer_count
///   6  temporal_jitter + 3 std140 pad scalars  (Phase 13.9)
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct CloudParamsGpu {
    /// Per-channel scattering coefficient (per metre). Phase 12.2b —
    /// promoted to Vec3 so cloud transmittance can carry chromatic
    /// information, which is what makes cloud edges acquire warm
    /// fringes at sunset rather than a uniform grey dimming.
    pub sigma_s: [f32; 3],
    /// std140 pad — WGSL packs the next scalar into vec3's trailing
    /// 4 bytes, but only when the scalar is a "real" member. Here we
    /// keep an explicit pad because the next "real" member is also a
    /// vec3 (sigma_a), which forces realignment.
    pub _pad_after_sigma_s: f32,
    /// Per-channel absorption coefficient (per metre). Near-zero for
    /// real water droplets across the visible spectrum.
    pub sigma_a: [f32; 3],
    /// Forward HG g — packed into sigma_a's trailing 4 bytes per
    /// std140 vec3 packing.
    pub g_forward: f32,

    /// Backward HG g (negative).
    pub g_backward: f32,
    /// Dual-lobe blend factor.
    pub g_blend: f32,
    /// Detail erosion strength.
    pub detail_strength: f32,
    /// Curl perturbation strength.
    pub curl_strength: f32,

    /// Beer-Powder lerp factor.
    pub powder_strength: f32,
    /// Multi-scatter octave attenuation: energy.
    pub multi_scatter_a: f32,
    /// Multi-scatter octave attenuation: optical depth.
    pub multi_scatter_b: f32,
    /// Multi-scatter octave attenuation: phase anisotropy.
    pub multi_scatter_c: f32,

    /// Sky-ambient strength multiplier.
    pub ambient_strength: f32,
    /// Base shape sample period (metres).
    pub base_scale_m: f32,
    /// Detail sample period (metres).
    pub detail_scale_m: f32,
    /// Weather map period (metres).
    pub weather_scale_m: f32,

    /// Light-march steps to the sun.
    pub light_steps: u32,
    /// Primary cloud march steps.
    pub cloud_steps: u32,
    /// Multi-scatter octaves.
    pub multi_scatter_octaves: u32,
    /// Number of valid entries in the layer array.
    pub cloud_layer_count: u32,

    /// Phase 13.9 — when `1`, the cloud march XORs the blue-noise
    /// sample coords with a frame-index-derived offset (16-frame
    /// rotation). Default `0`. Off when `freeze_time` so paused
    /// screenshots stay deterministic.
    pub temporal_jitter: u32,
    /// std140 pad — keeps the struct's tail on a vec4 boundary.
    pub _pad_temporal_jitter_0: u32,
    /// std140 pad — second of three trailing scalars.
    pub _pad_temporal_jitter_1: u32,
    /// std140 pad — third of three trailing scalars.
    pub _pad_temporal_jitter_2: u32,
}

impl Default for CloudParamsGpu {
    fn default() -> Self {
        Self {
            // Per-channel scattering coefficient. Real cloud Mie
            // scattering with droplets ~10 μm is nearly
            // wavelength-independent; smaller droplets (sub-µm —
            // fresh fog, thin cirrus, cumulus updraft tops) tilt
            // toward Rayleigh-like blue scattering, attenuating
            // blue more than red as sun light passes through. The
            // engine treats clouds as a single optically-thin
            // homogeneous medium, so we bias more aggressively than
            // pure Mie would prescribe — this is the parameter that
            // makes cirrus edges and cumulus tops acquire visibly
            // warm fringes at sunset rather than dimming uniformly:
            //   R: 0.030 / m   (red passes through more)
            //   G: 0.040 / m   (reference)
            //   B: 0.060 / m   (blue scatters more)
            //
            // Geometric mean ≈ 0.040 — matches the previous scalar
            // default so existing scene tunings still produce
            // comparable optical thickness in the achromatic
            // average.
            sigma_s: [0.030, 0.040, 0.060],
            _pad_after_sigma_s: 0.0,
            sigma_a: [0.0, 0.0, 0.0],
            g_forward: 0.8,

            g_backward: -0.3,
            g_blend: 0.5,
            // Schneider 2015 quotes 0.35 but the canonical remap formula
            // wipes coverage<0.5 layers entirely. Keep low here so the
            // default cumulus layer is visible; UI will expose this and
            // power users will dial it up alongside coverage.
            detail_strength: 0.05,
            curl_strength: 0.1,

            powder_strength: 1.0,
            multi_scatter_a: 0.5,
            multi_scatter_b: 0.5,
            multi_scatter_c: 0.5,

            ambient_strength: 1.0,
            base_scale_m: 4500.0,
            detail_scale_m: 800.0,
            weather_scale_m: 32_000.0,

            light_steps: 6,
            cloud_steps: 192,
            multi_scatter_octaves: 4,
            cloud_layer_count: 0,

            temporal_jitter: 0,
            _pad_temporal_jitter_0: 0,
            _pad_temporal_jitter_1: 0,
            _pad_temporal_jitter_2: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::CloudLayerGpu;

    #[test]
    fn cloud_params_size_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<CloudParamsGpu>() % 16, 0);
        // 96 in v1; bumped to 112 by Phase 13.9 temporal_jitter +
        // 3 std140 pads.
        assert_eq!(std::mem::size_of::<CloudParamsGpu>(), 112);
    }

    #[test]
    fn cloud_layer_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<CloudLayerGpu>(), 32);
    }
}
