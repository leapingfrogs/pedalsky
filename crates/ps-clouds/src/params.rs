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
///   1  sigma_a.r/g/b + droplet_diameter_bias
///   2  cone_light_sampling, forward_bias, detail_strength, curl_strength
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
    /// Global multiplier applied to every layer's
    /// `droplet_diameter_um`. Default 1.0 = use the synthesised
    /// per-cloud-type diameter unchanged. The Approximate Mie phase
    /// function in the cloud march evaluates its fit at
    /// `diameter * droplet_diameter_bias`, clamped to the paper's
    /// 5–50 µm range. Packed into sigma_a's trailing 4 bytes per
    /// std140 vec3 packing.
    pub droplet_diameter_bias: f32,

    /// Schneider/Nubis 2017 cone-tap light sampling toggle.
    /// `0` = use the original straight `march_to_light`;
    /// `1` = use the 5-cone + 1-long-tap kernel. Lives in the
    /// std140 slot that used to be `hg_backward_bias` so the struct
    /// size is unchanged.
    pub cone_light_sampling: u32,
    /// First-octave forward bias for the multi-scatter loop. Scales
    /// the primary (octave-0) energy contribution `a` by
    /// `(1 + forward_bias)` — concentrates light into the strongly
    /// forward-peaked primary scatter so the visible "in-cloud
    /// sun-shaft" effect reads more strongly through cloud bodies.
    /// 0.0 reproduces unbiased Hillaire multi-octave behaviour.
    /// Lives in the std140 slot that used to be `hg_blend_bias`.
    pub forward_bias: f32,
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
    /// Phase 14.C — multiplier on the wind-driven cloud advection
    /// offset (`wind(altitude) * simulated_seconds * wind_drift_strength`).
    /// Default `1.0` makes clouds drift at full physical speed; `0.0`
    /// freezes the cloud world-space lookup (used by bless-time
    /// golden renders so test scenes stay deterministic and by the
    /// freeze-time toggle so paused screenshots are stable). Lives in
    /// the std140 slot that used to be `_pad_temporal_jitter_0`; the
    /// struct size is unchanged.
    pub wind_drift_strength: f32,
    /// Phase 14.H — strength of the Schneider Nubis 2017 "skew with
    /// height" effect. The cloud march offsets noise lookups by
    /// `h * layer_thickness * wind_direction * wind_skew_strength`,
    /// so at `1.0` the top of a layer reads from one layer-thickness
    /// downwind of the base, giving visible cumulus lean / anvil
    /// tilt under directional shear. `0.0` disables the effect.
    /// Uses wind *direction* only — magnitude doesn't scale the lean
    /// because real clouds in strong wind shred rather than lean
    /// further. Lives in the slot that used to be
    /// `_pad_temporal_jitter_1`.
    pub wind_skew_strength: f32,
    /// Phase 18 — strength of the diurnal modulation applied to
    /// convective cloud types' `shape_bias` / `detail_bias`. The
    /// shader multiplies both per-layer biases by
    /// `smoothstep(-0.1, 0.4, sin(sun_altitude)) * diurnal_strength`
    /// for convective layers (Cumulus / Stratocumulus / Altocumulus
    /// / Cumulonimbus) so cumulus visibly grows convective character
    /// through the day and dissipates at night. `0.0` disables the
    /// effect (locks biases at their Phase 17 baseline). Non-
    /// convective cloud types ignore this scalar — stratus is
    /// smooth all day, cirrus is shear-driven not turbulence-driven.
    /// Lives in the slot that used to be `_pad_temporal_jitter_2`,
    /// so the struct stays at 112 B.
    pub diurnal_strength: f32,
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
            // Default 1.0 = "use the synthesised per-cloud-type
            // droplet diameter unchanged".
            droplet_diameter_bias: 1.0,

            cone_light_sampling: 0,
            forward_bias: 0.0,
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
            // Drift defaults on at full strength so the cloud march
            // animates clouds with the synthesised wind field as soon
            // as the subsystem is enabled. The shader auto-disables
            // when `simulated_seconds` is zero (golden bless runs) and
            // when `freeze_time` latches the clock.
            wind_drift_strength: 1.0,
            // Phase 14.H — moderate default lean. At 0.5 the cloud
            // top sits half a layer-thickness downwind of its base,
            // giving cumulus a clearly-visible tilt without
            // collapsing the cell sideways. 0.0 disables the effect.
            wind_skew_strength: 0.5,
            // Phase 18 — full strength by default so a windowed
            // user sees diurnal evolution out of the box. Bless
            // / golden runs zero this explicitly for determinism.
            diurnal_strength: 1.0,
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
        assert_eq!(std::mem::size_of::<CloudLayerGpu>(), 48);
    }
}
