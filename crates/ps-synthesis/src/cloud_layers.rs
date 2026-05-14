//! Phase 3 §3.2.2 — per-layer cloud envelope synthesis.
//!
//! Takes the parsed `Scene` cloud layers and produces a `CloudLayerGpu`
//! per layer plus the top-level non-overlap validation that the synthesis
//! stage repeats (independently of `Scene::validate` so synthesis is a
//! self-contained boundary).

use ps_core::{
    default_density_scale, default_droplet_diameter_um, CloudLayer, CloudLayerGpu, CloudType,
};

/// Synthesise GPU-ready cloud layers from the parsed scene's layers.
///
/// `shape_octave_bias` and `detail_octave_bias` are passed through to
/// the GPU `CloudLayerGpu.shape_bias` / `.detail_bias` fields where
/// the cloud march now reads them (Phase 13 follow-up — previously
/// these were declared but ignored). The semantics are:
///
/// - `shape_bias` shifts the base-shape low-frequency Worley FBM sum;
///   the visual change is subtle (slight redistribution of the bulk
///   density envelope).
/// - `detail_bias` shifts the high-frequency boundary erosion
///   strength. Because Schneider's remap **concentrates surviving
///   samples while culling more low ones**, positive bias produces a
///   denser-looking cloud volume (each surviving voxel hits cloud=1
///   sooner) rather than the "wispier" reading the name suggests.
///   Recommended useful range is small (~±0.1); larger magnitudes
///   trigger sky-wide saturation or near-total cloud loss.
///
/// No per-type defaults are applied — the plan §3.2.6 "Detail
/// erosion" classification needs a calibrated tuning bench before
/// it can be mapped to bias defaults safely. For now, scenes that
/// want type-specific edge character set their own bias values.
///
/// `anvil_bias` defaults to 1.0 for Cumulonimbus (matches the v1
/// hard-coded anvil multiplier in the NDF) and 0.0 for all others.
/// `density_scale` defaults to a per-type optical-depth value
/// calibrated against the meteorological literature
/// (Cumulus/Stratus/Stratocumulus = 1.0, Altocumulus = 0.85,
/// Altostratus = 0.7, Cirrus = 0.55, Cirrostratus = 0.4,
/// Cumulonimbus = 1.4 for the deep convective core). Scenes that
/// supply explicit values via the optional `density_scale` /
/// `anvil_bias` fields on `CloudLayer` override these defaults.
pub fn synthesise_cloud_layers(layers: &[CloudLayer]) -> Vec<CloudLayerGpu> {
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        let type_anvil_default = match layer.cloud_type {
            CloudType::Cumulonimbus => 1.0,
            _ => 0.0,
        };
        let anvil_bias = layer.anvil_bias.unwrap_or(type_anvil_default);
        let density_scale = layer
            .density_scale
            .unwrap_or_else(|| default_density_scale(layer.cloud_type));
        let droplet_diameter_um = layer
            .droplet_diameter_um
            .unwrap_or_else(|| default_droplet_diameter_um(layer.cloud_type));
        out.push(CloudLayerGpu {
            base_m: layer.base_m,
            top_m: layer.top_m,
            coverage: remap_coverage_to_visible_band(layer.coverage),
            density_scale,
            cloud_type: layer.cloud_type as u8 as u32,
            shape_bias: layer.shape_octave_bias,
            detail_bias: layer.detail_octave_bias,
            anvil_bias,
            droplet_diameter_um,
            _pad_after_droplets_0: 0.0,
            _pad_after_droplets_1: 0.0,
            _pad_after_droplets_2: 0.0,
        });
    }
    out
}


/// Map a scene-side coverage value in [0, 1] onto the Schneider 2015
/// remap's "visible band" so METAR-natural values (0.25 SCT, 0.5 BKN,
/// 1.0 OVC) produce the cloud structure a meteorologist would expect.
///
/// The Schneider remap (`cloud = remap(base, 1-coverage, 1, 0, 1) *
/// coverage`) silently wipes coverage values below ~`VISIBLE_LOW` (the
/// noise rarely exceeds `1 − coverage`) and produces dark structureless
/// "slab" bases above ~`VISIBLE_HIGH` (the medium becomes too thick
/// for sunlight to reach the camera-facing surface). The output is
/// therefore confined to a narrow band where the slider has visible
/// effect, with two ramps around it:
///
/// 1. A smooth knee at the low end (slider 0 → KNEE_END) so the empty
///    zone below the visible band ramps gradually instead of jumping.
///    This replaces the legacy step gate at `scene_coverage = 0.02`
///    that produced an abrupt cliff in the rendered output around that
///    slider value.
/// 2. Linear into the visible band from `KNEE_END` to slider = 1.0 so
///    `FEW` (input ~0.15) lands just inside the band, `BKN` (input
///    0.5) mid-band, and `OVC` (input 1.0) at the top edge.
///
/// `VISIBLE_LOW = 0.40` is calibrated so that the cover fractions
/// written by `ps-weather-feed` (NWP `cover_pct / 100.0`) keep
/// rendering visible cloud structure for FEW/SCT — dropping it lower
/// pushes those METAR conditions below the Schneider visible
/// threshold. Pushing past `VISIBLE_HIGH = 0.60` slides into the
/// dark-slab artifact; if that range is ever needed it requires a
/// separate tuning pass on `sigma_t`, Beer-Powder, and the multi-
/// octave decay so thick bases still admit ambient + multi-scatter
/// light.
fn remap_coverage_to_visible_band(scene_coverage: f32) -> f32 {
    const VISIBLE_LOW: f32 = 0.40;
    const VISIBLE_HIGH: f32 = 0.60;
    /// Width of the smoothstep knee at the low end (in slider units).
    /// Slider <= this lives in the "near-clear sparse-puff" regime
    /// where effective coverage is below the Schneider visible
    /// threshold; the cubic smoothstep gives that range a graceful
    /// taper so sweeping the slider feels continuous.
    const KNEE_END: f32 = 0.10;
    let t = scene_coverage.clamp(0.0, 1.0);
    if t <= 0.0 {
        return 0.0;
    }
    if t < KNEE_END {
        let s = t / KNEE_END;
        let smoothed = s * s * (3.0 - 2.0 * s);
        return smoothed * VISIBLE_LOW;
    }
    let band_t = ((t - KNEE_END) / (1.0 - KNEE_END)).clamp(0.0, 1.0);
    VISIBLE_LOW + band_t * (VISIBLE_HIGH - VISIBLE_LOW)
}

/// Verify the synthesised cloud-layer envelopes don't overlap vertically.
///
/// `Scene::validate` already enforces this; we repeat the check here so
/// `synthesise(...)` is robust against direct construction of layer
/// vectors that bypass scene validation. Returns the offending pair.
pub fn check_non_overlap(
    layers: &[CloudLayerGpu],
) -> Result<(), (usize, usize, CloudLayerGpu, CloudLayerGpu)> {
    for i in 0..layers.len() {
        for j in (i + 1)..layers.len() {
            let a = layers[i];
            let b = layers[j];
            // Half-open interval test.
            if a.base_m < b.top_m && b.base_m < a.top_m {
                return Err((i, j, a, b));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::CloudLayer;

    fn make_layer(t: CloudType, base: f32, top: f32) -> CloudLayer {
        CloudLayer {
            cloud_type: t,
            base_m: base,
            top_m: top,
            coverage: 0.5,
            density_scale: None,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
            anvil_bias: None,
            droplet_diameter_um: None,
        }
    }

    #[test]
    fn cumulus_layer_round_trips() {
        let layers = vec![make_layer(CloudType::Cumulus, 1500.0, 2300.0)];
        let gpu = synthesise_cloud_layers(&layers);
        assert_eq!(gpu.len(), 1);
        assert_eq!(gpu[0].cloud_type, 0);
        assert_eq!(gpu[0].base_m, 1500.0);
        assert_eq!(gpu[0].top_m, 2300.0);
        assert_eq!(gpu[0].anvil_bias, 0.0);
    }

    #[test]
    fn cumulonimbus_gets_anvil_bias() {
        let layers = vec![make_layer(CloudType::Cumulonimbus, 800.0, 11_000.0)];
        let gpu = synthesise_cloud_layers(&layers);
        assert_eq!(gpu[0].cloud_type, CloudType::Cumulonimbus as u8 as u32);
        assert_eq!(gpu[0].anvil_bias, 1.0);
    }

    /// Pin the per-type `density_scale` defaults — see
    /// `default_density_scale` for the calibration rationale.
    #[test]
    fn per_type_density_scale_defaults() {
        let cases = [
            (CloudType::Cumulus, 1.0),
            (CloudType::Stratus, 1.0),
            (CloudType::Stratocumulus, 1.0),
            (CloudType::Altocumulus, 0.85),
            (CloudType::Altostratus, 0.7),
            (CloudType::Cirrus, 0.55),
            (CloudType::Cirrostratus, 0.4),
            (CloudType::Cumulonimbus, 1.4),
        ];
        for (t, expected) in cases {
            let gpu = synthesise_cloud_layers(&[make_layer(t, 100.0, 200.0)]);
            assert_eq!(
                gpu[0].density_scale, expected,
                "default density_scale for {t:?}",
            );
        }
    }

    /// Scene-supplied `density_scale` wins over the per-type default.
    #[test]
    fn explicit_density_scale_overrides_default() {
        let mut layer = make_layer(CloudType::Cirrus, 8000.0, 9000.0);
        layer.density_scale = Some(2.5);
        let gpu = synthesise_cloud_layers(&[layer]);
        assert_eq!(gpu[0].density_scale, 2.5);
    }

    /// Per-cloud-type droplet effective diameter defaults. Values
    /// pulled from `ps_core::default_droplet_diameter_um`; the
    /// Approximate Mie phase function in the shader is fitted for
    /// 5–50 µm.
    #[test]
    fn per_type_droplet_diameter_defaults() {
        let cases = [
            (CloudType::Cumulus, 20.0_f32),
            (CloudType::Stratus, 16.0),
            (CloudType::Stratocumulus, 16.0),
            (CloudType::Altocumulus, 14.0),
            (CloudType::Altostratus, 30.0),
            (CloudType::Cirrus, 50.0),
            (CloudType::Cirrostratus, 50.0),
            (CloudType::Cumulonimbus, 20.0),
        ];
        for (t, expected) in cases {
            let gpu = synthesise_cloud_layers(&[make_layer(t, 100.0, 200.0)]);
            assert_eq!(
                gpu[0].droplet_diameter_um, expected,
                "default droplet diameter for {t:?}",
            );
        }
    }

    /// Scene-supplied `droplet_diameter_um` wins over the per-type
    /// default.
    #[test]
    fn explicit_droplet_diameter_overrides_default() {
        let mut layer = make_layer(CloudType::Cirrus, 8000.0, 9000.0);
        layer.droplet_diameter_um = Some(35.0);
        let gpu = synthesise_cloud_layers(&[layer]);
        assert_eq!(gpu[0].droplet_diameter_um, 35.0);
    }

    #[test]
    fn non_overlapping_layers_validate() {
        let layers = synthesise_cloud_layers(&[
            make_layer(CloudType::Stratus, 200.0, 600.0),
            make_layer(CloudType::Cumulus, 1500.0, 2300.0),
        ]);
        check_non_overlap(&layers).expect("disjoint layers must validate");
    }

    #[test]
    fn overlapping_layers_are_rejected() {
        let layers = synthesise_cloud_layers(&[
            make_layer(CloudType::Cumulus, 1000.0, 2000.0),
            make_layer(CloudType::Stratus, 1500.0, 2500.0),
        ]);
        let (a, b, _, _) = check_non_overlap(&layers).expect_err("must reject");
        assert_eq!((a, b), (0, 1));
    }

    /// Coverage remap must be monotonically non-decreasing across
    /// `[0, 1]` and free of step jumps. A step like the old gate at
    /// `0.02` made the slider feel discontinuous to users; this test
    /// pins the smooth-knee replacement.
    #[test]
    fn coverage_remap_is_monotone_and_continuous() {
        // 0 → 0; 1 → top of visible band.
        let zero = remap_coverage_to_visible_band(0.0);
        let one = remap_coverage_to_visible_band(1.0);
        assert!(zero.abs() < 1e-6, "remap(0.0) = {zero}, expected 0");
        assert!((one - 0.60).abs() < 1e-6, "remap(1.0) = {one}, expected 0.60");

        // Sweep at 1% increments; each step must not regress and must
        // not jump by more than a small fraction of the visible-band
        // width (catches future step-style cliffs).
        const MAX_STEP: f32 = 0.06;
        let mut prev = 0.0_f32;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let v = remap_coverage_to_visible_band(t);
            assert!(v >= prev - 1e-6, "non-monotone at t={t}: {prev} → {v}");
            let delta = v - prev;
            assert!(
                delta <= MAX_STEP,
                "step too large at t={t}: Δ = {delta} (over {MAX_STEP})",
            );
            prev = v;
        }
    }

    /// The top of the knee at slider 0.10 should land at `VISIBLE_LOW`
    /// (the bottom of the Schneider visible band). Above the knee the
    /// mapping is linear toward `VISIBLE_HIGH` at slider 1.0. Pinning
    /// these endpoints makes the look-and-feel deliberate rather than
    /// a happy accident of the smoothstep, and also locks the
    /// real-weather mapping: an `ps-weather-feed` METAR/NWP cover of
    /// e.g. 0.40 (SCT) must produce effective coverage above the
    /// Schneider visible threshold so the cloud march renders.
    #[test]
    fn coverage_remap_knee_endpoints() {
        let at_knee = remap_coverage_to_visible_band(0.10);
        assert!(
            (at_knee - 0.40).abs() < 1e-5,
            "knee endpoint at 0.10 = {at_knee}, expected 0.40 (VISIBLE_LOW)",
        );
        let mid_band = remap_coverage_to_visible_band(0.50);
        let expected_mid = 0.40 + (0.50 - 0.10) / (1.0 - 0.10) * (0.60 - 0.40);
        assert!(
            (mid_band - expected_mid).abs() < 1e-5,
            "mid-band slider 0.50 = {mid_band}, expected {expected_mid}",
        );
    }
}
