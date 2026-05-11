//! Phase 3 §3.2.2 — per-layer cloud envelope synthesis.
//!
//! Takes the parsed `Scene` cloud layers and produces a `CloudLayerGpu`
//! per layer plus the top-level non-overlap validation that the synthesis
//! stage repeats (independently of `Scene::validate` so synthesis is a
//! self-contained boundary).

use ps_core::{CloudLayer, CloudLayerGpu, CloudType};

/// Synthesise GPU-ready cloud layers from the parsed scene's layers.
///
/// Per-type defaults (plan §3.2.6 table) are applied **only when the
/// scene leaves the field at its struct default** (zero, since these are
/// optional bias fields). For v1 the scene's `shape_octave_bias` and
/// `detail_octave_bias` map directly to `shape_bias` and `detail_bias`;
/// `anvil_bias` is set from the cloud type (Cumulonimbus = 1.0, others 0).
pub fn synthesise_cloud_layers(layers: &[CloudLayer]) -> Vec<CloudLayerGpu> {
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        let anvil_bias = match layer.cloud_type {
            CloudType::Cumulonimbus => 1.0,
            _ => 0.0,
        };
        out.push(CloudLayerGpu {
            base_m: layer.base_m,
            top_m: layer.top_m,
            coverage: remap_coverage_to_visible_band(layer.coverage),
            density_scale: layer.density_scale,
            cloud_type: layer.cloud_type as u8 as u32,
            shape_bias: layer.shape_octave_bias,
            detail_bias: layer.detail_octave_bias,
            anvil_bias,
        });
    }
    out
}

/// Map a scene-side coverage value in [0, 1] onto the Schneider 2015
/// remap's "visible band" so that METAR-natural values (0.25 SCT, 0.5
/// BKN, 1.0 OVC) produce the cloud structure a meteorologist would
/// expect.
///
/// The Schneider remap (`cloud = remap(base, 1-coverage, 1, 0, 1) *
/// coverage`) silently wipes coverage values below ~0.55 (the remap
/// gives near-zero density) and saturates above ~0.78 (cloud bases
/// read as dark slabs because the medium is too thick for sun-light
/// to penetrate). This function pre-biases scene coverage onto the
/// `[0.60, 0.78]` band that produces visually meaningful clouds.
///
/// Coverage values <= 0.02 round to zero (truly clear sky); above
/// 0.02 the input maps linearly into the visible band. Documented
/// in followup #57.
fn remap_coverage_to_visible_band(scene_coverage: f32) -> f32 {
    // The cloud march in cloud_march.wgsl combines weather_map.r (now
    // a {0,1} spatial gate) with layer.coverage as a Schneider remap
    // threshold + multiplier. Empirically the *effective* coverage the
    // shader needs to fall inside [≈0.4, ≈0.6] for visible structure:
    // below ≈0.4 the remap silently kills density; above ≈0.6 the
    // medium becomes too thick for sun-light to penetrate and cloud
    // bases read as dark slabs.
    //
    // Map scene-side METAR-ish input [0.02, 1] linearly onto that
    // band so a BKN scene (input ~0.5) lands mid-band and an OVC
    // scene (input ~1.0) lands near the top edge.
    const VISIBLE_LOW: f32 = 0.40;
    const VISIBLE_HIGH: f32 = 0.60;
    const CLEAR_SKY_GATE: f32 = 0.02;
    if scene_coverage <= CLEAR_SKY_GATE {
        return 0.0;
    }
    let t = ((scene_coverage - CLEAR_SKY_GATE) / (1.0 - CLEAR_SKY_GATE)).clamp(0.0, 1.0);
    VISIBLE_LOW + t * (VISIBLE_HIGH - VISIBLE_LOW)
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
            density_scale: 1.0,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
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
}
