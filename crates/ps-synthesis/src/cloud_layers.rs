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
            coverage: layer.coverage,
            density_scale: layer.density_scale,
            cloud_type: layer.cloud_type as u8 as u32,
            shape_bias: layer.shape_octave_bias,
            detail_bias: layer.detail_octave_bias,
            anvil_bias,
        });
    }
    out
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
