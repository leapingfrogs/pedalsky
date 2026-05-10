//! Phase 1 Group A tests for `Scene` parsing and validation.

use ps_core::{CloudType, Scene, SceneError};

fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn scene_toml_loads_unchanged() {
    let path = workspace_root().join("scenes/broken_cumulus_afternoon.toml");
    let scene = Scene::load(&path).expect("scene should parse");
    scene.validate().expect("scene should validate");
    assert_eq!(scene.schema_version, 1);
    assert_eq!(scene.clouds.layers.len(), 1);
    assert_eq!(scene.clouds.layers[0].cloud_type, CloudType::Cumulus);
    assert_eq!(scene.clouds.layers[0].base_m, 1500.0);
    assert_eq!(scene.clouds.layers[0].top_m, 2300.0);
}

#[test]
fn cloud_type_round_trips() {
    use CloudType::*;
    for ct in [
        Cumulus,
        Stratus,
        Stratocumulus,
        Altocumulus,
        Altostratus,
        Cirrus,
        Cirrostratus,
        Cumulonimbus,
    ] {
        let toml_text = format!(
            r#"
schema_version = 1
[[clouds.layers]]
type = "{:?}"
base_m = 1000.0
top_m  = 2000.0
coverage = 0.3
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0
[precipitation]
type = "None"
intensity_mm_per_h = 0.0
[lightning]
strikes_per_min_per_km2 = 0.0
"#,
            ct
        );
        let scene = Scene::parse(&toml_text)
            .unwrap_or_else(|e| panic!("scene with type {ct:?} should parse: {e}"));
        assert_eq!(scene.clouds.layers[0].cloud_type, ct);
    }
}

#[test]
fn overlapping_cloud_layers_are_rejected() {
    let toml = r#"
schema_version = 1
[[clouds.layers]]
type = "Cumulus"
base_m = 1000.0
top_m  = 2000.0
coverage = 0.5
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0

[[clouds.layers]]
type = "Stratus"
base_m = 1500.0
top_m  = 2500.0
coverage = 0.4
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0

[precipitation]
type = "None"
intensity_mm_per_h = 0.0

[lightning]
strikes_per_min_per_km2 = 0.0
"#;
    let scene = Scene::parse(toml).expect("syntactically valid");
    let err = scene.validate().expect_err("overlapping layers must be rejected");
    match err {
        SceneError::OverlappingCloudLayers { a, b, .. } => {
            assert_eq!(a, 0);
            assert_eq!(b, 1);
        }
        other => panic!("expected OverlappingCloudLayers, got {other:?}"),
    }
}

#[test]
fn unknown_cloud_field_is_rejected() {
    let toml = r#"
schema_version = 1
[[clouds.layers]]
type = "Cumulus"
base_m = 1000.0
top_m  = 2000.0
coverage = 0.5
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0
mystery_field = 42

[precipitation]
type = "None"
intensity_mm_per_h = 0.0

[lightning]
strikes_per_min_per_km2 = 0.0
"#;
    let err = Scene::parse(toml).expect_err("unknown field should fail");
    assert!(format!("{err}").contains("mystery_field"));
}

#[test]
fn schema_version_other_than_1_is_rejected() {
    let scene = Scene::parse(
        r#"
schema_version = 2
[surface]
[[clouds.layers]]
type = "Cumulus"
base_m = 1000.0
top_m  = 2000.0
coverage = 0.5
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0
[precipitation]
type = "None"
intensity_mm_per_h = 0.0
[lightning]
strikes_per_min_per_km2 = 0.0
"#,
    )
    .expect("syntax ok");
    let err = scene.validate().expect_err("future schema version rejected");
    assert!(format!("{err}").contains("schema_version"));
}
