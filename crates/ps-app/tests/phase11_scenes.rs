//! Phase 11.1 — every reference scene under `tests/scenes/` parses
//! and validates via the canonical `Scene::load` + `Scene::validate`
//! path.

use std::path::PathBuf;

const SCENES: &[&str] = &[
    "clear_summer_noon.toml",
    "broken_cumulus_afternoon.toml",
    "overcast_drizzle.toml",
    "thunderstorm.toml",
    "high_cirrus_sunset.toml",
    "winter_overcast_snow.toml",
    "twilight_civil.toml",
    "mountain_wave_clouds.toml",
];

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("pedalsky.toml").is_file() {
            return dir;
        }
        if !dir.pop() {
            panic!("workspace root not found");
        }
    }
}

#[test]
fn all_reference_scenes_load_and_validate() {
    let root = workspace_root();
    for name in SCENES {
        let path = root.join("tests").join("scenes").join(name);
        let scene = ps_core::Scene::load(&path)
            .unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
        scene
            .validate()
            .unwrap_or_else(|e| panic!("validate {}: {e}", path.display()));
        eprintln!(
            "✓ {} — {} layer(s), precip = {:?}",
            name,
            scene.clouds.layers.len(),
            scene.precipitation.kind
        );
    }
}

#[test]
fn mountain_wave_coverage_grid_binary_exists() {
    let root = workspace_root();
    let path = root
        .join("tests")
        .join("scenes")
        .join("presets")
        .join("mountain_wave_lozenges_128x128.bin");
    let metadata = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("missing {}: {e}", path.display()));
    // 128 × 128 × 4 bytes (f32) = 65,536 bytes.
    assert_eq!(
        metadata.len(),
        65_536,
        "{} must be exactly 65,536 bytes",
        path.display()
    );
}
