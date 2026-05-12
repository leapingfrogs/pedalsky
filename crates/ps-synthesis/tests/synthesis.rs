//! Phase 3 integration tests for the synthesis pipeline.
//!
//! Most behavioural assertions (Koschmieder constant, NDF profiles, wind
//! power law, density mask growth with coverage) live in unit tests next
//! to each module. These integration tests cover the *end-to-end*
//! `synthesise` entry point against a real headless GPU.

use std::path::PathBuf;
use std::sync::OnceLock;

use ps_core::{Config, GpuContext, Scene, WorldState};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping synthesis integration tests — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn workspace_pedalsky_toml_synthesises() {
    let Some(gpu) = gpu() else { return };
    let root = workspace_root();
    let config = Config::load(&root.join("pedalsky.toml")).unwrap();
    config.validate_with_base(Some(&root)).unwrap();
    let scene_path = if config.paths.weather.is_absolute() {
        config.paths.weather.clone()
    } else {
        root.join(&config.paths.weather)
    };
    let scene = Scene::load(&scene_path).unwrap();
    scene.validate().unwrap();

    let world = WorldState::new(
        chrono::TimeZone::with_ymd_and_hms(&chrono::Utc, 2026, 5, 10, 13, 30, 0).unwrap(),
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );
    let weather =
        ps_synthesis::synthesise(&scene, &config, &world, gpu).expect("synthesise should succeed");
    assert_eq!(weather.cloud_layer_count, 1);
    // Visibility 30 km → β ≈ 1.304e-4
    let beta = weather.haze_extinction_per_m.x;
    assert!((beta - 1.304e-4).abs() < 1e-7, "got β = {beta}");
}

#[test]
fn overlapping_cloud_layers_rejected_at_synthesis() {
    let Some(gpu) = gpu() else { return };
    let mut scene = Scene::default();
    scene.clouds.layers.push(ps_core::CloudLayer {
        cloud_type: ps_core::CloudType::Cumulus,
        base_m: 1000.0,
        top_m: 2000.0,
        coverage: 0.5,
        density_scale: 1.0,
        shape_octave_bias: 0.0,
        detail_octave_bias: 0.0,
        anvil_bias: None,
    });
    scene.clouds.layers.push(ps_core::CloudLayer {
        cloud_type: ps_core::CloudType::Stratus,
        base_m: 1500.0,
        top_m: 2500.0,
        coverage: 0.4,
        density_scale: 1.0,
        shape_octave_bias: 0.0,
        detail_octave_bias: 0.0,
        anvil_bias: None,
    });
    let config = Config::default();
    let world = WorldState::default();
    let err =
        ps_synthesis::synthesise(&scene, &config, &world, gpu).expect_err("overlap should fail");
    let msg = format!("{err}");
    assert!(msg.contains("overlap"), "{msg}");
}

#[test]
fn weather_state_textures_have_expected_dimensions() {
    let Some(gpu) = gpu() else { return };
    let scene = Scene::default();
    let config = Config::default();
    let world = WorldState::default();
    let weather = ps_synthesis::synthesise(&scene, &config, &world, gpu).unwrap();
    assert_eq!(weather.textures.weather_map.size().width, 128);
    assert_eq!(weather.textures.weather_map.size().height, 128);
    assert_eq!(weather.textures.wind_field.size().width, 32);
    assert_eq!(weather.textures.wind_field.size().height, 16);
    assert_eq!(weather.textures.wind_field.size().depth_or_array_layers, 32);
    assert_eq!(weather.textures.top_down_density_mask.size().width, 128);
    assert_eq!(weather.textures.top_down_density_mask.size().height, 128);
}

#[test]
fn empty_scene_produces_zero_haze_when_visibility_zero() {
    let Some(gpu) = gpu() else { return };
    let mut scene = Scene::default();
    scene.surface.visibility_m = 0.0;
    let config = Config::default();
    let world = WorldState::default();
    let weather = ps_synthesis::synthesise(&scene, &config, &world, gpu).unwrap();
    assert_eq!(weather.haze_extinction_per_m, glam::Vec3::ZERO);
}

#[test]
fn sun_direction_propagates_from_world_state() {
    let Some(gpu) = gpu() else { return };
    let scene = Scene::default();
    let config = Config::default();
    let world = WorldState::default();
    let weather = ps_synthesis::synthesise(&scene, &config, &world, gpu).unwrap();
    // Default WorldState recomputes the sun for J2000.0; whatever the
    // exact value, it should match between WorldState and WeatherState.
    assert_eq!(weather.sun_direction, world.sun_direction_world);
}
