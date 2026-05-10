//! Phase 1 headless integration tests.
//!
//! Each test constructs a small (64×64) `App` from real factories
//! (Backdrop / Ground / Tint), renders one frame to an `Rgba8Unorm` output
//! texture sized to the HDR target, reads it back, and asserts on the
//! average pixel.
//!
//! On machines without a working GPU adapter the tests print a skip
//! message and pass (so CI stays green on adapter-less runners).

use std::path::PathBuf;
use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{Config, GpuContext};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping integration tests — headless adapter unavailable: {e}");
            None
        }
    })
    .as_ref()
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn config_with_ev0_passthrough() -> Config {
    let mut c = Config::default();
    c.render.ev100 = 0.0;
    c.render.tone_mapper = "Passthrough".into();
    // Disable subsystems whose register_passes() is currently a no-op
    // (atmosphere, clouds, precipitation are Phase 5/6/8 stubs). Leaving
    // them enabled is harmless but logs irrelevant constructions.
    c.render.subsystems.atmosphere = false;
    c.render.subsystems.clouds = false;
    c.render.subsystems.precipitation = false;
    c.render.subsystems.wet_surface = false;
    c
}

/// Phase 4 §4.4 acceptance: empty render graph still presents a valid
/// frame. With every subsystem disabled, ps-app's frame loop is just
/// "clear → tone-map → present" and the tone-mapper output should equal
/// the configured `[render].clear_color` (modulo the EV-0 passthrough).
#[test]
fn empty_render_graph_presents_clear_color() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.clear_color = [0.7, 0.0, 0.3, 1.0];

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let pixels = app.render_one_frame(gpu);

    let avg = average_rgb(&pixels);
    eprintln!("empty graph average RGB = {avg:?}");
    // Passthrough at EV0: linear * (1/1.2) → R ≈ 0.583, G ≈ 0, B ≈ 0.25.
    assert!(
        avg[0] > 0.45 && avg[0] < 0.7,
        "expected R≈0.58 (= 0.7/1.2), got {avg:?}"
    );
    assert!(avg[1] < 0.05, "expected near-zero G, got {avg:?}");
    assert!(
        avg[2] > 0.15 && avg[2] < 0.35,
        "expected B≈0.25 (= 0.3/1.2), got {avg:?}"
    );
}

#[test]
fn app_with_backdrop_renders_solid_color() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false; // pure backdrop test
    config.render.backdrop.color = [1.0, 0.0, 0.0];

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let pixels = app.render_one_frame(gpu);

    let avg = average_rgb(&pixels);
    eprintln!("backdrop average RGB = {avg:?}");
    assert!(avg[0] > 0.5, "expected red dominant; got {avg:?}");
    assert!(avg[1] < 0.05, "expected near-zero green; got {avg:?}");
    assert!(avg[2] < 0.05, "expected near-zero blue; got {avg:?}");
}

#[test]
fn app_with_tint_modifies_color() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = true;
    config.render.subsystems.ground = false;
    config.render.backdrop.color = [1.0, 0.0, 0.0];
    config.render.tint.multiplier = [0.5, 0.5, 0.5];

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let pixels = app.render_one_frame(gpu);

    let avg = average_rgb(&pixels);
    eprintln!("tint average RGB = {avg:?}");
    // Backdrop fills with [1, 0, 0] linear. Tint multiplies by 0.5.
    // Tone-map: ev100=0 → exposure ≈ 1/1.2; passthrough clamps to [0,1].
    // Expected linear R ≈ min(0.5 * 1.0 / 1.2, 1.0) ≈ 0.417.
    assert!(
        avg[0] > 0.2 && avg[0] < 0.6,
        "tint should halve the red; got {avg:?}"
    );
    assert!(avg[1] < 0.05);
    assert!(avg[2] < 0.05);
}

#[test]
fn disabling_subsystem_at_runtime_takes_effect() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = true;
    config.render.subsystems.ground = false;
    config.render.backdrop.color = [1.0, 0.0, 0.0];
    config.render.tint.multiplier = [0.5, 0.5, 0.5];

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let with_tint = app.render_one_frame(gpu);
    let with_tint_avg = average_rgb(&with_tint);

    // Now reconfigure to disable tint.
    config.render.subsystems.tint = false;
    app.reconfigure(gpu, &config).expect("reconfigure");
    let no_tint = app.render_one_frame(gpu);
    let no_tint_avg = average_rgb(&no_tint);

    eprintln!("with tint = {with_tint_avg:?}, no tint = {no_tint_avg:?}");
    assert!(
        no_tint_avg[0] > with_tint_avg[0] + 0.1,
        "removing tint should brighten the red: with={with_tint_avg:?} no={no_tint_avg:?}"
    );
}

#[test]
fn pedalsky_toml_at_workspace_root_boots_the_app() {
    let Some(gpu) = gpu() else { return };

    let path = workspace_root().join("pedalsky.toml");
    let mut config = Config::load(&path).expect("workspace pedalsky.toml should parse");
    config
        .validate()
        .expect("workspace pedalsky.toml should validate");

    // Disable Phase 5/6/8 stubs whose passes are no-ops anyway.
    config.render.subsystems.atmosphere = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;
    // Force passthrough + EV0 so the output is meaningful at this size.
    config.render.tone_mapper = "Passthrough".into();
    config.render.ev100 = 0.0;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    // Just constructing this should not panic.
    let _ = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new from workspace toml");
}

fn average_rgb(pixels: &[u8]) -> [f32; 3] {
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    for chunk in pixels.chunks_exact(4) {
        sum[0] += chunk[0] as u64;
        sum[1] += chunk[1] as u64;
        sum[2] += chunk[2] as u64;
        count += 1;
    }
    [
        sum[0] as f32 / count as f32 / 255.0,
        sum[1] as f32 / count as f32 / 255.0,
        sum[2] as f32 / count as f32 / 255.0,
    ]
}
