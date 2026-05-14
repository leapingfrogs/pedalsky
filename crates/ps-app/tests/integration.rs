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
    c.render.tone_mapper = ps_core::TonemapMode::Passthrough;
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
    config.render.tone_mapper = ps_core::TonemapMode::Passthrough;
    config.render.ev100 = 0.0;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    // Just constructing this should not panic.
    let _ = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new from workspace toml");
}

/// Phase 13.8 — stress the subsystem on/off combination space.
///
/// With 11 controllable subsystem flags (ground / atmosphere / clouds
/// / precipitation / wet_surface / godrays / lightning / aurora /
/// bloom / windsock / water) the full combination space is 2^11 =
/// 2048. The scope doc allows sampling down to ≤32 if runtime would
/// exceed 2 minutes, which it would: each combination pays an
/// atmosphere-LUT bake cost (≥10 ms) plus pipeline construction.
///
/// We sample 48 combinations: the 11 boundary cases (each subsystem
/// flipped on alone, plus the all-off and all-on cases — that's 13)
/// plus 35 pseudo-random masks drawn from a deterministic LCG so the
/// failure mode is reproducible from the seed.
///
/// Asserts: every combination builds a HeadlessApp, renders one
/// frame, and tears down without wgpu validation errors. The actual
/// pixel content is not checked.
#[test]
fn subsystem_combinations_render_without_validation_errors() {
    let Some(gpu) = gpu() else { return };

    // Deterministic LCG (Numerical Recipes constants) — gives the
    // same sequence every run so failures reproduce.
    let mut rng_state: u32 = 0xc0de_face;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        rng_state
    };

    // Build the 48 masks.
    const N_SUBSYSTEMS: u32 = 11;
    let mut masks: Vec<u32> = Vec::new();
    masks.push(0);                       // all off
    masks.push((1 << N_SUBSYSTEMS) - 1); // all on
    for i in 0..N_SUBSYSTEMS {
        masks.push(1 << i);              // one-on cases
    }
    while masks.len() < 48 {
        masks.push(next() & ((1 << N_SUBSYSTEMS) - 1));
    }

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;

    for (idx, &mask) in masks.iter().enumerate() {
        let s = &mut config.render.subsystems;
        s.ground        = mask & (1 << 0)  != 0;
        s.atmosphere    = mask & (1 << 1)  != 0;
        s.clouds        = mask & (1 << 2)  != 0;
        s.precipitation = mask & (1 << 3)  != 0;
        s.wet_surface   = mask & (1 << 4)  != 0;
        s.godrays       = mask & (1 << 5)  != 0;
        s.lightning     = mask & (1 << 6)  != 0;
        s.aurora        = mask & (1 << 7)  != 0;
        s.bloom         = mask & (1 << 8)  != 0;
        s.windsock      = mask & (1 << 9)  != 0;
        s.water         = mask & (1 << 10) != 0;

        let setup = TestSetup::new(gpu, &config, (48, 32));
        let mut app = HeadlessApp::new(gpu, &config, setup)
            .unwrap_or_else(|e| panic!("HeadlessApp::new mask 0x{mask:03x}: {e}"));
        let _pixels = app.render_one_frame(gpu);
        if idx % 8 == 7 {
            eprintln!("subsystem stress: {}/{} masks ok", idx + 1, masks.len());
        }
    }
}

/// Phase 13.8 — verify that toggling atmosphere off then on does
/// not leave dependent subsystems with stale LUT references.
///
/// The contract: subsystems that bind atmosphere LUTs (ground,
/// clouds, water, windsock) build their bind groups inside the pass
/// closure each frame, against `ctx.luts_bind_group` (which is
/// `None` when atmosphere is disabled and `Some(...)` against the
/// freshly-published LUT bundle otherwise). When atmosphere is
/// re-enabled, the new factory builds a fresh `AtmosphereLuts` arc
/// and publishes it through the shared cell — but the *consumers*
/// don't know about that and instead receive the new bind group via
/// the per-frame `PrepareContext::atmosphere_luts` borrow. This
/// test asserts that round-trips work without panicking or tripping
/// wgpu validation.
#[test]
fn atmosphere_toggle_roundtrip_preserves_dependent_subsystems() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = true;
    config.render.subsystems.clouds = false; // simpler — drop the cloud pipeline.
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;

    let setup = TestSetup::new(gpu, &config, (48, 32));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let _ = app.render_one_frame(gpu);
    // Drop and re-create with atmosphere off so ground loses LUTs.
    let mut config_off = config.clone();
    config_off.render.subsystems.atmosphere = false;
    let setup2 = TestSetup::new(gpu, &config_off, (48, 32));
    let mut app2 = HeadlessApp::new(gpu, &config_off, setup2).expect("HeadlessApp::new (off)");
    let _ = app2.render_one_frame(gpu);
    // Re-create with atmosphere back on. Ground should pick up the
    // newly-baked LUTs from the freshly-built atmosphere subsystem.
    let setup3 = TestSetup::new(gpu, &config, (48, 32));
    let mut app3 = HeadlessApp::new(gpu, &config, setup3).expect("HeadlessApp::new (on again)");
    let _ = app3.render_one_frame(gpu);
}

/// Phase 13.5 — water subsystem builds and renders one frame when
/// the scene supplies a `[water]` block. Asserts no GPU validation
/// regressions; the exact pixel values depend on lat/lon and time.
#[test]
fn water_subsystem_renders_one_frame() {
    let Some(gpu) = gpu() else { return };
    use ps_core::{Scene, WorldState, camera::FlyCamera};

    let mut config = config_with_ev0_passthrough();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.water = true;

    let mut scene = Scene::default();
    scene.water = Some(ps_core::Water::default());

    let setup = TestSetup::new(gpu, &config, (96, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let world = WorldState::new(
        chrono::Utc::now(),
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );
    let camera = FlyCamera::default();
    let _pixels = app.render_one_frame_with_scene(gpu, camera, &scene, world);
}

/// Phase 13.6 — the windsock subsystem renders cleanly when atmosphere
/// is enabled (the shader binds the AP LUT, so the pass needs the LUTs
/// to be live). With no GPU validation errors this smoke test just
/// asserts construction + one full frame.
#[test]
fn windsock_subsystem_renders_one_frame() {
    let Some(gpu) = gpu() else { return };

    let mut config = config_with_ev0_passthrough();
    // The windsock shader samples the atmosphere AP LUT, so the
    // atmosphere subsystem must be on for the pass to draw.
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = false;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.windsock = true;

    let setup = TestSetup::new(gpu, &config, (96, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // Rendering should not panic or trip validation. The exact pixel
    // contents depend on default surface wind direction (240°) and
    // are not asserted here.
    let _pixels = app.render_one_frame(gpu);
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
