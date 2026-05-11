//! Phase 5 acceptance tests for the atmosphere subsystem.
//!
//! These tests are part of the §5 acceptance ("daytime sky qualitatively
//! matches Hillaire's published screenshots"). They verify substrate-
//! correctness (the pipeline dispatches cleanly) and rough photometric
//! reasonableness (a midday sky pixel produces non-trivial luminance
//! after ACES tone-map at EV100=15).

use std::path::PathBuf;
use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{Config, GpuContext};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping atmosphere tests — no GPU adapter: {e}");
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
fn sky_pass_produces_non_black_output_at_midday() {
    let Some(gpu) = gpu() else { return };
    let path = workspace_root().join("pedalsky.toml");
    let mut config = Config::load(&path).expect("workspace config");
    config.validate_with_base(Some(&workspace_root())).unwrap();
    config.render.ev100 = 15.0;
    config.render.tone_mapper = "ACESFilmic".into();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // Tilt the camera up so the framebuffer is mostly upper hemisphere
    // (zenith / mid-sky) rather than near-horizon paths where Rayleigh
    // out-scattering reddens the colour balance.
    let camera = ps_core::camera::FlyCamera {
        pitch: 60_f32.to_radians(),
        ..ps_core::camera::FlyCamera::default()
    };
    let pixels = app.render_one_frame_with(gpu, camera);

    let avg = average_rgb(&pixels);
    eprintln!("midday sky avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    assert!(
        total > 0.05,
        "expected daytime sky to be visibly non-black; got {avg:?}"
    );
    // Sky should be blue-dominant: B channel > R channel by some margin.
    assert!(
        avg[2] > avg[0] + 0.05,
        "expected blue-dominant daytime sky; got {avg:?}"
    );
}

#[test]
fn ground_bounce_lifts_downward_sky_above_zero() {
    // Plan §5.3 (Hillaire 2020 §6): a ray that hits the planet inside the
    // sky pass picks up a ground-bounce term proportional to ground_albedo
    // and the sun-to-ground transmittance. Disable ground geometry so the
    // sky pass's downward output isn't overpainted, then look down and
    // confirm the result is non-trivially bright.
    let Some(gpu) = gpu() else { return };
    let mut config = Config::default();
    config.render.ev100 = 15.0;
    config.render.tone_mapper = "ACESFilmic".into();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // Pitch 60 degrees down so the camera frame is dominated by below-
    // horizon rays.
    let camera = ps_core::camera::FlyCamera {
        pitch: -60_f32.to_radians(),
        ..ps_core::camera::FlyCamera::default()
    };
    let pixels = app.render_one_frame_with(gpu, camera);
    let avg = average_rgb(&pixels);
    eprintln!("downward sky (ground bounce) avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    // Earth ground_albedo defaults to (0.18, 0.18, 0.18); midday sun gives
    // ~100k lux at the surface → bounce of ~5700 cd/m² per channel, well
    // above the EV100=15 noise floor after ACES.
    assert!(
        total > 0.05,
        "expected ground bounce to brighten downward sky; got {avg:?}"
    );
}

#[test]
fn atmosphere_pipeline_dispatches_cleanly() {
    let Some(gpu) = gpu() else { return };

    let mut config = Config::default();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let _ = app.render_one_frame(gpu);
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
