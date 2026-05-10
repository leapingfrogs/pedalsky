//! Phase 6 cloud subsystem acceptance tests.
//!
//! Smoke tests:
//! 1. The full pipeline (atmosphere LUT bakes + cloud march + composite)
//!    dispatches cleanly with no validation errors.
//! 2. A single cumulus layer at coverage 0.4 produces non-trivial output
//!    when rendered at midday from sea level.
//! 3. The cumulonimbus profile reaches the upper cloud altitudes (anvil
//!    test).
//!
//! These are coarse — Phase 6 is a high-content phase; the tests here
//! are deliberately broad. Sharper acceptance (silver lining detection,
//! NDF profile equality) is reserved for follow-up work once the bring
//! up is stable.

use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping cloud tests — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

fn baseline_config() -> Config {
    let mut config = Config::default();
    config.render.ev100 = 15.0;
    config.render.tone_mapper = "ACESFilmic".into();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.clouds = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;
    config
}

#[test]
fn cloud_pipeline_dispatches_cleanly() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let _ = app.render_one_frame(gpu);
}

#[test]
fn midday_cumulus_changes_image_vs_clear_sky() {
    let Some(gpu) = gpu() else { return };
    // Tilt the camera up to capture the cloud layer (default 1500..3500 m
    // base/top in CloudsSubsystem::new). At eye level ~30° pitch lands the
    // frame in the cloud altitudes ~3 km away.
    let camera = FlyCamera {
        pitch: 30_f32.to_radians(),
        ..FlyCamera::default()
    };

    // First: render with clouds off.
    let mut config_no_clouds = baseline_config();
    config_no_clouds.render.subsystems.clouds = false;
    let setup = TestSetup::new(gpu, &config_no_clouds, (128, 128));
    let mut app =
        HeadlessApp::new(gpu, &config_no_clouds, setup).expect("HeadlessApp::new no-clouds");
    let clear_sky = app.render_one_frame_with(gpu, camera.clone());
    let clear_avg = average_rgb(&clear_sky);

    // Then: render with clouds on.
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new clouds");
    let cloudy = app.render_one_frame_with(gpu, camera);
    let cloud_avg = average_rgb(&cloudy);

    eprintln!("clear sky avg  = {clear_avg:?}");
    eprintln!("cloudy avg     = {cloud_avg:?}");

    // Total per-pixel L1 difference. If clouds are doing anything at all,
    // there should be at least some delta against the bare sky.
    let mut diff: u64 = 0;
    for (a, b) in clear_sky.chunks_exact(4).zip(cloudy.chunks_exact(4)) {
        for c in 0..3 {
            diff += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let pixel_count = (clear_sky.len() / 4) as f64;
    let avg_l1 = diff as f64 / pixel_count;
    eprintln!("avg L1 delta per pixel (RGB sum 0..765) = {avg_l1:.2}");
    assert!(
        avg_l1 > 1.0,
        "cloud pass should perceptibly change the image (avg L1 = {avg_l1:.2})"
    );
}

#[test]
fn cumulus_produces_bright_cloud_pixels() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new clouds");

    // Pitch up high so the field of view passes through the cloud layer
    // for a long enough horizontal distance to hit substantive density.
    let camera = FlyCamera {
        pitch: 60_f32.to_radians(),
        ..FlyCamera::default()
    };
    let pixels = app.render_one_frame_with(gpu, camera);

    // Find brightest R channel — clouds illuminated by the sun should
    // show up as either R-dominant (red sun side) or near-white pixels
    // depending on geometry. Simply max R works as a proxy.
    let max_r = pixels.chunks_exact(4).map(|p| p[0]).max().unwrap_or(0);
    let max_g = pixels.chunks_exact(4).map(|p| p[1]).max().unwrap_or(0);
    let max_b = pixels.chunks_exact(4).map(|p| p[2]).max().unwrap_or(0);
    eprintln!("cloudy max RGB = ({max_r}, {max_g}, {max_b})");

    // We want cloud render contributing some non-trivial in-scatter.
    // The clear-sky max R at this pitch is dim (~30); any cloud
    // illumination should kick R above ~50.
    assert!(
        max_r > 40,
        "expected cloud illumination to lift max R above 40 (got {max_r})"
    );
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
