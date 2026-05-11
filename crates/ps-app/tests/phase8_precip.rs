//! Phase 8 precipitation acceptance tests.
//!
//! Plan §8 acceptance:
//! - Setting `intensity_mm_per_h = 5` while clouds are present produces
//!   visible rain streaks.
//! - Turning off clouds zeroes the top-down density mask and the rain
//!   disappears.
//!
//! Tests:
//! 1. Pipeline dispatches cleanly (rain enabled, intensity > 0).
//! 2. Rain renders perceptibly differ from no-rain image.
//! 3. With cloud mask zeroed, rain disappears (re-uses the existing
//!    no-cloud-mask path by injecting a zero-mask SurfaceParams via a
//!    custom path).

use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext, SurfaceParams};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping precip tests — no GPU adapter: {e}");
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
    config.render.subsystems.ground = true;
    config.render.subsystems.precipitation = true;
    config.render.subsystems.clouds = false;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.wet_surface = false;
    config
}

fn forward_camera() -> FlyCamera {
    // Look forward, slight downward — frame should include both sky and
    // ground so any rain streaks against the brighter sky are easy to
    // detect.
    FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: -10_f32.to_radians(),
        ..FlyCamera::default()
    }
}

#[test]
fn precip_pipeline_dispatches_cleanly() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let with_rain = SurfaceParams {
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };
    let _ = app.render_one_frame_with_surface(gpu, forward_camera(), Some(with_rain));
}

#[test]
fn rain_changes_image_vs_no_precip() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (256, 256));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // No precip baseline.
    let dry = SurfaceParams::default();
    let dry_pixels = app.render_one_frame_with_surface(gpu, forward_camera(), Some(dry));

    // Rain at 5 mm/h with cloud mask present (the stub uploads mask=1.0).
    let with_rain = SurfaceParams {
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };
    let rain_pixels =
        app.render_one_frame_with_surface(gpu, forward_camera(), Some(with_rain));

    let mut diff: u64 = 0;
    for (a, b) in dry_pixels.chunks_exact(4).zip(rain_pixels.chunks_exact(4)) {
        for c in 0..3 {
            diff += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let avg_l1 = diff as f64 / (dry_pixels.len() / 4) as f64;
    eprintln!("rain vs dry avg L1 = {avg_l1:.2}");
    // 5 mm/h is light rain; with 8000 particles spread across a 50 m
    // cylinder, only ~1500 land in the forward hemisphere of view at
    // pitch=-10°. The aggregate signal is small but non-zero, and it
    // disappears entirely when the cloud mask is gated off (verified by
    // rain_disappears_when_cloud_mask_zero).
    assert!(
        avg_l1 > 0.3,
        "rain at 5 mm/h with clouds should perceptibly modify the image \
         (avg L1 = {avg_l1:.2})"
    );
}

#[test]
fn rain_disappears_when_cloud_mask_zero() {
    // Plan §8.3 acceptance: "turning off clouds zeroes the top-down
    // density mask and the rain disappears." We simulate the absence of
    // clouds by overriding the mask to 0.0 while intensity stays at 5
    // mm/h. The expected behaviour: identical image to the no-precip
    // baseline.
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (256, 256));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let with_rain = SurfaceParams {
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };
    let dry = SurfaceParams::default();

    let dry_pixels = app.render_one_frame_with_surface_and_mask(
        gpu,
        forward_camera(),
        Some(dry),
        Some(0.0),
    );
    let no_clouds_pixels = app.render_one_frame_with_surface_and_mask(
        gpu,
        forward_camera(),
        Some(with_rain),
        Some(0.0),
    );

    let mut diff: u64 = 0;
    for (a, b) in dry_pixels
        .chunks_exact(4)
        .zip(no_clouds_pixels.chunks_exact(4))
    {
        for c in 0..3 {
            diff += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let avg_l1 = diff as f64 / (dry_pixels.len() / 4) as f64;
    eprintln!("no-clouds rain vs dry avg L1 = {avg_l1:.2}");
    assert!(
        avg_l1 < 0.1,
        "rain must disappear when cloud mask is zero (got avg L1 = {avg_l1:.2})"
    );
}

#[test]
fn rain_ripples_change_wet_ground_appearance() {
    // Plan §8.5: ripples render on the wet ground when wetness > 0.5
    // AND intensity > 0. With wetness=1, puddle_coverage=1, and
    // precip>0, the water-layer BRDF is evaluated with a perturbed
    // normal — different specular falls in different pixels than the
    // calm-water case.
    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    // Wet surface flag must be on for puddles + ripples.
    config.render.subsystems.wet_surface = true;
    let setup = TestSetup::new(gpu, &config, (256, 256));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        // Pitch down a bit so ground takes most of the frame.
        pitch: -25_f32.to_radians(),
        yaw: std::f32::consts::PI,
        ..FlyCamera::default()
    };

    let calm_water = SurfaceParams {
        ground_wetness: 1.0,
        puddle_coverage: 1.0,
        puddle_start: 0.0,
        precip_intensity_mm_per_h: 0.0,
        precip_kind: 0.0,
        ..SurfaceParams::default()
    };
    let rainy_water = SurfaceParams {
        ground_wetness: 1.0,
        puddle_coverage: 1.0,
        puddle_start: 0.0,
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };

    let calm = app.render_one_frame_with_surface_and_mask(
        gpu,
        camera.clone(),
        Some(calm_water),
        Some(1.0),
    );
    let rainy = app.render_one_frame_with_surface_and_mask(
        gpu,
        camera,
        Some(rainy_water),
        Some(1.0),
    );

    // The ripple-perturbed normal redistributes the specular highlight.
    // We expect a non-trivial difference in the ground region.
    let mut diff: u64 = 0;
    for (a, b) in calm.chunks_exact(4).zip(rainy.chunks_exact(4)) {
        for c in 0..3 {
            diff += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let avg_l1 = diff as f64 / (calm.len() / 4) as f64;
    eprintln!("calm vs ripply water avg L1 = {avg_l1:.2}");
    assert!(
        avg_l1 > 0.2,
        "ripples should perceptibly change wet-ground appearance \
         (avg L1 = {avg_l1:.2})"
    );
}

#[test]
fn marshall_palmer_intensity_scales_visibility() {
    // Heavy rain (50 mm/h) should produce a more visible image than
    // light rain (1 mm/h). Marshall-Palmer drop density ∝ I^0.21, so
    // the per-particle alpha rises monotonically with intensity.
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (256, 256));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let dry = SurfaceParams::default();
    let light_rain = SurfaceParams {
        precip_intensity_mm_per_h: 1.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };
    let heavy_rain = SurfaceParams {
        precip_intensity_mm_per_h: 50.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };

    let dry_pixels = app.render_one_frame_with_surface_and_mask(
        gpu,
        forward_camera(),
        Some(dry),
        Some(1.0),
    );
    let light_pixels = app.render_one_frame_with_surface_and_mask(
        gpu,
        forward_camera(),
        Some(light_rain),
        Some(1.0),
    );
    let heavy_pixels = app.render_one_frame_with_surface_and_mask(
        gpu,
        forward_camera(),
        Some(heavy_rain),
        Some(1.0),
    );

    let l1 = |a: &[u8], b: &[u8]| -> f64 {
        let mut d: u64 = 0;
        for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            for c in 0..3 {
                d += (x[c] as i32 - y[c] as i32).unsigned_abs() as u64;
            }
        }
        d as f64 / (a.len() / 4) as f64
    };
    let light_delta = l1(&dry_pixels, &light_pixels);
    let heavy_delta = l1(&dry_pixels, &heavy_pixels);
    eprintln!("light (1mm/h) L1 = {light_delta:.2}, heavy (50mm/h) L1 = {heavy_delta:.2}");
    assert!(
        heavy_delta > light_delta,
        "heavier rain should perturb the image more (light={light_delta:.2}, heavy={heavy_delta:.2})"
    );
}

#[test]
fn snow_kind_renders_distinct_pixels() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (256, 256));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let rain = SurfaceParams {
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 1.0,
        ..SurfaceParams::default()
    };
    let snow = SurfaceParams {
        precip_intensity_mm_per_h: 5.0,
        precip_kind: 2.0,
        ..SurfaceParams::default()
    };

    let rain_pixels = app.render_one_frame_with_surface(gpu, forward_camera(), Some(rain));
    let snow_pixels = app.render_one_frame_with_surface(gpu, forward_camera(), Some(snow));

    // Snow particles use a different sprite (round splat, white tint vs
    // rain's blue-grey streak). The two images should differ.
    let mut diff: u64 = 0;
    for (a, b) in rain_pixels.chunks_exact(4).zip(snow_pixels.chunks_exact(4)) {
        for c in 0..3 {
            diff += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let avg_l1 = diff as f64 / (rain_pixels.len() / 4) as f64;
    eprintln!("rain vs snow avg L1 = {avg_l1:.2}");
    assert!(
        avg_l1 > 0.2,
        "rain and snow should render distinct sprites (got avg L1 = {avg_l1:.2})"
    );
}
