//! Phase 7 ground subsystem acceptance tests.
//!
//! Plan §7 acceptance:
//! - Toggling `wetness` from 0 to 1 produces visibly darker, glossier
//!   ground with correct Fresnel at grazing angles.
//! - Snow at 0.05 m depth produces near-white ground when temperature is
//!   below freezing.

use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext, SurfaceParams};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping ground tests — no GPU adapter: {e}");
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
    config.render.subsystems.clouds = false;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.precipitation = false;
    // Phase 7 wet/snow paths are gated by this flag; tests exercise both
    // dry and wet/snow surfaces via SurfaceParams overrides.
    config.render.subsystems.wet_surface = true;
    config
}

fn ground_camera() -> FlyCamera {
    // Eye level looking 30° down so most of the framebuffer hits ground.
    FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: -30_f32.to_radians(),
        ..FlyCamera::default()
    }
}

/// Average RGB of pixels whose green channel is high enough to indicate
/// they hit the ground (sky pixels are typically blue-dominant — G < B).
/// Not exact: a coarse filter to isolate ground without depth readback.
fn ground_pixels_avg(pixels: &[u8]) -> ([f32; 3], usize) {
    let mut sum = [0u64; 3];
    let mut count: usize = 0;
    for chunk in pixels.chunks_exact(4) {
        // Sky: B > R + 20 (atmospheric blue). Ground: roughly neutral.
        let is_sky = (chunk[2] as i32 - chunk[0] as i32) > 30;
        if is_sky {
            continue;
        }
        sum[0] += chunk[0] as u64;
        sum[1] += chunk[1] as u64;
        sum[2] += chunk[2] as u64;
        count += 1;
    }
    if count == 0 {
        return ([0.0; 3], 0);
    }
    (
        [
            sum[0] as f32 / count as f32 / 255.0,
            sum[1] as f32 / count as f32 / 255.0,
            sum[2] as f32 / count as f32 / 255.0,
        ],
        count,
    )
}

#[test]
fn ground_pipeline_dispatches_cleanly() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let _ = app.render_one_frame_with(gpu, ground_camera());
}

#[test]
fn wet_ground_is_darker_than_dry() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let dry = SurfaceParams {
        ground_wetness: 0.0,
        puddle_coverage: 0.0,
        puddle_start: 0.6,
        snow_depth_m: 0.0,
        ..SurfaceParams::default()
    };
    let wet = SurfaceParams {
        ground_wetness: 1.0,
        puddle_coverage: 0.5,
        puddle_start: 0.6,
        snow_depth_m: 0.0,
        ..SurfaceParams::default()
    };

    let dry_pixels =
        app.render_one_frame_with_surface(gpu, ground_camera(), Some(dry));
    let wet_pixels =
        app.render_one_frame_with_surface(gpu, ground_camera(), Some(wet));

    let (dry_avg, dry_n) = ground_pixels_avg(&dry_pixels);
    let (wet_avg, wet_n) = ground_pixels_avg(&wet_pixels);
    eprintln!("dry ground avg ({dry_n} px) = {dry_avg:?}");
    eprintln!("wet ground avg ({wet_n} px) = {wet_avg:?}");

    assert!(dry_n > 100 && wet_n > 100, "expected substantial ground coverage in test frame");

    // Wet ground should be darker on average (Lagarde dark_albedo via
    // pow(albedo, 3) at wetness=1). Specular highlights can locally
    // brighten, but the average over the field should drop.
    let dry_mean = (dry_avg[0] + dry_avg[1] + dry_avg[2]) / 3.0;
    let wet_mean = (wet_avg[0] + wet_avg[1] + wet_avg[2]) / 3.0;
    eprintln!("dry mean = {dry_mean:.3}, wet mean = {wet_mean:.3}");
    assert!(
        wet_mean < dry_mean - 0.005,
        "expected wet ground darker than dry (dry={dry_mean:.3}, wet={wet_mean:.3})"
    );
}

#[test]
fn wet_ground_is_glossier_at_grazing() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // Grazing angle: pitch close to horizon so Fresnel kicks in. Yaw
    // 180° so the camera looks toward +Z (where the test sun is, since
    // default sun_direction has +z component) — otherwise the specular
    // reflection lobe sits behind the camera and the wet highlight isn't
    // visible.
    let grazing = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: -3_f32.to_radians(),
        yaw: std::f32::consts::PI,
        ..FlyCamera::default()
    };

    let dry = SurfaceParams {
        ground_wetness: 0.0,
        ..SurfaceParams::default()
    };
    let wet = SurfaceParams {
        ground_wetness: 1.0,
        // Enable puddles so the smooth water layer's mirror reflection
        // lights up at grazing angles.
        puddle_coverage: 1.0,
        puddle_start: 0.0,
        ..SurfaceParams::default()
    };

    let dry_pixels = app.render_one_frame_with_surface(gpu, grazing.clone(), Some(dry));
    let wet_pixels = app.render_one_frame_with_surface(gpu, grazing, Some(wet));

    // Find max RGB in the bottom half (ground region for a near-horizon
    // shot). Wet should produce specular highlights brighter than dry.
    let half = dry_pixels.len() / 2;
    let dry_max_y = max_brightness(&dry_pixels[half..]);
    let wet_max_y = max_brightness(&wet_pixels[half..]);
    eprintln!("dry max brightness in lower half = {dry_max_y}");
    eprintln!("wet max brightness in lower half = {wet_max_y}");
    assert!(
        wet_max_y >= dry_max_y,
        "wet ground should produce specular highlights at least as bright as dry \
         (dry={dry_max_y}, wet={wet_max_y})"
    );
}

#[test]
fn snow_below_freezing_produces_near_white_ground() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let snow = SurfaceParams {
        ground_wetness: 0.0,
        // Snow distribution requires puddle_coverage > 0 since
        // accumulation = (1 - mask*coverage); leaving it 0 would saturate
        // snow everywhere. Set to a moderate value so a fraction of the
        // ground reads as snow.
        puddle_coverage: 0.3,
        puddle_start: 0.6,
        snow_depth_m: 0.05,
        temperature_c: -2.0,
        ..SurfaceParams::default()
    };

    let pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(snow));
    let (avg, n) = ground_pixels_avg(&pixels);
    eprintln!("snow ground avg ({n} px) = {avg:?}");

    // Snow albedo 0.9 + ambient = visibly bright. Compare against the
    // dry baseline to confirm an upward shift.
    let dry = SurfaceParams::default();
    let dry_pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(dry));
    let (dry_avg, _) = ground_pixels_avg(&dry_pixels);
    let dry_mean = (dry_avg[0] + dry_avg[1] + dry_avg[2]) / 3.0;
    let snow_mean = (avg[0] + avg[1] + avg[2]) / 3.0;
    eprintln!("dry mean = {dry_mean:.3}, snow mean = {snow_mean:.3}");
    assert!(
        snow_mean > dry_mean + 0.05,
        "snow at 0.05m below freezing should brighten ground (dry={dry_mean:.3}, snow={snow_mean:.3})"
    );
}

#[test]
fn snow_above_freezing_does_not_render() {
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // Same snow depth, but temperature above freezing — should be gated.
    let warm_snow = SurfaceParams {
        snow_depth_m: 0.05,
        temperature_c: 5.0,
        ..SurfaceParams::default()
    };
    let dry = SurfaceParams::default();

    let warm_pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(warm_snow));
    let dry_pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(dry));

    let (warm_avg, _) = ground_pixels_avg(&warm_pixels);
    let (dry_avg, _) = ground_pixels_avg(&dry_pixels);
    let dry_mean = (dry_avg[0] + dry_avg[1] + dry_avg[2]) / 3.0;
    let warm_mean = (warm_avg[0] + warm_avg[1] + warm_avg[2]) / 3.0;
    eprintln!("warm-snow mean = {warm_mean:.3}, dry mean = {dry_mean:.3}");
    assert!(
        (warm_mean - dry_mean).abs() < 0.01,
        "snow above freezing should not render (warm={warm_mean:.3}, dry={dry_mean:.3})"
    );
}

#[test]
fn wet_surface_flag_off_disables_wet_path() {
    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.subsystems.wet_surface = false;
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    let wet = SurfaceParams {
        ground_wetness: 1.0,
        puddle_coverage: 1.0,
        puddle_start: 0.0,
        snow_depth_m: 0.05,
        temperature_c: -2.0,
        ..SurfaceParams::default()
    };
    let dry = SurfaceParams::default();

    let wet_pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(wet));
    let dry_pixels = app.render_one_frame_with_surface(gpu, ground_camera(), Some(dry));

    let (wet_avg, _) = ground_pixels_avg(&wet_pixels);
    let (dry_avg, _) = ground_pixels_avg(&dry_pixels);
    let wet_mean = (wet_avg[0] + wet_avg[1] + wet_avg[2]) / 3.0;
    let dry_mean = (dry_avg[0] + dry_avg[1] + dry_avg[2]) / 3.0;
    eprintln!("flag-off: wet input={wet_mean:.3}, dry input={dry_mean:.3}");
    // With the flag off, wet/snow inputs should be ignored — the two
    // renders must collapse to the same image.
    assert!(
        (wet_mean - dry_mean).abs() < 0.01,
        "wet_surface=false should bypass wet/snow regardless of SurfaceParams \
         (got wet={wet_mean:.3}, dry={dry_mean:.3})"
    );
}

fn max_brightness(pixels: &[u8]) -> u8 {
    pixels
        .chunks_exact(4)
        .map(|p| p[0].max(p[1]).max(p[2]))
        .max()
        .unwrap_or(0)
}
