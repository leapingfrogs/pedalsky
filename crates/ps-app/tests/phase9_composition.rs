//! Phase 9 acceptance tests.
//!
//! Plan §9.1: cloud march termination uses per-pixel scene depth as
//! `t_max` so clouds clip behind opaque geometry.
//! Plan §9.2: debug auto-exposure mode derives EV100 from average
//! log-luminance of the HDR target.
//! Plan §9.3 / §9.4 acceptance: midday EV=15, sunset EV=12, twilight
//! EV=8 produce believable photographic images (non-zero, non-clipped,
//! qualitatively right colour balance).

use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext, HdrFramebufferImpl, SurfaceParams};
use ps_postprocess::AutoExposure;

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping phase 9 tests — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

fn baseline_config() -> Config {
    let mut config = Config::default();
    config.render.tone_mapper = "ACESFilmic".into();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = true;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.wet_surface = false;
    config
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

// ---------------------------------------------------------------------------
// §9.1 — render order: tonemap registered as a PassStage::ToneMap pass
// ---------------------------------------------------------------------------

#[test]
fn tonemap_is_registered_as_in_graph_pass() {
    use ps_core::PassStage;
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // The HeadlessApp's internal `App` must include exactly one
    // `PassStage::ToneMap` pass — that's the in-graph tonemap subsystem.
    let stages = app.app_for_test().pass_stages();
    let count = stages.iter().filter(|s| **s == PassStage::ToneMap).count();
    assert_eq!(
        count, 1,
        "expected exactly one ToneMap-stage pass; got stages = {stages:?}"
    );
}

// ---------------------------------------------------------------------------
// §9.1 — depth-aware cloud termination
// ---------------------------------------------------------------------------

#[test]
fn clouds_clip_behind_ground_via_depth() {
    let Some(gpu) = gpu() else { return };
    // Enable both ground (opaque, 200x200 km quad at y=0) and clouds.
    // The default cumulus layer sits at 1500-3500 m AGL. With the
    // camera looking 30° down, every view ray hits the ground long
    // before reaching the cloud altitude — so depth-aware termination
    // should suppress cloud contributions in those pixels.
    let mut config = baseline_config();
    config.render.subsystems.clouds = true;
    config.render.ev100 = 15.0;
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // Camera looking down at the ground.
    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: -30_f32.to_radians(),
        ..FlyCamera::default()
    };
    let with_clouds = app.render_one_frame_with(gpu, camera.clone());

    // Same camera with clouds disabled.
    let mut config_no_clouds = config.clone();
    config_no_clouds.render.subsystems.clouds = false;
    let setup2 = TestSetup::new(gpu, &config_no_clouds, (128, 128));
    let mut app2 =
        HeadlessApp::new(gpu, &config_no_clouds, setup2).expect("HeadlessApp::new no-clouds");
    let no_clouds = app2.render_one_frame_with(gpu, camera);

    // For the *bottom half* of the frame (which is below the horizon →
    // ground only), the two images should be nearly identical because
    // depth-aware termination clips clouds behind the ground.
    let row_bytes = 128 * 4;
    let half_offset = 64 * row_bytes;
    let mut diff_bottom: u64 = 0;
    let bottom_a = &with_clouds[half_offset..];
    let bottom_b = &no_clouds[half_offset..];
    for (a, b) in bottom_a.chunks_exact(4).zip(bottom_b.chunks_exact(4)) {
        for c in 0..3 {
            diff_bottom += (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
        }
    }
    let avg_l1_bottom = diff_bottom as f64 / (bottom_a.len() / 4) as f64;
    eprintln!("bottom-half (ground-occluded) avg L1 = {avg_l1_bottom:.2}");
    // The cloud composite blends with `OneMinusSrcAlpha` on top of the
    // ground; with t1 clipped at the ground hit there's no cloud
    // density to integrate, so the cloud RT alpha is zero and the
    // composite leaves the ground untouched.
    assert!(
        avg_l1_bottom < 0.5,
        "clouds should not contribute to ground-occluded pixels (avg L1 = {avg_l1_bottom:.2})"
    );
}

// ---------------------------------------------------------------------------
// §9.2 — debug auto-exposure
// ---------------------------------------------------------------------------

#[test]
fn auto_exposure_derives_ev100_from_avg_log_luminance() {
    // Build a small HDR target, fill with a known constant luminance,
    // dispatch the auto-exposure pass, read back, check derived EV100.
    let Some(gpu) = gpu() else { return };
    let hdr = HdrFramebufferImpl::new(gpu, (64, 64));
    let ae = AutoExposure::new(&gpu.device, &hdr);

    // Fill the HDR target with a known luminance using textureCopy via
    // queue.write_texture. We pack 64x64 Rgba16Float pixels at exact
    // values that map to luminance = 1000 cd/m² (a daylight-ish value).
    let target_lum: f32 = 1000.0;
    let pixel = [
        half::f16::from_f32(target_lum),
        half::f16::from_f32(target_lum),
        half::f16::from_f32(target_lum),
        half::f16::from_f32(1.0),
    ];
    let bytes_per_pixel = 8usize;
    let mut data = Vec::with_capacity(64 * 64 * bytes_per_pixel);
    for _ in 0..(64 * 64) {
        data.extend_from_slice(bytemuck::cast_slice(&pixel));
    }
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &hdr.color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64 * bytes_per_pixel as u32),
            rows_per_image: Some(64),
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );

    // Drive the pipelined readback over two frames so the GPU has a
    // chance to complete the slot we just wrote. Sprint 1 made
    // `read_back_ev100` non-blocking: the first call after fresh
    // construction submits this dispatch's slot for mapping and reads
    // the opposite (still-Idle) slot, returning None. Wait between
    // frames so the GPU has time to complete the map_async.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ae-test"),
    });
    ae.dispatch(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let _ = ae.read_back_ev100(&gpu.device); // first call submits, returns None
    let _ = gpu.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }); // wait for the just-submitted map
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ae-test-2"),
    });
    ae.dispatch(&mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let ev100 = ae.read_back_ev100(&gpu.device).expect("readback");

    // Expected: ev100 = log2(avg_lum / 0.216). With Rec.709 luminance
    // weights summing to 1.0 and all channels = 1000, lum = 1000.
    let expected = (target_lum / 0.216_f32).log2();
    eprintln!("auto-exposure EV100 = {ev100:.3}, expected ≈ {expected:.3}");
    assert!(
        (ev100 - expected).abs() < 0.1,
        "EV100 derivation off (got {ev100:.3}, expected {expected:.3})"
    );
}

// ---------------------------------------------------------------------------
// §9.3/9.4 — photographic exposure acceptance
// ---------------------------------------------------------------------------

fn render_with(
    gpu: &GpuContext,
    config: Config,
    camera: FlyCamera,
    surface: SurfaceParams,
) -> Vec<u8> {
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    app.render_one_frame_with_surface(gpu, camera, Some(surface))
}

#[test]
fn midday_at_ev15_is_believable() {
    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.ev100 = 15.0;
    // Default test world has sun ~+0.92 elevation (noon equator).
    let camera = FlyCamera {
        pitch: 60_f32.to_radians(),
        ..FlyCamera::default()
    };
    let pixels = render_with(gpu, config, camera, SurfaceParams::default());
    let avg = average_rgb(&pixels);
    eprintln!("midday EV=15 avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    // Mid-day blue sky at EV=15 should be visibly bright, blue-dominant,
    // not clipped to white.
    assert!(total > 0.3 && total < 2.6, "total RGB out of range: {total:.3}");
    assert!(avg[2] > avg[0] + 0.1, "expected blue-dominant midday sky");
    // No fully-clipped pixels (we want headroom).
    let clipped = pixels
        .chunks_exact(4)
        .filter(|p| p[0] == 255 && p[1] == 255 && p[2] == 255)
        .count();
    let frac = clipped as f32 / (pixels.len() / 4) as f32;
    eprintln!("midday clipped fraction = {frac:.4}");
    assert!(frac < 0.05, "too many clipped pixels at EV=15: {frac}");
}

#[test]
fn twilight_at_ev8_is_visible() {
    // Twilight: sun ~5° below horizon. Build a WorldState with a
    // pre-dawn time and a far-northern latitude so the sun stays
    // below the local horizon. EV=8 (7 stops brighter than midday's
    // EV=15) lifts the dim atmospheric inscatter to a visible image.
    use chrono::TimeZone;

    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.ev100 = 8.0;

    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // 04:30 UTC at 60° N on the prime meridian, June solstice → sun at
    // ~−5° altitude (civil twilight). Specifying a custom UTC moves the
    // sun off the harness's default J2000 noon-equator zenith.
    let twilight_utc = chrono::Utc.with_ymd_and_hms(2026, 6, 21, 2, 30, 0).unwrap();
    let world = ps_core::WorldState::new(twilight_utc, 60.0, 0.0, 0.0);
    eprintln!("twilight sun_dir = {:?}", world.sun_direction_world);
    // Sanity: confirm the sun is below the horizon.
    assert!(
        world.sun_direction_world.y < 0.0,
        "expected sun below horizon, got y = {}",
        world.sun_direction_world.y
    );

    let camera = FlyCamera {
        // Look at the brighter twilight horizon (around the +Z direction
        // — yaw=0 looks down -Z so use yaw=π).
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: 5_f32.to_radians(),
        yaw: std::f32::consts::PI,
        ..FlyCamera::default()
    };

    let pixels = app.render_one_frame_full(
        gpu,
        camera,
        Some(SurfaceParams::default()),
        Some(0.0),
        Some(world),
    );
    let avg = average_rgb(&pixels);
    eprintln!("twilight EV=8 avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    // Twilight should be visible (not pure black) but not clipped.
    assert!(
        total > 0.05,
        "twilight EV=8 should be visible (total={total:.3})"
    );
    assert!(
        total < 2.5,
        "twilight EV=8 should not be fully clipped (total={total:.3})"
    );
}

#[test]
fn ev_scale_responds_monotonically() {
    // Sanity check that the EV100 slider is wired: lowering EV from 15
    // to 8 (7 stops brighter) on the same midday scene should saturate.
    let Some(gpu) = gpu() else { return };
    let camera = FlyCamera {
        pitch: 60_f32.to_radians(),
        ..FlyCamera::default()
    };

    let mut c8 = baseline_config();
    c8.render.ev100 = 8.0;
    let pixels = render_with(gpu, c8, camera, SurfaceParams::default());
    let avg = average_rgb(&pixels);
    eprintln!("EV=8 over midday scene avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    assert!(
        total > 2.0,
        "EV=8 over midday scene should saturate (total={total:.3})"
    );
}

#[test]
fn sunset_at_ev12_has_warm_tint() {
    // Sunset: sun close to the horizon. Pre-sunset civil time at 60° N
    // gives a low sun with strong Rayleigh out-scattering — sky should
    // tilt warm (R > B) when looking near the sun. EV=12 lifts the
    // dimmer sunset sky to a believable photographic image.
    use chrono::TimeZone;

    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.ev100 = 12.0;
    let setup = TestSetup::new(gpu, &config, (128, 128));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");

    // 20:30 UTC at 60° N on the prime meridian, June solstice → sun
    // close to the horizon (~5°). Late evening pushes the long-Rayleigh
    // path to its maximum, producing the strongest sunset tint.
    let sunset_utc = chrono::Utc.with_ymd_and_hms(2026, 6, 21, 20, 30, 0).unwrap();
    let world = ps_core::WorldState::new(sunset_utc, 60.0, 0.0, 0.0);
    eprintln!("sunset sun_dir = {:?}", world.sun_direction_world);
    assert!(
        world.sun_direction_world.y > 0.0 && world.sun_direction_world.y < 0.2,
        "expected sun very low above horizon, got y = {}",
        world.sun_direction_world.y
    );

    // Aim the camera straight at the sun's azimuth, near horizon.
    let sun = world.sun_direction_world;
    let sun_yaw = sun.x.atan2(-sun.z);
    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: sun.y.asin(),
        yaw: sun_yaw,
        ..FlyCamera::default()
    };
    let pixels = app.render_one_frame_full(
        gpu,
        camera,
        Some(SurfaceParams::default()),
        Some(0.0),
        Some(world),
    );
    let avg = average_rgb(&pixels);
    eprintln!("sunset EV=12 avg = {avg:?}");
    let total = avg[0] + avg[1] + avg[2];
    assert!(total > 0.2, "sunset EV=12 should be visible (total={total:.3})");

    // Compare R/B chromatic ratio against the equivalent midday scene at
    // the same EV. The sunset path traverses much more atmosphere (low
    // sun → grazing line of sight) so the relative red-to-blue ratio
    // should shift warmer (i.e. R/B higher) than midday.
    let mut midday_config = baseline_config();
    midday_config.render.ev100 = 12.0;
    let midday_setup = TestSetup::new(gpu, &midday_config, (128, 128));
    let mut midday_app =
        HeadlessApp::new(gpu, &midday_config, midday_setup).expect("HeadlessApp::new");
    let midday_camera = FlyCamera {
        pitch: 60_f32.to_radians(),
        ..FlyCamera::default()
    };
    let midday_pixels =
        midday_app.render_one_frame_with_surface(gpu, midday_camera, Some(SurfaceParams::default()));
    let midday_avg = average_rgb(&midday_pixels);
    eprintln!("midday EV=12 avg = {midday_avg:?}");

    let r_to_b = |a: [f32; 3]| a[0] / a[2].max(1e-3);
    let sunset_warmth = r_to_b(avg);
    let midday_warmth = r_to_b(midday_avg);
    eprintln!("R/B ratio: sunset = {sunset_warmth:.3}, midday = {midday_warmth:.3}");
    assert!(
        sunset_warmth > midday_warmth,
        "sunset sky should be warmer (higher R/B) than midday \
         (sunset={sunset_warmth:.3}, midday={midday_warmth:.3})"
    );
}
