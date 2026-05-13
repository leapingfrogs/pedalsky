//! Phase 10 acceptance tests.
//!
//! The egui UI subsystem itself requires a winit Window to construct
//! (egui_winit::State::new takes &Window). We can't easily mint a
//! headless Window in a unit test, so we cover the spirit of the
//! acceptance criteria with two narrower checks:
//!
//! 1. Slider-style edits routed through `App::reconfigure` produce a
//!    visible change in the next frame's render. This proves the
//!    "config edit → reconfigure → next-frame uniform refresh" path
//!    that powers Phase 10.3 sliders.
//! 2. Every Phase 5–9 uniform field listed in the plan exists on the
//!    `Config` types the UI panels mutate — a static coverage check.

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext};
use std::sync::OnceLock;

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping phase 10 tests — no GPU adapter: {e}");
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

#[test]
fn config_edit_via_reconfigure_changes_next_frame() {
    // Simulates a slider edit on EV100. Render once at EV=15, call
    // reconfigure with EV=10 (which is 5 stops brighter — the scene
    // should saturate), render again, verify the average lifted.
    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.ev100 = 15.0;
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let camera = FlyCamera {
        pitch: 60_f32.to_radians(),
        ..FlyCamera::default()
    };
    let pixels1 = app.render_one_frame_with(gpu, camera.clone());
    let avg1 = average_rgb(&pixels1);
    eprintln!("EV=15 avg = {avg1:?}");

    // Slider-equivalent edit through reconfigure.
    let mut new_config = config.clone();
    new_config.render.ev100 = 10.0;
    app.reconfigure(gpu, &new_config).expect("reconfigure");
    let pixels2 = app.render_one_frame_with(gpu, camera);
    let avg2 = average_rgb(&pixels2);
    eprintln!("EV=10 avg = {avg2:?}");

    let mean = |a: [f32; 3]| (a[0] + a[1] + a[2]) / 3.0;
    assert!(
        mean(avg2) > mean(avg1) + 0.05,
        "reconfigure to brighter EV should lift the image (was {:.3}, now {:.3})",
        mean(avg1),
        mean(avg2)
    );
}

#[test]
fn config_surfaces_every_phase5_to_9_tunable() {
    // Static coverage check: every plan-mentioned tunable should be
    // reachable from the engine config the UI panels mutate. Listing
    // them by *accessing* the field locks the type — if a future
    // refactor renames or removes a field, this test fails to compile.
    let mut c = Config::default();

    // Phase 5 atmosphere — config-level tuning.
    let _ = &mut c.render.atmosphere.multi_scattering;
    let _ = &mut c.render.atmosphere.sun_disk;
    let _ = &mut c.render.atmosphere.sun_angular_radius_deg;
    let _ = &mut c.render.atmosphere.ozone_enabled;

    // Phase 6 clouds.
    let _ = &mut c.render.clouds.cloud_steps;
    let _ = &mut c.render.clouds.light_steps;
    let _ = &mut c.render.clouds.multi_scatter_octaves;
    let _ = &mut c.render.clouds.detail_strength;
    let _ = &mut c.render.clouds.powder_strength;
    let _ = &mut c.render.clouds.freeze_time;
    let _ = &mut c.render.clouds.reprojection;
    let _ = &mut c.render.clouds.wind_drift_strength;
    let _ = &mut c.render.clouds.wind_skew_strength;
    let _ = &mut c.render.clouds.diurnal_strength;

    // Phase 8 precipitation.
    let _ = &mut c.render.precip.near_particle_count;
    let _ = &mut c.render.precip.far_layers;

    // Phase 9 composition.
    let _ = &mut c.render.ev100;
    let _ = &mut c.render.tone_mapper;
    let _ = &mut c.render.clear_color;
    let _ = &mut c.window.vsync;

    // Subsystem toggles.
    let _ = &mut c.render.subsystems.ground;
    let _ = &mut c.render.subsystems.atmosphere;
    let _ = &mut c.render.subsystems.clouds;
    let _ = &mut c.render.subsystems.precipitation;
    let _ = &mut c.render.subsystems.wet_surface;

    // Debug.
    let _ = &mut c.debug.gpu_validation;
    let _ = &mut c.debug.shader_hot_reload;
    let _ = &mut c.debug.auto_exposure;
    let _ = &mut c.debug.atmosphere_lut_overlay;

    // World controls.
    let _ = &mut c.world.latitude_deg;
    let _ = &mut c.world.longitude_deg;
    let _ = &mut c.world.ground_elevation_m;
    let _ = &mut c.time.year;
    let _ = &mut c.time.month;
    let _ = &mut c.time.day;
    let _ = &mut c.time.hour;
    let _ = &mut c.time.minute;
    let _ = &mut c.time.second;
    let _ = &mut c.time.timezone_offset_hours;
    let _ = &mut c.time.auto_advance;
    let _ = &mut c.time.time_scale;
}

#[test]
fn ui_state_round_trips_through_handle() {
    // The UiHandle's UiState is the bridge between panel logic and the
    // host. This test exercises the handle without any egui dependency.
    let initial = baseline_config();
    let handle = ps_ui::UiHandle::new(initial.clone());
    {
        let mut s = handle.lock();
        s.live_config.render.ev100 = 11.5;
        s.pending.config_dirty = true;
        s.pending.screenshot_png = true;
    }
    // Drain (host-side equivalent).
    let mut state = handle.lock();
    assert!(state.pending.config_dirty);
    assert!(state.pending.screenshot_png);
    assert_eq!(state.live_config.render.ev100, 11.5);
    let pending = std::mem::take(&mut state.pending);
    assert!(pending.config_dirty);
    // After drain the state's pending block should be default again.
    assert!(!state.pending.config_dirty);
    assert!(!state.pending.screenshot_png);
    assert_eq!(state.live_config.render.ev100, 11.5);
}

#[test]
fn probe_transmittance_returns_sensible_rgb() {
    // Phase 10.A4 — dispatch the probe compute, read back, check
    // values are in (0, 1) and finite. Centre-pixel through the
    // atmosphere from sea level looking forward should give non-zero
    // transmittance dominated by red (Rayleigh scatters blue more).
    let Some(gpu) = gpu() else { return };
    let mut config = baseline_config();
    config.render.subsystems.atmosphere = true;
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // Render a frame so the LUTs are baked.
    let _ = app.render_one_frame(gpu);

    // Now run the probe.
    let probe = ps_app::probe::ProbeReadback::new(gpu);
    let luts = app
        .atmosphere_luts_for_diag()
        .expect("LUTs published");
    // Build minimal frame + world bind groups via the test harness's
    // shared bindings.
    let result = {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe-test"),
        });
        probe.dispatch(
            &mut encoder,
            &gpu.queue,
            &gpu.device,
            (32, 32),
            app.frame_bind_group_for_test(),
            app.world_bind_group_for_test(),
            &luts.bind_group,
        );
        gpu.queue.submit([encoder.finish()]);
        probe.read(gpu).expect("read probe")
    };
    eprintln!("probe readout = {result:?}");
    for c in &result.transmittance {
        assert!(
            c.is_finite(),
            "transmittance must be finite (got {:?})",
            result.transmittance
        );
        assert!(
            *c >= 0.0 && *c <= 1.0,
            "transmittance out of [0,1]: {:?}",
            result.transmittance
        );
    }
    // Phase 13.10 — per-component OD must also be finite and
    // non-negative (it's an integral of non-negative extinction).
    for od in [&result.od_rayleigh, &result.od_mie, &result.od_ozone] {
        for c in od {
            assert!(c.is_finite(), "OD component non-finite: {od:?}");
            assert!(*c >= 0.0, "OD component negative: {od:?}");
        }
    }
}

#[test]
fn gpu_timestamps_drain_returns_per_pass_durations() {
    // App::drain_gpu_timings should return one entry per registered
    // pass when the device supports timestamps. NVIDIA's Vulkan driver
    // satisfies this; on adapters that don't expose
    // TIMESTAMP_QUERY_INSIDE_ENCODERS the Vec is empty (acceptable).
    let Some(gpu) = gpu() else { return };
    let config = baseline_config();
    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    // First frame populates the staging buffer.
    let _ = app.render_one_frame(gpu);
    // Second frame's drain reads the previous frame's timings.
    let _ = app.render_one_frame(gpu);
    let timings = app.app_for_test().drain_gpu_timings(gpu);
    eprintln!("gpu timings = {timings:?}");
    if !timings.is_empty() {
        // Names must be the registered pass names; durations should be
        // finite and >= 0.
        for (name, ms) in &timings {
            assert!(!name.is_empty(), "pass name should be set");
            assert!(ms.is_finite(), "duration should be finite");
            assert!(*ms >= 0.0, "duration should be non-negative");
        }
        // At least one pass should be detectable. Atmosphere LUT bakes
        // are reliably > 0.
        let any_above_zero = timings.iter().any(|(_, ms)| *ms > 0.0);
        assert!(any_above_zero, "at least one pass should report > 0 ms");
    }
}
