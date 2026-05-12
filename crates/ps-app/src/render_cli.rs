//! Phase 11.2 — `ps-app render` headless subcommand.
//!
//! Loads a scene TOML, parses an ISO-8601 time argument, builds a
//! headless `App` (no winit surface), renders one frame at 1280x720,
//! and writes:
//!
//! - `<output>.png` — tonemapped 8-bit sRGB
//! - `<output>.exr` — linear HDR (32-bit float)
//! - `<output>.weather_dump.json` — synthesised state summary
//! - `<output>.parameter_log.toml` — full Config + Scene round-trip
//!
//! Output files are derived from the `--output` argument's stem (the
//! caller passes a base path like `out/clear_summer_noon`; we append
//! the four extensions).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use ps_core::{camera::FlyCamera, Config, GpuContext, Scene, WorldState};

use crate::test_harness::{HeadlessApp, TestSetup};

/// Parsed `--render` invocation.
pub struct RenderArgs {
    /// Scene TOML path (relative to workspace root or absolute).
    pub scene: PathBuf,
    /// Output base path; we append `.png`/`.exr`/`.weather_dump.json`/
    /// `.parameter_log.toml`.
    pub output: PathBuf,
    /// Simulated render time as a UTC `DateTime`.
    pub time: DateTime<Utc>,
    /// Render width in pixels (default 1280).
    pub width: u32,
    /// Render height in pixels (default 720).
    pub height: u32,
    /// Camera yaw override in degrees (rotation around Y from due north).
    /// Default 0 (camera looks north). Useful for headless tests of
    /// effects like godrays where the sun must be in view.
    pub yaw_deg: f32,
    /// Camera pitch override in degrees (rotation about X). Default 5°
    /// (matches the existing reference scene rendering).
    pub pitch_deg: f32,
    /// Phase 12.3 — override the lightning RNG seed for deterministic
    /// strikes. `None` keeps the seed from the engine config block.
    pub seed: Option<u64>,
    /// Phase 12.5 — override observer latitude in degrees. `None`
    /// keeps the value from `pedalsky.toml [world] latitude_deg`.
    /// Useful for headless renders of high-latitude scenes (auroras
    /// at 65°N, polar twilight).
    pub latitude_deg: Option<f64>,
    /// Phase 13.7 — animation sequence: end-of-range time (inclusive).
    /// When set, `time` is the start of the range and `fps` is sampled
    /// at `1/fps`-second intervals. `<output>` becomes a base name and
    /// frames write to `<output>.NNNN.png` (zero-padded to four
    /// digits) plus the matching `.exr` next to each.
    pub time_end: Option<DateTime<Utc>>,
    /// Phase 13.7 — frames per second. Defaults to 24. Ignored when
    /// `time_end` is `None`.
    pub fps: f32,
    /// Phase 13.7 — re-synthesise `WeatherState` for every frame
    /// (default), or once at the start of the sequence. The
    /// "synthesise once" path is much faster but freezes the cloud
    /// noise advection and precip-driven scene mutations.
    pub synthesise_once: bool,
}

/// Detect `render <args...>` after the binary name. Returns `None`
/// when the subcommand isn't present.
pub fn parse_args(argv: &[String]) -> Option<RenderArgs> {
    let mut iter = argv.iter().skip(1);
    if iter.next().map(String::as_str) != Some("render") {
        return None;
    }
    let mut scene: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut time_str: Option<String> = None;
    let mut width: u32 = 1280;
    let mut height: u32 = 720;
    let mut yaw_deg: f32 = 0.0;
    let mut pitch_deg: f32 = 5.0;
    let mut seed: Option<u64> = None;
    let mut latitude_deg: Option<f64> = None;
    let mut time_end_str: Option<String> = None;
    let mut fps: f32 = 24.0;
    let mut synthesise_once = false;
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--scene" => scene = iter.next().map(PathBuf::from),
            "--time" => time_str = iter.next().cloned(),
            "--time-end" => time_end_str = iter.next().cloned(),
            "--fps" => {
                fps = iter.next().and_then(|s| s.parse().ok()).unwrap_or(24.0)
            }
            "--synthesise-once" => synthesise_once = true,
            "--output" => output = iter.next().map(PathBuf::from),
            "--width" => {
                width = iter.next().and_then(|s| s.parse().ok()).unwrap_or(1280)
            }
            "--height" => {
                height = iter.next().and_then(|s| s.parse().ok()).unwrap_or(720)
            }
            "--yaw-deg" => {
                yaw_deg = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0.0)
            }
            "--pitch-deg" => {
                pitch_deg = iter.next().and_then(|s| s.parse().ok()).unwrap_or(5.0)
            }
            "--seed" => {
                seed = iter.next().and_then(|s| s.parse().ok());
            }
            "--latitude-deg" => {
                latitude_deg = iter.next().and_then(|s| s.parse().ok());
            }
            _ => {
                eprintln!("unknown render flag: {arg}");
                return None;
            }
        }
    }
    let scene = scene?;
    let output = output?;
    // Default time = J2000 noon if not supplied (deterministic).
    let time = match time_str {
        Some(s) => DateTime::parse_from_rfc3339(&s)
            .ok()
            .map(|t| t.with_timezone(&Utc))
            .or_else(|| {
                Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).single()
            })?,
        None => Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).single()?,
    };
    let time_end = time_end_str.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .ok()
            .map(|t| t.with_timezone(&Utc))
    });
    Some(RenderArgs {
        scene,
        output,
        time,
        width,
        height,
        yaw_deg,
        pitch_deg,
        seed,
        latitude_deg,
        time_end,
        fps,
        synthesise_once,
    })
}

/// Run the render subcommand. Returns Ok(()) on success.
pub fn run(workspace_root: &Path, args: RenderArgs) -> Result<()> {
    // Load engine config + scene.
    let config_path = workspace_root.join("pedalsky.toml");
    let mut config = Config::load(&config_path)
        .with_context(|| format!("loading {}", config_path.display()))?;
    config
        .validate_with_base(config_path.parent())
        .context("validating engine config")?;
    let scene_path = if args.scene.is_absolute() {
        args.scene.clone()
    } else {
        workspace_root.join(&args.scene)
    };
    let scene = Scene::load(&scene_path)
        .with_context(|| format!("loading {}", scene_path.display()))?;
    scene.validate().context("validating scene")?;

    // Disable backdrop / debug subsystems for the headless render.
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;

    // Phase 12.3 — apply --seed override so headless renders of
    // lightning-bearing scenes are reproducible.
    if let Some(s) = args.seed {
        config.render.lightning.seed = s;
    }
    // Phase 12.5 — apply --latitude-deg override for high-latitude
    // headless renders (auroras, polar twilight).
    if let Some(lat) = args.latitude_deg {
        config.world.latitude_deg = lat;
    }

    // Init headless GPU.
    let gpu = ps_core::gpu::init_headless().context("init headless GPU")?;

    // Build a HeadlessApp at the requested resolution.
    let setup = TestSetup::new(&gpu, &config, (args.width, args.height));
    let mut app = HeadlessApp::new(&gpu, &config, setup).context("HeadlessApp::new")?;

    // Construct the WorldState at the requested time.
    let world = WorldState::new(
        args.time,
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );

    // Default camera — eye-level looking forward + up. Future versions
    // could plumb a per-scene camera override.
    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: args.pitch_deg.to_radians(),
        yaw: args.yaw_deg.to_radians(),
        ..FlyCamera::default()
    };

    // Phase 13.7 — animation-sequence branch. When `time_end` is set
    // the output is a numbered PNG (and EXR) sequence; the single-
    // frame fall-through writes the original four documentation
    // outputs.
    if let Some(time_end) = args.time_end {
        return run_sequence(&gpu, &mut app, &scene, &config, camera, args.time, time_end,
                            args.fps, args.synthesise_once, &args.output, args.width,
                            args.height);
    }

    // Render via the scene-synthesis path so cloud layers, wind field
    // and density mask actually reach the GPU (matches ps-bless).
    let pixels = app.render_one_frame_with_scene(&gpu, camera, &scene, world.clone());

    // Write the four documentation outputs.
    let png_path = output_with_ext(&args.output, "png");
    let exr_path = output_with_ext(&args.output, "exr");
    let json_path = output_with_ext(&args.output, "weather_dump.json");
    let toml_path = output_with_ext(&args.output, "parameter_log.toml");

    if let Some(dir) = png_path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    ps_ui::screenshot::write_png(&png_path, args.width, args.height, &pixels)
        .with_context(|| format!("write {}", png_path.display()))?;
    tracing::info!(target: "ps_app::render", path = %png_path.display(), "wrote PNG");

    // EXR — re-render directly from the live HDR target.
    let hdr_pixels = app.read_hdr_for_test(&gpu);
    ps_ui::screenshot::write_exr(&exr_path, args.width, args.height, &hdr_pixels)
        .with_context(|| format!("write {}", exr_path.display()))?;
    tracing::info!(target: "ps_app::render", path = %exr_path.display(), "wrote EXR");

    // weather_dump.json — minimal scene + world summary.
    let dump = WeatherDump::new(&scene, &world);
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&dump).context("encode weather dump")?,
    )
    .with_context(|| format!("write {}", json_path.display()))?;
    tracing::info!(target: "ps_app::render", path = %json_path.display(), "wrote JSON dump");

    // parameter_log.toml — engine config + scene round-trip.
    let log = ParameterLog {
        config: config.clone(),
        scene: scene.clone(),
    };
    std::fs::write(
        &toml_path,
        toml::to_string_pretty(&log).context("encode parameter log")?,
    )
    .with_context(|| format!("write {}", toml_path.display()))?;
    tracing::info!(target: "ps_app::render", path = %toml_path.display(), "wrote parameter log");

    Ok(())
}

fn output_with_ext(base: &Path, ext: &str) -> PathBuf {
    let parent = base.parent().unwrap_or(Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("render");
    parent.join(format!("{stem}.{ext}"))
}

#[derive(serde::Serialize)]
struct WeatherDump {
    scene_summary: SceneSummary,
    world: WorldSummary,
}

#[derive(serde::Serialize)]
struct SceneSummary {
    cloud_layer_count: usize,
    precip_kind: String,
    precip_intensity_mm_per_h: f32,
    surface_visibility_m: f32,
    surface_wetness: f32,
    surface_snow_depth_m: f32,
}

#[derive(serde::Serialize)]
struct WorldSummary {
    utc: String,
    latitude_deg: f64,
    longitude_deg: f64,
    sun_altitude_deg: f64,
    sun_azimuth_deg: f64,
    moon_altitude_deg: f64,
    moon_azimuth_deg: f64,
    julian_day: f64,
}

impl WeatherDump {
    fn new(scene: &Scene, world: &WorldState) -> Self {
        Self {
            scene_summary: SceneSummary {
                cloud_layer_count: scene.clouds.layers.len(),
                precip_kind: format!("{:?}", scene.precipitation.kind),
                precip_intensity_mm_per_h: scene.precipitation.intensity_mm_per_h,
                surface_visibility_m: scene.surface.visibility_m,
                surface_wetness: scene.surface.wetness.ground_wetness,
                surface_snow_depth_m: scene.surface.wetness.snow_depth_m,
            },
            world: WorldSummary {
                utc: world.clock.current_utc().to_rfc3339(),
                latitude_deg: world.latitude_deg,
                longitude_deg: world.longitude_deg,
                sun_altitude_deg: f64::from(world.sun.altitude_rad).to_degrees(),
                sun_azimuth_deg: f64::from(world.sun.azimuth_rad).to_degrees(),
                moon_altitude_deg: f64::from(world.moon.altitude_rad).to_degrees(),
                moon_azimuth_deg: f64::from(world.moon.azimuth_rad).to_degrees(),
                julian_day: ps_core::astro::julian_day_utc(world.clock.current_utc()),
            },
        }
    }
}

#[derive(serde::Serialize)]
struct ParameterLog {
    config: Config,
    scene: Scene,
}

/// Helper exposed so `main.rs` can hand the headless GPU pollster
/// runtime over.
pub fn use_gpu_context(_gpu: &GpuContext) {}

/// Phase 13.7 — render a PNG/EXR sequence over `[start, end]` at
/// `fps` frames per second. Single GPU init (re-uses the caller's
/// `HeadlessApp`); each frame writes `<output>.NNNN.png` and the
/// matching `.exr`. A single `parameter_log.toml` and
/// `weather_dump.json` are written for the start frame (the
/// animation's setup is the same across the sequence).
#[allow(clippy::too_many_arguments)]
fn run_sequence(
    gpu: &GpuContext,
    app: &mut HeadlessApp,
    scene: &Scene,
    config: &Config,
    camera: FlyCamera,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fps: f32,
    synthesise_once: bool,
    output_base: &Path,
    width: u32,
    height: u32,
) -> Result<()> {
    if end < start {
        anyhow::bail!("--time-end must be at or after --time (got {start} → {end})");
    }
    if fps <= 0.0 {
        anyhow::bail!("--fps must be positive (got {fps})");
    }
    let dt = 1.0 / fps as f64;
    let total_secs = (end - start).num_milliseconds() as f64 / 1000.0;
    let frame_count = (total_secs / dt).floor() as u32 + 1;
    let pad = (frame_count - 1).max(1).to_string().len().max(4);

    tracing::info!(
        target: "ps_app::render",
        frame_count, fps, synthesise_once,
        start = %start, end = %end,
        "rendering animation sequence"
    );

    // Pre-synthesise once when requested. The cached state holds GPU
    // buffers (cloud layers, wind field, density mask) so the inner
    // path skips re-creating them per frame.
    let mut cached_weather: Option<ps_core::WeatherState> = if synthesise_once {
        let world_start = WorldState::new(
            start,
            config.world.latitude_deg,
            config.world.longitude_deg,
            config.world.ground_elevation_m as f64,
        );
        Some(
            ps_synthesis::synthesise(scene, config, &world_start, gpu)
                .context("synthesise WeatherState (once)")?,
        )
    } else {
        None
    };

    if let Some(dir) = output_base.parent() {
        std::fs::create_dir_all(dir).ok();
    }

    // Always write the documentation pair for frame 0; they describe
    // the sequence's setup, not any single frame's appearance.
    let world_start = WorldState::new(
        start,
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );
    let json_path = output_with_ext(output_base, "weather_dump.json");
    let toml_path = output_with_ext(output_base, "parameter_log.toml");
    let dump = WeatherDump::new(scene, &world_start);
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&dump).context("encode weather dump")?,
    )
    .with_context(|| format!("write {}", json_path.display()))?;
    let log = ParameterLog {
        config: config.clone(),
        scene: scene.clone(),
    };
    std::fs::write(
        &toml_path,
        toml::to_string_pretty(&log).context("encode parameter log")?,
    )
    .with_context(|| format!("write {}", toml_path.display()))?;

    for frame in 0..frame_count {
        let secs_into = frame as f64 * dt;
        let frame_time = start + chrono::Duration::milliseconds((secs_into * 1000.0) as i64);
        let world = WorldState::new(
            frame_time,
            config.world.latitude_deg,
            config.world.longitude_deg,
            config.world.ground_elevation_m as f64,
        );

        let pixels = app.render_animation_frame(
            gpu,
            camera.clone(),
            scene,
            world,
            secs_into as f32,
            frame,
            cached_weather.as_mut(),
        );

        let png_path = sequence_path(output_base, "png", frame, pad);
        let exr_path = sequence_path(output_base, "exr", frame, pad);
        ps_ui::screenshot::write_png(&png_path, width, height, &pixels)
            .with_context(|| format!("write {}", png_path.display()))?;
        let hdr_pixels = app.read_hdr_for_test(gpu);
        ps_ui::screenshot::write_exr(&exr_path, width, height, &hdr_pixels)
            .with_context(|| format!("write {}", exr_path.display()))?;
        tracing::info!(
            target: "ps_app::render",
            frame, time = %frame_time, png = %png_path.display(),
            "rendered animation frame"
        );
    }

    tracing::info!(
        target: "ps_app::render",
        frame_count,
        "animation sequence complete"
    );
    Ok(())
}

/// Build the per-frame output path: `<stem>.NNNN.<ext>` where the
/// frame index is zero-padded to `pad` digits.
fn sequence_path(base: &Path, ext: &str, frame: u32, pad: usize) -> PathBuf {
    let parent = base.parent().unwrap_or(Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("render");
    parent.join(format!("{stem}.{frame:0>pad$}.{ext}", pad = pad))
}
