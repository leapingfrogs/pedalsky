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
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--scene" => scene = iter.next().map(PathBuf::from),
            "--time" => time_str = iter.next().cloned(),
            "--output" => output = iter.next().map(PathBuf::from),
            "--width" => {
                width = iter.next().and_then(|s| s.parse().ok()).unwrap_or(1280)
            }
            "--height" => {
                height = iter.next().and_then(|s| s.parse().ok()).unwrap_or(720)
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
    Some(RenderArgs {
        scene,
        output,
        time,
        width,
        height,
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
        pitch: 5_f32.to_radians(),
        ..FlyCamera::default()
    };

    // Render — uses the harness's full path (atmosphere LUTs, clouds,
    // ground, precip). Surface params from the scene flow through the
    // synthesis pipeline.
    let surface = ps_synthesis::synthesise(&scene, &config, &world, &gpu)
        .context("synthesise WeatherState")?;
    // Override the harness's stub WeatherState with the real one by
    // feeding through render_one_frame_full(world_override). The
    // surface scalars get pushed via the harness's
    // surface_override path. Cloud layers + atmosphere come from the
    // synthesised state once we wire them — for v1 of the headless
    // render path we let HeadlessApp use its stub WeatherState (which
    // honours surface_override but not cloud layers); a future change
    // (when test_harness supports a full-WeatherState override) lights
    // up clouds in headless renders too.
    let _ = surface;

    let pixels = app.render_one_frame_full(
        &gpu,
        camera,
        Some(scene_to_surface_params(&scene)),
        Some(1.0),
        Some(world.clone()),
    );

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

fn scene_to_surface_params(scene: &Scene) -> ps_core::SurfaceParams {
    let s = &scene.surface;
    ps_core::SurfaceParams {
        visibility_m: s.visibility_m,
        temperature_c: s.temperature_c,
        dewpoint_c: s.dewpoint_c,
        pressure_hpa: s.pressure_hpa,
        wind_dir_deg: s.wind_dir_deg,
        wind_speed_mps: s.wind_speed_mps,
        ground_wetness: s.wetness.ground_wetness,
        puddle_coverage: s.wetness.puddle_coverage,
        snow_depth_m: s.wetness.snow_depth_m,
        puddle_start: s.wetness.puddle_start,
        precip_intensity_mm_per_h: scene.precipitation.intensity_mm_per_h,
        precip_kind: match scene.precipitation.kind {
            ps_core::PrecipKind::None => 0.0,
            ps_core::PrecipKind::Rain => 1.0,
            ps_core::PrecipKind::Snow => 2.0,
            ps_core::PrecipKind::Sleet => 3.0,
        },
    }
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
