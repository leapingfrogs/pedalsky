//! Phase 11.3 helper — `ps-bless` regenerates the golden images for
//! the eight reference scenes, overwriting `tests/golden/<scene>.png`.
//!
//! Run when a *deliberate* visual change has landed. Review the diffs
//! versus version control before committing.

use std::path::PathBuf;

use anyhow::{Context, Result};

const RENDER_W: u32 = 1280;
const RENDER_H: u32 = 720;

/// (scene_name, time_iso8601, ev100, pitch_deg) — must match the
/// `SCENES` table in `crates/ps-app/tests/golden.rs`.
///
/// EV chosen per the Phase 9 acceptance bands: midday=15, sunset=12,
/// twilight=8. Overcast scenes use EV=14 (1 stop brighter than
/// midday to compensate for cloud darkening). Phase 13 follow-up D —
/// overhead-deck scenes use a steeper pitch (look up at the cloud
/// base) so the deck reads as a sheet rather than as a thin band
/// seen edge-on through 9+ km of haze.
const SCENES: &[(&str, &str, f32, f32)] = &[
    ("clear_summer_noon",        "2026-06-21T11:00:00Z", 14.0,  5.0),
    ("broken_cumulus_afternoon", "2026-05-10T14:30:00Z", 14.0,  5.0),
    ("overcast_drizzle",         "2026-04-12T10:00:00Z", 14.0, 45.0),
    ("thunderstorm",             "2026-08-16T16:00:00Z", 14.0,  5.0),
    ("high_cirrus_sunset",       "2026-09-22T17:30:00Z", 11.0,  5.0),
    ("winter_overcast_snow",     "2026-01-08T12:00:00Z", 14.0, 60.0),
    ("twilight_civil",           "2026-12-21T04:30:00Z",  8.0,  5.0),
    ("mountain_wave_clouds",     "2026-03-15T13:00:00Z", 14.0,  5.0),
];

fn workspace_root() -> Result<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("pedalsky.toml").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            anyhow::bail!("workspace root not found");
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let root = workspace_root()?;
    let golden_dir = root.join("tests").join("golden");
    std::fs::create_dir_all(&golden_dir).ok();
    let gpu = ps_core::gpu::init_headless().context("init headless GPU")?;

    for (name, time_iso, ev100, pitch_deg) in SCENES {
        let pixels = render_one(&gpu, &root, name, time_iso, *ev100, *pitch_deg)?;
        let img = image::RgbaImage::from_raw(RENDER_W, RENDER_H, pixels)
            .context("rgba buffer length")?;
        let path = golden_dir.join(format!("{name}.png"));
        img.save(&path)
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("blessed {} (EV={ev100})", path.display());
    }
    Ok(())
}

fn render_one(
    gpu: &ps_core::GpuContext,
    root: &std::path::Path,
    name: &str,
    time_iso: &str,
    ev100: f32,
    pitch_deg: f32,
) -> Result<Vec<u8>> {
    use chrono::{DateTime, Utc};
    use ps_app::test_harness::{HeadlessApp, TestSetup};
    use ps_core::{camera::FlyCamera, Config, Scene, WorldState};

    let config_path = root.join("pedalsky.toml");
    let mut config = Config::load(&config_path).context("load config")?;
    config
        .validate_with_base(config_path.parent())
        .context("validate config")?;
    let scene_path = root.join("tests").join("scenes").join(format!("{name}.toml"));
    let scene = Scene::load(&scene_path).context("load scene")?;
    scene.validate().context("validate scene")?;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.ev100 = ev100;

    let setup = TestSetup::new(gpu, &config, (RENDER_W, RENDER_H));
    let mut app = HeadlessApp::new(gpu, &config, setup).context("HeadlessApp::new")?;
    app.set_ev100(ev100);

    let utc: DateTime<Utc> = DateTime::parse_from_rfc3339(time_iso)
        .context("parse time")?
        .with_timezone(&Utc);
    let world = WorldState::new(
        utc,
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );
    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: pitch_deg.to_radians(),
        ..FlyCamera::default()
    };
    Ok(app.render_one_frame_with_scene(gpu, camera, &scene, world))
}
