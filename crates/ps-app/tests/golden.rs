//! Phase 11.3 — golden-image regression.
//!
//! For every reference scene, render headlessly at 1280×720 at the
//! scene's nominal time, then compare against `tests/golden/<scene>.png`
//! via SSIM. Tolerance: SSIM ≥ 0.99 (exact bit-comparison isn't
//! feasible because different GPU vendors round fp16 differently).
//!
//! Goldens are checked into `tests/golden/`. To regenerate after a
//! deliberate visual change, run:
//!
//!   cargo run --bin ps-bless
//!
//! On adapters that lack the GPU (e.g. CI without a graphics driver)
//! these tests skip silently.

use std::path::PathBuf;
use std::sync::OnceLock;

use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{camera::FlyCamera, Config, GpuContext, Scene, WorldState};

const SSIM_TOLERANCE: f64 = 0.99;
const RENDER_W: u32 = 1280;
const RENDER_H: u32 = 720;

/// (scene_filename, time_iso8601, ev100) — must match
/// `crates/ps-app/src/bin/ps-bless.rs` so the regenerated goldens
/// share the same render parameters.
const SCENES: &[(&str, &str, f32)] = &[
    ("clear_summer_noon", "2026-06-21T11:00:00Z", 15.0),
    ("broken_cumulus_afternoon", "2026-05-10T14:30:00Z", 15.0),
    ("overcast_drizzle", "2026-04-12T10:00:00Z", 14.0),
    ("thunderstorm", "2026-08-16T16:00:00Z", 14.0),
    ("high_cirrus_sunset", "2026-09-22T17:30:00Z", 11.0),
    ("winter_overcast_snow", "2026-01-08T09:00:00Z", 15.0),
    ("twilight_civil", "2026-12-21T04:30:00Z", 8.0),
    ("mountain_wave_clouds", "2026-03-15T13:00:00Z", 15.0),
];

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("pedalsky.toml").is_file() {
            return dir;
        }
        if !dir.pop() {
            panic!("workspace root not found");
        }
    }
}

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping golden tests — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

/// Render one scene at the canonical resolution + time + EV. Returns
/// raw `Rgba8Unorm` pixels.
pub fn render_scene(
    gpu: &GpuContext,
    scene_name: &str,
    time_iso: &str,
    ev100: f32,
) -> Vec<u8> {
    use chrono::{DateTime, Utc};
    let root = workspace_root();
    let config_path = root.join("pedalsky.toml");
    let mut config = Config::load(&config_path).expect("load engine config");
    config.validate_with_base(config_path.parent()).expect("validate config");
    let scene_path = root.join("tests").join("scenes").join(format!("{scene_name}.toml"));
    let scene = Scene::load(&scene_path).expect("load scene");
    scene.validate().expect("validate scene");
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.ev100 = ev100;

    let setup = TestSetup::new(gpu, &config, (RENDER_W, RENDER_H));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    app.set_ev100(ev100);

    let utc: DateTime<Utc> = DateTime::parse_from_rfc3339(time_iso)
        .expect("parse time")
        .with_timezone(&Utc);
    let world = WorldState::new(
        utc,
        config.world.latitude_deg,
        config.world.longitude_deg,
        config.world.ground_elevation_m as f64,
    );
    let camera = FlyCamera {
        position: glam::Vec3::new(0.0, 1.7, 0.0),
        pitch: 5_f32.to_radians(),
        ..FlyCamera::default()
    };
    app.render_one_frame_with_scene(gpu, camera, &scene, world)
}

/// SSIM via image-compare. Returns the RGB SSIM score in [0, 1].
fn ssim(a: &image::RgbaImage, b: &image::RgbaImage) -> f64 {
    let a_rgb = image::DynamicImage::ImageRgba8(a.clone()).to_rgb8();
    let b_rgb = image::DynamicImage::ImageRgba8(b.clone()).to_rgb8();
    let result = image_compare::rgb_hybrid_compare(&a_rgb, &b_rgb)
        .expect("image-compare failed");
    result.score
}

#[test]
fn all_scenes_match_goldens() {
    let Some(gpu) = gpu() else { return };
    let root = workspace_root();
    let golden_dir = root.join("tests").join("golden");
    if !golden_dir.exists() {
        eprintln!(
            "tests/golden/ does not exist — run `cargo run --bin ps-bless` to seed it."
        );
        return;
    }
    let mut failures = Vec::new();
    for (name, time_iso, ev100) in SCENES {
        let actual = render_scene(gpu, name, time_iso, *ev100);
        let actual_img = image::RgbaImage::from_raw(RENDER_W, RENDER_H, actual)
            .expect("rgba buffer length");
        let golden_path = golden_dir.join(format!("{name}.png"));
        if !golden_path.exists() {
            failures.push(format!(
                "{name}: missing golden at {} — run ps-bless",
                golden_path.display()
            ));
            continue;
        }
        let golden = image::open(&golden_path)
            .unwrap_or_else(|e| panic!("open {}: {e}", golden_path.display()))
            .to_rgba8();
        let score = ssim(&actual_img, &golden);
        eprintln!("{name}: SSIM = {score:.4}");
        if score < SSIM_TOLERANCE {
            // Write the diff actual image for inspection.
            let diff_dir = root.join("tests").join("golden-diffs");
            std::fs::create_dir_all(&diff_dir).ok();
            actual_img
                .save(diff_dir.join(format!("{name}-actual.png")))
                .ok();
            failures.push(format!(
                "{name}: SSIM {score:.4} < {SSIM_TOLERANCE}; actual saved to tests/golden-diffs/{name}-actual.png"
            ));
        }
    }
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} golden(s) failed:\n{}", failures.len(), failures.join("\n"));
    }
}
