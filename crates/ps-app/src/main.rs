//! PedalSky main binary.
//!
//! Phase 0 + Phase 1 scope: load `pedalsky.toml`, load the scene file pointed
//! to by `[paths] weather`, validate both, log a structured summary. Spinning
//! up the winit window + Phase 0 GPU pipeline lands as part of the Phase 0
//! finalisation work; Phase 1's contribution here is the load/validate path
//! and the AppBuilder wiring (Group B).

use std::path::PathBuf;

use anyhow::{Context, Result};
use ps_core::{Config, Scene};
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> Result<()> {
    // Tracing subscriber: pick up RUST_LOG, default to info.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();

    let workspace_root = workspace_root()?;
    info!(workspace_root = %workspace_root.display(), "starting ps-app");

    let config_path = workspace_root.join("pedalsky.toml");
    let config = Config::load(&config_path)
        .with_context(|| format!("loading {}", config_path.display()))?;
    config
        .validate()
        .with_context(|| format!("validating {}", config_path.display()))?;

    let scene_path = if config.paths.weather.is_absolute() {
        config.paths.weather.clone()
    } else {
        workspace_root.join(&config.paths.weather)
    };
    let scene = Scene::load(&scene_path)
        .with_context(|| format!("loading {}", scene_path.display()))?;
    scene
        .validate()
        .with_context(|| format!("validating {}", scene_path.display()))?;

    info!(
        cloud_layers = scene.clouds.layers.len(),
        precip = ?scene.precipitation.kind,
        "phase 1 load complete; rendering subsystems will land in subsequent phases"
    );

    Ok(())
}

/// Find the workspace root by walking up from `CARGO_MANIFEST_DIR` looking
/// for `pedalsky.toml`.
fn workspace_root() -> Result<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("pedalsky.toml").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            anyhow::bail!("could not find pedalsky.toml above CARGO_MANIFEST_DIR");
        }
    }
}
