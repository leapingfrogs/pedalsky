//! Engine root configuration. Mirrors the schema in plan Appendix A.
//!
//! All structs deny unknown fields, so a typo in `pedalsky.toml` surfaces as
//! a load error rather than silently using defaults. Every struct also
//! implements `Default` so partial files work.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

/// Errors returned by [`Config::load`] and [`Config::validate`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// `std::fs::read_to_string` failed.
    #[error("could not read config {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `toml::from_str` failed (syntax error, unknown field, type mismatch).
    #[error("could not parse config {path}: {source}")]
    Parse {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying TOML error (location info preserved).
        #[source]
        source: Box<toml::de::Error>,
    },
    /// Semantic validation failed.
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// Top-level engine configuration. Loaded from `pedalsky.toml` at the
/// workspace root (or any path passed to [`Config::load`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Window dimensions, title, vsync.
    pub window: WindowConfig,
    /// Observer location, ground albedo, planet geometry.
    pub world: WorldConfig,
    /// Wall-clock starting state.
    pub time: TimeConfig,
    /// Render targets, exposure, tone mapper, subsystem toggles, per-subsystem
    /// tuning.
    pub render: RenderConfig,
    /// Filesystem paths (scene, noise cache, screenshots).
    pub paths: PathsConfig,
    /// Debug toggles.
    pub debug: DebugConfig,
}

/// Window configuration block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WindowConfig {
    /// Window pixel width.
    pub width: u32,
    /// Window pixel height.
    pub height: u32,
    /// Title bar text.
    pub title: String,
    /// `true` → AutoVsync; `false` → Immediate.
    pub vsync: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            title: "PedalSky".into(),
            vsync: true,
        }
    }
}

/// World / observer configuration block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WorldConfig {
    /// Observer latitude (degrees, north positive). Range \[-90, 90\].
    pub latitude_deg: f64,
    /// Observer longitude (degrees, east positive). Range \[-180, 180\].
    pub longitude_deg: f64,
    /// Ground elevation above sea level in metres.
    pub ground_elevation_m: f32,
    /// Default ground albedo (linear sRGB).
    pub ground_albedo: [f32; 3],
    /// Planet radius in metres (Earth-like default 6_360_000).
    pub ground_radius_m: f32,
    /// Top of atmosphere altitude in metres (Earth-like default 100_000).
    pub atmosphere_top_m: f32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            latitude_deg: 56.1922,
            longitude_deg: -3.9645,
            ground_elevation_m: 60.0,
            ground_albedo: [0.18, 0.18, 0.18],
            ground_radius_m: 6_360_000.0,
            atmosphere_top_m: 100_000.0,
        }
    }
}

/// Wall-clock configuration block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TimeConfig {
    /// Year (Gregorian).
    pub year: i32,
    /// Month \[1, 12\].
    pub month: u32,
    /// Day of month \[1, 31\].
    pub day: u32,
    /// Hour \[0, 23\].
    pub hour: u32,
    /// Minute \[0, 59\].
    pub minute: u32,
    /// Second \[0, 59\].
    pub second: u32,
    /// Local-time offset in hours from UTC.
    pub timezone_offset_hours: f64,
    /// Whether the clock advances each frame.
    pub auto_advance: bool,
    /// Multiplier when `auto_advance = true`.
    pub time_scale: f64,
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            year: 2026,
            month: 5,
            day: 10,
            hour: 14,
            minute: 30,
            second: 0,
            timezone_offset_hours: 1.0,
            auto_advance: false,
            time_scale: 1.0,
        }
    }
}

/// Render configuration block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RenderConfig {
    /// HDR target format identifier (currently always `Rgba16Float`).
    pub hdr_format: String,
    /// Depth target format identifier (currently always `Depth32Float`).
    pub depth_format: String,
    /// Photographic EV at ISO 100.
    pub ev100: f32,
    /// Tone mapper choice: `ACESFilmic`, `Reinhard`, or `Passthrough`.
    pub tone_mapper: String,
    /// Linear-sRGB clear colour (ignored if any subsystem owns the SkyBackdrop stage).
    pub clear_color: [f32; 4],

    /// Subsystem on/off switches.
    pub subsystems: SubsystemFlags,
    /// Per-subsystem tuning.
    pub atmosphere: AtmosphereTuning,
    /// Per-subsystem tuning.
    pub clouds: CloudsTuning,
    /// Per-subsystem tuning.
    pub precip: PrecipTuning,
    /// Phase 1 demo: BackdropSubsystem tuning.
    pub backdrop: BackdropTuning,
    /// Phase 1 demo: TintSubsystem tuning.
    pub tint: TintTuning,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            hdr_format: "Rgba16Float".into(),
            depth_format: "Depth32Float".into(),
            ev100: 14.0,
            tone_mapper: "ACESFilmic".into(),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            subsystems: SubsystemFlags::default(),
            atmosphere: AtmosphereTuning::default(),
            clouds: CloudsTuning::default(),
            precip: PrecipTuning::default(),
            backdrop: BackdropTuning::default(),
            tint: TintTuning::default(),
        }
    }
}

/// Subsystem on/off switches.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SubsystemFlags {
    /// Phase 0 / Phase 7 ground subsystem.
    pub ground: bool,
    /// Phase 5 atmosphere subsystem.
    pub atmosphere: bool,
    /// Phase 6 volumetric clouds.
    pub clouds: bool,
    /// Phase 8 precipitation.
    pub precipitation: bool,
    /// Phase 7 wet surface.
    pub wet_surface: bool,
    /// Phase 1 demo: Backdrop (clears HDR target to a solid colour).
    pub backdrop: bool,
    /// Phase 1 demo: Tint (fullscreen RGB multiply).
    pub tint: bool,
}

impl Default for SubsystemFlags {
    fn default() -> Self {
        Self {
            ground: true,
            atmosphere: true,
            clouds: true,
            precipitation: false,
            wet_surface: false,
            backdrop: true,
            tint: false,
        }
    }
}

/// Atmosphere tuning block (Phase 5 reads this; Phase 1 just round-trips it).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AtmosphereTuning {
    /// Whether to include the multi-scattering LUT.
    pub multi_scattering: bool,
    /// Whether to render an analytic sun disk.
    pub sun_disk: bool,
    /// Sun angular radius in degrees (Earth ≈ 0.27).
    pub sun_angular_radius_deg: f32,
    /// Whether to include ozone absorption.
    pub ozone_enabled: bool,
}

impl Default for AtmosphereTuning {
    fn default() -> Self {
        Self {
            multi_scattering: true,
            sun_disk: true,
            sun_angular_radius_deg: 0.27,
            ozone_enabled: true,
        }
    }
}

/// Cloud tuning block (Phase 6 reads this).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CloudsTuning {
    /// Primary march steps along the camera ray.
    pub cloud_steps: u32,
    /// Shadow march steps to the sun per density sample.
    pub light_steps: u32,
    /// Multiple-scattering octaves for the Hillaire 2016 approximation.
    pub multi_scatter_octaves: u32,
    /// Detail erosion strength.
    pub detail_strength: f32,
    /// Beer-Powder lerp factor.
    pub powder_strength: f32,
    /// Reprojection mode (only `"off"` is supported in v1).
    pub reprojection: String,
    /// Pause `simulated_seconds` advancing for screenshot framing.
    pub freeze_time: bool,
}

impl Default for CloudsTuning {
    fn default() -> Self {
        Self {
            cloud_steps: 192,
            light_steps: 6,
            multi_scatter_octaves: 4,
            detail_strength: 0.35,
            powder_strength: 1.0,
            reprojection: "off".into(),
            freeze_time: false,
        }
    }
}

/// Precipitation tuning block (Phase 8 reads this).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrecipTuning {
    /// Particle pool size for near rain/snow.
    pub near_particle_count: u32,
    /// Number of layered streak textures for far rain.
    pub far_layers: u32,
}

impl Default for PrecipTuning {
    fn default() -> Self {
        Self {
            near_particle_count: 8000,
            far_layers: 3,
        }
    }
}

/// Phase 1 demo: clear-colour for the BackdropSubsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BackdropTuning {
    /// Linear-sRGB clear colour.
    pub color: [f32; 3],
}

impl Default for BackdropTuning {
    fn default() -> Self {
        Self {
            color: [0.05, 0.07, 0.12],
        }
    }
}

/// Phase 1 demo: per-channel multiplier for the TintSubsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TintTuning {
    /// Linear-sRGB multiplier applied to every fragment.
    pub multiplier: [f32; 3],
}

impl Default for TintTuning {
    fn default() -> Self {
        Self {
            multiplier: [1.0, 1.0, 1.0],
        }
    }
}

/// Filesystem paths block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PathsConfig {
    /// Scene config (relative to workspace root or absolute).
    pub weather: PathBuf,
    /// Cached compute-baked noise volumes.
    pub noise_cache: PathBuf,
    /// Where the UI's screenshot button writes PNG/EXR pairs.
    pub screenshot_dir: PathBuf,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            weather: PathBuf::from("scenes/broken_cumulus_afternoon.toml"),
            noise_cache: PathBuf::from("assets/noise"),
            screenshot_dir: PathBuf::from("screenshots"),
        }
    }
}

/// Debug toggles block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DebugConfig {
    /// Enable wgpu validation in release builds (debug builds always enable it).
    pub gpu_validation: bool,
    /// Watch `shaders/` and rebuild affected pipelines on change.
    pub shader_hot_reload: bool,
    /// `tracing` filter (`trace | debug | info | warn | error`).
    pub log_level: String,
    /// Optional auto-exposure (off by default).
    #[serde(default)]
    pub auto_exposure: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            gpu_validation: true,
            shader_hot_reload: true,
            log_level: "info".into(),
            auto_exposure: false,
        }
    }
}

impl Config {
    /// Read and parse a config from `path`. Does **not** call [`validate`](Self::validate).
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let bytes = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&bytes).map_err(|e| match e {
            ConfigError::Parse { source, .. } => ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            },
            other => other,
        })
    }

    /// Parse a config from an in-memory string. Used by tests; the load path
    /// wraps this with IO.
    pub fn parse(s: &str) -> Result<Self, ConfigError> {
        toml::from_str::<Self>(s).map_err(|source| ConfigError::Parse {
            path: PathBuf::from("<memory>"),
            source: Box::new(source),
        })
    }

    /// Semantic validation: range checks on lat/lon, time consistency.
    ///
    /// Equivalent to [`Self::validate_with_base`] with `None`; skips
    /// file-existence checks. Used by tests and code that doesn't have a
    /// configured base directory.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_with_base(None)
    }

    /// Like [`Self::validate`] but additionally checks that file paths in
    /// `[paths]` exist on disk, resolved against `base` (typically the
    /// directory containing the config file).
    pub fn validate_with_base(&self, base: Option<&Path>) -> Result<(), ConfigError> {
        if !(-90.0..=90.0).contains(&self.world.latitude_deg) {
            return Err(ConfigError::Invalid(format!(
                "world.latitude_deg = {} (must be in [-90, 90])",
                self.world.latitude_deg
            )));
        }
        if !(-180.0..=180.0).contains(&self.world.longitude_deg) {
            return Err(ConfigError::Invalid(format!(
                "world.longitude_deg = {} (must be in [-180, 180])",
                self.world.longitude_deg
            )));
        }
        if self.world.ground_radius_m <= 0.0 {
            return Err(ConfigError::Invalid(
                "world.ground_radius_m must be > 0".into(),
            ));
        }
        if self.world.atmosphere_top_m <= 0.0 {
            return Err(ConfigError::Invalid(
                "world.atmosphere_top_m must be > 0".into(),
            ));
        }
        if !(1..=12).contains(&self.time.month) {
            return Err(ConfigError::Invalid(format!(
                "time.month = {} (must be in [1, 12])",
                self.time.month
            )));
        }
        if !(1..=31).contains(&self.time.day) {
            return Err(ConfigError::Invalid(format!(
                "time.day = {} (must be in [1, 31])",
                self.time.day
            )));
        }
        if self.time.hour > 23 {
            return Err(ConfigError::Invalid(format!(
                "time.hour = {} (must be in [0, 23])",
                self.time.hour
            )));
        }
        if self.time.minute > 59 {
            return Err(ConfigError::Invalid(format!(
                "time.minute = {} (must be in [0, 59])",
                self.time.minute
            )));
        }
        if self.time.second > 59 {
            return Err(ConfigError::Invalid(format!(
                "time.second = {} (must be in [0, 59])",
                self.time.second
            )));
        }
        if self.window.width == 0 || self.window.height == 0 {
            return Err(ConfigError::Invalid(format!(
                "window.{{width, height}} must be > 0 (got {}×{})",
                self.window.width, self.window.height
            )));
        }
        if self.render.clouds.reprojection != "off" {
            return Err(ConfigError::Invalid(format!(
                "render.clouds.reprojection = {:?} — only \"off\" is supported in v1",
                self.render.clouds.reprojection
            )));
        }
        // File existence: only run if we have a base directory and the path
        // is non-empty. Resolved relative to `base` (or used absolutely when
        // already absolute).
        if let Some(base) = base {
            if !self.paths.weather.as_os_str().is_empty() {
                let resolved = if self.paths.weather.is_absolute() {
                    self.paths.weather.clone()
                } else {
                    base.join(&self.paths.weather)
                };
                if !resolved.is_file() {
                    return Err(ConfigError::Invalid(format!(
                        "paths.weather = {:?} not found at {}",
                        self.paths.weather,
                        resolved.display()
                    )));
                }
            }
        }
        info!(
            target: "ps_core::config",
            window = ?self.window,
            world.lat = self.world.latitude_deg,
            world.lon = self.world.longitude_deg,
            subsystems.backdrop = self.render.subsystems.backdrop,
            subsystems.tint = self.render.subsystems.tint,
            subsystems.atmosphere = self.render.subsystems.atmosphere,
            subsystems.clouds = self.render.subsystems.clouds,
            scene = %self.paths.weather.display(),
            "config validated"
        );
        Ok(())
    }
}
