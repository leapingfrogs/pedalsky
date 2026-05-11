//! Weather scene configuration. Mirrors the schema in plan Appendix B.
//!
//! Loaded via `[paths] weather` from the engine config. Phase 3 turns the
//! parsed `Scene` into a GPU-resident `WeatherState`; Phase 1 only parses
//! and validates.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

/// Errors returned by [`Scene::load`] and [`Scene::validate`].
#[derive(Debug, Error)]
pub enum SceneError {
    /// `std::fs::read_to_string` failed.
    #[error("could not read scene {path}: {source}")]
    Io {
        /// Path that failed to read.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// `toml::from_str` failed (syntax error, unknown field, type mismatch).
    #[error("could not parse scene {path}: {source}")]
    Parse {
        /// Path that failed to parse.
        path: PathBuf,
        /// Underlying TOML error.
        #[source]
        source: Box<toml::de::Error>,
    },
    /// Two or more cloud layers overlap in altitude. v1 requires layers to be
    /// vertically disjoint (plan §3.2.2).
    #[error(
        "cloud layers {a} ({a_base_m}-{a_top_m} m) and {b} ({b_base_m}-{b_top_m} m) overlap; \
         v1 requires vertically disjoint layers"
    )]
    OverlappingCloudLayers {
        /// Index of the first overlapping layer.
        a: usize,
        /// Base of layer `a` in metres.
        a_base_m: f32,
        /// Top of layer `a` in metres.
        a_top_m: f32,
        /// Index of the second overlapping layer.
        b: usize,
        /// Base of layer `b` in metres.
        b_base_m: f32,
        /// Top of layer `b` in metres.
        b_top_m: f32,
    },
    /// Other semantic validation failure.
    #[error("invalid scene: {0}")]
    Invalid(String),
}

/// Cloud type enum. Field name in TOML is the PascalCase variant identifier
/// (`type = "Cumulus"`).
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CloudType {
    /// Fair-weather puffy cumulus.
    Cumulus = 0,
    /// Low, layered, featureless.
    Stratus = 1,
    /// Low, lumpy stratiform.
    Stratocumulus = 2,
    /// Mid-level cumuliform.
    Altocumulus = 3,
    /// Mid-level stratiform sheet.
    Altostratus = 4,
    /// High, wispy ice cloud.
    Cirrus = 5,
    /// High, thin ice sheet.
    Cirrostratus = 6,
    /// Towering thunderstorm with anvil.
    Cumulonimbus = 7,
}

/// Top-level weather scene. Loaded from the path configured under
/// `[paths] weather` in `pedalsky.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Scene {
    /// Schema version (currently 1).
    pub schema_version: u32,
    /// Surface conditions block.
    pub surface: Surface,
    /// Cloud layers block.
    #[serde(default)]
    pub clouds: Clouds,
    /// Precipitation block.
    pub precipitation: Precipitation,
    /// Lightning block (placeholder; Phase 8+ uses it).
    pub lightning: Lightning,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            schema_version: 1,
            surface: Surface::default(),
            clouds: Clouds::default(),
            precipitation: Precipitation::default(),
            lightning: Lightning::default(),
        }
    }
}

/// `[surface]` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Surface {
    /// Meteorological visibility in metres (Koschmieder β = 3.912 / V).
    pub visibility_m: f32,
    /// Surface temperature in °C.
    pub temperature_c: f32,
    /// Dew-point in °C.
    pub dewpoint_c: f32,
    /// Mean sea-level pressure in hPa.
    pub pressure_hpa: f32,
    /// Wind direction in degrees (meteorological — direction wind comes from).
    pub wind_dir_deg: f32,
    /// Wind speed in m/s at 10 m AGL.
    pub wind_speed_mps: f32,
    /// Wetness sub-block (defaulted if absent).
    pub wetness: Wetness,
}

impl Default for Surface {
    fn default() -> Self {
        Self {
            visibility_m: 30000.0,
            temperature_c: 17.0,
            dewpoint_c: 7.0,
            pressure_hpa: 1018.0,
            wind_dir_deg: 240.0,
            wind_speed_mps: 5.0,
            wetness: Wetness::default(),
        }
    }
}

/// `[surface.wetness]` sub-block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Wetness {
    /// Continuous wetness scalar in \[0, 1\] (drives the Lagarde 2013 wet BRDF).
    pub ground_wetness: f32,
    /// Fraction of ground covered by puddles in \[0, 1\].
    pub puddle_coverage: f32,
    /// Wetness threshold above which puddles begin forming.
    pub puddle_start: f32,
    /// Snow depth in metres.
    pub snow_depth_m: f32,
}

impl Default for Wetness {
    fn default() -> Self {
        Self {
            ground_wetness: 0.0,
            puddle_coverage: 0.0,
            puddle_start: 0.6,
            snow_depth_m: 0.0,
        }
    }
}

/// `[clouds]` block: a list of layers + an optional gridded coverage texture.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Clouds {
    /// Per-layer envelope parameters.
    #[serde(default, rename = "layers")]
    pub layers: Vec<CloudLayer>,
    /// Optional gridded coverage override.
    #[serde(default)]
    pub coverage_grid: Option<CoverageGrid>,
}

/// One cloud layer (`[[clouds.layers]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CloudLayer {
    /// Cloud type (matches the WGSL `ndf()` switch).
    #[serde(rename = "type")]
    pub cloud_type: CloudType,
    /// Layer base in metres AGL.
    pub base_m: f32,
    /// Layer top in metres AGL.
    pub top_m: f32,
    /// Coverage in \[0, 1\] before per-pixel weather-map modulation.
    pub coverage: f32,
    /// Optical density multiplier.
    pub density_scale: f32,
    /// Bias on the shape-noise frequency (negative = blockier, positive = wispier).
    pub shape_octave_bias: f32,
    /// Bias on the detail-noise frequency.
    pub detail_octave_bias: f32,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            cloud_type: CloudType::Cumulus,
            base_m: 1500.0,
            top_m: 2300.0,
            coverage: 0.5,
            density_scale: 1.0,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
        }
    }
}

/// Optional `[clouds.coverage_grid]` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CoverageGrid {
    /// Spatial extent in metres (the grid covers `[-extent/2, +extent/2]` on
    /// each axis, centred on the world origin).
    pub extent_m: f32,
    /// Pixel dimensions (width, height).
    pub size: [u32; 2],
    /// Path to a row-major little-endian f32 raw file.
    pub data_path: PathBuf,
    /// Optional companion u8 type-index file.
    #[serde(default)]
    pub type_path: Option<PathBuf>,
}

impl Default for CoverageGrid {
    fn default() -> Self {
        Self {
            extent_m: 32_000.0,
            size: [128, 128],
            data_path: PathBuf::new(),
            type_path: None,
        }
    }
}

/// Precipitation kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PrecipKind {
    /// No precipitation.
    None,
    /// Liquid rain.
    Rain,
    /// Frozen snow.
    Snow,
    /// Mixed phase.
    Sleet,
}

/// `[precipitation]` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Precipitation {
    /// Kind of precipitation.
    #[serde(rename = "type")]
    pub kind: PrecipKind,
    /// Intensity in mm/hour.
    pub intensity_mm_per_h: f32,
}

impl Default for Precipitation {
    fn default() -> Self {
        Self {
            kind: PrecipKind::None,
            intensity_mm_per_h: 0.0,
        }
    }
}

/// `[lightning]` block (placeholder).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Lightning {
    /// Strikes per minute per square kilometre.
    pub strikes_per_min_per_km2: f32,
}

impl Default for Lightning {
    fn default() -> Self {
        Self {
            strikes_per_min_per_km2: 0.0,
        }
    }
}

impl Scene {
    /// Read and parse a scene from `path`. Does **not** call [`validate`](Self::validate).
    ///
    /// Relative paths inside `[clouds.coverage_grid]` (`data_path`,
    /// `type_path`) are resolved against `path`'s parent directory so
    /// that downstream synthesis can open them by absolute path
    /// without needing the caller to remember the scene's base.
    pub fn load(path: &Path) -> Result<Self, SceneError> {
        let bytes = std::fs::read_to_string(path).map_err(|source| SceneError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut scene = Self::parse(&bytes).map_err(|e| match e {
            SceneError::Parse { source, .. } => SceneError::Parse {
                path: path.to_path_buf(),
                source,
            },
            other => other,
        })?;
        if let Some(base) = path.parent() {
            scene.resolve_relative_paths(base);
        }
        Ok(scene)
    }

    /// Rewrite relative paths inside `[clouds.coverage_grid]` so they
    /// are absolute (or at least anchored to a known base). Idempotent
    /// when paths are already absolute.
    fn resolve_relative_paths(&mut self, base: &Path) {
        if let Some(grid) = self.clouds.coverage_grid.as_mut() {
            if grid.data_path.is_relative() {
                grid.data_path = base.join(&grid.data_path);
            }
            if let Some(tp) = grid.type_path.as_mut() {
                if tp.is_relative() {
                    *tp = base.join(&*tp);
                }
            }
        }
    }

    /// Parse a scene from an in-memory string.
    pub fn parse(s: &str) -> Result<Self, SceneError> {
        toml::from_str::<Self>(s).map_err(|source| SceneError::Parse {
            path: PathBuf::from("<memory>"),
            source: Box::new(source),
        })
    }

    /// Semantic validation: schema version, layer ordering, cloud layer
    /// non-overlap (plan §3.2.2), value ranges.
    pub fn validate(&self) -> Result<(), SceneError> {
        if self.schema_version != 1 {
            return Err(SceneError::Invalid(format!(
                "schema_version = {} (only version 1 is supported in v1)",
                self.schema_version
            )));
        }
        if self.surface.visibility_m <= 0.0 {
            return Err(SceneError::Invalid(
                "surface.visibility_m must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.surface.wetness.ground_wetness) {
            return Err(SceneError::Invalid(
                "surface.wetness.ground_wetness must be in [0, 1]".into(),
            ));
        }
        for (i, layer) in self.clouds.layers.iter().enumerate() {
            if layer.top_m <= layer.base_m {
                return Err(SceneError::Invalid(format!(
                    "clouds.layers[{i}]: top_m ({}) <= base_m ({})",
                    layer.top_m, layer.base_m
                )));
            }
            if !(0.0..=1.0).contains(&layer.coverage) {
                return Err(SceneError::Invalid(format!(
                    "clouds.layers[{i}].coverage = {} (must be in [0, 1])",
                    layer.coverage
                )));
            }
        }
        // Check for vertical overlap between any pair of layers.
        for a in 0..self.clouds.layers.len() {
            for b in (a + 1)..self.clouds.layers.len() {
                let la = &self.clouds.layers[a];
                let lb = &self.clouds.layers[b];
                let overlap = la.base_m < lb.top_m && lb.base_m < la.top_m;
                if overlap {
                    return Err(SceneError::OverlappingCloudLayers {
                        a,
                        a_base_m: la.base_m,
                        a_top_m: la.top_m,
                        b,
                        b_base_m: lb.base_m,
                        b_top_m: lb.top_m,
                    });
                }
            }
        }
        if self.precipitation.intensity_mm_per_h < 0.0 {
            return Err(SceneError::Invalid(
                "precipitation.intensity_mm_per_h must be >= 0".into(),
            ));
        }
        info!(
            target: "ps_core::scene",
            schema_version = self.schema_version,
            cloud_layers = self.clouds.layers.len(),
            precip = ?self.precipitation.kind,
            visibility_m = self.surface.visibility_m,
            "scene validated"
        );
        Ok(())
    }
}
