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
    /// Phase 12.5 aurora block. Defaults to `kp_index = 0`, which
    /// gates the subsystem off — pre-12.5 scenes that omit the
    /// `[aurora]` section render unchanged.
    #[serde(default)]
    pub aurora: Aurora,
    /// Phase 13.5 — optional water plane. When `None` (the default
    /// for pre-13.5 scenes) the water subsystem renders nothing.
    /// When `Some`, the ps-water subsystem draws a rectangular
    /// water surface with GGX specular + Fresnel sky reflection over
    /// the configured bounds.
    #[serde(default)]
    pub water: Option<Water>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            schema_version: 1,
            surface: Surface::default(),
            clouds: Clouds::default(),
            precipitation: Precipitation::default(),
            lightning: Lightning::default(),
            aurora: Aurora::default(),
            water: None,
        }
    }
}

/// `[scene.water]` block — Phase 13.5.
///
/// A flat rectangular water surface. Renders with GGX/Smith specular,
/// Fresnel-weighted sky reflection (sampled from the sky-view LUT at
/// the reflected ray), and a procedural normal map advected with the
/// surface wind direction. Refraction is explicitly out of scope —
/// there is no DEM, so there's nothing under the water to refract
/// toward.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Water {
    /// Minimum X-coordinate (metres, world space).
    pub xmin: f32,
    /// Maximum X-coordinate (metres, world space).
    pub xmax: f32,
    /// Minimum Z-coordinate (metres, world space).
    pub zmin: f32,
    /// Maximum Z-coordinate (metres, world space).
    pub zmax: f32,
    /// Water surface altitude in metres (world Y).
    pub altitude_m: f32,
    /// GGX base roughness for the calm-water case. Pinned low so the
    /// specular highlight is tight at minimum wind. Typical real-world
    /// water sits in `[0.02, 0.10]` — choppy water leans toward 0.10,
    /// lake-glass toward 0.02. The shader interpolates between
    /// `roughness_min` (calm) and `roughness_max` (windy) by surface
    /// wind speed.
    pub roughness_min: f32,
    /// GGX roughness ceiling for fully wind-roughed water.
    pub roughness_max: f32,
}

impl Default for Water {
    fn default() -> Self {
        // 100m × 100m pond centred on the origin. The altitude sits a
        // hair above the ground plane (y=0) so coplanar depth ties
        // don't lose the water pass to the ground pass.
        Self {
            xmin: -50.0,
            xmax: 50.0,
            zmin: -50.0,
            zmax: 50.0,
            altitude_m: 0.1,
            roughness_min: 0.02,
            roughness_max: 0.10,
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
    /// Phase 13.4 — surface material. Drives the Voronoi palette,
    /// base roughness, and Fresnel F0 in the ground PBR shader.
    /// Defaults to `Grass` so pre-13.4 scenes render unchanged.
    #[serde(default)]
    pub material: SurfaceMaterial,
    /// Phase 14 — winds at pressure levels above the surface, sorted
    /// by ascending altitude. Populated by the Open-Meteo feed at
    /// 850 / 700 / 500 / 300 hPa; empty by default so synthetic /
    /// scene-only renders keep using the procedural 1/7 power-law +
    /// Ekman profile in `ps-synthesis`. When non-empty, the wind-field
    /// synthesis interpolates these samples directly instead of
    /// extrapolating from the surface scalar.
    #[serde(default)]
    pub winds_aloft: Vec<WindAloftSample>,
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
            material: SurfaceMaterial::default(),
            winds_aloft: Vec::new(),
        }
    }
}

/// Phase 14 — one upper-air wind sample at a fixed pressure level.
/// Open-Meteo exposes wind speed + direction at the standard
/// pressure levels (1000 / 925 / 850 / 700 / 500 / 300 / 200 hPa);
/// each sample is converted to the height of that pressure under the
/// international standard atmosphere and the speed is normalised to
/// m/s. Samples on `Surface.winds_aloft` are stored sorted by
/// ascending `altitude_m` so the wind-field synthesis can binary-bracket
/// without sorting per frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindAloftSample {
    /// Pressure level (hPa). Kept for diagnostics / UI labelling;
    /// not consumed by the synthesis itself which reads
    /// `altitude_m` instead.
    pub pressure_hpa: u16,
    /// Standard-atmosphere height of `pressure_hpa` in metres AGL
    /// (approximated against the surface elevation).
    pub altitude_m: f32,
    /// Wind speed in metres / second.
    pub speed_mps: f32,
    /// Meteorological wind direction in degrees (direction wind
    /// comes from). Same convention as `Surface.wind_dir_deg`.
    pub dir_deg: f32,
}

/// Phase 13.4 — discrete surface materials. Each variant maps to a
/// distinct palette/roughness/F0 in the ground PBR shader. Stored
/// in `SurfaceParams.material` as a small integer (encoded as `f32`
/// to share the existing uniform layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SurfaceMaterial {
    /// Generic green grass — the v1 default.
    #[default]
    Grass,
    /// Dry bare soil / plough.
    BareSoil,
    /// Asphalt / road surface — darker, rougher than grass.
    Tarmac,
    /// Light beach sand — lighter, slightly more specular.
    Sand,
    /// Thin blue stripe along the visible horizon — placeholder for
    /// shoreline scenes pending the Phase 13.5 water plane.
    WaterEdge,
}

/// Per-cloud-type `density_scale` default used when a scene leaves
/// the `density_scale` field on a layer unset.
///
/// **See `docs/cloud_calibration.md`** for the full per-cloud-type
/// reference table (densities, HG triples, optical-depth ranges,
/// altitudes, droplet radii) with verified source citations.
///
/// Values calibrated against:
///
/// - Wallace & Hobbs *Atmospheric Science* (2006), Houze *Cloud
///   Dynamics* (2014) for optical-depth ranges per WMO type;
/// - Schneider 2015 / Hillaire 2016 baseline sigma_s ≈ 0.04 /m;
/// - The open-source consensus across Schneider HZD, Frostbite,
///   TrueSky, SilverLining, and Bevy / Unity / Godot community
///   reimplementations (PedalSky research synthesis, 2026-05-12).
///
/// Optical-depth targets (vertical, visible): Cumulus 5–30,
/// Stratus 8–40, Stratocumulus 8–25, Altocumulus 3–10,
/// Altostratus 5–30, Cirrus 0.03–1.0, Cirrostratus 0.5–3,
/// Cumulonimbus 30–200+. The values below place each type near
/// the middle of its band when combined with default `coverage`
/// and layer thickness.
pub fn default_density_scale(cloud_type: CloudType) -> f32 {
    match cloud_type {
        CloudType::Cumulus | CloudType::Stratus | CloudType::Stratocumulus => 1.0,
        CloudType::Altocumulus => 0.85,
        CloudType::Altostratus => 0.7,
        CloudType::Cirrus => 0.55,
        CloudType::Cirrostratus => 0.4,
        CloudType::Cumulonimbus => 1.4,
    }
}

/// Default per-cloud-type droplet effective diameter (µm), used
/// when a scene leaves the `droplet_diameter_um` field on a layer
/// unset. Drives the Jendersie–d'Eon 2023 Approximate Mie phase
/// function in the cloud march.
///
/// The values match published cloud-microphysics literature on
/// effective droplet / crystal size per WMO cloud type:
///
/// - **Water clouds** (Cu / St / Sc / Ac, plus Cb in the
///   convective core): r_e ≈ 5–15 µm → d ≈ 10–30 µm. Marine
///   stratocumulus ~10 µm; continental polluted Sc ~6 µm (Twomey
///   effect, Wood 2012 MWR review). Pruppacher & Klett 1997 is the
///   canonical reference.
/// - **Mixed-phase altostratus**: r_e ≈ 15–25 µm → d ≈ 30–50 µm.
/// - **Ice clouds** (Ci / Cs): r_e ≈ 20–60 µm → d ≈ 40–120 µm.
///   Generalised effective diameter D_ge in the paper's notation;
///   verified against multiple cirrus retrieval papers
///   (acp.copernicus.org/22:15179/2022, JAS 68(2)).
/// - **Cumulonimbus**: water-droplet (~20 µm) in the convective
///   core; the shader interpolates toward ice (~50 µm) inside the
///   anvil region using the same height-based blend that drove the
///   previous HG transition. See
///   `dual_lobe_hg_with_g_scale` in `shaders/clouds/cloud_march.wgsl`.
///
/// The Approximate Mie fit (Jendersie & d'Eon 2023, Eqs. 4–7) is
/// valid for **5 ≤ d ≤ 50 µm**. Values below clamp to that range
/// inside the shader; values above (ice) extrapolate the fit, which
/// the paper notes is reasonable in practice but increases
/// approximation error. See `docs/cloud_calibration.md` for the
/// full sources.
pub fn default_droplet_diameter_um(cloud_type: CloudType) -> f32 {
    match cloud_type {
        // Cumulus: r_e ~10 µm typical, d ≈ 20 µm.
        CloudType::Cumulus => 20.0,
        // Stratus, Stratocumulus: similar to cumulus, slightly
        // smaller in marine boundary-layer clouds.
        CloudType::Stratus | CloudType::Stratocumulus => 16.0,
        // Altocumulus: mid-level, often slightly smaller droplets.
        CloudType::Altocumulus => 14.0,
        // Altostratus (mixed-phase): water at base, ice in upper
        // deck. The single scalar sits between the two regimes.
        CloudType::Altostratus => 30.0,
        // Cirrus, Cirrostratus: ice crystals. Generalised effective
        // diameter ≈ 50 µm; this is at the upper end of the paper's
        // 5–50 µm fit range so the shader clamps.
        CloudType::Cirrus | CloudType::Cirrostratus => 50.0,
        // Cumulonimbus: water-droplet at the base (~10 µm r_e).
        // The shader blends toward ice in the anvil region.
        CloudType::Cumulonimbus => 20.0,
    }
}

impl SurfaceMaterial {
    /// Numeric encoding stored in `SurfaceParams.material` (mirrors
    /// the WGSL constants in `pbr.wgsl`). Stable across versions.
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Grass => 0,
            Self::BareSoil => 1,
            Self::Tarmac => 2,
            Self::Sand => 3,
            Self::WaterEdge => 4,
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
    /// Per-layer optical density multiplier. `None` (the default)
    /// selects the per-cloud-type default in `ps_synthesis`
    /// (Cumulus/Stratus/Stratocumulus = 1.0, Altocumulus = 0.85,
    /// Altostratus = 0.7, Cirrus = 0.55, Cirrostratus = 0.4,
    /// Cumulonimbus = 1.4). The defaults track the published
    /// per-type optical-depth ranges (Cumulus 5–30, Cirrus 0.03–1.0,
    /// Cumulonimbus core 30–200+) so scenes that omit the field
    /// land in the right ballpark for their cloud type. `Some(v)`
    /// overrides.
    #[serde(default)]
    pub density_scale: Option<f32>,
    /// Per-layer bias on the base-shape low-frequency Worley FBM
    /// weighting. Subtle: redistributes the bulk density envelope.
    /// Useful range \[-1, 1\] but most effect lives near 0.
    pub shape_octave_bias: f32,
    /// Per-layer bias on the high-frequency boundary erosion
    /// strength. The Schneider remap is non-monotonic in apparent
    /// density: positive bias culls more low-density samples *and*
    /// concentrates surviving samples, so the cloud reads as either
    /// thinner or denser depending on the layer's coverage and
    /// density_scale. Recommended useful range is small (~±0.1);
    /// larger magnitudes trigger sky-wide saturation or near-total
    /// cloud loss. Plan §3.2.6 reserves per-type defaults here, but
    /// those need a calibrated tuning bench to land safely; for now
    /// the field is a pure per-scene knob.
    pub detail_octave_bias: f32,
    /// Phase 13 follow-up — anvil bias for the Cumulonimbus NDF.
    /// `None` (the default) selects the per-type default
    /// (Cumulonimbus = 1.0, others = 0.0); `Some(v)` overrides. Range
    /// 0..2 — 0 suppresses the anvil entirely, 2 doubles its strength
    /// relative to v1. Non-cumulonimbus layers ignore this field.
    #[serde(default)]
    pub anvil_bias: Option<f32>,
    /// Droplet effective diameter in micrometres. `None` (the
    /// default) selects the per-cloud-type default from
    /// `default_droplet_diameter_um` (water clouds ≈ 14–20 µm, ice
    /// clouds ≈ 50 µm). `Some(v)` overrides. Drives the Jendersie–
    /// d'Eon 2023 Approximate Mie phase function in the cloud march.
    /// Fitted range 5–50 µm; values outside that range extrapolate
    /// the paper's fit.
    #[serde(default)]
    pub droplet_diameter_um: Option<f32>,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            cloud_type: CloudType::Cumulus,
            base_m: 1500.0,
            top_m: 2300.0,
            coverage: 0.5,
            density_scale: None,
            shape_octave_bias: 0.0,
            detail_octave_bias: 0.0,
            anvil_bias: None,
            droplet_diameter_um: None,
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

/// Phase 12.5 — `[aurora]` block. Solar-activity inputs that drive
/// the aurora subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Aurora {
    /// Planetary K-index (0..9). Higher = stronger geomagnetic
    /// activity → brighter, broader curtains. The default scene
    /// `Aurora::default()` has `kp_index = 0` so the subsystem
    /// renders nothing visible unless a scene overrides it.
    pub kp_index: f32,
    /// Optional override for the curtain emission scalar in `[0, 1]`.
    /// `None` (encoded as a negative sentinel) means "derive from
    /// kp_index". Use this to dial in a specific intensity that
    /// doesn't follow the kp curve — useful for screenshots.
    pub intensity_override: f32,
    /// Predominant emission colour. Real auroras are line-spectrum
    /// emission so the engine internally mixes three line colours;
    /// this enum biases the mix.
    pub predominant_colour: AuroraColour,
}

/// Predominant aurora emission colour. Matches the historical TOML
/// strings (`"green"`, `"red"`, `"purple"`, `"mixed"`) so existing
/// scene files round-trip without edits.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuroraColour {
    /// Canonical 557.7 nm O₂ green-dominant aurora.
    #[default]
    Green,
    /// Red-dominant (high-altitude N₂ band) aurora.
    Red,
    /// Purple-dominant aurora.
    Purple,
    /// Balanced mix of green/red/purple emission lines.
    Mixed,
}

impl AuroraColour {
    /// Normalised RGB bias used inside the aurora shader to weight the
    /// three emission lines.
    pub fn bias(self) -> [f32; 3] {
        match self {
            Self::Red => [0.30, 0.20, 0.50],
            Self::Purple => [0.20, 0.30, 0.55],
            Self::Mixed => [0.55, 0.20, 0.25],
            Self::Green => [0.10, 0.85, 0.05],
        }
    }
}

impl Default for Aurora {
    fn default() -> Self {
        Self {
            kp_index: 0.0,
            intensity_override: -1.0, // sentinel "no override"
            predominant_colour: AuroraColour::Green,
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
