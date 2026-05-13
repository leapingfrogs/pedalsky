//! Shared UI state — the bridge between panel logic (which writes
//! edits) and the host frame loop (which reads pending requests and
//! drives `App::reconfigure`, screenshots, preset I/O).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use ps_core::{AtmosphereParams, Config, Scene, WindAloftSample};

/// Side-channel state populated by panel logic and drained by the host.
#[derive(Default)]
pub struct UiPending {
    /// Set when any slider edit changed `live_config`. Host calls
    /// `App::reconfigure(&live_config, &gpu)` and clears the flag.
    pub config_dirty: bool,

    /// Pending world-clock edit: jump to this UTC.
    pub set_world_utc: Option<DateTime<Utc>>,
    /// Pending time-scale change (× wall-clock seconds → simulated).
    pub set_time_scale: Option<f64>,
    /// Pending pause state.
    pub set_paused: Option<bool>,
    /// Pending observer (latitude, longitude) override in degrees.
    pub set_lat_lon: Option<(f64, f64)>,

    /// User clicked Screenshot tonemapped (PNG). Host writes the
    /// post-tonemap swapchain to `paths.screenshot_dir`.
    pub screenshot_png: bool,
    /// User clicked Screenshot HDR (EXR). Host writes the linear HDR
    /// target to `paths.screenshot_dir`.
    pub screenshot_exr: bool,

    /// Load scene from file. Host re-parses and re-synthesises.
    pub load_scene: Option<PathBuf>,
    /// Save current state to file. Host writes both pedalsky.toml and
    /// the scene TOML.
    pub save_scene: Option<PathBuf>,

    /// Phase 10.A1: live atmosphere parameter override. Host applies
    /// directly to WeatherState.atmosphere and triggers the
    /// atmosphere subsystem's static-LUT re-bake.
    pub live_atmosphere: Option<AtmosphereParams>,

    /// Phase 10.A2/A3: live scene override (wetness, snow, cloud
    /// layers). Host replaces the in-memory Scene and re-synthesises
    /// WeatherState.
    pub live_scene: Option<Scene>,

    /// Plan §0.4 — pending camera fov/near/speed edit from the World
    /// panel. Host applies to its FlyCamera on the next frame.
    pub live_camera: Option<CameraSettings>,

    /// User clicked "Fetch real weather…". Host issues an
    /// `ps_weather_feed::fetch_scene` call on a background thread
    /// and, on success, replaces the in-memory scene through the
    /// same path `live_scene` uses. While the fetch is in-flight
    /// the host stamps `UiState.weather_fetch.in_flight = true` so
    /// the button can render its "Fetching…" state.
    pub fetch_real_weather: Option<WeatherFetchRequest>,

    /// Phase 16.B — user clicked "Search" in the World panel's
    /// location box. Host issues an
    /// `ps_weather_feed::geocoding::search` call on a background
    /// thread and pushes the result list back via
    /// `UiState.geocode.results`. While the search is in-flight,
    /// `UiState.geocode.in_flight = true`.
    pub geocode_query: Option<GeocodeRequest>,
}

/// One-shot weather fetch request. Sent from the UI to the host
/// when the user clicks the "Fetch real weather…" button.
#[derive(Debug, Clone)]
pub struct WeatherFetchRequest {
    /// Observer latitude (degrees north).
    pub lat: f64,
    /// Observer longitude (degrees east).
    pub lon: f64,
    /// Target time (UTC). Defaults to the world clock's current
    /// time at the moment the button is clicked.
    pub time: DateTime<Utc>,
    /// If true, the host fetches the nearest METAR station and
    /// applies surface + present-weather enrichment.
    pub enrich_with_metar: bool,
    /// Phase 15.A — if true, the host fetches the NOAA SWPC
    /// planetary Kp index and writes it onto `scene.aurora.kp_index`.
    /// Best-effort; the aurora subsystem stays dark if the fetch
    /// fails.
    pub fetch_kp_index: bool,
    /// Phase 15.B — if true, the host also fetches the SWPC OVATION
    /// nowcast and samples it at `(lon, lat)` for a per-location
    /// aurora-intensity override. Best-effort.
    pub fetch_ovation: bool,
}

/// Status of a weather fetch — host writes; UI reads.
#[derive(Default, Debug, Clone)]
pub struct WeatherFetchStatus {
    /// True while a fetch is in flight (button shows "Fetching…",
    /// disabled).
    pub in_flight: bool,
    /// Most recent error string, if any. Cleared on next successful
    /// fetch.
    pub last_error: Option<String>,
    /// Most recent successful fetch's source description (e.g.
    /// "Open-Meteo + METAR EGPF, 0.4° away"). Cleared on error.
    pub last_summary: Option<String>,
}

/// Phase 16.B — one-shot geocoding request, sent from the UI to the
/// host when the user clicks "Search" in the World panel.
#[derive(Debug, Clone)]
pub struct GeocodeRequest {
    /// Free-form place name (e.g. `"Dunblane"`, `"Tromsø"`).
    pub query: String,
    /// Maximum number of result rows to surface in the dropdown.
    /// Open-Meteo accepts up to 100; 10 is plenty for picking from
    /// a UI list.
    pub count: usize,
}

/// Lightweight UI mirror of `ps_weather_feed::GeocodeResult` — the
/// host's worker thread converts to this so the UI crate doesn't
/// re-export the feed types verbatim.
#[derive(Default, Debug, Clone)]
pub struct GeocodeMatch {
    /// GeoNames id — useful as a stable dedup / list key.
    pub id: u64,
    /// Place name.
    pub name: String,
    /// First-level admin division (e.g. `"Scotland"`, `"Texas"`).
    pub admin1: String,
    /// Full country name.
    pub country: String,
    /// Latitude (degrees north).
    pub latitude: f64,
    /// Longitude (degrees east).
    pub longitude: f64,
    /// Surface elevation in metres AMSL.
    pub elevation: f32,
    /// IANA timezone, e.g. `"Europe/London"`.
    pub timezone: String,
    /// Resident population. `0` if upstream omits it.
    pub population: u64,
}

/// Status of a geocoding search — host writes; UI reads.
#[derive(Default, Debug, Clone)]
pub struct GeocodeStatus {
    /// True while the search is in flight (button disabled,
    /// dropdown shows "Searching…").
    pub in_flight: bool,
    /// Last query the user issued — kept so the dropdown's header
    /// can echo it.
    pub last_query: String,
    /// Most recent result list. Empty for no-match queries.
    pub results: Vec<GeocodeMatch>,
    /// Most recent error string, cleared on next successful search.
    pub last_error: Option<String>,
    /// Persistent text-input buffer so the field doesn't reset
    /// every frame.
    pub input_buffer: String,
}

/// Snapshot of the host's camera configuration that the World panel
/// edits. Plan §0.4 — fov, near, speed sliders.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct CameraSettings {
    /// Vertical field of view in radians.
    pub fov_y_rad: f32,
    /// Near-plane distance in metres.
    pub near_m: f32,
    /// Movement speed in metres per second.
    pub speed_mps: f32,
}

/// Read-only frame stats the host pushes into the UI for the Debug panel.
#[derive(Default, Clone, Debug)]
pub struct UiFrameStats {
    /// Wall-clock ms last frame.
    pub frame_ms: f32,
    /// Frames per second over the last accumulator window.
    pub fps: f32,
    /// Per-pass GPU timestamps in milliseconds: `(pass_name, ms)`.
    /// Empty when timestamp queries are unsupported or unavailable
    /// for the most recent frame.
    pub gpu_passes: Vec<(String, f32)>,
}

/// World read-out the host pushes each frame for the World panel.
///
/// Phase 14.E — gained a `Vec<WindAloftSample>` field, so this is no
/// longer `Copy`. The compass-rose panel reads it by reference; no
/// other consumer needs the previous `Copy` semantics.
#[derive(Default, Clone, Debug)]
pub struct UiWorldReadout {
    /// Sun altitude above the local horizon, degrees.
    pub sun_alt_deg: f64,
    /// Sun azimuth (from north, clockwise), degrees.
    pub sun_az_deg: f64,
    /// Moon altitude above the local horizon, degrees.
    pub moon_alt_deg: f64,
    /// Moon azimuth (from north, clockwise), degrees.
    pub moon_az_deg: f64,
    /// Julian day for the current simulated UTC.
    pub julian_day: f64,
    /// Phase 13.6 — camera heading (azimuth, clockwise from north),
    /// degrees. Used by the compass rose overlay.
    pub camera_yaw_deg: f64,
    /// Phase 13.6 — surface wind direction (meteorological — the
    /// direction the wind is *coming from*), clockwise from north,
    /// degrees. Mirror of `weather.surface.wind_dir_deg` so the
    /// compass rose can draw a barb without re-reading WeatherState.
    pub wind_dir_deg: f32,
    /// Phase 13.6 — wind speed in m/s.
    pub wind_speed_mps: f32,
    /// Phase 14.E — winds aloft mirrored from `Scene.surface.winds_aloft`
    /// so the compass rose can overlay extra barbs at the standard
    /// pressure levels (850 / 700 / 500 / 300 hPa). Empty when no
    /// upper-air samples are present (synthetic scenes); the compass
    /// rose falls back to the surface-only barb.
    pub winds_aloft: Vec<WindAloftSample>,
}

/// The full shared state cell.
pub struct UiState {
    /// Live mutable copy of the engine config. Panel logic edits this;
    /// the host reads it on `config_dirty`.
    pub live_config: Config,
    /// Pending one-shot requests + flags.
    pub pending: UiPending,
    /// Latest frame stats (host writes per frame).
    pub frame_stats: UiFrameStats,
    /// Latest world read-out (host writes per frame).
    pub world_readout: UiWorldReadout,
    /// Cached debug-panel selections (LUT viewer modes etc.).
    pub debug: UiDebugSelection,
    /// Mirror of the host's live `Scene` (host pushes each frame).
    /// Atmosphere / wet-surface / cloud-layer panels read current
    /// values from here; their slider edits clone, mutate, and write
    /// back via `pending.live_scene`.
    pub latest_scene: Option<Scene>,
    /// Mirror of the host's live `WeatherState.atmosphere` (host pushes
    /// each frame). The atmosphere coefficient panel reads current
    /// values from here.
    pub latest_atmosphere: Option<AtmosphereParams>,
    /// Mirror of the host's live `FlyCamera` settings (fov, near,
    /// speed) — host pushes each frame; the World panel reads from
    /// here and writes edits to `pending.live_camera`.
    pub latest_camera: Option<CameraSettings>,

    /// Real-weather fetch status — host writes after each fetch
    /// attempt; UI reads to render the "Fetch real weather…"
    /// button's state and any error message.
    pub weather_fetch: WeatherFetchStatus,

    /// Phase 16.B — geocoding search status. Host writes after each
    /// search; UI reads to render the result dropdown and any
    /// error.
    pub geocode: GeocodeStatus,
}

/// Debug-panel toggles that don't belong in `Config`.
#[derive(Default, Clone, Copy, Debug)]
pub struct UiDebugSelection {
    /// Which LUT viewer to show full-screen, 0 = none.
    pub lut_viewer_mode: u32,
    /// AP LUT depth slice 0..1.
    pub ap_depth_slice: f32,
    /// Pixel coordinate for the probe-point read-out.
    pub probe_pixel: (u32, u32),
    /// Latest probe transmittance (RGB) from the previous frame, written
    /// by the host for display in the panel.
    pub probe_transmittance: [f32; 3],
    /// Phase 13.10 — per-component optical depth contributions at the
    /// probe pixel. Each entry is RGB. Total OD = `-ln(transmittance)`
    /// is computed in the panel from `probe_transmittance`.
    pub probe_od_rayleigh: [f32; 3],
    /// Phase 13.10 — Mie OD contribution at the probe pixel (RGB).
    pub probe_od_mie: [f32; 3],
    /// Phase 13.10 — ozone OD contribution at the probe pixel (RGB).
    pub probe_od_ozone: [f32; 3],
}

impl UiState {
    /// Construct from an initial config.
    pub fn new(config: Config) -> Self {
        Self {
            live_config: config,
            pending: UiPending::default(),
            frame_stats: UiFrameStats::default(),
            world_readout: UiWorldReadout::default(),
            debug: UiDebugSelection::default(),
            latest_scene: None,
            latest_atmosphere: None,
            latest_camera: None,
            weather_fetch: WeatherFetchStatus::default(),
            geocode: GeocodeStatus::default(),
        }
    }
}

/// Shared handle threaded between the UI subsystem and the host. Cheap
/// to clone (internal Arc).
#[derive(Clone)]
pub struct UiHandle {
    inner: Arc<Mutex<UiState>>,
}

impl UiHandle {
    /// Construct.
    pub fn new(initial_config: Config) -> Self {
        Self {
            inner: Arc::new(Mutex::new(UiState::new(initial_config))),
        }
    }
    /// Lock the shared state. Panic on poisoning.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, UiState> {
        self.inner.lock().expect("UiState lock poisoned")
    }
}
