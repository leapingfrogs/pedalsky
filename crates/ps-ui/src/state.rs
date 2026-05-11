//! Shared UI state — the bridge between panel logic (which writes
//! edits) and the host frame loop (which reads pending requests and
//! drives `App::reconfigure`, screenshots, preset I/O).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use ps_core::{AtmosphereParams, Config, Scene};

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
#[derive(Default, Clone, Copy, Debug)]
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
