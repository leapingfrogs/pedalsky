//! PedalSky main binary.
//!
//! Phase 0/1 scope: load `pedalsky.toml`, load the scene, validate both,
//! open a winit window, build a `ps_core::App` from registered subsystem
//! factories (backdrop, ground, tint), and drive a per-frame:
//!
//! ```text
//! frame-clear  (always: depth = 0.0; colour = [render].clear_color when
//!               backdrop is disabled, else loaded)
//! App::frame   → backdrop (SkyBackdrop) → ground (Opaque) → tint (PostProcess)
//! tone-map     → swapchain
//! ```
//!
//! The `HotReload` watcher emits debounced events; on `ConfigChanged` the
//! app re-loads + validates the config and calls `App::reconfigure`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{Vec3, Vec4};
use ps_app::config_initial_utc;
use ps_app::main_helpers::encode_frame_clear;
use ps_atmosphere::{
    lut_overlay::{LutOverlay, LutOverlayUniforms},
    lut_viewer::{LutViewer, LutViewerUniforms},
    AtmosphereFactory,
};
use ps_backdrop::BackdropFactory;
use ps_aurora::AuroraFactory;
use ps_bloom::BloomFactory;
use ps_godrays::GodraysFactory;
use ps_lightning::{LightningFactory, LightningPublish};
use ps_clouds::CloudsFactory;
use ps_core::{
    App, AppBuilder, Config, FrameUniforms, HdrFramebufferImpl, HotReload, PrepareContext,
    RenderContext, Scene, WatchEvent, DEFAULT_DEBOUNCE,
};
use ps_ground::GroundFactory;
use ps_postprocess::{Tonemap, TonemapMode};
use ps_precip::PrecipFactory;
use ps_tint::TintFactory;
use ps_water::WaterFactory;
use ps_windsock::WindsockFactory;
use tracing::{debug, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();

    let workspace_root = workspace_root()?;
    info!(workspace_root = %workspace_root.display(), "starting ps-app");

    // CLI: `--headless-dump <out_dir>` runs the Phase 3 §3.3 dump and exits.
    let argv: Vec<String> = std::env::args().collect();
    if let Some(out_dir) = ps_app::headless_dump::parse_args(&argv) {
        return ps_app::headless_dump::run(&workspace_root, &out_dir);
    }

    // CLI: `render --scene <toml> --time <ISO8601> --output <path>`
    // Phase 11.2 headless rendering subcommand.
    if let Some(args) = ps_app::render_cli::parse_args(&argv) {
        return ps_app::render_cli::run(&workspace_root, args);
    }

    // CLI override: `--lut-overlay` flips `config.debug.atmosphere_lut_overlay`
    // on independently of the config file.
    let cli_lut_overlay = argv.iter().any(|a| a == "--lut-overlay");

    // CLI override: `--gpu-trace <dir>` enables wgpu API call tracing
    // to the named directory (plan §GPU debugging).
    let gpu_trace_dir = argv
        .iter()
        .position(|a| a == "--gpu-trace")
        .and_then(|i| argv.get(i + 1))
        .map(PathBuf::from);

    // CLI override: `--seed <u64>` sets the determinism seed
    // (plan §Cross-Cutting/Determinism). Falls back to
    // `[debug] seed` from the config when absent.
    let cli_seed = argv
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| argv.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok());

    // CLI override: `--full-screen` (or `-F`) launches in exclusive fullscreen
    // on the primary monitor using its highest-resolution video mode.
    let cli_fullscreen = argv.iter().any(|a| a == "--full-screen" || a == "-F");

    let config_path = workspace_root.join("pedalsky.toml");
    let mut config =
        Config::load(&config_path).with_context(|| format!("loading {}", config_path.display()))?;
    if cli_lut_overlay {
        config.debug.atmosphere_lut_overlay = true;
    }
    if let Some(seed) = cli_seed {
        config.debug.seed = seed;
    }
    config
        .validate_with_base(config_path.parent())
        .with_context(|| format!("validating {}", config_path.display()))?;

    let scene_path = if config.paths.weather.is_absolute() {
        config.paths.weather.clone()
    } else {
        workspace_root.join(&config.paths.weather)
    };
    let scene =
        Scene::load(&scene_path).with_context(|| format!("loading {}", scene_path.display()))?;
    scene
        .validate()
        .with_context(|| format!("validating {}", scene_path.display()))?;

    info!(
        cloud_layers = scene.clouds.layers.len(),
        precip = ?scene.precipitation.kind,
        "configuration loaded; opening window"
    );

    let event_loop = EventLoop::new().context("create EventLoop")?;
    let mut shell = AppShell::new(
        config,
        scene,
        config_path,
        scene_path,
        gpu_trace_dir,
        cli_fullscreen,
    );
    event_loop.run_app(&mut shell).context("event loop")?;
    Ok(())
}

/// Resolve the workspace `shaders/` directory from a known-inside-
/// the-workspace path (we use the loaded config path).
fn workspace_root_for_shaders(config_path: &std::path::Path) -> PathBuf {
    config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
        .join("shaders")
}

/// Pick a `Fullscreen::Exclusive` video mode for `--full-screen`.
///
/// Strategy: primary monitor's highest pixel-count mode, breaking ties on
/// refresh rate. Falls back to `Fullscreen::Borderless(None)` if the
/// monitor reports no enumerable video modes (e.g. Wayland), and to
/// `None` if there is no primary monitor — winit will then open windowed.
fn pick_exclusive_fullscreen(event_loop: &ActiveEventLoop) -> Option<Fullscreen> {
    let monitor = event_loop.primary_monitor().or_else(|| event_loop.available_monitors().next())?;
    let mode = monitor
        .video_modes()
        .max_by_key(|m| {
            let sz = m.size();
            (sz.width as u64 * sz.height as u64, m.refresh_rate_millihertz())
        });
    match mode {
        Some(m) => {
            info!(
                target: "ps_app",
                size = ?m.size(),
                refresh_mhz = m.refresh_rate_millihertz(),
                "--full-screen: exclusive video mode selected"
            );
            Some(Fullscreen::Exclusive(m))
        }
        None => {
            warn!(target: "ps_app", "--full-screen: no enumerable video modes; falling back to borderless");
            Some(Fullscreen::Borderless(Some(monitor)))
        }
    }
}

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

struct AppShell {
    config: Config,
    scene: Scene,
    config_path: PathBuf,
    scene_path: PathBuf,
    /// Plan §GPU debugging — wgpu API trace output directory if `--gpu-trace`
    /// was passed on the command line.
    gpu_trace_dir: Option<PathBuf>,
    /// `--full-screen` / `-F`: open the window in exclusive fullscreen.
    fullscreen: bool,
    state: Option<RunState>,
}

impl AppShell {
    fn new(
        config: Config,
        scene: Scene,
        config_path: PathBuf,
        scene_path: PathBuf,
        gpu_trace_dir: Option<PathBuf>,
        fullscreen: bool,
    ) -> Self {
        Self {
            config,
            scene,
            config_path,
            scene_path,
            gpu_trace_dir,
            fullscreen,
            state: None,
        }
    }
}

struct RunState {
    window: Arc<Window>,
    windowed_gpu: ps_core::gpu::WindowedGpu<'static>,
    hdr: HdrFramebufferImpl,
    /// Phase 4 §4.2 — bind groups 0 (FrameUniforms) and 1 (WorldUniforms).
    /// Updated each frame from `frame_uniforms` and
    /// `weather.atmosphere`.
    bindings: ps_core::FrameWorldBindings,

    /// Phase 9.1 tonemap (and Phase 9.2 auto-exposure) shared with the
    /// in-graph `TonemapSubsystem`. Stored as `Arc`s so the host can
    /// rebuild bindings on resize while the subsystem holds independent
    /// references.
    tonemap: std::sync::Arc<Tonemap>,
    auto_exposure: std::sync::Arc<ps_postprocess::AutoExposure>,
    /// Side-channel handle to push per-frame state into the
    /// in-graph TonemapSubsystem and to drain its auto-exposure
    /// staging buffer after `queue.submit`.
    tonemap_handle: ps_postprocess::TonemapHandle,
    /// Phase 10: shared UI state cell. The host reads pending edits
    /// after each frame and either reconfigures, drains screenshot
    /// requests, or applies world-clock changes.
    ui_handle: ps_ui::UiHandle,
    /// Reference back to the constructed `UiSubsystem` (also held by
    /// `app` via the factory dispatch path). Used to feed winit events
    /// into egui and to call `build_ui_frame` each frame.
    ui_bridge: std::sync::Arc<ps_ui::UiBridge>,
    /// Phase 10.A4 probe-pixel transmittance readback.
    probe: ps_app::probe::ProbeReadback,
    /// `App` constructed from factories. Owns backdrop/ground/tint.
    app: App,
    /// Phase 5: LUTs handle published by `AtmosphereFactory`. `None`
    /// when `[render.subsystems].atmosphere = false`.
    atmosphere_luts: Option<std::sync::Arc<ps_core::AtmosphereLuts>>,
    /// Phase 12.3: snapshot publish handle from `LightningFactory`.
    /// `None` when lightning was disabled at build. Read after
    /// `App::frame` to splice cloud illumination into the next
    /// `FrameUniforms` upload.
    lightning_publish: Option<LightningPublish>,
    /// Phase 5 debug overlay drawing the four LUTs onto the swapchain.
    /// `Some` when `[debug].atmosphere_lut_overlay = true` (or
    /// `--lut-overlay`) AND atmosphere is enabled.
    lut_overlay: Option<LutOverlay>,
    /// Phase 10 fullscreen LUT viewer (gated by ui_state.debug.lut_viewer_mode).
    lut_viewer: Option<LutViewer>,

    camera: ps_core::camera::FlyCamera,
    keys: KeyState,
    cursor_grabbed: bool,
    mouse_delta: (f64, f64),
    ev100: f32,
    tonemap_mode: TonemapMode,
    /// Simulated world clock + observer + sun + moon (Phase 2).
    world: ps_core::WorldState,
    /// Synthesised weather state (Phase 3) — atmosphere, cloud layers,
    /// surface params, weather map, wind field, top-down density mask.
    weather: ps_core::WeatherState,
    start: Instant,
    last_frame: Instant,
    /// World-space camera position at the start of the previous frame.
    /// Used to derive `camera_velocity_world` for Phase 8 far-rain
    /// scrolling. `None` until the first frame finishes (so the very
    /// first frame reports zero velocity rather than a garbage delta).
    last_camera_position: Option<glam::Vec3>,
    /// Previous frame's `view_proj`. `None` until the first frame's
    /// matrices are set — on that first frame the TAA pass sees
    /// `prev_view_proj == view_proj` so its reprojection is identity
    /// (and the history is invalid anyway, so the shader uses the
    /// current sample directly).
    prev_view_proj: Option<glam::Mat4>,
    frame_index: u32,
    fps_accum_dt: f32,
    fps_accum_frames: u32,
    title_prefix: String,

    hot_reload: HotReload,
    /// Plan §Cross-Cutting/Shader hot-reload. `None` when the
    /// `[debug] shader_hot_reload` flag is off or the watcher failed
    /// to start.
    shader_hot_reload: Option<ps_core::ShaderHotReload>,

    /// Phase 14 — receiver for in-flight real-weather fetch
    /// requests. `Some(rx)` when a background thread is currently
    /// fetching; the host polls in `drain_ui_pending`. The
    /// `JoinHandle` is held only to keep the thread alive; we
    /// don't actually join — drop on RunState teardown is fine.
    weather_fetch_rx: Option<crossbeam_channel::Receiver<WeatherFetchResult>>,

    /// Phase 16.B — receiver for in-flight geocoding searches.
    /// Same gating as the weather fetch: only one in flight at a
    /// time; the UI's "Searching…" button disables clicks while
    /// `Some`.
    geocode_rx: Option<crossbeam_channel::Receiver<GeocodeWorkerResult>>,
}

/// Result delivered from the weather-fetch worker thread.
struct WeatherFetchResult {
    /// The fetched scene, or the error string.
    outcome: Result<ps_core::Scene, String>,
    /// A short summary string for the UI ("Open-Meteo, …" or the
    /// METAR station ICAO when enrichment fired). Empty on error.
    summary: String,
}

/// Result delivered from the geocoding worker thread.
struct GeocodeWorkerResult {
    /// The original query string — echoed back so the UI can label
    /// the result list ("Matches for \"Dunblane\":").
    query: String,
    /// The result rows, or the error string.
    outcome: Result<Vec<ps_weather_feed::GeocodeResult>, String>,
}

#[derive(Default)]
struct KeyState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    roll_left: bool,
    roll_right: bool,
    sprint: bool,
}

impl ApplicationHandler for AppShell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match RunState::new(
            event_loop,
            &self.config,
            &self.scene,
            &self.config_path,
            &self.scene_path,
            self.gpu_trace_dir.clone(),
            self.fullscreen,
        ) {
            Ok(s) => self.state = Some(s),
            Err(e) => {
                tracing::error!(error = %e, "failed to create RunState; exiting");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        state.handle_window_event(event_loop, event, &mut self.config, &mut self.scene);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += delta.0;
                state.mouse_delta.1 += delta.1;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        state.poll_hot_reload(&mut self.config, &mut self.scene);
        state.window.request_redraw();
    }
}

impl RunState {
    fn new(
        event_loop: &ActiveEventLoop,
        config: &Config,
        scene: &Scene,
        config_path: &std::path::Path,
        scene_path: &std::path::Path,
        gpu_trace_dir: Option<PathBuf>,
        fullscreen: bool,
    ) -> Result<Self> {
        let initial_size = (config.window.width, config.window.height);
        let fullscreen_mode = if fullscreen {
            pick_exclusive_fullscreen(event_loop)
        } else {
            None
        };
        let mut attrs = Window::default_attributes()
            .with_title(&config.window.title)
            .with_inner_size(PhysicalSize::new(initial_size.0, initial_size.1));
        if fullscreen_mode.is_some() {
            attrs = attrs.with_fullscreen(fullscreen_mode.clone());
        }
        let window = Arc::new(event_loop.create_window(attrs).context("create_window")?);

        // Use the configured size for the initial swapchain rather than
        // window.inner_size(): on Windows with HiDPI scaling, the latter
        // can return a larger physical extent than the OS will let us
        // present at, triggering Vulkan validation (currentExtent vs
        // imageExtent mismatch). winit fires a Resized event during the
        // first frame loop with the OS-actual physical size; the resize
        // handler then reconfigures the swapchain and HDR target.
        let physical = window.inner_size();
        let size = (
            physical.width.min(initial_size.0).max(1),
            physical.height.min(initial_size.1).max(1),
        );
        let windowed = ps_core::gpu::init_windowed(
            window.clone(),
            size,
            config.window.vsync,
            config.debug.gpu_validation,
            gpu_trace_dir,
        )
        .context("init_windowed")?;

        let hdr = HdrFramebufferImpl::new(&windowed.gpu, size);
        let bindings = ps_core::FrameWorldBindings::new(&windowed.gpu.device);
        let tonemap = std::sync::Arc::new(Tonemap::new(
            &windowed.gpu.device,
            &hdr,
            windowed.surface_config.format,
        ));
        let auto_exposure = std::sync::Arc::new(ps_postprocess::AutoExposure::new(
            &windowed.gpu.device,
            &hdr,
        ));

        let probe = ps_app::probe::ProbeReadback::new(&windowed.gpu);
        let ui_handle = ps_ui::UiHandle::new(config.clone());
        let (app, atmosphere_luts_cell, lightning_cell, tonemap_handle, ui_bridge) = build_app(
            config,
            &windowed.gpu,
            tonemap.clone(),
            auto_exposure.clone(),
            ui_handle.clone(),
            window.clone(),
            windowed.surface_config.format,
        )?;
        let lightning_publish = lightning_cell
            .lock()
            .map_err(|e| anyhow::anyhow!("lightning cell poisoned: {e}"))?
            .clone();
        let atmosphere_luts = atmosphere_luts_cell
            .lock()
            .map_err(|e| anyhow::anyhow!("luts cell poisoned: {e}"))?
            .clone();
        // Build the LUT overlay unconditionally; the runtime check
        // against config.debug.atmosphere_lut_overlay happens each
        // frame so the toggle goes live without a restart.
        let lut_overlay = atmosphere_luts.as_ref().map(|_| {
            info!(target: "ps_app", "LUT overlay constructed (toggle gated per-frame)");
            LutOverlay::new(&windowed.gpu, windowed.surface_config.format)
        });
        // Phase 10 fullscreen LUT viewer.
        let lut_viewer = atmosphere_luts
            .as_ref()
            .map(|_| LutViewer::new(&windowed.gpu, windowed.surface_config.format));

        let camera = ps_core::camera::FlyCamera {
            position: Vec3::new(0.0, 1.7, 5.0),
            ..ps_core::camera::FlyCamera::default()
        };

        let hot_reload = HotReload::watch(config_path, scene_path, DEFAULT_DEBOUNCE)
            .context("starting hot-reload watcher")?;

        // Plan §Cross-Cutting/Shader hot-reload — when the flag is on,
        // register the workspace's `shaders/` directory with the loader
        // so subsystems read live source, then start a recursive
        // watcher that fires `App::reconfigure` on change.
        let shader_hot_reload = if config.debug.shader_hot_reload {
            let shaders_root = workspace_root_for_shaders(config_path);
            ps_core::shaders::set_workspace_root(Some(shaders_root.clone()));
            match ps_core::ShaderHotReload::watch(&shaders_root, DEFAULT_DEBOUNCE) {
                Ok(w) => {
                    info!(
                        target: "ps_app",
                        path = %shaders_root.display(),
                        "shader hot-reload watching"
                    );
                    Some(w)
                }
                Err(e) => {
                    warn!(error = %e, "failed to start shader hot-reload watcher");
                    None
                }
            }
        } else {
            None
        };

        let initial_utc = config_initial_utc(config);
        let world = ps_core::WorldState::new(
            initial_utc,
            config.world.latitude_deg,
            config.world.longitude_deg,
            config.world.ground_elevation_m as f64,
        );
        info!(
            target: "ps_app",
            sim_utc = %world.clock.current_utc(),
            sun_alt_deg = world.sun.altitude_rad.to_degrees(),
            sun_az_deg = world.sun.azimuth_rad.to_degrees(),
            "WorldState initialised"
        );

        // Phase 3: synthesise the GPU-resident weather state from the
        // scene + config + world. Re-loaded by hot-reload on
        // SceneChanged / ConfigChanged events.
        let weather = ps_synthesis::synthesise(scene, config, &world, &windowed.gpu)
            .context("synthesise WeatherState")?;
        info!(
            target: "ps_app",
            cloud_layers = weather.cloud_layer_count,
            haze_per_m = weather.haze_extinction_per_m.x,
            "WeatherState synthesised"
        );

        let now = Instant::now();
        let title_prefix = config.window.title.clone();
        Ok(Self {
            window,
            windowed_gpu: windowed,
            hdr,
            bindings,
            tonemap,
            auto_exposure,
            tonemap_handle,
            ui_handle,
            ui_bridge,
            probe,
            app,
            camera,
            keys: KeyState::default(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            ev100: config.render.ev100,
            tonemap_mode: config.render.tone_mapper,
            world,
            weather,
            atmosphere_luts,
            lightning_publish,
            lut_overlay,
            lut_viewer,
            start: now,
            last_frame: now,
            last_camera_position: None,
            prev_view_proj: None,
            frame_index: 0,
            fps_accum_dt: 0.0,
            fps_accum_frames: 0,
            title_prefix,
            hot_reload,
            shader_hot_reload,
            weather_fetch_rx: None,
            geocode_rx: None,
        })
    }

    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
        config: &mut Config,
        scene: &mut Scene,
    ) {
        // Phase 10: forward the event to egui first. If egui consumes
        // it (e.g. focus on a slider, mouse over a panel), don't pass
        // it on to the camera / cursor-grab logic.
        let response = self.ui_bridge.on_window_event(&self.window, &event);
        if response.consumed {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let size = (size.width.max(1), size.height.max(1));
                self.windowed_gpu.resize(size);
                self.hdr.resize(&self.windowed_gpu.gpu, size);
                self.tonemap
                    .rebuild_bindings(&self.windowed_gpu.gpu.device, &self.hdr);
                self.auto_exposure
                    .rebuild_bindings(&self.windowed_gpu.gpu.device, &self.hdr);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                self.handle_key(event_loop, code, state == ElementState::Pressed);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.set_cursor_grab(true);
            }
            WindowEvent::Focused(false) => {
                self.set_cursor_grab(false);
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw(config, scene) {
                    tracing::error!(error = %e, "draw failed");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::Escape if pressed => self.set_cursor_grab(false),
            KeyCode::KeyW => self.keys.forward = pressed,
            KeyCode::KeyS => self.keys.back = pressed,
            KeyCode::KeyA => self.keys.left = pressed,
            KeyCode::KeyD => self.keys.right = pressed,
            KeyCode::Space => self.keys.up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.keys.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.sprint = pressed,
            KeyCode::KeyQ => self.keys.roll_left = pressed,
            KeyCode::KeyE => self.keys.roll_right = pressed,
            KeyCode::F4 if pressed => event_loop.exit(),
            _ => {}
        }
    }

    fn set_cursor_grab(&mut self, grab: bool) {
        if grab == self.cursor_grabbed {
            return;
        }
        let result = if grab {
            self.window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked))
        } else {
            self.window.set_cursor_grab(CursorGrabMode::None)
        };
        if let Err(e) = result {
            warn!(error = %e, "cursor grab failed");
            return;
        }
        self.window.set_cursor_visible(!grab);
        self.cursor_grabbed = grab;
        self.mouse_delta = (0.0, 0.0);
    }

    fn update_camera(&mut self, dt: f32) {
        let sensitivity = 0.0025_f32;
        if self.cursor_grabbed {
            self.camera.yaw += self.mouse_delta.0 as f32 * sensitivity;
            self.camera.pitch -= self.mouse_delta.1 as f32 * sensitivity;
            self.camera.pitch = self
                .camera
                .pitch
                .clamp(-89_f32.to_radians(), 89_f32.to_radians());
        }
        self.mouse_delta = (0.0, 0.0);

        let mut wish = Vec3::ZERO;
        let f = self.camera.forward();
        let r = self.camera.right();
        if self.keys.forward {
            wish += f;
        }
        if self.keys.back {
            wish -= f;
        }
        if self.keys.right {
            wish += r;
        }
        if self.keys.left {
            wish -= r;
        }
        if self.keys.up {
            wish += Vec3::Y;
        }
        if self.keys.down {
            wish -= Vec3::Y;
        }
        if wish.length_squared() > 0.0 {
            let speed = if self.keys.sprint { 5.0 } else { 1.0 } * self.camera.speed_mps;
            self.camera.position += wish.normalize() * speed * dt;
        }
        let roll_speed = 1.0_f32;
        if self.keys.roll_left {
            self.camera.roll += roll_speed * dt;
        }
        if self.keys.roll_right {
            self.camera.roll -= roll_speed * dt;
        }
    }

    fn poll_hot_reload(&mut self, config: &mut Config, scene: &mut Scene) {
        // Plan §Cross-Cutting/Shader hot-reload — drain shader-change
        // events and rebuild affected pipelines via App::reconfigure.
        // The reconfigure path is the same as the config-change path,
        // so subsystems re-read their WGSL via load_shader (which now
        // returns the freshly-edited file content).
        if let Some(watcher) = self.shader_hot_reload.as_ref() {
            let mut any = false;
            let mut last_path: Option<PathBuf> = None;
            for event in watcher.events().try_iter() {
                match event {
                    ps_core::ShaderWatchEvent::Changed(path) => {
                        any = true;
                        last_path = Some(path);
                    }
                    ps_core::ShaderWatchEvent::Error(msg) => {
                        warn!(%msg, "shader hot-reload watcher error");
                    }
                }
            }
            if any {
                info!(target: "ps_app", path = ?last_path,
                      "shader changed — rebuilding subsystems");
                if let Err(e) = self.app.reconfigure(config, &self.windowed_gpu.gpu) {
                    warn!(error = %e, "App::reconfigure (shader change) failed");
                }
            }
        }

        loop {
            match self.hot_reload.events().try_recv() {
                Ok(WatchEvent::ConfigChanged(path)) => {
                    info!(?path, "hot-reload: pedalsky.toml changed");
                    let load_result = Config::load(&path)
                        .and_then(|c| c.validate_with_base(path.parent()).map(|_| c));
                    match load_result {
                        Ok(new_config) => {
                            self.ev100 = new_config.render.ev100;
                            self.tonemap_mode =
                                new_config.render.tone_mapper;
                            if let Err(e) =
                                self.app.reconfigure(&new_config, &self.windowed_gpu.gpu)
                            {
                                warn!(error = %e, "App::reconfigure failed");
                            } else {
                                info!("App::reconfigure applied");
                            }
                            // Re-synthesise the weather state since
                            // atmosphere params and ground albedo come
                            // from the config too.
                            self.resynthesise_weather(scene, &new_config);
                            *config = new_config;
                        }
                        Err(e) => warn!(error = %e, "ignoring invalid config update"),
                    }
                }
                Ok(WatchEvent::SceneChanged(path)) => {
                    info!(?path, "hot-reload: scene changed — re-synthesising weather");
                    let load_result = Scene::load(&path).and_then(|s| s.validate().map(|_| s));
                    match load_result {
                        Ok(new_scene) => {
                            *scene = new_scene;
                            self.resynthesise_weather(scene, config);
                        }
                        Err(e) => warn!(error = %e, "ignoring invalid scene update"),
                    }
                }
                Ok(WatchEvent::Error(msg)) => warn!(%msg, "hot-reload watcher error"),
                Err(_) => break,
            }
        }
    }

    /// Re-run the synthesis pipeline. On error keep the previous state.
    fn resynthesise_weather(&mut self, scene: &Scene, config: &Config) {
        match ps_synthesis::synthesise(scene, config, &self.world, &self.windowed_gpu.gpu) {
            Ok(w) => {
                self.weather = w;
                info!(
                    target: "ps_app",
                    cloud_layers = self.weather.cloud_layer_count,
                    "WeatherState re-synthesised"
                );
            }
            Err(e) => warn!(error = %e, "synthesise failed; keeping previous WeatherState"),
        }
    }

    /// Phase 14 — spawn a background thread that fetches live
    /// weather. Multiple concurrent fetches are gated: if a
    /// previous fetch is in flight (`weather_fetch_rx` is `Some`),
    /// the new request is dropped on the floor. The UI's "Fetching…"
    /// button state already disables the click while in flight, so
    /// this is a belt-and-braces guard.
    fn spawn_weather_fetch(&mut self, req: ps_ui::WeatherFetchRequest) {
        if self.weather_fetch_rx.is_some() {
            tracing::debug!(
                target: "ps_app::weather_fetch",
                "fetch already in flight; ignoring new request"
            );
            return;
        }
        let (tx, rx) = crossbeam_channel::bounded::<WeatherFetchResult>(1);
        self.weather_fetch_rx = Some(rx);
        // Mark the UI as in-flight so the button repaints.
        self.ui_handle.lock().weather_fetch.in_flight = true;
        self.ui_handle.lock().weather_fetch.last_error = None;
        self.ui_handle.lock().weather_fetch.last_summary = None;

        let cache_dir = match workspace_root() {
            Ok(p) => p.join("cache").join("weather"),
            Err(_) => std::path::PathBuf::from("cache").join("weather"),
        };
        let opts = ps_weather_feed::FetchOptions {
            lat: req.lat,
            lon: req.lon,
            time: req.time,
            enrich_with_metar: req.enrich_with_metar,
            fetch_kp_index: req.fetch_kp_index,
            fetch_ovation: req.fetch_ovation,
            cache_dir,
        };
        std::thread::Builder::new()
            .name("ps-weather-feed".into())
            .spawn(move || {
                let result = match ps_weather_feed::fetch_scene(&opts) {
                    Ok(scene) => {
                        let summary = format!(
                            "Open-Meteo @ {:.3}, {:.3} ({} cloud layers)",
                            opts.lat, opts.lon, scene.clouds.layers.len()
                        );
                        WeatherFetchResult {
                            outcome: Ok(scene),
                            summary,
                        }
                    }
                    Err(e) => WeatherFetchResult {
                        outcome: Err(format!("weather fetch failed: {e}")),
                        summary: String::new(),
                    },
                };
                let _ = tx.send(result);
            })
            .expect("spawn weather-fetch thread");
    }

    /// Drain a completed weather fetch (if any). Replaces the
    /// live scene + triggers a synthesis re-bake when the worker
    /// returns a successful Scene.
    fn poll_weather_fetch(&mut self, scene: &mut Scene, config: &Config) {
        let Some(rx) = &self.weather_fetch_rx else {
            return;
        };
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(crossbeam_channel::TryRecvError::Empty) => return,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                // Worker panicked or completed without sending —
                // clear the in-flight state so the user can retry.
                self.weather_fetch_rx = None;
                let mut h = self.ui_handle.lock();
                h.weather_fetch.in_flight = false;
                h.weather_fetch.last_error =
                    Some("fetch thread disconnected without result".into());
                return;
            }
        };
        // We have a result. Clear in-flight state.
        self.weather_fetch_rx = None;
        match result.outcome {
            Ok(new_scene) => {
                info!(
                    target: "ps_app",
                    cloud_layers = new_scene.clouds.layers.len(),
                    "real-weather fetch succeeded"
                );
                *scene = new_scene;
                self.resynthesise_weather(scene, config);
                let mut h = self.ui_handle.lock();
                h.weather_fetch.in_flight = false;
                h.weather_fetch.last_summary = Some(result.summary);
                h.weather_fetch.last_error = None;
            }
            Err(err_msg) => {
                warn!(target: "ps_app", error = %err_msg, "real-weather fetch failed");
                let mut h = self.ui_handle.lock();
                h.weather_fetch.in_flight = false;
                h.weather_fetch.last_error = Some(err_msg);
                h.weather_fetch.last_summary = None;
            }
        }
    }

    /// Phase 16.B — spawn a background thread that runs a geocoding
    /// search. Drops the new request if a previous search is still in
    /// flight, matching the weather-fetch gating.
    fn spawn_geocode(&mut self, req: ps_ui::GeocodeRequest) {
        if self.geocode_rx.is_some() {
            tracing::debug!(
                target: "ps_app::geocode",
                "search already in flight; ignoring new request"
            );
            return;
        }
        let (tx, rx) = crossbeam_channel::bounded::<GeocodeWorkerResult>(1);
        self.geocode_rx = Some(rx);
        {
            let mut h = self.ui_handle.lock();
            h.geocode.in_flight = true;
            h.geocode.last_error = None;
        }
        let cache_dir = match workspace_root() {
            Ok(p) => p.join("cache").join("weather"),
            Err(_) => std::path::PathBuf::from("cache").join("weather"),
        };
        std::thread::Builder::new()
            .name("ps-geocode".into())
            .spawn(move || {
                let outcome = ps_weather_feed::geocoding::search(
                    &cache_dir,
                    &req.query,
                    req.count,
                    ps_weather_feed::geocoding::DEFAULT_TTL,
                )
                .map_err(|e| format!("geocoding search failed: {e}"));
                let _ = tx.send(GeocodeWorkerResult {
                    query: req.query,
                    outcome,
                });
            })
            .expect("spawn geocode thread");
    }

    /// Phase 16.B — drain a completed geocoding search and push the
    /// results into the shared UI state.
    fn poll_geocode(&mut self) {
        let Some(rx) = &self.geocode_rx else {
            return;
        };
        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(crossbeam_channel::TryRecvError::Empty) => return,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.geocode_rx = None;
                let mut h = self.ui_handle.lock();
                h.geocode.in_flight = false;
                h.geocode.last_error =
                    Some("geocode thread disconnected without result".into());
                return;
            }
        };
        self.geocode_rx = None;
        let mut h = self.ui_handle.lock();
        h.geocode.in_flight = false;
        h.geocode.last_query = result.query;
        match result.outcome {
            Ok(rows) => {
                h.geocode.last_error = None;
                h.geocode.results = rows
                    .into_iter()
                    .map(|r| ps_ui::GeocodeMatch {
                        id: r.id,
                        name: r.name,
                        admin1: r.admin1,
                        country: r.country,
                        latitude: r.latitude,
                        longitude: r.longitude,
                        elevation: r.elevation,
                        timezone: r.timezone,
                        population: r.population,
                    })
                    .collect();
                info!(
                    target: "ps_app",
                    count = h.geocode.results.len(),
                    query = %h.geocode.last_query,
                    "geocoding search succeeded"
                );
            }
            Err(err_msg) => {
                warn!(target: "ps_app", error = %err_msg, "geocoding search failed");
                h.geocode.last_error = Some(err_msg);
                h.geocode.results.clear();
            }
        }
    }

    fn draw(&mut self, config: &mut Config, scene: &mut Scene) -> Result<()> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().max(1e-6);
        self.last_frame = now;

        self.update_camera(dt);

        // Phase 2: drive the world clock; Phase 3: refresh per-frame sun
        // direction / illuminance on the live WeatherState (cheap; full
        // re-synthesis happens only on hot-reload).
        self.world.tick(dt as f64);
        self.weather.sun_direction = self.world.sun_direction_world;
        self.weather.sun_illuminance = glam::Vec3::splat(self.world.toa_illuminance_lux);

        let (w, h) = (
            self.windowed_gpu.surface_config.width,
            self.windowed_gpu.surface_config.height,
        );
        let aspect = w as f32 / h as f32;
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(aspect);
        // Camera velocity (m/s) for Phase 8 far-rain scrolling. Derived
        // from frame-to-frame position delta divided by wall-clock dt.
        let cam_velocity = match self.last_camera_position {
            Some(prev) if dt > 1e-4 => (self.camera.position - prev) / dt,
            _ => glam::Vec3::ZERO,
        };
        self.last_camera_position = Some(self.camera.position);
        let mut frame_uniforms = FrameUniforms {
            camera_position_world: Vec4::new(
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
                0.0,
            ),
            camera_velocity_world: Vec4::new(
                cam_velocity.x,
                cam_velocity.y,
                cam_velocity.z,
                0.0,
            ),
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: (now - self.start).as_secs_f32(),
            simulated_seconds: self.world.clock.simulated_seconds() as f32,
            frame_index: self.frame_index,
            ev100: self.ev100,
            ..FrameUniforms::default()
        };
        frame_uniforms.set_matrices(view, proj);
        // Cloud TAA reprojection. On the first frame `self.prev_view_proj`
        // is None — feed the current `view_proj` so the reprojection is
        // identity (history is also flagged invalid by the cloud
        // subsystem on its first run, so the TAA shader uses the
        // current sample directly). After that the cached value is
        // last frame's `view_proj`.
        frame_uniforms.prev_view_proj =
            self.prev_view_proj.unwrap_or(frame_uniforms.view_proj);
        // Sun angular radius from config (degrees → radians).
        // sun_disk toggle: setting angular radius to zero hides the
        // analytic disk (sky shader's `if (cos_view_sun > cos_disk)`
        // branch is never taken when cos_disk = 1).
        let sun_radius_rad = if config.render.atmosphere.sun_disk {
            config.render.atmosphere.sun_angular_radius_deg.to_radians()
        } else {
            0.0
        };
        frame_uniforms.set_sun(
            self.world.sun_direction_world,
            sun_radius_rad,
            self.weather.sun_illuminance,
            self.world.toa_illuminance_lux,
        );

        // Phase 12.3 — splice the previous frame's published lightning
        // snapshot into the uniforms. The lightning subsystem's
        // prepare() runs *after* this upload (PrepareContext exposes
        // FrameUniforms by `&`), so we accept a single-frame lag — at
        // 60 Hz that is ~16ms against the strike's 200 ms lifetime,
        // visually imperceptible.
        if let Some(publish) = &self.lightning_publish {
            let snap = *publish.lock().expect("lightning publish lock");
            frame_uniforms.lightning_illuminance = snap.illuminance;
            frame_uniforms.lightning_origin_world = snap.origin_world;
        }

        // Phase 5 diagnostic: log the key uniforms once at startup so we can
        // verify atmosphere bakes are receiving non-zero values. After the
        // first frame this is silent (frame_index > 0).
        if self.frame_index == 0 {
            info!(
                target: "ps_app::phase5_check",
                camera_pos = ?frame_uniforms.camera_position_world,
                sun_dir = ?frame_uniforms.sun_direction,
                sun_illum = ?frame_uniforms.sun_illuminance,
                atmo_planet_r = self.weather.atmosphere.planet_radius_m,
                atmo_top = self.weather.atmosphere.atmosphere_top_m,
                atmo_rayleigh = ?self.weather.atmosphere.rayleigh_scattering,
                "uniforms at first dispatch"
            );
        }
        // Apply atmosphere tuning toggles before uploading WorldUniforms.
        // - ozone_enabled: zero ozone_absorption when off.
        // The multi_scattering toggle is honoured inside
        // AtmosphereSubsystem (via `tuning_multi_scattering` field).
        let mut atmo = self.weather.atmosphere;
        if !config.render.atmosphere.ozone_enabled {
            atmo.ozone_absorption = glam::Vec4::ZERO;
        }
        // Upload the per-frame uniforms (groups 0 and 1). The world
        // (group 1) upload is diff-gated inside `write()` so it only
        // hits the wgpu queue when `atmo` actually changes — audit
        // §H2.
        self.bindings.write(
            &self.windowed_gpu.gpu.queue,
            &frame_uniforms,
            &atmo,
        );

        let frame = match self.windowed_gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                debug!("surface lost/outdated — reconfiguring");
                self.windowed_gpu.resize((w, h));
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("surface validation error from get_current_texture");
            }
        };
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.windowed_gpu
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });

        // Pre-frame clear: depth always; colour only if backdrop is disabled.
        encode_frame_clear(
            &mut encoder,
            &self.hdr,
            !config.render.subsystems.backdrop,
            config.render.clear_color,
        );

        // Audit §3.2 — propagate the cloud-subsystem on/off into the
        // WeatherState so the overcast modulation in sky/ground sees a
        // zeroed field when clouds aren't rendered. Without this, a
        // synthesised overcast deck still drives a pale-grey sky after
        // the user toggles the clouds subsystem off.
        self.weather.cloud_render_active = config.render.subsystems.clouds;
        self.weather
            .refresh_overcast_field_visibility(&self.windowed_gpu.gpu.queue);

        // Drive the App.
        let luts_ref = self.atmosphere_luts.as_deref();
        let luts_bind_group = luts_ref.map(|l| &l.bind_group);
        let mut prepare_ctx = PrepareContext {
            device: &self.windowed_gpu.gpu.device,
            queue: &self.windowed_gpu.gpu.queue,
            world: &self.world,
            weather: &self.weather,
            frame_uniforms: &frame_uniforms,
            atmosphere_luts: luts_ref,
            dt_seconds: dt,
        };
        let render_ctx = RenderContext {
            device: &self.windowed_gpu.gpu.device,
            queue: &self.windowed_gpu.gpu.queue,
            framebuffer: &self.hdr,
            frame_bind_group: &self.bindings.frame_bind_group,
            world_bind_group: &self.bindings.world_bind_group,
            luts_bind_group,
            frame_uniforms: &frame_uniforms,
            weather: &self.weather,
            tonemap_target: Some(&swap_view),
            tonemap_target_format: self.windowed_gpu.surface_config.format,
        };
        // Phase 9.1: push per-frame state into the in-graph tonemap
        // subsystem. The actual dispatch happens inside App::frame at
        // PassStage::ToneMap, with auto-exposure dispatched before
        // tonemap when the debug flag is on.
        self.tonemap_handle
            .set_state(ps_postprocess::TonemapState {
                ev100: self.ev100,
                mode: self.tonemap_mode,
                auto_exposure_enabled: config.debug.auto_exposure,
            });
        // Phase 10: refresh per-frame UI inputs and run the panel logic
        // before the in-graph Overlay pass picks up the paint output.
        {
            let mut state = self.ui_handle.lock();
            state.frame_stats.frame_ms = dt * 1000.0;
            if dt > 1e-4 {
                state.frame_stats.fps = 1.0 / dt;
            }
            state.world_readout = ps_ui::UiWorldReadout {
                sun_alt_deg: f64::from(self.world.sun.altitude_rad).to_degrees(),
                sun_az_deg: f64::from(self.world.sun.azimuth_rad).to_degrees(),
                moon_alt_deg: f64::from(self.world.moon.altitude_rad).to_degrees(),
                moon_az_deg: f64::from(self.world.moon.azimuth_rad).to_degrees(),
                julian_day: ps_core::astro::julian_day_utc(self.world.clock.current_utc()),
                // Camera yaw in the FlyCamera is in radians, with
                // yaw=0 → looking down −Z (= north in PedalSky's
                // +X east, +Z south right-handed frame). Convert to
                // a compass azimuth (clockwise from north).
                camera_yaw_deg: f64::from(self.camera.yaw).to_degrees(),
                wind_dir_deg: self.weather.surface.wind_dir_deg,
                wind_speed_mps: self.weather.surface.wind_speed_mps,
                // Phase 14.E — mirror the parsed scene's winds aloft
                // into the readout so the compass-rose overlay can draw
                // per-level barbs. Cloned each frame; the vec is at
                // most ~4 entries so the allocation is negligible.
                winds_aloft: scene.surface.winds_aloft.clone(),
            };
        }
        self.ui_bridge.build_ui_frame(&self.window);
        self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);

        // Phase 10.A4: dispatch the probe-pixel transmittance probe.
        // No-op if atmosphere is disabled (no LUT bind group).
        if let Some(luts_bg) = luts_bind_group {
            let probe_pixel = self.ui_handle.lock().debug.probe_pixel;
            self.probe.dispatch(
                &mut encoder,
                &self.windowed_gpu.gpu.queue,
                &self.windowed_gpu.gpu.device,
                probe_pixel,
                &self.bindings.frame_bind_group,
                &self.bindings.world_bind_group,
                luts_bg,
            );
        }

        // Phase 5 debug overlay (after tone-map; writes to the swapchain
        // with LoadOp::Load so the tone-mapped scene shows through where
        // the overlay isn't drawing).
        if config.debug.atmosphere_lut_overlay {
            if let (Some(overlay), Some(luts)) = (&self.lut_overlay, &self.atmosphere_luts) {
                let uniforms = LutOverlayUniforms::default();
                overlay.render(
                    &mut encoder,
                    &self.windowed_gpu.gpu.queue,
                    &self.windowed_gpu.gpu.device,
                    &swap_view,
                    luts,
                    &uniforms,
                );
            }
        }
        // Phase 10 fullscreen LUT viewer (gated by ui debug state).
        let (viewer_mode, viewer_depth) = {
            let s = self.ui_handle.lock();
            (s.debug.lut_viewer_mode, s.debug.ap_depth_slice)
        };
        if viewer_mode != 0 {
            if let (Some(viewer), Some(luts)) = (&self.lut_viewer, &self.atmosphere_luts) {
                // Per-mode scale: transmittance is in [0,1]; per-unit-
                // illuminance LUTs need amplification to be visible.
                let scale = match viewer_mode {
                    1 => 1.0,
                    2 => 1000.0,
                    3 => 1.0 / 5000.0,
                    4 => 1.0 / 5000.0,
                    _ => 1.0,
                };
                viewer.render(
                    &mut encoder,
                    &self.windowed_gpu.gpu.queue,
                    &self.windowed_gpu.gpu.device,
                    &swap_view,
                    luts,
                    &LutViewerUniforms {
                        mode: viewer_mode,
                        _pad: 0,
                        depth_slice: viewer_depth,
                        scale,
                    },
                );
            }
        }

        self.windowed_gpu.gpu.queue.submit([encoder.finish()]);
        // Phase 9.2: drain the auto-exposure staging buffer (one-frame
        // lag). The handle no-ops when auto-exposure is off.
        self.tonemap_handle
            .drain_auto_exposure(&self.windowed_gpu.gpu);
        // Phase 10 GPU timestamps: drain into the UI's frame stats.
        let gpu_passes = self.app.drain_gpu_timings(&self.windowed_gpu.gpu);
        if !gpu_passes.is_empty() {
            self.ui_handle.lock().frame_stats.gpu_passes = gpu_passes;
        }
        // Phase 10.A4 / 13.10: drain probe-pixel readout
        // (transmittance + per-component OD).
        if luts_bind_group.is_some() {
            if let Ok(r) = self.probe.read(&self.windowed_gpu.gpu) {
                let mut h = self.ui_handle.lock();
                h.debug.probe_transmittance = r.transmittance;
                h.debug.probe_od_rayleigh = r.od_rayleigh;
                h.debug.probe_od_mie = r.od_mie;
                h.debug.probe_od_ozone = r.od_ozone;
            }
        }
        // Phase 10.3 / 10.4: drain UI pending requests.
        self.drain_ui_pending(config, scene)?;
        frame.present();

        self.frame_index = self.frame_index.wrapping_add(1);
        // Capture this frame's view_proj for next frame's cloud TAA
        // reprojection. (Computed before the `set_matrices` call near
        // the top of `frame`; the value mirrors `frame_uniforms.view_proj`.)
        self.prev_view_proj = Some(proj * view);
        self.fps_accum_dt += dt;
        self.fps_accum_frames += 1;
        if self.fps_accum_dt >= 0.5 {
            let fps = self.fps_accum_frames as f32 / self.fps_accum_dt;
            self.window
                .set_title(&format!("{} — {:.0} fps", self.title_prefix, fps));
            self.fps_accum_dt = 0.0;
            self.fps_accum_frames = 0;
        }
        Ok(())
    }

    /// Phase 10.3/10.4/10.A — drain pending UI requests after each frame.
    fn drain_ui_pending(&mut self, config: &mut Config, scene: &mut Scene) -> Result<()> {
        let pending = std::mem::take(&mut self.ui_handle.lock().pending);

        // 10.3: slider edits → reconfigure.
        if pending.config_dirty {
            let new_config = self.ui_handle.lock().live_config.clone();
            // Vsync is a surface-level config; apply before
            // App::reconfigure so the next frame presents on the new
            // mode.
            if new_config.window.vsync != config.window.vsync {
                self.windowed_gpu.set_vsync(new_config.window.vsync);
            }
            *config = new_config.clone();
            self.ev100 = new_config.render.ev100;
            self.tonemap_mode = new_config.render.tone_mapper;
            self.app
                .reconfigure(&new_config, &self.windowed_gpu.gpu)
                .context("App::reconfigure (ui)")?;
        }

        // World-clock edits.
        if let Some(utc) = pending.set_world_utc {
            self.world.clock.set_utc(utc);
            self.world.recompute();
        }
        if let Some(scale) = pending.set_time_scale {
            self.world.clock.set_time_scale(scale);
        }
        if let Some(paused) = pending.set_paused {
            self.world.clock.set_paused(paused);
        }
        if let Some((lat, lon)) = pending.set_lat_lon {
            self.world.latitude_deg = lat;
            self.world.longitude_deg = lon;
            self.world.recompute();
        }

        // 10.4: screenshots (PNG / EXR).
        if pending.screenshot_png {
            if let Err(e) = self.write_png_screenshot(config) {
                warn!(error = %e, "PNG screenshot failed");
            }
        }
        if pending.screenshot_exr {
            if let Err(e) = self.write_exr_screenshot(config) {
                warn!(error = %e, "EXR screenshot failed");
            }
        }

        // 10.4: scene load/save.
        if let Some(path) = pending.load_scene {
            match Scene::load(&path) {
                Ok(loaded) => {
                    info!(target: "ps_app", path = %path.display(), "loading scene");
                    *scene = loaded;
                    self.resynthesise_weather(scene, config);
                }
                Err(e) => warn!(error = %e, "load scene failed"),
            }
        }
        if let Some(path) = pending.save_scene {
            if let Err(e) = self.write_scene_toml(&path, config, scene) {
                warn!(error = %e, "save scene failed");
            }
        }

        // 10.A1 — atmosphere coefficient edits land directly into the
        // live WeatherState; the atmosphere subsystem detects the
        // change at next frame's prepare via reconfigure (which marks
        // its static LUTs dirty).
        if let Some(atmo) = pending.live_atmosphere {
            self.weather.atmosphere = atmo;
            // Force atmosphere subsystem to re-bake static LUTs.
            self.app
                .reconfigure(config, &self.windowed_gpu.gpu)
                .context("App::reconfigure (live atmo)")?;
        }

        // 10.A2 / 10.A3 — scene-side edits trigger a re-synthesise of
        // the WeatherState (cloud layers, surface wetness etc.).
        if let Some(new_scene) = pending.live_scene {
            *scene = new_scene;
            self.resynthesise_weather(scene, config);
        }

        // Plan §0.4 — apply camera fov/near/speed slider edits.
        if let Some(cam) = pending.live_camera {
            self.camera.fov_y = cam.fov_y_rad;
            self.camera.near_m = cam.near_m;
            self.camera.speed_mps = cam.speed_mps;
        }

        // Phase 14 — real-weather fetch request from the UI's
        // "Fetch real weather" button.
        if let Some(req) = pending.fetch_real_weather {
            self.spawn_weather_fetch(req);
        }
        // Poll the worker channel for completed fetches and
        // splice the result back into the scene path.
        self.poll_weather_fetch(scene, config);

        // Phase 16.B — geocoding search request from the World
        // panel's "Search" button (or Enter inside the text input).
        if let Some(req) = pending.geocode_query {
            self.spawn_geocode(req);
        }
        self.poll_geocode();

        // Push live mirrors into the UI for the next frame's panels.
        {
            let mut s = self.ui_handle.lock();
            s.latest_atmosphere = Some(self.weather.atmosphere);
            s.latest_scene = Some(scene.clone());
            s.latest_camera = Some(ps_ui::CameraSettings {
                fov_y_rad: self.camera.fov_y,
                near_m: self.camera.near_m,
                speed_mps: self.camera.speed_mps,
            });
        }

        Ok(())
    }

    fn write_png_screenshot(&self, config: &Config) -> Result<()> {
        let dir = if config.paths.screenshot_dir.is_absolute() {
            config.paths.screenshot_dir.clone()
        } else {
            workspace_root()?.join(&config.paths.screenshot_dir)
        };
        std::fs::create_dir_all(&dir).ok();
        let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let path = dir.join(format!("pedalsky-{stamp}.png"));
        // Read the current swapchain output via tonemap target. Easiest:
        // re-encode the HDR target into an offscreen Rgba8Unorm and read.
        // Here we read the HDR target's tonemapped equivalent through a
        // dedicated copy. Simpler: read the previously-presented frame
        // from the swapchain by way of a copy-to-buffer at end of frame.
        // For now, write the HDR target through tonemap into an
        // offscreen Rgba8 and save that.
        let (w, h) = self.hdr.size;
        let (rgba, w_out, h_out) = self.read_tonemapped_into_rgba8(w, h)?;
        ps_ui::screenshot::write_png(&path, w_out, h_out, &rgba)?;
        info!(target: "ps_app", path = %path.display(), "wrote PNG screenshot");
        Ok(())
    }

    fn write_exr_screenshot(&self, config: &Config) -> Result<()> {
        let dir = if config.paths.screenshot_dir.is_absolute() {
            config.paths.screenshot_dir.clone()
        } else {
            workspace_root()?.join(&config.paths.screenshot_dir)
        };
        std::fs::create_dir_all(&dir).ok();
        let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let path = dir.join(format!("pedalsky-{stamp}.exr"));
        let (w, h) = self.hdr.size;
        let pixels = self.read_hdr_into_f16_rgba(w, h)?;
        ps_ui::screenshot::write_exr(&path, w, h, &pixels)?;
        info!(target: "ps_app", path = %path.display(), "wrote EXR screenshot");
        Ok(())
    }

    fn read_hdr_into_f16_rgba(&self, w: u32, h: u32) -> Result<Vec<half::f16>> {
        let bytes_per_pixel = 8u32; // Rgba16Float
        let unpadded = w * bytes_per_pixel;
        let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = self.windowed_gpu.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr-screenshot-staging"),
            size: (aligned * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .windowed_gpu
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hdr-screenshot-copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.hdr.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.windowed_gpu.gpu.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.windowed_gpu
            .gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        // Audit §3.5 — propagate the `map_async` callback's Result so a
        // failed mapping returns an error instead of panicking in
        // `get_mapped_range`. The screenshot button is user-triggered;
        // a panic here would crash the app on a transient mapping
        // failure (out of memory, device lost, etc.).
        let map_result = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("HDR screenshot: map_async channel closed: {e}"))?;
        map_result.map_err(|e| anyhow::anyhow!("HDR screenshot: map_async failed: {e}"))?;
        let bytes = slice.get_mapped_range().to_vec();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let row = &bytes[(y * aligned) as usize..(y * aligned + unpadded) as usize];
            for chunk in row.chunks_exact(2) {
                out.push(half::f16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        Ok(out)
    }

    fn read_tonemapped_into_rgba8(&self, w: u32, h: u32) -> Result<(Vec<u8>, u32, u32)> {
        // Allocate an offscreen Rgba8Unorm + run tonemap into it.
        let device = &self.windowed_gpu.gpu.device;
        let queue = &self.windowed_gpu.gpu.queue;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot-target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot-tonemap"),
        });
        // Reuse the live Tonemap with current EV/mode settings.
        self.tonemap.render(&mut encoder, queue, &target_view, self.ev100, self.tonemap_mode);
        // Copy target → staging.
        let bytes_per_pixel = 4u32;
        let unpadded = w * bytes_per_pixel;
        let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot-staging"),
            size: (aligned * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        // Audit §3.5 — see `read_hdr_into_*` for the rationale.
        let map_result = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("LDR screenshot: map_async channel closed: {e}"))?;
        map_result.map_err(|e| anyhow::anyhow!("LDR screenshot: map_async failed: {e}"))?;
        let bytes = slice.get_mapped_range().to_vec();
        let mut out = Vec::with_capacity((w * h * bytes_per_pixel) as usize);
        for y in 0..h {
            let row = &bytes[(y * aligned) as usize..(y * aligned + unpadded) as usize];
            out.extend_from_slice(row);
        }
        Ok((out, w, h))
    }

    fn write_scene_toml(
        &self,
        path: &std::path::Path,
        config: &Config,
        scene: &Scene,
    ) -> Result<()> {
        // Save the live Config + the live Scene as a TOML pair. Plan
        // §10.4 / Appendices A & B.
        //
        // - If the user-supplied path looks like a scene file
        //   (`<name>.scene.toml`), write the scene there directly and
        //   write the engine config alongside as `<name>.toml`.
        // - Otherwise treat the supplied path as the engine config and
        //   sidecar the scene as `<stem>.scene.toml`.
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pedalsky");
        let user_picked_scene_path = path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.contains(".scene."));

        let (config_path, scene_path): (std::path::PathBuf, std::path::PathBuf) =
            if user_picked_scene_path {
                let stem_no_scene =
                    stem.strip_suffix(".scene").unwrap_or(stem);
                (
                    parent.join(format!("{stem_no_scene}.toml")),
                    path.to_path_buf(),
                )
            } else {
                (
                    path.to_path_buf(),
                    parent.join(format!("{stem}.scene.toml")),
                )
            };

        let config_toml = toml::to_string_pretty(config).context("toml encode config")?;
        std::fs::write(&config_path, config_toml)
            .with_context(|| format!("write {}", config_path.display()))?;
        info!(target: "ps_app", path = %config_path.display(), "wrote engine config TOML");

        let scene_toml = toml::to_string_pretty(scene).context("toml encode scene")?;
        std::fs::write(&scene_path, scene_toml)
            .with_context(|| format!("write {}", scene_path.display()))?;
        info!(target: "ps_app", path = %scene_path.display(), "wrote scene TOML");
        Ok(())
    }
}

/// Cell into which `AtmosphereFactory` deposits its `AtmosphereLuts`
/// handle once the subsystem is constructed.
type AtmosphereLutsCell =
    std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<ps_core::AtmosphereLuts>>>>;

/// Cell into which `LightningFactory` deposits its snapshot publish
/// handle. The host reads the inner `LightningPublish` after each
/// `App::frame` to splice cloud illumination into `FrameUniforms`
/// before the GPU upload.
type LightningPublishCell = std::sync::Arc<std::sync::Mutex<Option<LightningPublish>>>;

/// Build the app. Returns the `App` plus the LUTs cell published by
/// `AtmosphereFactory` (Some after build if atmosphere is enabled).
#[allow(clippy::type_complexity)]
fn build_app(
    config: &Config,
    gpu: &ps_core::GpuContext,
    tonemap: std::sync::Arc<Tonemap>,
    auto_exposure: std::sync::Arc<ps_postprocess::AutoExposure>,
    ui_handle: ps_ui::UiHandle,
    window: std::sync::Arc<winit::window::Window>,
    tonemap_target_format: wgpu::TextureFormat,
) -> Result<(
    App,
    AtmosphereLutsCell,
    LightningPublishCell,
    ps_postprocess::TonemapHandle,
    std::sync::Arc<ps_ui::UiBridge>,
)> {
    let (atmosphere_factory, luts_cell) = AtmosphereFactory::new();
    let (lightning_factory, lightning_cell) = LightningFactory::new();
    let clouds_factory = CloudsFactory::with_atmosphere_luts(luts_cell.clone());
    let (tonemap_factory, tonemap_handle) = ps_postprocess::TonemapFactory::new();
    tonemap_handle.inject(
        tonemap,
        auto_exposure,
        ps_postprocess::TonemapState {
            ev100: config.render.ev100,
            mode: config.render.tone_mapper,
            auto_exposure_enabled: config.debug.auto_exposure,
        },
    );
    let (ui_factory, ui_injector, ui_bridge) = ps_ui::UiFactory::new();
    ui_injector.inject(tonemap_target_format, window, ui_handle);
    let app = AppBuilder::new()
        .with_factory(Box::new(BackdropFactory))
        .with_factory(Box::new(atmosphere_factory))
        .with_factory(Box::new(GroundFactory))
        // Phase 13.5 — water draws after ground at PassStage::Opaque
        // (registration order tie-break). Gated by both the
        // [render.subsystems].water flag and the scene having a
        // [water] block.
        .with_factory(Box::new(WaterFactory))
        .with_factory(Box::new(clouds_factory))
        .with_factory(Box::new(PrecipFactory))
        .with_factory(Box::new(TintFactory))
        // Phase 12.4 — godrays sit at PostProcess (before
        // tone-map). Built unconditionally; the GodraysFactory
        // gates on `[render.subsystems].godrays` itself, and the
        // pass closure no-ops when the sun is off-screen.
        .with_factory(Box::new(GodraysFactory))
        // Phase 12.3 — lightning sits at Translucent (after clouds).
        // Built unconditionally; LightningFactory gates on
        // `[render.subsystems].lightning` itself.
        .with_factory(Box::new(lightning_factory))
        // Phase 12.5 — aurora curtains, also at Translucent. Gated
        // on `[render.subsystems].aurora` and the latitude/kp curve.
        .with_factory(Box::new(AuroraFactory))
        // Phase 13.3 — HDR bloom at PostProcess (before tone-map).
        // Gated on `[render.subsystems].bloom`.
        .with_factory(Box::new(BloomFactory))
        // Phase 13.6 — windsock geometry at Opaque (after ground).
        // Off by default; gated on `[render.subsystems].windsock`.
        .with_factory(Box::new(WindsockFactory))
        .with_factory(Box::new(tonemap_factory))
        .with_factory(Box::new(ui_factory))
        .build(config, gpu)
        .context("AppBuilder::build")?;
    Ok((app, luts_cell, lightning_cell, tonemap_handle, ui_bridge))
}
