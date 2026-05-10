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
use ps_atmosphere::{AtmosphereFactory, lut_overlay::{LutOverlay, LutOverlayUniforms}};
use ps_backdrop::BackdropFactory;
use ps_clouds::CloudsFactory;
use ps_core::{
    App, AppBuilder, Config, FrameUniforms, HdrFramebufferImpl, HotReload, PrepareContext,
    RenderContext, Scene, WatchEvent, DEFAULT_DEBOUNCE,
};
use ps_ground::GroundFactory;
use ps_postprocess::{Tonemap, TonemapMode};
use ps_precip::PrecipFactory;
use ps_tint::TintFactory;
use tracing::{debug, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

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

    // CLI override: `--lut-overlay` flips `config.debug.atmosphere_lut_overlay`
    // on independently of the config file.
    let cli_lut_overlay = argv.iter().any(|a| a == "--lut-overlay");

    let config_path = workspace_root.join("pedalsky.toml");
    let mut config =
        Config::load(&config_path).with_context(|| format!("loading {}", config_path.display()))?;
    if cli_lut_overlay {
        config.debug.atmosphere_lut_overlay = true;
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
    let mut shell = AppShell::new(config, scene, config_path, scene_path);
    event_loop.run_app(&mut shell).context("event loop")?;
    Ok(())
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
    state: Option<RunState>,
}

impl AppShell {
    fn new(config: Config, scene: Scene, config_path: PathBuf, scene_path: PathBuf) -> Self {
        Self {
            config,
            scene,
            config_path,
            scene_path,
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

    tonemap: Tonemap,
    /// `App` constructed from factories. Owns backdrop/ground/tint.
    app: App,
    /// Phase 5: LUTs handle published by `AtmosphereFactory`. `None`
    /// when `[render.subsystems].atmosphere = false`.
    atmosphere_luts: Option<std::sync::Arc<ps_core::AtmosphereLuts>>,
    /// Phase 5 debug overlay drawing the four LUTs onto the swapchain.
    /// `Some` when `[debug].atmosphere_lut_overlay = true` (or
    /// `--lut-overlay`) AND atmosphere is enabled.
    lut_overlay: Option<LutOverlay>,

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
    frame_index: u32,
    fps_accum_dt: f32,
    fps_accum_frames: u32,
    title_prefix: String,

    hot_reload: HotReload,
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
        state.handle_window_event(event_loop, event, &mut self.config);
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
    ) -> Result<Self> {
        let initial_size = (config.window.width, config.window.height);
        let attrs = Window::default_attributes()
            .with_title(&config.window.title)
            .with_inner_size(PhysicalSize::new(initial_size.0, initial_size.1));
        let window = Arc::new(event_loop.create_window(attrs).context("create_window")?);

        let physical = window.inner_size();
        let size = (physical.width.max(1), physical.height.max(1));
        let windowed = ps_core::gpu::init_windowed(
            window.clone(),
            size,
            config.window.vsync,
            config.debug.gpu_validation,
        )
        .context("init_windowed")?;

        let hdr = HdrFramebufferImpl::new(&windowed.gpu, size);
        let bindings = ps_core::FrameWorldBindings::new(&windowed.gpu.device);
        let tonemap = Tonemap::new(&windowed.gpu.device, &hdr, windowed.surface_config.format);

        let (app, atmosphere_luts_cell) = build_app(config, &windowed.gpu)?;
        let atmosphere_luts = atmosphere_luts_cell
            .lock()
            .map_err(|e| anyhow::anyhow!("luts cell poisoned: {e}"))?
            .clone();
        let lut_overlay = if config.debug.atmosphere_lut_overlay && atmosphere_luts.is_some() {
            info!(target: "ps_app", "LUT overlay enabled");
            Some(LutOverlay::new(
                &windowed.gpu,
                windowed.surface_config.format,
            ))
        } else {
            None
        };

        let camera = ps_core::camera::FlyCamera {
            position: Vec3::new(0.0, 1.7, 5.0),
            ..ps_core::camera::FlyCamera::default()
        };

        let hot_reload = HotReload::watch(config_path, scene_path, DEFAULT_DEBOUNCE)
            .context("starting hot-reload watcher")?;

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
            app,
            camera,
            keys: KeyState::default(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            ev100: config.render.ev100,
            tonemap_mode: TonemapMode::from_config(&config.render.tone_mapper),
            world,
            weather,
            atmosphere_luts,
            lut_overlay,
            start: now,
            last_frame: now,
            frame_index: 0,
            fps_accum_dt: 0.0,
            fps_accum_frames: 0,
            title_prefix,
            hot_reload,
        })
    }

    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
        config: &mut Config,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let size = (size.width.max(1), size.height.max(1));
                self.windowed_gpu.resize(size);
                self.hdr.resize(&self.windowed_gpu.gpu, size);
                self.tonemap
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
                if let Err(e) = self.draw(config) {
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
                                TonemapMode::from_config(&new_config.render.tone_mapper);
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

    fn draw(&mut self, config: &Config) -> Result<()> {
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
        let mut frame_uniforms = FrameUniforms {
            camera_position_world: Vec4::new(
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
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
        // Sun angular radius from config (degrees → radians).
        let sun_radius_rad = config.render.atmosphere.sun_angular_radius_deg.to_radians();
        frame_uniforms.set_sun(
            self.world.sun_direction_world,
            sun_radius_rad,
            self.weather.sun_illuminance,
            self.world.toa_illuminance_lux,
        );

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
        // Upload the per-frame uniforms (groups 0 and 1).
        self.bindings.write(
            &self.windowed_gpu.gpu.queue,
            &frame_uniforms,
            &self.weather.atmosphere,
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
        };
        self.app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);

        // Tone-map.
        self.tonemap.render(
            &mut encoder,
            &self.windowed_gpu.gpu.queue,
            &swap_view,
            self.ev100,
            self.tonemap_mode,
        );

        // Phase 5 debug overlay (after tone-map; writes to the swapchain
        // with LoadOp::Load so the tone-mapped scene shows through where
        // the overlay isn't drawing).
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

        self.windowed_gpu.gpu.queue.submit([encoder.finish()]);
        frame.present();

        self.frame_index = self.frame_index.wrapping_add(1);
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
}

/// Cell into which `AtmosphereFactory` deposits its `AtmosphereLuts`
/// handle once the subsystem is constructed.
type AtmosphereLutsCell =
    std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<ps_core::AtmosphereLuts>>>>;

/// Build the app. Returns the `App` plus the LUTs cell published by
/// `AtmosphereFactory` (Some after build if atmosphere is enabled).
fn build_app(config: &Config, gpu: &ps_core::GpuContext) -> Result<(App, AtmosphereLutsCell)> {
    let (atmosphere_factory, luts_cell) = AtmosphereFactory::new();
    let app = AppBuilder::new()
        .with_factory(Box::new(BackdropFactory))
        .with_factory(Box::new(atmosphere_factory))
        .with_factory(Box::new(GroundFactory))
        .with_factory(Box::new(CloudsFactory))
        .with_factory(Box::new(PrecipFactory))
        .with_factory(Box::new(TintFactory))
        .build(config, gpu)
        .context("AppBuilder::build")?;
    Ok((app, luts_cell))
}
