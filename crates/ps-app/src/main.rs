//! PedalSky main binary.
//!
//! Phase 0 + Phase 1 scope: load `pedalsky.toml`, load the scene file pointed
//! to by `[paths] weather`, validate both, then open a winit window and
//! render the Phase 0 placeholder ground plane through the ACES Filmic
//! tone-mapper. Phase 1's AppBuilder/factory wiring lands here once the
//! Backdrop/Tint demo subsystems are written.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{Vec3, Vec4};
use ps_core::{Config, FrameUniforms, HdrFramebufferImpl, Scene};
use ps_ground::CheckerGround;
use ps_postprocess::{Tonemap, TonemapMode};
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
        "configuration loaded; opening window"
    );

    let event_loop = EventLoop::new().context("create EventLoop")?;
    let mut app = AppShell::new(config);
    event_loop.run_app(&mut app).context("event loop")?;
    Ok(())
}

/// Walk up from `CARGO_MANIFEST_DIR` until a `pedalsky.toml` is found.
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

/// Outer shell holding the loaded config plus an option of the live
/// rendering state. winit 0.30 only gives us a real `ActiveEventLoop` inside
/// `resumed()`, so the GPU plumbing is constructed lazily there.
struct AppShell {
    config: Config,
    state: Option<RunState>,
}

impl AppShell {
    fn new(config: Config) -> Self {
        Self { config, state: None }
    }
}

/// Live rendering state: window, GPU, framebuffer, render passes, camera.
struct RunState {
    window: Arc<Window>,
    windowed_gpu: ps_core::gpu::WindowedGpu<'static>,
    hdr: HdrFramebufferImpl,
    ground: CheckerGround,
    tonemap: Tonemap,
    camera: ps_core::camera::FlyCamera,
    /// Pressed-key state for movement.
    keys: KeyState,
    /// Whether the cursor is currently grabbed for mouse-look.
    cursor_grabbed: bool,
    /// Pending mouse delta accumulated since the last frame.
    mouse_delta: (f64, f64),
    /// EV100 from config (UI sliders edit this in Phase 10).
    ev100: f32,
    /// Tone-map mode from config.
    tonemap_mode: TonemapMode,
    /// Wall-clock start, for `time_seconds` in FrameUniforms.
    start: Instant,
    /// Last frame's wall-clock instant, for dt.
    last_frame: Instant,
    /// Monotonic frame counter.
    frame_index: u32,
    /// Title-bar fps smoother.
    fps_accum_dt: f32,
    fps_accum_frames: u32,
    /// Cached title prefix.
    title_prefix: String,
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
        match RunState::new(event_loop, &self.config) {
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
        let Some(state) = self.state.as_mut() else { return };
        state.handle_window_event(event_loop, event);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += delta.0;
                state.mouse_delta.1 += delta.1;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.window.request_redraw();
        }
    }
}

impl RunState {
    fn new(event_loop: &ActiveEventLoop, config: &Config) -> Result<Self> {
        let initial_size = (config.window.width, config.window.height);
        let attrs = Window::default_attributes()
            .with_title(&config.window.title)
            .with_inner_size(PhysicalSize::new(initial_size.0, initial_size.1));
        let window = Arc::new(event_loop.create_window(attrs).context("create_window")?);

        let physical = window.inner_size();
        let size = (physical.width.max(1), physical.height.max(1));
        let windowed = ps_core::gpu::init_windowed(window.clone(), size, config.window.vsync)
            .context("init_windowed")?;

        let hdr = HdrFramebufferImpl::new(&windowed.gpu, size);
        let ground = CheckerGround::new(&windowed.gpu.device);
        let tonemap = Tonemap::new(&windowed.gpu.device, &hdr, windowed.surface_config.format);

        let camera = ps_core::camera::FlyCamera {
            position: Vec3::new(0.0, 1.7, 5.0),
            ..ps_core::camera::FlyCamera::default()
        };

        let now = Instant::now();
        let title_prefix = config.window.title.clone();
        Ok(Self {
            window,
            windowed_gpu: windowed,
            hdr,
            ground,
            tonemap,
            camera,
            keys: KeyState::default(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            ev100: config.render.ev100,
            tonemap_mode: TonemapMode::from_config(&config.render.tone_mapper),
            start: now,
            last_frame: now,
            frame_index: 0,
            fps_accum_dt: 0.0,
            fps_accum_frames: 0,
            title_prefix,
        })
    }

    fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
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
                event: KeyEvent {
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
                if let Err(e) = self.draw() {
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
        // Mouse-look (pixels → radians at ~0.1° per pixel).
        let sensitivity = 0.0025_f32;
        if self.cursor_grabbed {
            self.camera.yaw -= self.mouse_delta.0 as f32 * sensitivity;
            self.camera.pitch -= self.mouse_delta.1 as f32 * sensitivity;
            self.camera.pitch = self
                .camera
                .pitch
                .clamp(-89_f32.to_radians(), 89_f32.to_radians());
        }
        self.mouse_delta = (0.0, 0.0);

        // Movement.
        let mut wish = Vec3::ZERO;
        let f = self.camera.forward();
        let r = self.camera.right();
        if self.keys.forward { wish += f; }
        if self.keys.back    { wish -= f; }
        if self.keys.right   { wish += r; }
        if self.keys.left    { wish -= r; }
        if self.keys.up      { wish += Vec3::Y; }
        if self.keys.down    { wish -= Vec3::Y; }
        if wish.length_squared() > 0.0 {
            let speed = if self.keys.sprint { 5.0 } else { 1.0 } * self.camera.speed_mps;
            self.camera.position += wish.normalize() * speed * dt;
        }

        // Roll.
        let roll_speed = 1.0_f32;
        if self.keys.roll_left  { self.camera.roll += roll_speed * dt; }
        if self.keys.roll_right { self.camera.roll -= roll_speed * dt; }
    }

    fn draw(&mut self) -> Result<()> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().max(1e-6);
        self.last_frame = now;

        self.update_camera(dt);

        let (w, h) = (self.windowed_gpu.surface_config.width, self.windowed_gpu.surface_config.height);
        let aspect = w as f32 / h as f32;
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(aspect);
        let frame_uniforms = FrameUniforms {
            view,
            proj,
            view_proj: proj * view,
            camera_position_world: self.camera.position,
            viewport_size: Vec4::new(w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32),
            time_seconds: (now - self.start).as_secs_f32(),
            simulated_seconds: 0.0,
            frame_index: self.frame_index,
            ev100: self.ev100,
        };

        // Acquire the next swapchain image.
        let frame = match self.windowed_gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                debug!("surface lost/outdated — reconfiguring");
                self.windowed_gpu.resize((w, h));
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                // Skip the frame; will retry on the next redraw request.
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("surface validation error from get_current_texture");
            }
        };
        let swap_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .windowed_gpu
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });

        // Phase 0 render graph: ground (clears HDR + depth) → tone-map.
        self.ground.render(&mut encoder, &self.windowed_gpu.gpu.queue, &self.hdr, &frame_uniforms);
        self.tonemap.render(
            &mut encoder,
            &self.windowed_gpu.gpu.queue,
            &swap_view,
            self.ev100,
            self.tonemap_mode,
        );

        self.windowed_gpu.gpu.queue.submit([encoder.finish()]);
        frame.present();

        // FPS in title bar (smoothed over ~0.5 s).
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
