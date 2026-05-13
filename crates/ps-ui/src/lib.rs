//! PedalSky UI overlay (Phase 10: egui).
//!
//! Architecture:
//! - [`UiSubsystem`] owns `egui::Context`, `egui_winit::State`, and
//!   `egui_wgpu::Renderer`. It registers a single `PassStage::Overlay`
//!   pass that paints the UI on top of the tonemapped swapchain.
//! - [`UiHandle`] is the shared state cell threaded between the UI and
//!   the host frame loop. Panel logic mutates `UiHandle.live_config`
//!   and sets `pending` flags; the host drains them after `App::frame`.
//! - The host pumps winit events via [`UiSubsystem::on_window_event`]
//!   and pushes per-frame raw input via [`UiSubsystem::take_egui_input`].
//! - Plan §10.3: slider edits set `pending.config_dirty`, the host calls
//!   `App::reconfigure(&live_config, &gpu)` — the same path as
//!   file-watch hot-reload.

#![deny(missing_docs)]

pub mod panels;
pub mod screenshot;
pub mod state;

use std::sync::{Arc, Mutex};

use egui_wgpu::{Renderer, ScreenDescriptor};
use ps_core::{
    Config, GpuContext, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory,
};

pub use state::{
    CameraSettings, GeocodeMatch, GeocodeRequest, GeocodeStatus, UiDebugSelection, UiFrameStats,
    UiHandle, UiPending, UiState, WeatherFetchRequest, WeatherFetchStatus, UiWorldReadout,
};

/// Stable subsystem name.
pub const NAME: &str = "ui";

/// Captured paint output from `egui_ctx.end_pass()`. Owned by an `Arc<Mutex>`
/// so the (`prepare()` → pass-closure) hand-off works without unique
/// ownership.
struct PaintFrame {
    paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    pixels_per_point: f32,
}

/// Phase 10 UI subsystem.
///
/// Most of the user-facing control surface is exposed through
/// [`UiBridge`] (created by [`UiFactory::new`]) — the subsystem itself
/// only owns GPU resources and the `Overlay`-stage pass.
pub struct UiSubsystem {
    enabled: bool,

    /// egui core context. Cheap to clone (Arc inside).
    ctx: egui::Context,
    /// Per-window winit input → egui events bridge. Owned by an
    /// `Arc<Mutex>` so the [`UiBridge`] (host-facing) and the subsystem
    /// share one state.
    shared_winit_state: Arc<Mutex<egui_winit::State>>,
    /// egui → wgpu painter. Held in `Arc<Mutex>` so the pass closure
    /// can hold a reference independent of the subsystem's lifetime.
    renderer: Arc<Mutex<Renderer>>,

    /// Captured paint output from the most recent `build_ui_frame`. The
    /// pass closure drains this and renders.
    paint: Arc<Mutex<Option<PaintFrame>>>,
}

impl UiSubsystem {
    /// Construct.
    fn new(
        gpu: &GpuContext,
        target_format: wgpu::TextureFormat,
        window: &winit::window::Window,
        _handle: UiHandle,
    ) -> Self {
        let ctx = egui::Context::default();
        let viewport_id = ctx.viewport_id();
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = Renderer::new(
            &gpu.device,
            target_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: true,
                predictable_texture_filtering: false,
            },
        );
        Self {
            enabled: true,
            ctx,
            shared_winit_state: Arc::new(Mutex::new(winit_state)),
            renderer: Arc::new(Mutex::new(renderer)),
            paint: Arc::new(Mutex::new(None)),
        }
    }
}

impl RenderSubsystem for UiSubsystem {
    fn name(&self) -> &'static str {
        "ui"
    }

    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {
        // The host calls build_ui_frame before App::frame so raw input
        // and the panel state are populated. Nothing to do here.
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let renderer = self.renderer.clone();
        let paint = self.paint.clone();
        vec![RegisteredPass {
            name: "ui::overlay",
            stage: PassStage::Overlay,
            run: Box::new(move |encoder, ctx| {
                let Some(target) = ctx.tonemap_target else {
                    return;
                };
                let mut paint_guard = paint.lock().expect("ui paint lock");
                let Some(frame_paint) = paint_guard.take() else {
                    // No build_ui_frame this turn — leave overlay alone.
                    return;
                };
                let (w, h) = (ctx.framebuffer.size.0, ctx.framebuffer.size.1);
                let screen_desc = ScreenDescriptor {
                    size_in_pixels: [w, h],
                    pixels_per_point: frame_paint.pixels_per_point,
                };
                let mut renderer = renderer.lock().expect("ui renderer lock");
                for (id, image_delta) in &frame_paint.textures_delta.set {
                    renderer.update_texture(ctx.device, ctx.queue, *id, image_delta);
                }
                renderer.update_buffers(
                    ctx.device,
                    ctx.queue,
                    encoder,
                    &frame_paint.paint_jobs,
                    &screen_desc,
                );
                {
                    let mut pass = encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ui::overlay"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        })
                        .forget_lifetime();
                    renderer.render(&mut pass, &frame_paint.paint_jobs, &screen_desc);
                }
                for id in &frame_paint.textures_delta.free {
                    renderer.free_texture(id);
                }
            }),
        }]
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`. Pair with [`UiHandle`] via
/// [`UiFactory::new`].
pub struct UiFactory {
    inject_cell: Arc<Mutex<Option<UiInjection>>>,
}

struct UiInjection {
    target_format: wgpu::TextureFormat,
    window: Arc<winit::window::Window>,
    handle: UiHandle,
    bridge: Arc<UiBridge>,
}

/// Side-channel handle the host uses to push winit events into egui and
/// to drive the per-frame `build_ui_frame` call. Internally an
/// `Arc<Mutex<Option<...>>>` populated by the factory after build.
pub struct UiBridge {
    /// `Arc` reference to the constructed UI subsystem's egui state.
    /// Populated by `UiFactory::build`.
    ctx: Mutex<Option<UiCtx>>,
}

struct UiCtx {
    egui_ctx: egui::Context,
    winit_state: Arc<Mutex<egui_winit::State>>,
    paint: Arc<Mutex<Option<PaintFrame>>>,
    handle: UiHandle,
}

impl UiBridge {
    /// Construct an empty bridge.
    pub fn new() -> Self {
        Self {
            ctx: Mutex::new(None),
        }
    }

    /// Forward a winit window event to egui. Returns whether egui
    /// consumed the event (e.g. clicked a button or focused a slider).
    pub fn on_window_event(
        &self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        let guard = self.ctx.lock().expect("ui bridge ctx lock");
        let Some(ctx) = guard.as_ref() else {
            return egui_winit::EventResponse {
                consumed: false,
                repaint: false,
            };
        };
        let mut winit_guard = ctx.winit_state.lock().expect("ui winit state lock");
        winit_guard.on_window_event(window, event)
    }

    /// Build the UI for the upcoming frame: pump raw input, run the
    /// panel logic, capture paint output. The host then calls
    /// `App::frame` and the registered Overlay pass renders the paint.
    pub fn build_ui_frame(&self, window: &winit::window::Window) {
        let guard = self.ctx.lock().expect("ui bridge ctx lock");
        let Some(ctx) = guard.as_ref() else {
            return;
        };
        let raw_input = ctx
            .winit_state
            .lock()
            .expect("ui winit state lock")
            .take_egui_input(window);
        ctx.egui_ctx.begin_pass(raw_input);
        {
            let mut state = ctx.handle.lock();
            crate::panels::ui(&ctx.egui_ctx, &mut state);
        }
        let full_output = ctx.egui_ctx.end_pass();
        ctx.winit_state
            .lock()
            .expect("ui winit state lock")
            .handle_platform_output(window, full_output.platform_output);
        let paint_jobs = ctx
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        *ctx.paint.lock().expect("ui paint lock") = Some(PaintFrame {
            paint_jobs,
            textures_delta: full_output.textures_delta,
            pixels_per_point: full_output.pixels_per_point,
        });
    }
}

impl Default for UiBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl UiFactory {
    /// Construct an unconfigured factory; pair with a `UiInjector` the
    /// host fills before `AppBuilder::build`. Also returns an empty
    /// `Arc<UiBridge>` the host fills the same way.
    pub fn new() -> (Self, UiInjector, Arc<UiBridge>) {
        let cell = Arc::new(Mutex::new(None));
        let bridge = Arc::new(UiBridge::new());
        (
            Self {
                inject_cell: cell.clone(),
            },
            UiInjector {
                cell,
                bridge: bridge.clone(),
            },
            bridge,
        )
    }
}

/// Side-channel injection handle for [`UiFactory`]. Host calls
/// [`Self::inject`] before `AppBuilder::build`.
pub struct UiInjector {
    cell: Arc<Mutex<Option<UiInjection>>>,
    bridge: Arc<UiBridge>,
}

impl UiInjector {
    /// Provide the target swapchain format, window, and shared state
    /// handle.
    pub fn inject(
        &self,
        target_format: wgpu::TextureFormat,
        window: Arc<winit::window::Window>,
        handle: UiHandle,
    ) {
        *self.cell.lock().expect("ui inject lock") = Some(UiInjection {
            target_format,
            window,
            handle,
            bridge: self.bridge.clone(),
        });
    }
}

impl SubsystemFactory for UiFactory {
    fn name(&self) -> &'static str {
        "ui"
    }
    fn enabled(&self, _config: &Config) -> bool {
        true
    }
    fn build(
        &self,
        _config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        let injection = self
            .inject_cell
            .lock()
            .expect("ui inject lock")
            .take()
            .ok_or_else(|| {
                anyhow::anyhow!("UiFactory: nothing injected before AppBuilder::build")
            })?;
        let subsys = UiSubsystem::new(
            gpu,
            injection.target_format,
            &injection.window,
            injection.handle.clone(),
        );
        // Publish a UiCtx into the bridge so the host can call
        // build_ui_frame / on_window_event without going through `App`.
        *injection.bridge.ctx.lock().expect("ui bridge ctx lock") = Some(UiCtx {
            egui_ctx: subsys.ctx.clone(),
            // Move the winit_state Mutex out of the subsystem into a
            // shared Arc so both the bridge and the subsystem reference
            // the same state. The subsystem only uses winit_state via
            // build_ui_frame in the bridge path now, so we strip the
            // local field by replacing it with a shared Arc.
            winit_state: subsys.shared_winit_state.clone(),
            paint: subsys.paint.clone(),
            handle: injection.handle,
        });
        Ok(Box::new(subsys))
    }
}
