//! Phase 9.1 — register the tone-map (and optional auto-exposure)
//! pass into the unified render-graph executor at `PassStage::ToneMap`.
//!
//! Pattern: the host constructs `Tonemap` and `AutoExposure` (because
//! both bind to the HDR framebuffer, which is host-owned and resized
//! by the host on `WindowEvent::Resized`). It then hands `Arc`s of
//! both to a [`TonemapFactory`] via [`TonemapHandle::inject`]. The
//! factory builds a [`TonemapSubsystem`] that registers the pass; per-
//! frame state (EV100, tonemap mode, auto-exposure enabled) is updated
//! by the host via [`TonemapHandle::set_state`] before each frame.

use std::sync::{Arc, Mutex};

use ps_core::{
    Config, GpuContext, PassDescriptor, PassId, PassStage, PrepareContext, RenderContext,
    RenderSubsystem, SubsystemFactory,
};

use crate::{auto_exposure::AutoExposure, tonemap::Tonemap, tonemap::TonemapMode};

const PASS_TONEMAP: PassId = 0;

/// Per-frame state the host pushes into the subsystem.
#[derive(Debug, Clone, Copy)]
pub struct TonemapState {
    /// User-configured EV100 (used when auto-exposure is off).
    pub ev100: f32,
    /// Tone-map curve.
    pub mode: TonemapMode,
    /// `true` → dispatch the auto-exposure compute pass and use the
    /// previous frame's derived EV100. `false` → use `ev100`.
    pub auto_exposure_enabled: bool,
}

impl Default for TonemapState {
    fn default() -> Self {
        Self {
            ev100: 14.0,
            mode: TonemapMode::AcesFilmic,
            auto_exposure_enabled: false,
        }
    }
}

/// Shared inner state of the tonemap subsystem. Held behind an `Arc`
/// so both the executor (via `RenderSubsystem`) and the host (via
/// `TonemapHandle`) can mutate without exclusive ownership.
struct TonemapShared {
    tonemap: Arc<Tonemap>,
    auto_exposure: Arc<AutoExposure>,
    state: Mutex<TonemapState>,
    /// Cached EV100 derived by the previous frame's auto-exposure
    /// read-back. `None` until the first read completes.
    auto_ev100: Mutex<Option<f32>>,
}

/// Phase 9.1 tone-map subsystem.
pub struct TonemapSubsystem {
    inner: Arc<TonemapShared>,
}

impl TonemapSubsystem {
    fn new(
        tonemap: Arc<Tonemap>,
        auto_exposure: Arc<AutoExposure>,
        initial_state: TonemapState,
    ) -> Self {
        Self {
            inner: Arc::new(TonemapShared {
                tonemap,
                auto_exposure,
                state: Mutex::new(initial_state),
                auto_ev100: Mutex::new(None),
            }),
        }
    }
}

impl RenderSubsystem for TonemapSubsystem {
    fn name(&self) -> &'static str {
        "tonemap"
    }

    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![PassDescriptor {
            name: "tonemap",
            stage: PassStage::ToneMap,
            id: PASS_TONEMAP,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        let Some(target) = ctx.tonemap_target else {
            return;
        };
        let st = *self.inner.state.lock().expect("tonemap state lock");
        let ev = if st.auto_exposure_enabled {
            self.inner.auto_exposure.dispatch(encoder);
            self.inner
                .auto_ev100
                .lock()
                .expect("auto ev lock")
                .unwrap_or(st.ev100)
        } else {
            st.ev100
        };
        self.inner
            .tonemap
            .render(encoder, ctx.queue, target, ev, st.mode);
    }
}

/// Side-channel handle for host-side state updates.
///
/// Returned by [`TonemapFactory::new`] paired with the factory itself.
/// The host calls [`Self::inject`] before `AppBuilder::build`, then
/// uses [`Self::set_state`] each frame and [`Self::drain_auto_exposure`]
/// after `queue.submit`.
#[derive(Clone)]
pub struct TonemapHandle {
    inject_cell: Arc<Mutex<Option<TonemapInjection>>>,
    /// Populated by the factory's `build()` after construction.
    inner: Arc<Mutex<Option<Arc<TonemapShared>>>>,
}

struct TonemapInjection {
    tonemap: Arc<Tonemap>,
    auto_exposure: Arc<AutoExposure>,
    initial_state: TonemapState,
}

impl TonemapHandle {
    /// Provide the host's `Tonemap` + `AutoExposure` + initial state.
    /// Must be called before `AppBuilder::build`.
    pub fn inject(
        &self,
        tonemap: Arc<Tonemap>,
        auto_exposure: Arc<AutoExposure>,
        initial_state: TonemapState,
    ) {
        *self.inject_cell.lock().expect("tonemap inject lock") = Some(TonemapInjection {
            tonemap,
            auto_exposure,
            initial_state,
        });
    }

    /// Update the per-frame state. Cheap (mutex lock + small struct copy).
    pub fn set_state(&self, state: TonemapState) {
        if let Some(inner) = self.inner.lock().expect("tonemap inner lock").as_ref() {
            *inner.state.lock().expect("tonemap state lock") = state;
        }
    }

    /// Drain the staging buffer. Call after `queue.submit(...)` so the
    /// previous frame's auto-exposure result is available for the next
    /// frame's tonemap. Blocks on `device.poll(WaitIndefinitely)`. No-op
    /// when auto-exposure is off.
    pub fn drain_auto_exposure(&self, gpu: &GpuContext) {
        let Some(inner) = self.inner.lock().expect("tonemap inner lock").clone() else {
            return;
        };
        let enabled = inner
            .state
            .lock()
            .expect("tonemap state lock")
            .auto_exposure_enabled;
        if !enabled {
            return;
        }
        if let Some(ev) = inner.auto_exposure.read_back_ev100(&gpu.device) {
            *inner.auto_ev100.lock().expect("auto ev lock") = Some(ev);
        }
    }
}

/// Factory wired by `AppBuilder`. Pair with [`TonemapHandle`] via
/// [`TonemapFactory::new`].
pub struct TonemapFactory {
    inject_cell: Arc<Mutex<Option<TonemapInjection>>>,
    inner_cell: Arc<Mutex<Option<Arc<TonemapShared>>>>,
}

impl TonemapFactory {
    /// Construct a paired (factory, handle).
    pub fn new() -> (Self, TonemapHandle) {
        let inject_cell = Arc::new(Mutex::new(None));
        let inner_cell = Arc::new(Mutex::new(None));
        (
            Self {
                inject_cell: inject_cell.clone(),
                inner_cell: inner_cell.clone(),
            },
            TonemapHandle {
                inject_cell,
                inner: inner_cell,
            },
        )
    }
}

impl SubsystemFactory for TonemapFactory {
    fn name(&self) -> &'static str {
        "tonemap"
    }

    /// Tonemap is mandatory per plan §9.1; ignore any (absent)
    /// `[render.subsystems].tonemap` flag.
    fn enabled(&self, _config: &Config) -> bool {
        true
    }

    fn build(
        &self,
        _config: &Config,
        _gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        let injection = self
            .inject_cell
            .lock()
            .expect("tonemap inject lock")
            .take()
            .ok_or_else(|| {
                anyhow::anyhow!("TonemapFactory: nothing injected before AppBuilder::build")
            })?;
        let subsys = TonemapSubsystem::new(
            injection.tonemap,
            injection.auto_exposure,
            injection.initial_state,
        );
        // Publish the inner Arc so the host's TonemapHandle can drive
        // per-frame state without re-fetching the subsystem from `App`.
        *self.inner_cell.lock().expect("tonemap inner lock") = Some(subsys.inner.clone());
        Ok(Box::new(subsys))
    }
}
