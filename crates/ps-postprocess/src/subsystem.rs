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

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ps_core::{
    Config, GpuContext, PassDescriptor, PassId, PassStage, PrepareContext, RenderContext,
    RenderSubsystem, SubsystemFactory,
};

use crate::{auto_exposure::AutoExposure, tonemap::Tonemap, tonemap::TonemapMode};

/// Sentinel bit-pattern for `auto_ev100_bits` meaning "no successful
/// readback yet". Using `u32::MAX` (the bit-pattern of a quiet NaN with
/// every payload bit set) means a real `f32` EV-100 will never collide.
const AUTO_EV100_NONE: u32 = u32::MAX;

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
///
/// Audit §M1 — `TonemapState` is decomposed into three atomics so the
/// per-pass read and the host-side `set_state` write are both
/// lock-free. Same for the auto-exposure EV100 cache (`AtomicU32`
/// bit-cast of the `f32`, with `u32::MAX` as the "None" sentinel).
struct TonemapShared {
    tonemap: Arc<Tonemap>,
    auto_exposure: Arc<AutoExposure>,
    /// `f32::to_bits()` of the user-configured EV-100.
    ev100_bits: AtomicU32,
    /// `TonemapMode::as_u32()` of the selected curve.
    mode: AtomicU32,
    /// Mirror of the host's auto-exposure toggle.
    auto_exposure_enabled: AtomicBool,
    /// Cached EV100 derived by the previous frame's auto-exposure
    /// read-back. `AUTO_EV100_NONE` (= `u32::MAX`) means "no value
    /// yet"; any other value is the bit-pattern of a valid `f32`.
    auto_ev100_bits: AtomicU32,
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
                ev100_bits: AtomicU32::new(initial_state.ev100.to_bits()),
                mode: AtomicU32::new(initial_state.mode.as_u32()),
                auto_exposure_enabled: AtomicBool::new(initial_state.auto_exposure_enabled),
                auto_ev100_bits: AtomicU32::new(AUTO_EV100_NONE),
            }),
        }
    }
}

/// Helper — turn `mode.as_u32()` back into a `TonemapMode`. Mirrors the
/// 1-to-1 mapping inside [`TonemapMode::as_u32`].
fn tonemap_mode_from_u32(v: u32) -> TonemapMode {
    match v {
        1 => TonemapMode::Passthrough,
        _ => TonemapMode::AcesFilmic,
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
        // Audit §M1 — lock-free reads via atomics.
        let inner = &self.inner;
        let user_ev = f32::from_bits(inner.ev100_bits.load(Ordering::Relaxed));
        let mode = tonemap_mode_from_u32(inner.mode.load(Ordering::Relaxed));
        let auto_enabled = inner.auto_exposure_enabled.load(Ordering::Relaxed);
        let ev = if auto_enabled {
            inner.auto_exposure.dispatch(encoder);
            let bits = inner.auto_ev100_bits.load(Ordering::Relaxed);
            if bits == AUTO_EV100_NONE {
                user_ev
            } else {
                f32::from_bits(bits)
            }
        } else {
            user_ev
        };
        inner.tonemap.render(encoder, ctx.queue, target, ev, mode);
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

    /// Update the per-frame state. Lock-free atomics after audit §M1.
    pub fn set_state(&self, state: TonemapState) {
        if let Some(inner) = self.inner.lock().expect("tonemap inner lock").as_ref() {
            inner.ev100_bits.store(state.ev100.to_bits(), Ordering::Relaxed);
            inner.mode.store(state.mode.as_u32(), Ordering::Relaxed);
            inner
                .auto_exposure_enabled
                .store(state.auto_exposure_enabled, Ordering::Relaxed);
        }
    }

    /// Drain the staging buffer. Call after `queue.submit(...)` so the
    /// previous frame's auto-exposure result is available for the next
    /// frame's tonemap. No-op when auto-exposure is off.
    pub fn drain_auto_exposure(&self, gpu: &GpuContext) {
        let Some(inner) = self.inner.lock().expect("tonemap inner lock").clone() else {
            return;
        };
        if !inner.auto_exposure_enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(ev) = inner.auto_exposure.read_back_ev100(&gpu.device) {
            inner.auto_ev100_bits.store(ev.to_bits(), Ordering::Relaxed);
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
