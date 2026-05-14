//! The `RenderSubsystem` trait, the `PassStage` ordering enum, and the
//! lightweight `PassDescriptor` value the trait returns. See plan §1.3.

use crate::config::Config;
use crate::contexts::{GpuContext, PrepareContext, RenderContext};

/// Coarse render-graph ordering. Within a stage, registration order is
/// preserved.
///
/// Variants are listed in increasing dependency order: anything `Compute`
/// (LUT bakes etc.) runs before any `SkyBackdrop` pass; everything in
/// `Translucent` runs after every `Opaque` pass; `ToneMap` runs exactly once
/// after every `PostProcess` pass; `Overlay` runs last (post tone-map, into
/// the swapchain).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PassStage {
    /// Compute work (e.g. atmosphere LUT bakes) before any rasterisation.
    Compute,
    /// Sky / atmosphere fullscreen pass at far depth.
    SkyBackdrop,
    /// Depth-tested opaque geometry (ground, terrain).
    Opaque,
    /// Translucent volumes (clouds, precipitation).
    Translucent,
    /// Pre-tone-map post-process passes (bloom, tint demo).
    PostProcess,
    /// The single tone-mapping pass.
    ToneMap,
    /// Post-tone-map overlays (egui).
    Overlay,
}

/// Subsystem-local identifier for a single registered pass.
///
/// Subsystems pick their own values (typically a small enum cast to `u32`,
/// or sequential 0/1/2 if they don't need stable names). The executor
/// just round-trips it back to the subsystem inside `dispatch_pass`.
pub type PassId = u32;

/// Lightweight description of a render-graph pass. Returned once at
/// registration time; the executor stores it by value and dispatches
/// to the owning subsystem's [`RenderSubsystem::dispatch_pass`] when
/// the pass should run.
///
/// Replaces the previous `RegisteredPass { run: Box<dyn Fn...> }` model
/// (audit §2.4): with the closure gone, subsystems hold their own
/// per-frame mutable state as plain `&mut self` fields instead of
/// `Arc<Mutex<…>>` cells that only existed to satisfy the closure's
/// `Send + Sync + 'static` bounds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PassDescriptor {
    /// Debug-friendly pass name (used in tracing logs and wgpu labels).
    pub name: &'static str,
    /// Coarse ordering bucket.
    pub stage: PassStage,
    /// Subsystem-local identifier. Passed back to the owning subsystem
    /// via `dispatch_pass` so it knows which of its passes to run.
    pub id: PassId,
}

/// Trait implemented by every renderable subsystem (atmosphere, clouds,
/// ground, precipitation, wet surface, plus the Phase 1 demo subsystems).
///
/// Subsystems are constructed via a [`crate::app::SubsystemFactory`]; no
/// `Default::new()` constructors are permitted.
pub trait RenderSubsystem: Send + Sync {
    /// Stable subsystem identifier (matches the `[render.subsystems].<name>`
    /// flag in `pedalsky.toml`).
    fn name(&self) -> &'static str;

    /// Called once per frame, before any render passes. May write to GPU
    /// buffers and rebuild bind groups. Subsystems MUST rebuild bind groups
    /// here rather than caching across frames; another subsystem may have
    /// been recreated by hot-reload, invalidating cached references.
    ///
    /// **Phase 13.8 bind-group rebuild invariant.** Subsystems that bind
    /// shared atmosphere LUTs (ground, clouds, water, windsock) must not
    /// cache the bind group across `prepare()` calls — the LUT bundle
    /// gets re-published as a fresh `Arc<AtmosphereLuts>` whenever the
    /// atmosphere subsystem is toggled off and back on, and any cached
    /// reference would point at a dropped texture. Build the LUT bind
    /// group inside `dispatch_pass` (or each frame in `prepare`) from
    /// the live `PrepareContext::atmosphere_luts` / `RenderContext::
    /// luts_bind_group` reference.
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>);

    /// Render-graph passes this subsystem contributes. Called once at
    /// registration (and on reconfigure); the executor flattens, sorts
    /// by stage, and stores the result for the per-frame loop.
    fn register_passes(&self) -> Vec<PassDescriptor>;

    /// Execute a single pass previously announced by `register_passes`.
    /// `id` is the same `PassId` the subsystem chose when describing
    /// the pass, so a match on `id` selects the right code path.
    fn dispatch_pass(
        &mut self,
        id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    );

    /// Optional UI panel.
    ///
    /// Phase 1 stubs this — Phase 10 wires egui.
    #[allow(unused_variables)]
    fn ui(&mut self) {}

    /// Re-apply changed config values without dropping the subsystem.
    ///
    /// Default is a no-op; subsystems holding heavy GPU resources implement
    /// this to avoid full recreation when only tunable parameters change.
    /// If the change is structural (e.g. resolution), return an error and
    /// the runtime drops + recreates the subsystem via its factory.
    #[allow(unused_variables)]
    fn reconfigure(&mut self, config: &Config, gpu: &GpuContext) -> anyhow::Result<()> {
        Ok(())
    }
}
