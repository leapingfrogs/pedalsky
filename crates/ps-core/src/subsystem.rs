//! The `RenderSubsystem` trait, the `PassStage` ordering enum, and the
//! `RegisteredPass` value the trait returns. See plan §1.3.

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

/// Closure type executed once per frame for a registered pass. Boxed and
/// trait-objected so the runtime can store heterogeneous passes in a single
/// `Vec`.
pub type PassFn = dyn Fn(&mut wgpu::CommandEncoder, &RenderContext<'_>) + Send + Sync;

/// A single render-graph pass registered by a subsystem. A subsystem may
/// register many — see plan §1.3.
pub struct RegisteredPass {
    /// Debug-friendly pass name (used in tracing logs and wgpu labels).
    pub name: &'static str,
    /// Coarse ordering bucket.
    pub stage: PassStage,
    /// Closure executed once per frame with an active command encoder.
    pub run: Box<PassFn>,
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
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>);

    /// Render-graph passes this subsystem contributes. Called once at
    /// registration; the executor flattens and sorts these into the
    /// per-frame command sequence.
    fn register_passes(&self) -> Vec<RegisteredPass>;

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

    /// Whether the subsystem is currently active.
    fn enabled(&self) -> bool;

    /// Set the enabled flag. Disabled subsystems still receive `prepare()`
    /// calls (so they can keep their resources warm) but their passes are
    /// skipped by the render-graph executor.
    fn set_enabled(&mut self, enabled: bool);
}
