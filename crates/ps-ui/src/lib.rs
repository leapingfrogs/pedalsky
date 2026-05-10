//! PedalSky UI overlay (Phase 10 stub).
//!
//! Phase 10 wires egui-wgpu and exposes every tunable subsystem parameter
//! as a slider plus the world clock controls. Until then this is a no-op
//! `RenderSubsystem` so plan §0.1 is satisfied. There is intentionally no
//! `[render.subsystems].ui` flag — Phase 10 will decide whether to gate
//! through config or always-on.

#![deny(missing_docs)]

use ps_core::{PassStage, PrepareContext, RegisteredPass, RenderSubsystem};

/// Stable subsystem name.
pub const NAME: &str = "ui";

/// Phase 10 UI overlay subsystem. Currently a no-op; reserves
/// `PassStage::Overlay` for the egui draw.
pub struct UiSubsystem {
    enabled: bool,
}

impl UiSubsystem {
    /// Construct.
    pub fn new() -> Self {
        Self { enabled: false }
    }
}

impl Default for UiSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderSubsystem for UiSubsystem {
    fn name(&self) -> &'static str {
        "ui"
    }
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}
    fn register_passes(&self) -> Vec<RegisteredPass> {
        // Phase 10 will register an `Overlay`-stage pass; reserve the slot
        // by returning an empty Vec for now.
        let _ = PassStage::Overlay;
        Vec::new()
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
