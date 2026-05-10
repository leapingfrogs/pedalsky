//! PedalSky atmosphere subsystem (Phase 5 stub).
//!
//! Phase 5 implements Hillaire 2020 sky + atmosphere with transmittance,
//! multi-scatter, sky-view, and aerial-perspective LUTs. Until then this
//! is a no-op `RenderSubsystem` so the workspace layout in plan §0.1 is
//! satisfied and `AppBuilder` has a factory it can wire when
//! `[render.subsystems].atmosphere = true`.

#![deny(missing_docs)]

use ps_core::{
    Config, GpuContext, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};

/// Stable subsystem name (matches `[render.subsystems].atmosphere`).
pub const NAME: &str = "atmosphere";

/// Phase 5 atmosphere subsystem. Currently a no-op — registers no passes.
pub struct AtmosphereSubsystem {
    enabled: bool,
}

impl AtmosphereSubsystem {
    /// Construct.
    pub fn new(_config: &Config, _gpu: &GpuContext) -> Self {
        Self { enabled: true }
    }
}

impl RenderSubsystem for AtmosphereSubsystem {
    fn name(&self) -> &'static str {
        "atmosphere"
    }
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}
    fn register_passes(&self) -> Vec<RegisteredPass> {
        Vec::new()
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`.
pub struct AtmosphereFactory;

impl SubsystemFactory for AtmosphereFactory {
    fn name(&self) -> &'static str {
        "atmosphere"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(AtmosphereSubsystem::new(config, gpu)))
    }
}
