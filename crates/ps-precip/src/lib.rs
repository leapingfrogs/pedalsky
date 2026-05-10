//! PedalSky precipitation subsystem (Phase 8 stub).
//!
//! Phase 8 implements compute-driven near rain particles, three layered
//! screen-space far-rain streak textures, snow with terminal velocity ~1
//! m/s, and animated normal-map ripples on wet surfaces. Until then this
//! is a no-op `RenderSubsystem` so plan §0.1 is satisfied.

#![deny(missing_docs)]

use ps_core::{
    Config, GpuContext, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};

/// Stable subsystem name (matches `[render.subsystems].precipitation`).
pub const NAME: &str = "precipitation";

/// Phase 8 precipitation subsystem. Currently a no-op — registers no passes.
pub struct PrecipSubsystem {
    enabled: bool,
}

impl PrecipSubsystem {
    /// Construct.
    pub fn new(_config: &Config, _gpu: &GpuContext) -> Self {
        Self { enabled: true }
    }
}

impl RenderSubsystem for PrecipSubsystem {
    fn name(&self) -> &'static str {
        "precipitation"
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
pub struct PrecipFactory;

impl SubsystemFactory for PrecipFactory {
    fn name(&self) -> &'static str {
        "precipitation"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(PrecipSubsystem::new(config, gpu)))
    }
}
