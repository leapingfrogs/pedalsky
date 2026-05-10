//! PedalSky volumetric cloud subsystem (Phase 6 stub).
//!
//! Phase 6 implements Schneider Nubis (2015/2017/2022) + Hillaire 2016
//! energy-conserving integration with multi-octave multiple-scattering.
//! Until then this is a no-op `RenderSubsystem` so plan §0.1 is satisfied.

#![deny(missing_docs)]

use ps_core::{
    Config, GpuContext, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};

/// Stable subsystem name (matches `[render.subsystems].clouds`).
pub const NAME: &str = "clouds";

/// Phase 6 volumetric cloud subsystem. Currently a no-op — registers no passes.
pub struct CloudsSubsystem {
    enabled: bool,
}

impl CloudsSubsystem {
    /// Construct.
    pub fn new(_config: &Config, _gpu: &GpuContext) -> Self {
        Self { enabled: true }
    }
}

impl RenderSubsystem for CloudsSubsystem {
    fn name(&self) -> &'static str {
        "clouds"
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
pub struct CloudsFactory;

impl SubsystemFactory for CloudsFactory {
    fn name(&self) -> &'static str {
        "clouds"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(CloudsSubsystem::new(config, gpu)))
    }
}
