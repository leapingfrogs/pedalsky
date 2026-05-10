//! PedalSky volumetric cloud subsystem (Phase 6: Schneider/Hillaire).
//!
//! Implements Schneider Nubis (2015/2017/2022) base/detail/curl noise
//! volumes plus the Hillaire 2016 multi-octave multiple-scattering
//! approximation. Wired as a `Translucent`-stage `RenderSubsystem`
//! that composites premultiplied cloud luminance over the HDR target.

#![deny(missing_docs)]

pub mod noise;

use ps_core::{
    Config, GpuContext, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};

pub use noise::CloudNoise;

/// Stable subsystem name (matches `[render.subsystems].clouds`).
pub const NAME: &str = "clouds";

/// Phase 6 volumetric cloud subsystem. Currently noise volumes only;
/// march pipeline lands in Phase 6.6.
pub struct CloudsSubsystem {
    enabled: bool,
    #[allow(dead_code)]
    noise: CloudNoise,
}

impl CloudsSubsystem {
    /// Construct + bake noise volumes.
    pub fn new(_config: &Config, gpu: &GpuContext) -> Self {
        let noise = CloudNoise::bake(gpu);
        Self {
            enabled: true,
            noise,
        }
    }

    /// Reference to the baked noise textures (for diagnostic readback).
    pub fn noise(&self) -> &CloudNoise {
        &self.noise
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
