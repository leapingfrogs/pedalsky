//! PedalSky core: GPU primitives, configuration, world state, frame uniforms,
//! the [`RenderSubsystem`] trait and the `AppBuilder`/`App` runtime that wires
//! subsystems into a render graph.
//!
//! See `Pedalsky corrected implementation plan.md` at the workspace root for
//! the architectural specification.

#![deny(missing_docs)]

pub mod app;
pub mod astro;
pub mod camera;
pub mod config;
pub mod contexts;
pub mod framebuffer;
pub mod gpu;
pub mod hot_reload;
pub mod scene;
pub mod subsystem;
pub mod time;
pub mod weather;
pub mod world;

pub use app::{App, AppBuilder, AppError, SubsystemFactory};
pub use config::{Config, ConfigError};
pub use contexts::{
    AtmosphereLuts, FrameUniforms, FrameUniformsGpu, GpuContext, HdrFramebuffer, PrepareContext,
    RenderContext,
};
pub use framebuffer::HdrFramebufferImpl;
pub use hot_reload::{HotReload, WatchEvent, DEFAULT_DEBOUNCE};
pub use scene::{
    CloudLayer, CloudType, Clouds, CoverageGrid, Lightning, PrecipKind, Precipitation, Scene,
    SceneError, Surface, Wetness,
};
pub use subsystem::{PassStage, RegisteredPass, RenderSubsystem};
pub use weather::{AtmosphereParams, CloudLayerGpu, SurfaceParams, WeatherState, WeatherTextures};
pub use world::WorldState;
