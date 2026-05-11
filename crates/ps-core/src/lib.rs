//! PedalSky core: GPU primitives, configuration, world state, frame uniforms,
//! the [`RenderSubsystem`] trait and the `AppBuilder`/`App` runtime that wires
//! subsystems into a render graph.
//!
//! See `Pedalsky corrected implementation plan.md` at the workspace root for
//! the architectural specification.

#![deny(missing_docs)]

pub mod app;
pub mod astro;
pub mod atmosphere_luts;
pub mod bind_groups;
pub mod camera;
pub mod config;
pub mod contexts;
pub mod frame_uniforms;
pub mod framebuffer;
pub mod gpu;
pub mod hot_reload;
pub mod scene;
pub mod shaders;
pub mod subsystem;
pub mod time;
pub mod weather;
pub mod world;

pub use app::{App, AppBuilder, AppError, SubsystemFactory};
pub use atmosphere_luts::{
    atmosphere_lut_bind_group_layout, atmosphere_static_only_bind_group,
    atmosphere_static_only_bind_group_layout, atmosphere_transmittance_only_bind_group,
    atmosphere_transmittance_only_bind_group_layout, AtmosphereLuts, AP_FAR_M, AP_NEAR_M,
    AP_SIZE,
};
pub use bind_groups::{frame_bind_group_layout, world_bind_group_layout, FrameWorldBindings};
pub use config::{Config, ConfigError};
pub use contexts::{
    FrameUniforms, FrameUniformsGpu, GpuContext, HdrFramebuffer, PrepareContext, RenderContext,
};
pub use framebuffer::HdrFramebufferImpl;
pub use hot_reload::{
    HotReload, ShaderHotReload, ShaderWatchEvent, WatchEvent, DEFAULT_DEBOUNCE,
};
pub use scene::{
    aurora_colour_bias, Aurora, CloudLayer, CloudType, Clouds, CoverageGrid, Lightning, PrecipKind,
    Precipitation, Scene, SceneError, Surface, Wetness,
};
pub use subsystem::{PassStage, RegisteredPass, RenderSubsystem};
pub use weather::{
    AtmosphereParams, CloudLayerGpu, SurfaceParams, WeatherState, WeatherTextures, WorldUniformsGpu,
};
pub use world::WorldState;
