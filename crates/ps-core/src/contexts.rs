//! Per-frame contexts (`PrepareContext`, `RenderContext`) passed to each
//! [`crate::RenderSubsystem`] and the placeholder types they reference.
//!
//! Per the plan §1.4, both contexts borrow from long-lived state. Subsystems
//! must not retain any borrowed reference past the call that produced it.
//!
//! Several types here are intentionally placeholders that will gain real
//! fields in later phases. They exist now so Phase 1 can lock the trait
//! shape without depending on incomplete subsystems.

pub use crate::framebuffer::HdrFramebufferImpl as HdrFramebuffer;
pub use crate::gpu::GpuContext;

/// Frame-level uniforms. Defined in [`crate::frame_uniforms`]; re-exported
/// here so `PrepareContext` continues to refer to them from one place.
pub use crate::frame_uniforms::{FrameUniforms, FrameUniformsGpu};

/// Synthesised weather state. Defined in [`crate::weather`]; re-exported
/// here so `PrepareContext` continues to refer to it from one place.
pub use crate::weather::WeatherState;
/// World state: clock, sun/moon position, observer location, TOA solar
/// illuminance. Defined in [`crate::world`]; re-exported here so
/// `PrepareContext` continues to refer to it from one place.
pub use crate::world::WorldState;

/// Atmosphere LUTs. Defined in [`crate::atmosphere_luts`]; re-exported
/// here so `PrepareContext` continues to refer to it from one place.
pub use crate::atmosphere_luts::AtmosphereLuts;

/// Per-frame context handed to each subsystem's `prepare()` call.
///
/// Borrows are valid only for the duration of `prepare()`. Subsystems must
/// rebuild bind groups every frame rather than caching them — another
/// subsystem may have been recreated by hot-reload, invalidating cached
/// references.
pub struct PrepareContext<'a> {
    /// Shared `wgpu::Device`.
    pub device: &'a wgpu::Device,
    /// Shared `wgpu::Queue`.
    pub queue: &'a wgpu::Queue,
    /// World clock + sun/moon state.
    pub world: &'a WorldState,
    /// Synthesised weather resources.
    pub weather: &'a WeatherState,
    /// Per-frame uniforms (matrices, sun, exposure, time).
    pub frame_uniforms: &'a FrameUniforms,
    /// Atmosphere LUTs (`Some` once Phase 5 lands; `None` until then).
    pub atmosphere_luts: Option<&'a AtmosphereLuts>,
    /// Frame delta in seconds (wall clock).
    pub dt_seconds: f32,
}

/// Per-pass context handed to each `RegisteredPass::run` closure.
///
/// `frame_bind_group` (group 0) and `world_bind_group` (group 1) are
/// presented separately so a subsystem that does not need world data can
/// skip binding it. `luts_bind_group` is `Some` once Phase 5 has built
/// its LUTs.
pub struct RenderContext<'a> {
    /// Shared `wgpu::Device`.
    pub device: &'a wgpu::Device,
    /// Shared `wgpu::Queue`.
    pub queue: &'a wgpu::Queue,
    /// HDR target this frame.
    pub framebuffer: &'a HdrFramebuffer,
    /// Bind group 0: `FrameUniforms`.
    pub frame_bind_group: &'a wgpu::BindGroup,
    /// Bind group 1: `WorldUniforms`.
    pub world_bind_group: &'a wgpu::BindGroup,
    /// Bind group 3: shared atmosphere LUTs (`None` until Phase 5).
    pub luts_bind_group: Option<&'a wgpu::BindGroup>,
    /// Same uniforms passed to `prepare()`, for closures that need scalars.
    pub frame_uniforms: &'a FrameUniforms,
}
