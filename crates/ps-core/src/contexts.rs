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

/// Frame-level uniforms shared by every pass via bind group 0.
///
/// TODO: Phase 4 will populate this with view/projection matrices, sun direction,
/// illuminance, time, frame index, EV100, etc., and define the std140 layout.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameUniforms {
    /// Wall-clock seconds since application start.
    pub time_seconds: f32,
    /// Pause-aware accumulated simulated seconds.
    pub simulated_seconds: f32,
    /// Monotonic frame counter.
    pub frame_index: u32,
    /// Photographic EV at ISO 100.
    pub ev100: f32,
}

/// Synthesised weather state, GPU resources for clouds/precip/wet surface.
///
/// TODO: Phase 3 owns the real definition (atmosphere params, weather map,
/// cloud-layer storage buffer, wind field, surface params, etc.).
#[derive(Debug, Default, Clone, Copy)]
pub struct WeatherState;

/// World state: clock, sun/moon position, observer location.
///
/// TODO: Phase 2 owns the real definition.
#[derive(Debug, Default, Clone, Copy)]
pub struct WorldState;

/// Atmosphere LUTs (transmittance, multi-scatter, sky-view, aerial perspective)
/// shared via bind group 3.
///
/// TODO: Phase 5 owns the real definition.
#[derive(Debug, Default, Clone, Copy)]
pub struct AtmosphereLuts;

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
