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

/// Frame-level uniforms (CPU side) shared by every pass via bind group 0.
///
/// Phase 0 populates `view`/`proj`/`view_proj`/`viewport_size` for the ground
/// shader. TODO: Phase 4 will widen this with sun direction, illuminance,
/// frame index, EV100, etc., and lock down the std140 layout against
/// `FrameUniformsGpu` via a naga linter pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameUniforms {
    /// View matrix (world → camera).
    pub view: glam::Mat4,
    /// Reverse-Z infinite-far perspective projection.
    pub proj: glam::Mat4,
    /// `proj * view`.
    pub view_proj: glam::Mat4,
    /// World-space camera position.
    pub camera_position_world: glam::Vec3,
    /// `(width, height, 1/width, 1/height)`.
    pub viewport_size: glam::Vec4,
    /// Wall-clock seconds since application start.
    pub time_seconds: f32,
    /// Pause-aware accumulated simulated seconds.
    pub simulated_seconds: f32,
    /// Monotonic frame counter.
    pub frame_index: u32,
    /// Photographic EV at ISO 100.
    pub ev100: f32,
}

/// `#[repr(C)]` GPU-side mirror of [`FrameUniforms`].
///
/// This is the std140-compatible struct uploaded to bind group 0. Phase 4
/// owns this fully; Phase 0 lays it out with just enough to drive the
/// checker ground.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniformsGpu {
    /// View matrix (column-major).
    pub view: [[f32; 4]; 4],
    /// Reverse-Z infinite-far perspective projection.
    pub proj: [[f32; 4]; 4],
    /// `proj * view`.
    pub view_proj: [[f32; 4]; 4],
    /// `xyz` = world-space camera position; `w` unused.
    pub camera_position_world: [f32; 4],
    /// `(width, height, 1/width, 1/height)`.
    pub viewport_size: [f32; 4],
    /// Wall-clock seconds since application start.
    pub time_seconds: f32,
    /// Pause-aware accumulated simulated seconds.
    pub simulated_seconds: f32,
    /// Monotonic frame counter.
    pub frame_index: u32,
    /// Photographic EV at ISO 100.
    pub ev100: f32,
}

impl FrameUniformsGpu {
    /// Pack a CPU [`FrameUniforms`] into the GPU layout.
    pub fn from_cpu(u: &FrameUniforms) -> Self {
        Self {
            view: u.view.to_cols_array_2d(),
            proj: u.proj.to_cols_array_2d(),
            view_proj: u.view_proj.to_cols_array_2d(),
            camera_position_world: [
                u.camera_position_world.x,
                u.camera_position_world.y,
                u.camera_position_world.z,
                0.0,
            ],
            viewport_size: u.viewport_size.to_array(),
            time_seconds: u.time_seconds,
            simulated_seconds: u.simulated_seconds,
            frame_index: u.frame_index,
            ev100: u.ev100,
        }
    }
}

/// Synthesised weather state. Defined in [`crate::weather`]; re-exported
/// here so `PrepareContext` continues to refer to it from one place.
pub use crate::weather::WeatherState;
/// World state: clock, sun/moon position, observer location, TOA solar
/// illuminance. Defined in [`crate::world`]; re-exported here so
/// `PrepareContext` continues to refer to it from one place.
pub use crate::world::WorldState;

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
