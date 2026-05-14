//! Per-frame uniforms (bind group 0). Plan §4.1.
//!
//! [`FrameUniforms`] is the CPU-side bundle the host populates each frame
//! ([`crate::PrepareContext::frame_uniforms`]). [`FrameUniformsGpu`] is the
//! `#[repr(C)]` Pod mirror that lands in the uniform buffer at group 0.
//!
//! Layout is locked against the canonical WGSL declaration in
//! `shaders/common/uniforms.wgsl` by the std140 cross-check test in
//! `crates/ps-core/tests/uniform_layout.rs`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Frame-level uniforms (CPU side). Plan §4.1.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameUniforms {
    /// View matrix (world → camera).
    pub view: Mat4,
    /// Reverse-Z infinite-far perspective projection.
    pub proj: Mat4,
    /// `proj * view`.
    pub view_proj: Mat4,
    /// `(proj * view)⁻¹` — used by `compute_view_ray` in `shaders/common/math.wgsl`.
    pub inv_view_proj: Mat4,
    /// World-space camera position (`w` unused).
    pub camera_position_world: Vec4,
    /// World-space camera velocity in m/s (`w` unused). Used by Phase 8.2
    /// far-rain streaks to scroll by `wind − camera_velocity`. The host
    /// computes this as `(position_now − position_prev) / dt`.
    pub camera_velocity_world: Vec4,
    /// `(xyz)` = world-space sun direction (unit vector).
    /// `w` = sun angular radius in radians (Earth ≈ 0.27° = 4.71e-3 rad).
    pub sun_direction: Vec4,
    /// `(rgb)` = sun illuminance in cd/m²·sr proxy units.
    /// `w` = top-of-atmosphere illuminance in lux.
    pub sun_illuminance: Vec4,
    /// Phase 12.3 — lightning flash illumination uniform.
    /// `(rgb)` = current aggregated lightning emission in cd/m²·sr
    /// proxy units, sampled from all active strikes' two-pulse
    /// envelopes; `w` = a horizontal falloff radius in metres so
    /// the cloud shader can attenuate by distance from the bolt
    /// origin packed in `lightning_origin_world`.
    pub lightning_illuminance: Vec4,
    /// Phase 12.3 — world-space origin of the strongest currently
    /// active strike. `(xyz)` = position; `w` = unused. The cloud
    /// shader uses this to localise the flash so distant cumulus
    /// don't all light up uniformly when only one strike is firing.
    pub lightning_origin_world: Vec4,
    /// `(width, height, 1/width, 1/height)`.
    pub viewport_size: Vec4,
    /// Wall-clock seconds since application start.
    pub time_seconds: f32,
    /// Pause-aware accumulated simulated seconds.
    pub simulated_seconds: f32,
    /// Monotonic frame counter.
    pub frame_index: u32,
    /// Photographic EV at ISO 100.
    pub ev100: f32,
    /// Previous frame's `proj * view`. Used by the cloud TAA pass to
    /// reproject the previous frame's resolved cloud RT into the
    /// current frame's screen space. On the first frame the host
    /// should set this to the current frame's `view_proj` so the
    /// reprojection is identity (no offset — correctly maps to
    /// "history sample = current pixel"). After each frame the host
    /// captures the current `view_proj` and stores it here for the
    /// next frame's prepare step.
    pub prev_view_proj: Mat4,
}

impl FrameUniforms {
    /// Set `view`, `proj`, `view_proj`, `inv_view_proj` from the four basic
    /// matrices. Convenience for hosts that only carry view + proj.
    ///
    /// Does **not** touch `prev_view_proj` — that's a stateful slot the
    /// host must manage explicitly across frames via
    /// [`FrameUniforms::shift_view_proj_history`]. Setting matrices for
    /// the first frame should be followed by a call to
    /// `shift_view_proj_history` to make the prev/current slots equal
    /// (so the TAA reprojection is identity on frame 0).
    pub fn set_matrices(&mut self, view: Mat4, proj: Mat4) {
        self.view = view;
        self.proj = proj;
        self.view_proj = proj * view;
        self.inv_view_proj = self.view_proj.inverse();
    }

    /// Latch the current `view_proj` into `prev_view_proj`. Call this at
    /// the **start** of each prepare, before recomputing `view_proj` for
    /// the new frame, so the previous-frame value is preserved for the
    /// cloud TAA pass. On the first frame the host should call
    /// [`Self::set_matrices`] first and then either call this method
    /// (which makes prev == current — identity reprojection) or set
    /// `prev_view_proj` directly to the same value.
    pub fn shift_view_proj_history(&mut self) {
        self.prev_view_proj = self.view_proj;
    }

    /// Set sun direction + angular radius (radians) from a unit world-space vector.
    pub fn set_sun(
        &mut self,
        direction_world: Vec3,
        angular_radius_rad: f32,
        illuminance_rgb: Vec3,
        toa_lux: f32,
    ) {
        self.sun_direction = Vec4::new(
            direction_world.x,
            direction_world.y,
            direction_world.z,
            angular_radius_rad,
        );
        self.sun_illuminance = Vec4::new(
            illuminance_rgb.x,
            illuminance_rgb.y,
            illuminance_rgb.z,
            toa_lux,
        );
    }
}

/// `#[repr(C)]` GPU-side mirror of [`FrameUniforms`]. Layout pinned by
/// `shaders/common/uniforms.wgsl`.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable)]
pub struct FrameUniformsGpu {
    /// View matrix (column-major).
    pub view: [[f32; 4]; 4],
    /// Reverse-Z infinite-far perspective projection.
    pub proj: [[f32; 4]; 4],
    /// `proj * view`.
    pub view_proj: [[f32; 4]; 4],
    /// `(proj * view)⁻¹`.
    pub inv_view_proj: [[f32; 4]; 4],
    /// World-space camera position; `w` unused.
    pub camera_position_world: [f32; 4],
    /// World-space camera velocity in m/s; `w` unused.
    pub camera_velocity_world: [f32; 4],
    /// `xyz` = sun direction; `w` = sun angular radius (radians).
    pub sun_direction: [f32; 4],
    /// `rgb` = sun illuminance proxy; `w` = TOA lux.
    pub sun_illuminance: [f32; 4],
    /// Phase 12.3 — `rgb` = lightning aggregated emission proxy;
    /// `w` = horizontal falloff radius (m).
    pub lightning_illuminance: [f32; 4],
    /// Phase 12.3 — `xyz` = strongest active strike origin
    /// (world-space); `w` unused.
    pub lightning_origin_world: [f32; 4],
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
    /// Previous frame's `proj * view`, for the cloud TAA pass'
    /// reprojection step. See [`FrameUniforms::prev_view_proj`] for
    /// the host-side lifecycle.
    pub prev_view_proj: [[f32; 4]; 4],
}

impl FrameUniformsGpu {
    /// Pack a CPU [`FrameUniforms`] into the GPU layout.
    pub fn from_cpu(u: &FrameUniforms) -> Self {
        Self {
            view: u.view.to_cols_array_2d(),
            proj: u.proj.to_cols_array_2d(),
            view_proj: u.view_proj.to_cols_array_2d(),
            inv_view_proj: u.inv_view_proj.to_cols_array_2d(),
            camera_position_world: u.camera_position_world.to_array(),
            camera_velocity_world: u.camera_velocity_world.to_array(),
            sun_direction: u.sun_direction.to_array(),
            sun_illuminance: u.sun_illuminance.to_array(),
            lightning_illuminance: u.lightning_illuminance.to_array(),
            lightning_origin_world: u.lightning_origin_world.to_array(),
            viewport_size: u.viewport_size.to_array(),
            time_seconds: u.time_seconds,
            simulated_seconds: u.simulated_seconds,
            frame_index: u.frame_index,
            ev100: u.ev100,
            prev_view_proj: u.prev_view_proj.to_cols_array_2d(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout pin: drift here forces a deliberate review against the WGSL
    /// declaration in `shaders/common/uniforms.wgsl`.
    #[test]
    fn frame_uniforms_size_pinned() {
        // 4 mat4 (4×64=256) + 7 vec4 (7×16=112) + 4 scalars (16) = 384 bytes
        // baseline; +1 mat4 (64) for the cloud-TAA `prev_view_proj` slot
        // takes us to 448.
        assert_eq!(std::mem::size_of::<FrameUniformsGpu>(), 448);
        assert_eq!(std::mem::size_of::<FrameUniformsGpu>() % 16, 0);
    }

    #[test]
    fn from_cpu_populates_inv_view_proj() {
        let mut u = FrameUniforms::default();
        let view = Mat4::IDENTITY;
        let proj = Mat4::perspective_infinite_reverse_rh(60_f32.to_radians(), 16.0 / 9.0, 0.1);
        u.set_matrices(view, proj);
        let g = FrameUniformsGpu::from_cpu(&u);
        // inv_view_proj * view_proj ≈ identity.
        let inv = Mat4::from_cols_array_2d(&g.inv_view_proj);
        let vp = Mat4::from_cols_array_2d(&g.view_proj);
        let prod = inv * vp;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = prod.col(i)[j];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "[{i},{j}] = {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn set_sun_packs_w_components() {
        let mut u = FrameUniforms::default();
        u.set_sun(
            Vec3::new(0.0, 1.0, 0.0),
            0.00471, // ~0.27°
            Vec3::new(120_000.0, 110_000.0, 100_000.0),
            127_500.0,
        );
        assert_eq!(u.sun_direction.w, 0.00471);
        assert_eq!(u.sun_illuminance.w, 127_500.0);
    }
}
