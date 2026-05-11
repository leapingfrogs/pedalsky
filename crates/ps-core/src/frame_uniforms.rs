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
}

impl FrameUniforms {
    /// Set `view`, `proj`, `view_proj`, `inv_view_proj` from the four basic
    /// matrices. Convenience for hosts that only carry view + proj.
    pub fn set_matrices(&mut self, view: Mat4, proj: Mat4) {
        self.view = view;
        self.proj = proj;
        self.view_proj = proj * view;
        self.inv_view_proj = self.view_proj.inverse();
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
            inv_view_proj: u.inv_view_proj.to_cols_array_2d(),
            camera_position_world: u.camera_position_world.to_array(),
            camera_velocity_world: u.camera_velocity_world.to_array(),
            sun_direction: u.sun_direction.to_array(),
            sun_illuminance: u.sun_illuminance.to_array(),
            viewport_size: u.viewport_size.to_array(),
            time_seconds: u.time_seconds,
            simulated_seconds: u.simulated_seconds,
            frame_index: u.frame_index,
            ev100: u.ev100,
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
        // 4 mat4 (4×64=256) + 5 vec4 (5×16=80) + 4 scalars (16) = 352 bytes.
        assert_eq!(std::mem::size_of::<FrameUniformsGpu>(), 352);
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
