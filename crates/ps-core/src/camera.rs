//! Right-handed reverse-Z fly camera. See plan §0.4.
//!
//! World coordinates: +X east, +Y up, +Z south (right-handed). Position is
//! stored in metres. `glam`'s `_rh` projection variants are used so the depth
//! convention matches the engine's reverse-Z.

use glam::{Mat4, Vec3};

/// Free-flying camera controlled by WASD + mouse + Space/Ctrl + Q/E.
#[derive(Debug, Clone)]
pub struct FlyCamera {
    /// World-space position in metres.
    pub position: Vec3,
    /// Pitch in radians.
    pub pitch: f32,
    /// Yaw in radians (0 looks down −Z, +90° looks +X).
    pub yaw: f32,
    /// Optional roll in radians (Q/E).
    pub roll: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near-plane distance in metres.
    pub near_m: f32,
    /// Movement speed in metres per second.
    pub speed_mps: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.7, 0.0),
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
            fov_y: 60_f32.to_radians(),
            near_m: 0.1,
            speed_mps: 5.0,
        }
    }
}

impl FlyCamera {
    /// Forward direction in world space (right-handed).
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, -cp * cy).normalize()
    }

    /// Right (camera +X) in world space.
    pub fn right(&self) -> Vec3 {
        let f = self.forward();
        f.cross(Vec3::Y).normalize_or_zero()
    }

    /// View matrix (world → camera).
    pub fn view_matrix(&self) -> Mat4 {
        let f = self.forward();
        let up = Vec3::Y;
        let mut view = Mat4::look_to_rh(self.position, f, up);
        if self.roll != 0.0 {
            view = Mat4::from_rotation_z(self.roll) * view;
        }
        view
    }

    /// Reverse-Z infinite-far perspective projection.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, self.near_m)
    }
}
