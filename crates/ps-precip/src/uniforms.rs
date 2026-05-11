//! Phase 8 precipitation uniforms.
//!
//! Mirrored on the GPU side by `precip/particle_advance.comp.wgsl::PrecipUniforms`.

use bytemuck::{Pod, Zeroable};

/// CPU-side mirror of the WGSL `PrecipUniforms` struct.
///
/// 64 bytes; std140 boundary respected.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct PrecipUniformsGpu {
    /// Camera world position (xyz; w unused).
    pub camera_position: [f32; 4],
    /// Wind velocity at the camera in m/s (xyz; w = turbulence).
    pub wind_velocity: [f32; 4],
    /// Precipitation intensity in mm/hour.
    pub intensity_mm_per_h: f32,
    /// Frame delta in seconds (wall clock).
    pub dt_seconds: f32,
    /// Simulated seconds (for deterministic respawn seeds).
    pub simulated_seconds: f32,
    /// Particle kind (0 = rain, 1 = snow).
    pub kind: u32,
    /// Live particle count.
    pub particle_count: u32,
    /// Spawn cylinder radius in metres.
    pub spawn_radius_m: f32,
    /// Spawn cylinder top above the camera in metres.
    pub spawn_top_m: f32,
    /// Terminal fall speed in m/s.
    pub fall_speed_mps: f32,
    /// User-supplied determinism seed (plan §Cross-Cutting/Determinism).
    /// XOR'd into the per-particle respawn jitter so a fixed `--seed`
    /// argument produces bit-identical particle distributions across
    /// runs. Defaults to 0 (which still gives a deterministic but
    /// arbitrary distribution).
    pub user_seed: u32,
    /// _pad to bring the struct to a 16 B multiple.
    pub _pad: [f32; 3],
}

impl Default for PrecipUniformsGpu {
    fn default() -> Self {
        Self {
            camera_position: [0.0; 4],
            wind_velocity: [0.0; 4],
            intensity_mm_per_h: 0.0,
            dt_seconds: 1.0 / 60.0,
            simulated_seconds: 0.0,
            kind: 0,
            particle_count: 0,
            spawn_radius_m: 50.0,
            spawn_top_m: 30.0,
            fall_speed_mps: 6.0,
            user_seed: 0,
            _pad: [0.0; 3],
        }
    }
}

/// Per-far-rain-layer uniform mirror.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct FarRainLayerGpu {
    /// Virtual depth in metres.
    pub depth_m: f32,
    /// Approximate streak count per screen height.
    pub streak_density: f32,
    /// Streak length in pixels (rough; the shader normalises against
    /// the cell size).
    pub streak_length_px: f32,
    /// Per-layer alpha multiplier.
    pub intensity_scale: f32,
}
