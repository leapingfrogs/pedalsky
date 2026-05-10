//! Phase 6 cloud march uniform parameters.
//!
//! Mirrored on the GPU side by `cloud_uniforms.wgsl::CloudParams`.

use bytemuck::{Pod, Zeroable};

/// Maximum number of cloud layers handled per frame. Matches the WGSL
/// `MAX_CLOUD_LAYERS` constant in `cloud_uniforms.wgsl`.
pub const MAX_CLOUD_LAYERS: u32 = 8;

/// CPU-side mirror of the WGSL `CloudParams` struct.
///
/// 80 bytes; std140 boundary respected — every scalar is 4 B, the layout
/// is contiguous f32/u32 with 4 B alignment, and the trailing `_pad`
/// rounds the total to a 16 B multiple.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct CloudParamsGpu {
    /// Scattering coefficient (per metre).
    pub sigma_s: f32,
    /// Absorption coefficient (per metre).
    pub sigma_a: f32,
    /// Forward HG g.
    pub g_forward: f32,
    /// Backward HG g (negative).
    pub g_backward: f32,

    /// Dual-lobe blend factor.
    pub g_blend: f32,
    /// Detail erosion strength.
    pub detail_strength: f32,
    /// Curl perturbation strength.
    pub curl_strength: f32,
    /// Beer-Powder lerp factor.
    pub powder_strength: f32,

    /// Multi-scatter octave attenuation: energy.
    pub multi_scatter_a: f32,
    /// Multi-scatter octave attenuation: optical depth.
    pub multi_scatter_b: f32,
    /// Multi-scatter octave attenuation: phase anisotropy.
    pub multi_scatter_c: f32,
    /// Sky-ambient strength multiplier.
    pub ambient_strength: f32,

    /// Base shape sample period (metres).
    pub base_scale_m: f32,
    /// Detail sample period (metres).
    pub detail_scale_m: f32,
    /// Weather map period (metres).
    pub weather_scale_m: f32,
    /// Light-march steps to the sun.
    pub light_steps: u32,

    /// Primary cloud march steps.
    pub cloud_steps: u32,
    /// Multi-scatter octaves.
    pub multi_scatter_octaves: u32,
    /// Number of valid entries in the layer array.
    pub cloud_layer_count: u32,
    /// Padding to round struct to 16B boundary.
    pub _pad: u32,
}

impl Default for CloudParamsGpu {
    fn default() -> Self {
        Self {
            sigma_s: 0.04,
            sigma_a: 0.0,
            g_forward: 0.8,
            g_backward: -0.3,

            g_blend: 0.5,
            // Schneider 2015 quotes 0.35 but the canonical remap formula
            // wipes coverage<0.5 layers entirely. Keep low here so the
            // default cumulus layer is visible; UI will expose this and
            // power users will dial it up alongside coverage.
            detail_strength: 0.05,
            curl_strength: 0.1,
            powder_strength: 1.0,

            multi_scatter_a: 0.5,
            multi_scatter_b: 0.5,
            multi_scatter_c: 0.5,
            ambient_strength: 1.0,

            base_scale_m: 4500.0,
            detail_scale_m: 800.0,
            weather_scale_m: 32_000.0,
            light_steps: 6,

            cloud_steps: 192,
            multi_scatter_octaves: 4,
            cloud_layer_count: 0,
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::CloudLayerGpu;

    #[test]
    fn cloud_params_size_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<CloudParamsGpu>() % 16, 0);
        assert_eq!(std::mem::size_of::<CloudParamsGpu>(), 80);
    }

    #[test]
    fn cloud_layer_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<CloudLayerGpu>(), 32);
    }
}
