//! Parameter struct for the erosion augment.
//!
//! Field names + defaults track Section 1 of
//! `docs/pedalback_terrain_pipeline_spec.md`. Tuning notes live with
//! each field's doc-comment so the UI can surface them as tooltips.

use bytemuck::{Pod, Zeroable};

/// All Section 1 parameters. Defaults match the spec; the UI exposes
/// the "Parameter summary" subset prominently and the rest behind an
/// Advanced collapse.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ErosionParams {
    // --- Stage 1.1 — upsample ----------------------------------------
    /// Working resolution in metres per cell after Stage 1.1 upsample.
    /// `0.0` keeps the source resolution; otherwise the source is
    /// bicubically resampled to (source_dim * source_gsd /
    /// target_resolution_m).
    pub target_resolution_m: f32,

    /// Hard cap on the working-grid side length to keep the compute
    /// passes within wgpu's texture-size limit and within reasonable
    /// runtime budgets. 2048 is the universal wgpu downlevel minimum;
    /// raise on adapters with 8k+ texture support if you want sharper
    /// detail.
    pub max_working_dim: u32,

    // --- Stage 1.2 — hydraulic erosion -------------------------------
    /// Number of full hydraulic erosion steps. 50 = preview; 200 =
    /// standard; 500+ approaches steady state.
    pub iterations: u32,
    /// Time step in seconds. Smaller is more stable but slower.
    /// Subject to a CFL condition; halve if oscillation / NaNs appear.
    pub dt: f32,
    /// Metres of rain added per cell per second of simulated time.
    pub rainfall_rate: f32,
    /// Water column fraction lost per second.
    pub evaporation_rate: f32,
    /// Virtual pipe cross-section in m². Scales flux response to
    /// height differences.
    pub pipe_cross_section: f32,
    /// Virtual pipe length, normally equal to `target_resolution_m`.
    pub pipe_length: f32,
    /// Standard gravity.
    pub gravity: f32,
    /// **Highest-impact parameter for visual character.** Sediment
    /// holding capacity per (velocity × slope × depth) unit.
    pub sediment_capacity_constant: f32,
    /// Rate at which sediment is picked up from the bed when capacity
    /// exceeds current load.
    pub dissolution_rate: f32,
    /// Rate at which sediment is dropped when capacity falls below
    /// current load.
    pub deposition_rate: f32,
    /// Minimum effective slope to avoid zero-capacity pathologies.
    pub min_slope: f32,
    /// Water depth below which the capacity formula is attenuated.
    pub shallow_water_threshold: f32,

    // --- Stage 1.3 — thermal erosion ---------------------------------
    /// Angle of repose in degrees. Slopes steeper than this slough
    /// material to the lower side. 35° = scree / loose dirt;
    /// 45° = blockier / cohesive.
    pub talus_angle_degrees: f32,
    /// Fraction of "excess" material moved per thermal pass.
    pub thermal_erosion_rate: f32,
    /// Thermal passes per thermal cycle.
    pub thermal_iterations_per_cycle: u32,
    /// Hydraulic iterations between thermal cycles.
    pub hydraulic_iterations_between_thermal: u32,

    // --- Stage 1.4 — fractal detail ----------------------------------
    /// Maximum fractal detail amplitude in metres, before masking.
    /// `0.0` disables the stage.
    pub fractal_amplitude_m: f32,
    /// Frequency of the lowest octave in cycles per metre.
    pub fractal_base_frequency: f32,
    /// Number of frequency-doubled octaves.
    pub fractal_octaves: u32,
    /// Frequency multiplier per octave.
    pub fractal_lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub fractal_persistence: f32,
    /// If non-zero, transforms each octave with `1 - |noise|` to
    /// produce ridged features. Stored as `f32` so it round-trips
    /// through the GPU uniform cleanly.
    pub fractal_ridged: f32,
    /// Slope-mask strength. `1.0` = flat areas get zero fractal noise.
    pub slope_mask_strength: f32,
    /// Slope angle (degrees) below which detail is fully attenuated.
    pub slope_mask_threshold_degrees: f32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            target_resolution_m: 1.0,
            max_working_dim: 1024,

            iterations: 200,
            dt: 0.02,
            rainfall_rate: 0.012,
            evaporation_rate: 0.015,
            pipe_cross_section: 1.0,
            pipe_length: 1.0,
            gravity: 9.81,
            sediment_capacity_constant: 0.5,
            dissolution_rate: 0.5,
            deposition_rate: 1.0,
            min_slope: 0.01,
            shallow_water_threshold: 0.05,

            talus_angle_degrees: 35.0,
            thermal_erosion_rate: 0.3,
            thermal_iterations_per_cycle: 1,
            hydraulic_iterations_between_thermal: 10,

            fractal_amplitude_m: 0.4,
            fractal_base_frequency: 0.5,
            fractal_octaves: 5,
            fractal_lacunarity: 2.0,
            fractal_persistence: 0.5,
            fractal_ridged: 0.0,
            slope_mask_strength: 1.0,
            slope_mask_threshold_degrees: 5.0,
        }
    }
}

/// GPU uniform mirror of the hydraulic-erosion parameters used by the
/// Stage 1.2 compute shaders. The full `ErosionParams` carries
/// host-only fields (iterations counter, thermal cadence) that don't
/// belong in the per-iteration uniform.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub(super) struct HydraulicUniformGpu {
    pub cell_size: f32,
    pub dt: f32,
    pub rainfall_rate: f32,
    pub evaporation_rate: f32,

    pub pipe_cross_section: f32,
    pub pipe_length: f32,
    pub gravity: f32,
    pub sediment_capacity_constant: f32,

    pub dissolution_rate: f32,
    pub deposition_rate: f32,
    pub min_slope: f32,
    pub shallow_water_threshold: f32,
}

impl HydraulicUniformGpu {
    pub fn from_params(params: &ErosionParams, cell_size: f32) -> Self {
        Self {
            cell_size,
            dt: params.dt,
            rainfall_rate: params.rainfall_rate,
            evaporation_rate: params.evaporation_rate,
            pipe_cross_section: params.pipe_cross_section,
            pipe_length: params.pipe_length.max(0.0001),
            gravity: params.gravity,
            sediment_capacity_constant: params.sediment_capacity_constant,
            dissolution_rate: params.dissolution_rate,
            deposition_rate: params.deposition_rate,
            min_slope: params.min_slope,
            shallow_water_threshold: params.shallow_water_threshold,
        }
    }
}

/// GPU uniform for the thermal-erosion compute pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub(super) struct ThermalUniformGpu {
    pub cell_size: f32,
    /// Pre-computed `tan(talus_angle_degrees)`.
    pub talus_tan: f32,
    pub erosion_rate: f32,
    pub _pad: f32,
}

impl ThermalUniformGpu {
    pub fn from_params(params: &ErosionParams, cell_size: f32) -> Self {
        Self {
            cell_size,
            talus_tan: params.talus_angle_degrees.to_radians().tan(),
            erosion_rate: params.thermal_erosion_rate,
            _pad: 0.0,
        }
    }
}

/// GPU uniform for the fractal-detail compute pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub(super) struct FractalUniformGpu {
    pub cell_size: f32,
    pub amplitude_m: f32,
    pub base_frequency: f32,
    pub lacunarity: f32,

    pub persistence: f32,
    pub octaves: u32,
    pub ridged: f32,
    pub slope_mask_strength: f32,

    pub slope_mask_threshold_tan: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl FractalUniformGpu {
    pub fn from_params(params: &ErosionParams, cell_size: f32) -> Self {
        Self {
            cell_size,
            amplitude_m: params.fractal_amplitude_m,
            base_frequency: params.fractal_base_frequency,
            lacunarity: params.fractal_lacunarity,
            persistence: params.fractal_persistence,
            octaves: params.fractal_octaves.max(1),
            ridged: params.fractal_ridged,
            slope_mask_strength: params.slope_mask_strength,
            slope_mask_threshold_tan: params.slope_mask_threshold_degrees.to_radians().tan(),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

/// GPU uniform for the normal-map compute pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub(super) struct NormalMapUniformGpu {
    pub cell_size: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl NormalMapUniformGpu {
    pub fn from_cell_size(cell_size: f32) -> Self {
        Self {
            cell_size,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}
