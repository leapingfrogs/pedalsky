// LUT sampling helpers for atmosphere shaders.
// Only included by shaders that bind `transmittance_lut`, optionally
// `multiscatter_lut` / `skyview_lut` / `aerial_perspective_lut`, and
// `lut_sampler`. See `atmosphere.wgsl` for the parametrisation function.

fn sample_transmittance_lut(p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let uv = transmittance_lut_uv(p, dir);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

/// Phase 13.1 — aerial-perspective LUT depth-axis parametrisation.
/// Mirrors the bake's exponential spacing
/// (`shaders/atmosphere/aerialperspective.comp.wgsl`) so consumers
/// can convert a world-space distance to the matching `w` UV
/// coordinate. Distances below `AP_NEAR_M` snap to slice 0; above
/// `AP_FAR_M` clamp to slice (Z-1).
const AP_LUT_NEAR_M: f32 = 50.0;
const AP_LUT_FAR_M:  f32 = 100000.0;

fn ap_depth_uv(d_world: f32) -> f32 {
    let d_safe = max(d_world, AP_LUT_NEAR_M);
    let z_norm = log(d_safe / AP_LUT_NEAR_M)
               / log(AP_LUT_FAR_M / AP_LUT_NEAR_M);
    return clamp(z_norm, 0.0, 1.0);
}
