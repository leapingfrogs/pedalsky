// LUT sampling helpers for atmosphere shaders.
// Only included by shaders that bind `transmittance_lut`, optionally
// `multiscatter_lut` / `skyview_lut` / `aerial_perspective_lut`, and
// `lut_sampler`. See `atmosphere.wgsl` for the parametrisation function.

fn sample_transmittance_lut(p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let uv = transmittance_lut_uv(p, dir);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}
