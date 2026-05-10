// Phase 5.2.2 — Multi-scattering LUT bake.
//
// Output: 32 × 32 Rgba16Float. UV maps to (sun_cos, h_normalised).
// For each texel, integrate over 8×8 sample directions on the sphere; for
// each direction march 20 steps along the view ray, accumulating the
// 2nd-order in-scattered radiance L_2 and the transfer factor f_ms.
// Closed-form geometric series gives the full multi-scatter contribution
// `L_full = L_2 · 1 / (1 − f_ms)` (Hillaire 2020 Eq 9).
//
// Bindings:
//   group 1 binding 0 — WorldUniforms
//   group 3 binding 0 — transmittance LUT (texture_2d)
//   group 3 binding 4 — sampler
//   group 2 binding 0 — output multi-scatter LUT (storage)

@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const SIZE: vec2<u32> = vec2<u32>(32u, 32u);
const N_DIR_SQRT: u32 = 8u;
const N_DIRS: u32 = 64u; // = 8 * 8
const N_STEPS: u32 = 20u;
// PI comes from common/atmosphere.wgsl.

/// Generate a Fibonacci-uniform direction on the sphere indexed by `i`.
fn sphere_sample(i: u32) -> vec3<f32> {
    let golden = 0.6180339887498949; // 1/phi
    let z = 1.0 - 2.0 * (f32(i) + 0.5) / f32(N_DIRS);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 2.0 * PI * fract(f32(i) * golden);
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE.x || gid.y >= SIZE.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(SIZE);

    // U = cos(sun zenith), V = normalised altitude.
    let sun_cos = clamp(uv.x * 2.0 - 1.0, -1.0, 1.0);
    let sun_sin = sqrt(max(1.0 - sun_cos * sun_cos, 0.0));
    let h_norm = uv.y;
    let r = world.planet_radius_m + h_norm * (world.atmosphere_top_m - world.planet_radius_m);
    let p0 = vec3<f32>(0.0, r, 0.0);
    let sun_dir = vec3<f32>(sun_sin, sun_cos, 0.0);

    var l_2nd_acc = vec3<f32>(0.0);
    var f_ms_acc = vec3<f32>(0.0);

    // p0 is on the planet's up axis at radius r.  Pre-compute for the
    // numerically-stable height inside the inner loop.
    let h0 = r - world.planet_radius_m;

    for (var d = 0u; d < N_DIRS; d = d + 1u) {
        let view_dir = sphere_sample(d);
        let t_max = distance_to_atmosphere_boundary(p0, view_dir);
        if (t_max <= 0.0) { continue; }
        let dt = t_max / f32(N_STEPS);
        let cos_view = dot(p0 / max(r, 1.0), view_dir);
        var transmittance = vec3<f32>(1.0);
        var l_dir = vec3<f32>(0.0);
        var f_dir = vec3<f32>(0.0);
        for (var s = 0u; s < N_STEPS; s = s + 1u) {
            let t = (f32(s) + 0.5) * dt;
            let p = p0 + view_dir * t;
            // Stable height: avoid `length(p) - planet_radius` cancellation.
            // |pi|² = r0² + 2t·r0·cos_view + t², so
            // |pi| − r0 = (2t·r0·cos_view + t²) / (r0 + |pi|).
            let r_delta_num = 2.0 * t * r * cos_view + t * t;
            let r_delta = r_delta_num / (max(r, 1.0) + length(p));
            let h = h0 + r_delta;
            let sigma_t = extinction_at(h);
            let sigma_s = scattering_at(h);
            let sample_transmit = exp(-sigma_t * dt);

            // Sun visibility through transmittance LUT.
            let r_p = r + r_delta;
            let cos_sun = dot(p, sun_dir) / max(r_p, 1.0);
            // Avoid sampling the LUT when the sun is below the horizon
            // (occluded by the planet).
            var sun_visibility = vec3<f32>(0.0);
            if (cos_sun > -sqrt(max(1.0 - (world.planet_radius_m / r_p) * (world.planet_radius_m / r_p), 0.0))) {
                sun_visibility = sample_transmittance_lut(p, sun_dir);
            }

            // Single scatter contribution at this sample (isotropic phase
            // is folded in via 1/(4π) factor that cancels with the
            // sphere-integration normalisation `4π / N_DIRS`).
            let in_scatter = sigma_s * sun_visibility;
            // Use Hillaire's energy-conserving step integral:
            // S_int = (S - S * Tr) / σ_t.
            let safe_sigma_t = max(sigma_t, vec3<f32>(1e-7));
            let s_int = (in_scatter - in_scatter * sample_transmit) / safe_sigma_t;
            l_dir = l_dir + transmittance * s_int;

            // f_ms: the fraction of the sample's scattering that re-enters
            // the integration loop; same shape but with no sun visibility,
            // i.e. the response to a unit isotropic source.
            let in_iso = sigma_s;
            let s_int_iso = (in_iso - in_iso * sample_transmit) / safe_sigma_t;
            f_dir = f_dir + transmittance * s_int_iso;

            transmittance = transmittance * sample_transmit;
        }
        l_2nd_acc = l_2nd_acc + l_dir;
        f_ms_acc = f_ms_acc + f_dir;
    }

    // Sphere normalisation: 4π · (1 / N_DIRS).
    let norm = (4.0 * PI) / f32(N_DIRS);
    let l_2 = l_2nd_acc * norm * (1.0 / (4.0 * PI));
    let f_ms = f_ms_acc * norm * (1.0 / (4.0 * PI));

    // Closed-form geometric series. Clamp f_ms so we never divide by
    // a value ≥ 1 (would imply the atmosphere amplifies energy).
    let f_clamped = clamp(f_ms, vec3<f32>(0.0), vec3<f32>(0.95));
    let l_full = l_2 / (vec3<f32>(1.0) - f_clamped);

    textureStore(
        output,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(l_full, 1.0),
    );
}
