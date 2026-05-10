// Phase 5.2.3 — Sky-view LUT bake.
//
// Output: 192 × 108 Rgba16Float.
//
// UV → view direction:
//   u = wrap((azimuth − sun_azimuth) / (2π))
//   v = 0.5 + 0.5 · sign(lat) · sqrt(|lat| / (π/2))
//
// The non-linear v concentrates samples around the horizon (where
// Rayleigh detail matters most). 32 march steps; samples transmittance
// + multi-scatter LUTs (so this is the integral of single-scatter +
// the closed-form multi-scatter contribution).
//
// Bindings:
//   group 0 binding 0 — FrameUniforms (provides sun_direction)
//   group 1 binding 0 — WorldUniforms
//   group 3 binding 0 — transmittance LUT
//   group 3 binding 1 — multi-scatter LUT
//   group 3 binding 4 — sampler
//   group 2 binding 0 — output sky-view LUT (storage)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const SIZE: vec2<u32> = vec2<u32>(192u, 108u);
const N_STEPS: u32 = 32u;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE.x || gid.y >= SIZE.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(SIZE);

    // V → latitude. Inverse of v = 0.5 + 0.5 · sign(lat) · sqrt(|lat|/(π/2)).
    let v_centred = uv.y * 2.0 - 1.0;
    let lat_sign = sign(v_centred);
    let lat_norm = v_centred * v_centred; // = |lat|/(π/2)
    let latitude = lat_sign * lat_norm * (PI * 0.5);

    // U → azimuth offset from the sun.
    let azimuth = uv.x * 2.0 * PI;

    // Build the view direction in atmosphere-local frame: latitude is
    // angle above the horizon plane (zenith=π/2, horizon=0), azimuth is
    // measured around the up-axis.
    let cos_lat = cos(latitude);
    let sin_lat = sin(latitude);
    let view_dir = vec3<f32>(cos_lat * sin(azimuth), sin_lat, cos_lat * cos(azimuth));

    // Camera position in atmosphere-local frame: just above the planet
    // surface for sky-view. The sky-view LUT is invariant in horizontal
    // translation so any longitude works; pick the camera's actual
    // altitude for accuracy.
    let p_world = frame.camera_position_world.xyz;
    var p = world_to_atmosphere_pos(p_world);
    // Avoid degenerate rays when the camera is below the planet surface.
    let r = length(p);
    if (r < world.planet_radius_m) {
        p = normalize(p) * world.planet_radius_m;
    }

    // Sun direction in atmosphere-local frame is the same as world-space
    // since translation doesn't change directions.
    let sun_dir = frame.sun_direction.xyz;
    let cos_theta = dot(view_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, world.mie_g);

    let t_max = distance_to_atmosphere_boundary(p, view_dir);
    if (t_max <= 0.0) {
        textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let dt = t_max / f32(N_STEPS);

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);

    for (var s = 0u; s < N_STEPS; s = s + 1u) {
        let t = (f32(s) + 0.5) * dt;
        let pi = p + view_dir * t;
        let h = length(pi) - world.planet_radius_m;
        let sigma_t = extinction_at(h);
        let sample_transmit = exp(-sigma_t * dt);

        // Direct light at sample point.
        let sun_visibility = sample_transmittance_lut(pi, sun_dir);
        let scat = scattering_pair(h);
        let sun_inscatter = (scat.rayleigh * phase_r + scat.mie * phase_m) * sun_visibility;

        // Multi-scatter contribution from precomputed LUT, sampled at
        // the local altitude and sun zenith cosine.
        let r_p = length(pi);
        let h_norm = clamp((r_p - world.planet_radius_m)
            / max(world.atmosphere_top_m - world.planet_radius_m, 1.0), 0.0, 1.0);
        let sun_cos = clamp(dot(pi / max(r_p, 1.0), sun_dir), -1.0, 1.0);
        let ms_uv = vec2<f32>(sun_cos * 0.5 + 0.5, h_norm);
        let l_ms = textureSampleLevel(multiscatter_lut, lut_sampler, ms_uv, 0.0).rgb;
        // L_ms is already a multi-scattered radiance (Hillaire 2020 §5.2);
        // weight by the local scattering coefficient (sum r+m).
        let multi = (scat.rayleigh + scat.mie) * l_ms;

        let in_scatter = sun_inscatter + multi;
        // Energy-conserving step.
        let safe_sigma_t = max(sigma_t, vec3<f32>(1e-7));
        let s_int = (in_scatter - in_scatter * sample_transmit) / safe_sigma_t;
        luminance = luminance + transmittance * s_int;
        transmittance = transmittance * sample_transmit;
    }

    // Multiply by sun illuminance — the LUT stores radiance per unit
    // solar irradiance, then sky raymarch scales by the actual sun.
    // To make the LUT directly usable without reading frame.sun_illuminance,
    // we bake the scaling here.
    luminance = luminance * frame.sun_illuminance.rgb;

    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(luminance, 1.0));
}
