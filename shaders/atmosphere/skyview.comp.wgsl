// Phase 5.2.3 — Sky-view LUT bake.
//
// Output: 192 × 108 Rgba16Float.
//
// UV → view direction (Hillaire 2020 reference parametrisation,
// `SkyViewLutParamsToUv` in sebh/UnrealEngineSkyAtmosphere):
//   u = wrap((azimuth − sun_azimuth) / (2π))
//   v ∈ [0,   0.5): above-horizon rays
//     coord = (uv.y − 0)/0.5      ∈ [0, 1)
//     vza   = coord² · ZenithHorizonAngle
//             where ZenithHorizonAngle = π − acos(R / r) is the angle
//             from local zenith down to the geometric horizon for a
//             viewer at radius r.  vza is the angle from zenith to the
//             ray (0 = straight up, ZenithHorizonAngle = horizon).
//   v ∈ [0.5, 1]: below-horizon rays
//     coord = (uv.y − 0.5)/0.5    ∈ [0, 1]
//     vza_below = coord² · (π − ZenithHorizonAngle)
//     vza       = ZenithHorizonAngle + vza_below
//
// The non-linear (quadratic) v spacing concentrates samples around the
// geometric horizon, where Rayleigh extinction changes fastest.  Note
// v=0.5 corresponds to the *camera-altitude* horizon, not the local
// horizon plane — this is the critical difference from a flat-Earth
// parametrisation.
//
// 32 march steps; samples transmittance + multi-scatter LUTs.
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

    // Camera position in atmosphere-local frame.  Compute the camera's
    // radius now since the horizon angle depends on it.
    let p_world = frame.camera_position_world.xyz;
    var p = world_to_atmosphere_pos(p_world);
    var r = length(p);
    // Guard against numerical badness at or below the surface.
    if (r < world.planet_radius_m + 1.0) {
        p = normalize(p) * (world.planet_radius_m + 1.0);
        r = world.planet_radius_m + 1.0;
    }

    // Reference parametrisation: v=0 at zenith, v=0.5 at the geometric
    // horizon (as seen from radius r), v=1 at nadir.  For a viewer at
    // radius r outside the planet of radius R, the tangent (horizon)
    // ray makes an angle `asin(R/r)` with the nadir; the angle from
    // zenith is `π − asin(R/r)`.  At r = R this is π/2 (horizontal);
    // from orbit it approaches π (horizon collapses to the nadir).
    let sin_horizon = clamp(world.planet_radius_m / r, 0.0, 1.0);
    let zenith_horizon_angle = PI - asin(sin_horizon);
    var vza: f32;
    if (uv.y < 0.5) {
        let coord = uv.y * 2.0;                 // 0..1
        vza = coord * coord * zenith_horizon_angle;
    } else {
        let coord = (uv.y - 0.5) * 2.0;         // 0..1
        let below_range = PI - zenith_horizon_angle;
        vza = zenith_horizon_angle + coord * coord * below_range;
    }
    // U is the azimuth offset *from the sun* (Hillaire 2020). The
    // stored view_dir's absolute azimuth is therefore the sun's
    // azimuth plus uv.x*2π, so the sky shader's `(view_az - sun_az)`
    // lookup recovers the same value.
    let sun_dir = frame.sun_direction.xyz;
    let sun_az = atan2(sun_dir.x, sun_dir.z);
    let azimuth = sun_az + uv.x * 2.0 * PI;

    // Build the view direction in atmosphere-local frame.  vza is the
    // angle from local zenith (+Y), azimuth is around the up axis.
    // For vza=0 we want view_dir = +Y; for vza=π/2 a horizontal ring;
    // for vza=π we want view_dir = −Y.
    let cos_v = cos(vza);
    let sin_v = sin(vza);
    let view_dir = vec3<f32>(sin_v * sin(azimuth), cos_v, sin_v * cos(azimuth));
    let cos_theta = dot(view_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, world.mie_g);

    let t_max = distance_to_atmosphere_boundary(p, view_dir);
    if (t_max <= 0.0) {
        textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let dt = t_max / f32(N_STEPS);

    // For the numerically-stable height calculation inside the march.
    let cos_view = dot(p / max(r, 1.0), view_dir);
    let h0 = r - world.planet_radius_m;

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);

    for (var s = 0u; s < N_STEPS; s = s + 1u) {
        let t = (f32(s) + 0.5) * dt;
        let pi = p + view_dir * t;
        // Stable height: avoid `length(pi) - planet_radius` cancellation.
        // |pi|² = r² + 2t·r·cos_view + t², so
        // |pi| − r = (2t·r·cos_view + t²) / (r + |pi|).
        let r_delta_num = 2.0 * t * r * cos_view + t * t;
        let r_delta = r_delta_num / (max(r, 1.0) + length(pi));
        let h = h0 + r_delta;
        let sigma_t = extinction_at(h);
        let sample_transmit = exp(-sigma_t * dt);

        // Direct light at sample point.
        let sun_visibility = sample_transmittance_lut(pi, sun_dir);
        let scat = scattering_pair(h);
        let sun_inscatter = (scat.rayleigh * phase_r + scat.mie * phase_m) * sun_visibility;

        // Multi-scatter contribution from precomputed LUT, sampled at
        // the local altitude and sun zenith cosine.
        let r_p = r + r_delta;
        let h_norm = clamp(h
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

    // Ground-bounce term (Hillaire 2020 §6). If the ray hits the planet
    // surface inside the atmosphere, add the sun-illuminated bounce
    // attenuated by ground albedo / π and the integrated transmittance
    // back to the camera.
    var t_planet: vec2<f32>;
    let hit_planet = ray_sphere_intersect_origin(p, view_dir, world.planet_radius_m, &t_planet);
    if (hit_planet && t_planet.x > 0.0) {
        // Sample point on the planet surface.
        let p_ground = p + view_dir * t_planet.x;
        let n_ground = normalize(p_ground);
        let sun_visibility_ground = sample_transmittance_lut(p_ground, sun_dir);
        let n_dot_l = max(dot(n_ground, sun_dir), 0.0);
        // Per-unit-illuminance bounce radiance reaching the camera.
        let bounce = world.ground_albedo.rgb * (1.0 / 3.14159265)
                   * n_dot_l * sun_visibility_ground;
        luminance = luminance + transmittance * bounce;
    }

    // Store per-unit-illuminance radiance.  The sky shader multiplies
    // by `frame.sun_illuminance.rgb` at sample time.  Keeping the LUT
    // in unit space means the f16 storage range comfortably fits
    // values for both daytime and twilight without saturating at the
    // f16 max (65504).
    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(luminance, 1.0));
}
