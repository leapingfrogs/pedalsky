// Phase 13.5 — water surface.
//
// A flat rectangular plane at `params.altitude_m`, rendered with GGX
// specular and a Fresnel-weighted sky reflection sampled from the
// sky-view LUT at the reflected view direction. Normals are
// perturbed by a tiny 2D value-noise field advected with the surface
// wind, so the highlight breaks up at wind-driven scale. No
// refraction (the v1 plan has no DEM, so there's nothing under the
// water to refract toward).
//
// Bindings:
//   group 0 binding 0      FrameUniforms
//   group 1 binding 0      WorldUniforms (planet radius for the sky-view
//                                          LUT cam_r reconstruction)
//   group 2 binding 0      WaterParams uniform (this crate)
//   group 3 binding {0..4} atmosphere LUTs

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;

struct WaterParams {
    // x = xmin, y = xmax, z = zmin, w = zmax — world bounds.
    bounds: vec4<f32>,
    // x = altitude (world Y), y = roughness_min, z = roughness_max,
    // w = simulated time (seconds).
    config: vec4<f32>,
    // x = wind_dir_deg (meteorological — direction wind comes from),
    // y = wind_speed_mps, zw unused.
    wind: vec4<f32>,
};

@group(2) @binding(0) var<uniform> params: WaterParams;

@group(3) @binding(0) var transmittance_lut:      texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut:       texture_2d<f32>;
@group(3) @binding(2) var skyview_lut:            texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler:            sampler;

const WPI: f32 = 3.14159265358979;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos_xz: vec2<f32>) -> VsOut {
    // `pos_xz` lives in normalised [0,1]² (the mesh is a unit grid).
    // Remap into the world bounds at the configured altitude.
    let xmin = params.bounds.x;
    let xmax = params.bounds.y;
    let zmin = params.bounds.z;
    let zmax = params.bounds.w;
    let x = mix(xmin, xmax, pos_xz.x);
    let z = mix(zmin, zmax, pos_xz.y);
    let y = params.config.x;
    var out: VsOut;
    out.world_pos = vec3<f32>(x, y, z);
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// 2D value noise (small, deterministic)
// ---------------------------------------------------------------------------

fn hash22(p: vec2<f32>) -> vec2<f32> {
    // Cheap deterministic hash → vec2 in [0,1].
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let pi = floor(p);
    let pf = p - pi;
    let u = pf * pf * (3.0 - 2.0 * pf);
    let a = hash22(pi).x;
    let b = hash22(pi + vec2<f32>(1.0, 0.0)).x;
    let c = hash22(pi + vec2<f32>(0.0, 1.0)).x;
    let d = hash22(pi + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Wave normal: gradient of two octaves of value noise advected by
// wind. Sampling distance scales with the wind speed so calm water
// reads as glass-flat (frequency drops to near zero); windy water
// shows visibly broken specular.
fn wave_normal(p_xz: vec2<f32>, time_s: f32) -> vec3<f32> {
    let wind_dir_deg = params.wind.x;
    let wind_speed = params.wind.y;
    let theta = wind_dir_deg * (WPI / 180.0);
    // Wind direction is meteorological (FROM); waves travel TO. Drift
    // velocity is opposite to wind_from.
    let drift = vec2<f32>(sin(theta + WPI), -cos(theta + WPI));

    let amp = clamp(wind_speed / 12.0, 0.05, 1.0);

    // Octave 1 — coarse waves at ~1.5 m spatial scale.
    let f1 = 0.7;
    let v1 = drift * (time_s * 0.4);
    let n1 = value_noise(p_xz * f1 + v1);

    // Octave 2 — finer ripples at ~30 cm.
    let f2 = 4.0;
    let v2 = drift * (time_s * 0.9);
    let n2 = value_noise(p_xz * f2 + v2);

    // Compute finite-difference gradients per octave.
    let eps = 0.05;
    let dx1 = value_noise((p_xz + vec2<f32>(eps, 0.0)) * f1 + v1)
            - value_noise((p_xz - vec2<f32>(eps, 0.0)) * f1 + v1);
    let dy1 = value_noise((p_xz + vec2<f32>(0.0, eps)) * f1 + v1)
            - value_noise((p_xz - vec2<f32>(0.0, eps)) * f1 + v1);
    let dx2 = value_noise((p_xz + vec2<f32>(eps, 0.0)) * f2 + v2)
            - value_noise((p_xz - vec2<f32>(eps, 0.0)) * f2 + v2);
    let dy2 = value_noise((p_xz + vec2<f32>(0.0, eps)) * f2 + v2)
            - value_noise((p_xz - vec2<f32>(0.0, eps)) * f2 + v2);

    let slope_scale = 0.6 * amp;
    let dx = (dx1 + 0.35 * dx2) * slope_scale;
    let dz = (dy1 + 0.35 * dy2) * slope_scale;

    // Phony assignment to keep the noise samples in scope for the
    // optimiser (without forcing the compiler to drop the texture
    // reads as dead code on backends that don't constant-fold them).
    _ = n1;
    _ = n2;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

// ---------------------------------------------------------------------------
// Sky-view LUT sampling (mirrors the ground/cloud helper).
// ---------------------------------------------------------------------------

fn skyview_uv_local(view_dir: vec3<f32>, sun_dir: vec3<f32>, cam_r: f32) -> vec2<f32> {
    let cos_v = clamp(view_dir.y, -1.0, 1.0);
    let vza = acos(cos_v);
    let sin_horizon = clamp(world.planet_radius_m / cam_r, 0.0, 1.0);
    let zenith_horizon_angle = WPI - asin(sin_horizon);
    var v: f32;
    if (vza < zenith_horizon_angle) {
        let coord = sqrt(clamp(vza / max(zenith_horizon_angle, 1e-6), 0.0, 1.0));
        v = coord * 0.5;
    } else {
        let below_range = max(WPI - zenith_horizon_angle, 1e-6);
        let coord = sqrt(clamp((vza - zenith_horizon_angle) / below_range, 0.0, 1.0));
        v = 0.5 + coord * 0.5;
    }
    let view_az = atan2(view_dir.x, view_dir.z);
    let sun_az  = atan2(sun_dir.x, sun_dir.z);
    var du = (view_az - sun_az) / (2.0 * WPI);
    du = du - floor(du);
    return vec2<f32>(du, v);
}

fn sample_sky_reflection(reflected: vec3<f32>, sun_dir: vec3<f32>, p_world: vec3<f32>) -> vec3<f32> {
    // Build cam_r from the surface point as if it were the observer.
    let centre = vec3<f32>(0.0, -world.planet_radius_m, 0.0);
    let cam_r = length(p_world - centre);
    let uv = skyview_uv_local(reflected, sun_dir, cam_r);
    return textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb
         * frame.sun_illuminance.rgb;
}

fn ap_depth_uv(d_world: f32) -> f32 {
    let d_safe = max(d_world, 50.0);
    let z_norm = log(d_safe / 50.0) / log(100000.0 / 50.0);
    return clamp(z_norm, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// GGX/Smith specular
// ---------------------------------------------------------------------------

fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(WPI * denom * denom, 1e-6);
}

fn smith_g1(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 1e-6);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    let x5 = x * x * x * x * x;
    return f0 + (vec3<f32>(1.0) - f0) * x5;
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos;
    let v = normalize(frame.camera_position_world.xyz - p);
    let sun_dir = normalize(frame.sun_direction.xyz);

    // Wind-driven normal map.
    let n = wave_normal(p.xz, frame.simulated_seconds);

    // Roughness interpolates between the configured min/max by wind speed.
    let speed_norm = clamp(params.wind.y / 12.0, 0.0, 1.0);
    let rough = mix(params.config.y, params.config.z, speed_norm);

    // Fresnel-weighted sky reflection.
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0_water = vec3<f32>(0.02);   // n=1.33 dielectric.
    let f_at_view = fresnel_schlick(n_dot_v, f0_water);

    // Reflected ray direction (about n).
    var reflected = reflect(-v, n);
    // Clamp the reflected ray to the upper hemisphere so we don't
    // sample the bottom of the sky-view (which represents the ground
    // viewed from a point above the surface). For a tilted normal
    // the reflection can occasionally dip below horizontal; nudge it
    // back upward.
    if (reflected.y < 0.02) {
        reflected.y = 0.02;
        reflected = normalize(reflected);
    }
    let sky_rgb = sample_sky_reflection(reflected, sun_dir, p);
    let reflection = sky_rgb;

    // GGX direct sun specular.
    let h = normalize(v + sun_dir);
    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let d = ggx_d(n_dot_h, rough);
    let g = smith_g1(n_dot_v, rough) * smith_g1(max(n_dot_l, 1e-4), rough);
    let f_sun = fresnel_schlick(v_dot_h, f0_water);
    let specular_sun = (d * g * f_sun) / max(4.0 * n_dot_v * max(n_dot_l, 1e-4), 1e-4);
    let direct = specular_sun * frame.sun_illuminance.rgb * n_dot_l;

    // Diffuse water body tint — a small "deep" blue absorbing the
    // refracted light path that doesn't return. Without refraction
    // we proxy this as a fixed-colour Lambertian term weighted by
    // (1 - Fresnel) so it's only visible where reflection is weak.
    let body = vec3<f32>(0.020, 0.060, 0.090);
    let body_lit = body * (1.0 - f_at_view) * frame.sun_illuminance.rgb * 0.0005;

    var lit = f_at_view * reflection + direct + body_lit;

    // Aerial perspective.
    let ndc_xy = (in.clip_pos.xy / frame.viewport_size.xy) * 2.0 - 1.0;
    let d_world = length(p - frame.camera_position_world.xyz);
    let ap_uvw = vec3<f32>(
        ndc_xy.x * 0.5 + 0.5,
        ndc_xy.y * 0.5 + 0.5,
        ap_depth_uv(d_world),
    );
    let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);
    let final_color = lit * ap.a + ap.rgb;
    return vec4<f32>(final_color, 1.0);
}
