// Phase 7 — physically based ground with wet surface and snow.
//
// Replaces Phase 0's procedural checker. Standard GGX/Smith specular +
// Lambertian diffuse over a Voronoi-tiled albedo. Lagarde 2013 wet
// surface chain (darkened albedo + reduced roughness; thin water layer
// above puddle_start). Optional snow layer when temp_c < 0.5 and
// snow_depth_m > 0.
//
// Ambient comes from the Phase 5 sky-view LUT sampled at the local up
// vector. Aerial perspective applied in-shader (plan §5.4 / §7.4).
//
// Bindings (the pipeline layout in ps-ground wires these explicitly):
//   group 0 binding 0      FrameUniforms (frame)
//   group 1 binding 0      WorldUniforms (world)
//   group 2 binding 0      SurfaceParamsGpu (surface)
//   group 3 binding {0..4} atmosphere LUTs

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(0) var<uniform> surface: SurfaceParams;

@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut:  texture_2d<f32>;
@group(3) @binding(2) var skyview_lut:       texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler:       sampler;

const GR_PI: f32 = 3.14159265358979;
const AP_FAR_M: f32 = 32000.0;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.world_pos = pos;
    out.clip_pos = frame.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// Hash + Voronoi (3-entry palette)
// ---------------------------------------------------------------------------

fn hash21(p: vec2<i32>) -> u32 {
    var x = u32(p.x) * 0x27d4eb2du + u32(p.y) * 0x165667b1u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    return x ^ (x >> 15u);
}

fn rand22(p: vec2<i32>) -> vec2<f32> {
    let h0 = hash21(p);
    let h1 = hash21(p + vec2<i32>(7919, 1)); // distinct salt for Y
    return vec2<f32>(
        f32(h0 & 0xffffu) / 65535.0,
        f32(h1 & 0xffffu) / 65535.0,
    );
}

/// 2D Voronoi with cell side `cell_size_m`. Returns (cell_id_hash, jitter_dist).
fn voronoi(p: vec2<f32>, cell_size_m: f32) -> vec2<u32> {
    let scaled = p / cell_size_m;
    let cell = vec2<i32>(floor(scaled));
    let frac = scaled - floor(scaled);
    var min_d2 = 4.0;
    var best = vec2<i32>(0, 0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let neighbour = cell + vec2<i32>(dx, dy);
            let jitter = rand22(neighbour);
            let centre = vec2<f32>(f32(dx), f32(dy)) + jitter;
            let diff = centre - frac;
            let d2 = dot(diff, diff);
            if (d2 < min_d2) {
                min_d2 = d2;
                best = neighbour;
            }
        }
    }
    return vec2<u32>(hash21(best), u32(min_d2 * 65535.0));
}

fn voronoi_palette(cell_id: u32) -> vec3<f32> {
    let bucket = cell_id % 3u;
    if (bucket == 0u) {
        return vec3<f32>(0.18, 0.18, 0.18);
    } else if (bucket == 1u) {
        return vec3<f32>(0.22, 0.20, 0.16);
    }
    return vec3<f32>(0.14, 0.16, 0.18);
}

// ---------------------------------------------------------------------------
// 2D value noise (puddle / snow distribution)
// ---------------------------------------------------------------------------

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let c = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u = vec2<f32>(fade(f.x), fade(f.y));
    let h00 = f32(hash21(c) & 0xffffu) / 65535.0;
    let h10 = f32(hash21(c + vec2<i32>(1, 0)) & 0xffffu) / 65535.0;
    let h01 = f32(hash21(c + vec2<i32>(0, 1)) & 0xffffu) / 65535.0;
    let h11 = f32(hash21(c + vec2<i32>(1, 1)) & 0xffffu) / 65535.0;
    return mix(mix(h00, h10, u.x), mix(h01, h11, u.x), u.y);
}

// ---------------------------------------------------------------------------
// GGX/Smith BRDF
// ---------------------------------------------------------------------------

fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (GR_PI * denom * denom + 1e-6);
}

fn smith_g1(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

fn smith_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return smith_g1(max(n_dot_v, 1e-4), roughness)
         * smith_g1(max(n_dot_l, 1e-4), roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    let x5 = x * x * x * x * x;
    return f0 + (vec3<f32>(1.0) - f0) * x5;
}

/// Cook-Torrance BRDF. Returns the RGB BRDF (not multiplied by n_dot_l).
fn brdf(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = ggx_d(n_dot_h, roughness);
    let g = smith_g(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    let spec = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f);
    let diffuse = kd * albedo / GR_PI;
    return diffuse + spec;
}

// ---------------------------------------------------------------------------
// Lagarde 2013 wet material
// ---------------------------------------------------------------------------

struct WetMaterial {
    albedo: vec3<f32>,
    roughness: f32,
};

fn wet_material(albedo: vec3<f32>, roughness: f32, wetness: f32) -> WetMaterial {
    let dark_albedo = pow(albedo, vec3<f32>(mix(1.0, 3.0, wetness)));
    let wet_rough = mix(roughness, max(roughness * 0.5, 0.05), wetness);
    return WetMaterial(dark_albedo, wet_rough);
}

// ---------------------------------------------------------------------------
// Sky-view ambient sample (mirrors the cloud march helper)
// ---------------------------------------------------------------------------

fn skyview_uv_local(view_dir: vec3<f32>, sun_dir: vec3<f32>, cam_r: f32) -> vec2<f32> {
    let cos_v = clamp(view_dir.y, -1.0, 1.0);
    let vza = acos(cos_v);
    let sin_horizon = clamp(world.planet_radius_m / cam_r, 0.0, 1.0);
    let zenith_horizon_angle = GR_PI - asin(sin_horizon);
    var v: f32;
    if (vza < zenith_horizon_angle) {
        let coord = sqrt(clamp(vza / max(zenith_horizon_angle, 1e-6), 0.0, 1.0));
        v = coord * 0.5;
    } else {
        let below_range = max(GR_PI - zenith_horizon_angle, 1e-6);
        let coord = sqrt(clamp((vza - zenith_horizon_angle) / below_range, 0.0, 1.0));
        v = 0.5 + coord * 0.5;
    }
    let view_az = atan2(view_dir.x, view_dir.z);
    let sun_az = atan2(sun_dir.x, sun_dir.z);
    var du = (view_az - sun_az) / (2.0 * GR_PI);
    du = du - floor(du);
    return vec2<f32>(du, v);
}

fn sample_sky_at(p_world: vec3<f32>, up: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let centre = vec3<f32>(0.0, -world.planet_radius_m, 0.0);
    let cam_r = length(p_world - centre);
    let uv = skyview_uv_local(up, sun_dir, cam_r);
    return textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb
         * frame.sun_illuminance.rgb;
}

// ---------------------------------------------------------------------------
// Sun visibility (transmittance through atmosphere from p to sun)
// ---------------------------------------------------------------------------

fn sun_visibility(p_world: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let p_atm = p_world + vec3<f32>(0.0, world.planet_radius_m, 0.0);
    return sample_transmittance_lut(p_atm, sun_dir);
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = vec3<f32>(0.0, 1.0, 0.0); // ground normal
    let p = in.world_pos;
    let v = normalize(frame.camera_position_world.xyz - p);
    let sun_dir = frame.sun_direction.xyz;

    // Voronoi base albedo, modulated by world ground_albedo so the global
    // tint slider still applies.
    let cell = voronoi(p.xz, 5.0);
    let palette = voronoi_palette(cell.x);
    let base_albedo = palette * (world.ground_albedo.rgb * 3.0); // palette already
                                                                  // averages ~0.18.
    let base_roughness = 0.85;
    let dielectric_f0 = vec3<f32>(0.04);

    let wetness = clamp(surface.ground_wetness, 0.0, 1.0);
    let wm = wet_material(base_albedo, base_roughness, wetness);

    // Compute the dry/wet diffuse-and-specular ground (before puddles + snow).
    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let sun_t = sun_visibility(p, sun_dir);
    let direct_E = frame.sun_illuminance.rgb * sun_t * n_dot_l;
    let f_dry = brdf(n, v, sun_dir, wm.albedo, wm.roughness, dielectric_f0);
    let direct_lit = f_dry * direct_E;

    // Ambient: sky-view at local up, weighted by Lambertian diffuse only
    // (cheap stand-in for proper diffuse irradiance from the upper
    // hemisphere; Hillaire 2020 §6 quotes this approximation).
    let sky_irr = sample_sky_at(p, n, sun_dir);
    let ambient_lit = (wm.albedo / GR_PI) * sky_irr;

    var lit = direct_lit + ambient_lit;

    // Puddle layer: thin water with F0 = 0.02, modulated by a noise mask
    // gated to flat surfaces (Lagarde 2013).
    let puddle_start = surface.puddle_start;
    if (wetness > puddle_start) {
        let puddle_t = clamp((wetness - puddle_start) / max(1.0 - puddle_start, 1e-3), 0.0, 1.0);
        let mask_noise = value_noise(p.xz * 0.15);
        let n_up = step(0.95, dot(n, vec3<f32>(0.0, 1.0, 0.0)));
        let coverage_gate = clamp(surface.puddle_coverage, 0.0, 1.0);
        let puddle_mask = mask_noise * n_up * coverage_gate * puddle_t;
        if (puddle_mask > 0.01) {
            // Smooth water layer: roughness ~0.05, F0 = 0.02.
            let water_roughness = 0.05;
            let water_f0 = vec3<f32>(0.02);
            let f_water = brdf(n, v, sun_dir, vec3<f32>(0.0), water_roughness, water_f0);
            let water_direct = f_water * direct_E;
            // Specular reflection of sky off the water surface; sample
            // sky-view LUT at the reflected view direction.
            let r = reflect(-v, n);
            let sky_refl = sample_sky_at(p, r, sun_dir);
            let n_dot_v = max(dot(n, v), 1e-4);
            let f_v = fresnel_schlick(n_dot_v, water_f0);
            let water_lit = water_direct + f_v * sky_refl;
            lit = mix(lit, water_lit, puddle_mask);
        }
    }

    // Snow layer (gated by temperature + depth).
    let snow_visible = (surface.temperature_c < 0.5) && (surface.snow_depth_m > 0.0);
    if (snow_visible) {
        // Snow accumulates on flat surfaces and away from puddles.
        let mask_noise = value_noise(p.xz * 0.15);
        let coverage_gate = clamp(surface.puddle_coverage, 0.0, 1.0);
        let inv_puddle = 1.0 - mask_noise * coverage_gate;
        let n_up = step(0.95, dot(n, vec3<f32>(0.0, 1.0, 0.0)));
        // Coverage saturates at 5 cm depth.
        let snow_amount = clamp(surface.snow_depth_m / 0.05, 0.0, 1.0)
                        * n_up * inv_puddle;
        if (snow_amount > 0.01) {
            let snow_albedo = vec3<f32>(0.9);
            let snow_roughness = 0.85;
            let f_snow = brdf(n, v, sun_dir, snow_albedo, snow_roughness, dielectric_f0);
            let snow_direct = f_snow * direct_E;
            let snow_ambient = (snow_albedo / GR_PI) * sky_irr;
            let snow_lit = snow_direct + snow_ambient;
            lit = mix(lit, snow_lit, snow_amount);
        }
    }

    // Aerial perspective (plan §5.4): camera-relative AP LUT lookup, then
    //   final = lit · ap.a + ap.rgb.
    let ndc_xy = (in.clip_pos.xy / frame.viewport_size.xy) * 2.0 - 1.0;
    let d_world = length(p - frame.camera_position_world.xyz);
    let ap_uvw = vec3<f32>(
        ndc_xy.x * 0.5 + 0.5,
        ndc_xy.y * 0.5 + 0.5,
        clamp(d_world / AP_FAR_M, 0.0, 1.0),
    );
    let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);
    let final_color = lit * ap.a + ap.rgb;
    return vec4<f32>(final_color, 1.0);
}
