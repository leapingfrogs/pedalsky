// Phase 5 §5 — common atmosphere helpers used by transmittance / multi-
// scatter / sky-view / aerial-perspective compute shaders and by the sky
// raymarch fragment shader.
//
// Conventions:
// - All distances in metres.
// - Planet centre is at the origin in *atmosphere-local* coordinates.
//   The render-side world origin is `(0, -planet_radius, 0)`; we shift
//   incoming world rays so they're planet-centred before tracing.
// - `world: WorldUniforms` and `frame: FrameUniforms` must be in scope
//   (the shader that includes this file is responsible for binding them).

const PI: f32 = 3.14159265358979;
const ISOTROPIC_PHASE: f32 = 0.0795774715459477; // 1 / (4*PI)

// ---------------------------------------------------------------------------
// Density profiles
// ---------------------------------------------------------------------------

/// Per-component density at altitude `h` (metres above sea level).
/// Returns `(rayleigh, mie, ozone)` densities (unitless, normalised to 1
/// at the relevant peak altitude).
fn atmosphere_density(h: f32) -> vec3<f32> {
    let r = exp(-h / world.rayleigh_scale_height_m);
    let m = exp(-h / world.mie_scale_height_m);
    // Ozone: tent profile centred on `ozone_center_m` with `ozone_thickness_m` width.
    let half_thickness = world.ozone_thickness_m * 0.5;
    let dz = abs(h - world.ozone_center_m);
    let o = max(0.0, 1.0 - dz / max(half_thickness, 1.0));
    return vec3<f32>(r, m, o);
}

/// Total extinction `σ_t` at altitude `h` (per metre, RGB).
fn extinction_at(h: f32) -> vec3<f32> {
    let d = atmosphere_density(h);
    return world.rayleigh_scattering.rgb * d.x
         + (world.mie_scattering.rgb + world.mie_absorption.rgb) * d.y
         + world.ozone_absorption.rgb * d.z;
}

/// Total scattering `σ_s` at altitude `h` (per metre, RGB). Excludes
/// absorption (Mie absorption + ozone absorption).
fn scattering_at(h: f32) -> vec3<f32> {
    let d = atmosphere_density(h);
    return world.rayleigh_scattering.rgb * d.x + world.mie_scattering.rgb * d.y;
}

/// Rayleigh and Mie scattering coefficients separately at altitude `h`.
/// Returned as `mat2x4` would be ideal but WGSL is limited; we return
/// a struct.
struct ScatteringPair {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
};
fn scattering_pair(h: f32) -> ScatteringPair {
    let d = atmosphere_density(h);
    var r: ScatteringPair;
    r.rayleigh = world.rayleigh_scattering.rgb * d.x;
    r.mie = world.mie_scattering.rgb * d.y;
    return r;
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-4), 1.5));
}

// ---------------------------------------------------------------------------
// Ray / planet geometry
// ---------------------------------------------------------------------------

/// Intersect ray with a sphere centred at origin, radius `r`. Returns the
/// far intersection distance along `rd`, or -1.0 if no intersection.
fn ray_sphere_far(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b + sqrt(disc);
}

/// Intersect ray with a sphere centred at origin. Returns near, far in t.
/// Result is meaningful only if `*hit` is true.
fn ray_sphere_intersect_origin(ro: vec3<f32>, rd: vec3<f32>, r: f32,
                                t: ptr<function, vec2<f32>>) -> bool {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) { return false; }
    let sq = sqrt(disc);
    *t = vec2<f32>(-b - sq, -b + sq);
    return true;
}

/// Distance from `p` to atmosphere top along `dir`. Returns 0 if `p`
/// already outside the atmosphere or `dir` doesn't intersect it.
fn distance_to_top_atmosphere(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    var t: vec2<f32>;
    let hit = ray_sphere_intersect_origin(p, dir, world.atmosphere_top_m, &t);
    if (!hit) { return 0.0; }
    return max(t.y, 0.0);
}

/// Distance from `p` along `dir` to the next atmosphere boundary
/// (planet surface or top of atmosphere, whichever comes first).
fn distance_to_atmosphere_boundary(p: vec3<f32>, dir: vec3<f32>) -> f32 {
    let r_top = world.atmosphere_top_m;
    let r_planet = world.planet_radius_m;
    var t_top: vec2<f32>;
    var t_planet: vec2<f32>;
    let hit_top = ray_sphere_intersect_origin(p, dir, r_top, &t_top);
    let hit_planet = ray_sphere_intersect_origin(p, dir, r_planet, &t_planet);
    if (!hit_top) {
        return 0.0;
    }
    var t_max = t_top.y;
    if (hit_planet && t_planet.x > 0.0) {
        t_max = min(t_max, t_planet.x);
    }
    return max(t_max, 0.0);
}

/// Convert PedalSky world-space position (`(0, -planet_radius, 0)` is
/// planet centre) to atmosphere-local (planet-centred).
fn world_to_atmosphere_pos(p_world: vec3<f32>) -> vec3<f32> {
    return p_world + vec3<f32>(0.0, world.planet_radius_m, 0.0);
}

// ---------------------------------------------------------------------------
// Transmittance LUT mapping
// ---------------------------------------------------------------------------

/// Map (height ∈ [0, atmosphere_thickness], view_zenith_cos ∈ [-1, 1]) →
/// (u, v) ∈ [0, 1]². Hillaire 2020 Eq 5.
fn transmittance_lut_uv(p: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let r = length(p);
    let h_top = world.atmosphere_top_m;
    let h_planet = world.planet_radius_m;
    let H = sqrt(max(h_top * h_top - h_planet * h_planet, 0.0));
    let rho = sqrt(max(r * r - h_planet * h_planet, 0.0));
    let mu = clamp(dot(p, dir) / max(r, 1.0), -1.0, 1.0);
    let d = max(distance_to_top_atmosphere(p, dir), 0.0);
    let d_min = h_top - r;
    let d_max = rho + H;
    let x_mu = clamp((d - d_min) / max(d_max - d_min, 1.0), 0.0, 1.0);
    let x_r = rho / max(H, 1.0);
    return vec2<f32>(x_mu, x_r);
}

/// Inverse mapping for the transmittance LUT compute pass.
fn transmittance_lut_uv_to_pos_dir(uv: vec2<f32>, p: ptr<function, vec3<f32>>,
                                    dir: ptr<function, vec3<f32>>) {
    let h_top = world.atmosphere_top_m;
    let h_planet = world.planet_radius_m;
    let H = sqrt(max(h_top * h_top - h_planet * h_planet, 0.0));
    let rho = uv.y * H;
    let r = sqrt(rho * rho + h_planet * h_planet);
    let d_min = h_top - r;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if (d != 0.0) {
        mu = clamp((H * H - rho * rho - d * d) / max(2.0 * r * d, 1e-6), -1.0, 1.0);
    }
    *p = vec3<f32>(0.0, r, 0.0);
    let sin_t = sqrt(max(1.0 - mu * mu, 0.0));
    *dir = vec3<f32>(sin_t, mu, 0.0);
}

// ---------------------------------------------------------------------------
// Numerical transmittance (used by the LUT bake)
// ---------------------------------------------------------------------------

/// Trapezoidal integration of total optical depth from `p` along `dir`
/// to the next atmosphere boundary. `n_steps` is fixed at 40 per plan §5.2.1.
/// Uses a numerically-stable height calculation: at planet scale,
/// `length(pi) - planet_radius_m` suffers catastrophic cancellation
/// (the squared term dwarfs the small height offset in fp32).  The
/// identity sqrt(1+x) − 1 = x/(sqrt(1+x)+1) lets us extract the height
/// delta directly.
fn integrate_optical_depth(p: vec3<f32>, dir: vec3<f32>, n_steps: u32) -> vec3<f32> {
    let t_max = distance_to_atmosphere_boundary(p, dir);
    if (t_max <= 0.0) { return vec3<f32>(0.0); }
    let dt = t_max / f32(n_steps);
    let r0 = length(p);
    let cos_view = dot(p / max(r0, 1.0), dir);
    let h0 = r0 - world.planet_radius_m;
    var optical_depth = vec3<f32>(0.0);
    var prev = extinction_at(h0);
    for (var i = 1u; i <= n_steps; i = i + 1u) {
        let t = f32(i) * dt;
        let pi = p + dir * t;
        // |pi|² = r0² + 2t·r0·cos_view + t², so
        // |pi| − r0 = (2t·r0·cos_view + t²) / (r0 + |pi|).
        let r_delta_num = 2.0 * t * r0 * cos_view + t * t;
        let r_delta = r_delta_num / (max(r0, 1.0) + length(pi));
        let h = h0 + r_delta;
        let cur = extinction_at(h);
        optical_depth = optical_depth + 0.5 * (prev + cur) * dt;
        prev = cur;
    }
    return optical_depth;
}

// `sample_transmittance_lut` lives in `common/atmosphere_lut_sampling.wgsl`
// and is only included by shaders that bind `transmittance_lut` +
// `lut_sampler`.
