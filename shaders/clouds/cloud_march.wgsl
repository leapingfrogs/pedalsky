// Phase 6.6 — cloud raymarch fragment shader.
//
// Bindings (the pipeline layout in ps-clouds wires these explicitly):
//   group 0 binding 0   FrameUniforms (frame)
//   group 1 binding 0   WorldUniforms (world)
//   group 2 binding 0   noise_base    (texture_3d<f32>)
//   group 2 binding 1   noise_detail  (texture_3d<f32>)
//   group 2 binding 2   noise_curl    (texture_2d<f32>)
//   group 2 binding 3   blue_noise    (texture_2d<f32>)  - sampled via textureLoad
//   group 2 binding 4   noise_sampler (linear-repeat)
//   group 2 binding 5   nearest_sampler
//   group 3 binding 0   transmittance_lut  (texture_2d<f32>)
//   group 3 binding 1   multiscatter_lut   (texture_2d<f32>)
//   group 3 binding 2   skyview_lut        (texture_2d<f32>)
//   group 3 binding 3   aerial_perspective_lut (texture_3d<f32>)
//   group 3 binding 4   lut_sampler        (filtering)
//   immediate constants slot 0..N - CloudParams (uniform buffer @group 4)
//
// The shader writes to a dedicated cloud RT in premultiplied-alpha form:
// rgb = transmittance-attenuated luminance, a = 1 - T_lum.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;

@group(2) @binding(0) var noise_base:    texture_3d<f32>;
@group(2) @binding(1) var noise_detail:  texture_3d<f32>;
@group(2) @binding(2) var noise_curl:    texture_2d<f32>;
@group(2) @binding(3) var blue_noise:    texture_2d<f32>;
@group(2) @binding(4) var noise_sampler: sampler;
@group(2) @binding(5) var nearest_sampler: sampler;
@group(2) @binding(6) var<uniform> params: CloudParams;
@group(2) @binding(7) var<storage, read> cloud_layers: CloudLayerArray;
@group(2) @binding(8) var weather_map: texture_2d<f32>;

@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut:  texture_2d<f32>;
@group(3) @binding(2) var skyview_lut:       texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler:       sampler;

const HG_PI: f32 = 3.14159265358979;

// ---------------------------------------------------------------------------
// Fullscreen triangle vertex shader (writes UV in NDC space).
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * HG_PI * pow(max(denom, 1e-4), 1.5));
}

/// Dual-lobe HG with anisotropy g scaled by `g_scale`. Used by the
/// multi-octave multiple-scattering approximation: cos_theta is geometric
/// and is NOT scaled; only g is.
fn dual_lobe_hg_with_g_scale(cos_theta: f32, g_scale: f32) -> f32 {
    let gf = params.g_forward * g_scale;
    let gb = params.g_backward * g_scale;
    return mix(
        henyey_greenstein(cos_theta, gf),
        henyey_greenstein(cos_theta, gb),
        params.g_blend,
    );
}

// ---------------------------------------------------------------------------
// Density-height profile (NDF, 8 cloud types)
// ---------------------------------------------------------------------------

fn ndf(h: f32, t: u32) -> f32 {
    switch t {
        case 0u: { // Cumulus: bell, peak ~0.4
            return smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.2, h);
        }
        case 1u: { // Stratus: top-heavy, low and thin
            return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.6, 1.0, h));
        }
        case 2u: { // Stratocumulus: mid-heavy
            return smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.4, h);
        }
        case 3u: { // Altocumulus: mid-heavy, thinner overall
            return smoothstep(0.0, 0.20, h) * smoothstep(1.0, 0.3, h) * 0.8;
        }
        case 4u: { // Altostratus: top-heavy sheet
            return smoothstep(0.0, 0.30, h) * (1.0 - smoothstep(0.7, 1.0, h)) * 0.6;
        }
        case 5u: { // Cirrus: thin, top-heavy, wispy
            return smoothstep(0.0, 0.40, h) * (1.0 - smoothstep(0.6, 1.0, h)) * 0.4;
        }
        case 6u: { // Cirrostratus: very thin sheet, highest
            return smoothstep(0.0, 0.50, h) * (1.0 - smoothstep(0.8, 1.0, h)) * 0.3;
        }
        case 7u: { // Cumulonimbus: bottom-heavy + anvil top
            let base  = smoothstep(0.0, 0.05, h);
            let mid   = smoothstep(0.95, 0.5, h);
            let anvil = smoothstep(0.7, 0.9, h) * 1.5;
            let mix_t = smoothstep(0.65, 0.8, h);
            return base * mix(mid, anvil, mix_t);
        }
        default: { return 0.0; }
    }
}

fn ambient_height_gradient(h: f32) -> f32 {
    return mix(0.3, 1.0, h);
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Atmosphere-local planet centre. PedalSky world origin sits on the
/// planet surface at (0, 0, 0), so the centre is at (0, -planet_radius, 0).
fn planet_centre() -> vec3<f32> {
    return vec3<f32>(0.0, -world.planet_radius_m, 0.0);
}

/// Stable altitude (metres above sea level) given a precomputed entry
/// radius `r0`, view-direction cosine `cos_view`, base altitude `h0`,
/// world-space sample point `p_world`, and march distance `t` from the
/// ray entry. Avoids `length(p) - planet_r` cancellation at planet scale.
fn altitude_from_entry(p_world: vec3<f32>, r0: f32, cos_view: f32, h0: f32, t: f32) -> f32 {
    let pi = p_world - planet_centre();
    let r_delta_num = 2.0 * t * r0 * cos_view + t * t;
    let r_delta = r_delta_num / (max(r0, 1.0) + length(pi));
    return h0 + r_delta;
}

/// Per-layer ray-shell intersection. Returns true if the ray's [t0, t1]
/// interval is non-empty. Handles camera below, inside, or above the
/// layer (the shell is in atmosphere-local coordinates).
fn ray_layer_intersect(
    ro: vec3<f32>, rd: vec3<f32>, layer: CloudLayerGpu,
    t0: ptr<function, f32>, t1: ptr<function, f32>,
) -> bool {
    let centre = planet_centre();
    let r_bottom = world.planet_radius_m + layer.base_m;
    let r_top    = world.planet_radius_m + layer.top_m;
    var t_top: vec2<f32>;
    var t_bot: vec2<f32>;
    let hit_top = ray_sphere_intersect(ro, rd, centre, r_top, &t_top);
    let hit_bot = ray_sphere_intersect(ro, rd, centre, r_bottom, &t_bot);
    if (!hit_top) {
        return false;
    }
    // Inside-out ordering depends on whether the camera is below, inside,
    // or above the shell. Compute the camera's altitude (atmosphere-local)
    // to branch.
    let r_cam = length(ro - centre);
    var t_a: f32;
    var t_b: f32;
    if (r_cam < r_bottom) {
        // Below shell: enter at bottom-far, exit at top-far.
        if (!hit_bot) { return false; }
        t_a = max(t_bot.y, 0.0);
        t_b = max(t_top.y, 0.0);
    } else if (r_cam > r_top) {
        // Above shell: enter at top-near, exit at bottom-near (or top-far
        // if ray skims past the bottom shell entirely).
        t_a = max(t_top.x, 0.0);
        if (hit_bot && t_bot.x > 0.0) {
            t_b = max(t_bot.x, 0.0);
        } else {
            t_b = max(t_top.y, 0.0);
        }
    } else {
        // Inside shell: enter at 0 (the camera), exit at the next boundary.
        t_a = 0.0;
        if (hit_bot && t_bot.x > 0.0) {
            t_b = min(max(t_top.y, 0.0), t_bot.x);
        } else {
            t_b = max(t_top.y, 0.0);
        }
    }
    *t0 = t_a;
    *t1 = t_b;
    return t_b > t_a;
}

// ---------------------------------------------------------------------------
// Density sampling
// ---------------------------------------------------------------------------

fn world_to_weather_uv(xz: vec2<f32>) -> vec2<f32> {
    return xz / max(params.weather_scale_m, 1.0) + vec2<f32>(0.5);
}

/// Schneider 2015/2017 cloud density at world position `p`.
/// `p_alt` is the precomputed altitude (m) at `p`.
fn sample_density(
    p: vec3<f32>, p_alt: f32, layer: CloudLayerGpu, weather: vec4<f32>,
) -> f32 {
    let h = (p_alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
    if (h < 0.0 || h > 1.0) { return 0.0; }

    let base_uv = p / max(params.base_scale_m, 1.0);
    let base = textureSampleLevel(noise_base, noise_sampler, base_uv, 0.0);
    // Schneider remap: low-frequency Worley FBM erodes the Perlin-Worley.
    let lf_fbm = base.g * 0.625 + base.b * 0.25 + base.a * 0.125;
    let base_cloud = remap(base.r, -(1.0 - lf_fbm), 1.0, 0.0, 1.0);

    let profile = ndf(h, layer.cloud_type);
    var cloud = base_cloud * profile;

    // Coverage from synthesised weather × per-layer scalar.
    let coverage = clamp(weather.r * layer.coverage, 0.0, 1.0);
    cloud = remap(cloud, 1.0 - coverage, 1.0, 0.0, 1.0);
    cloud = cloud * coverage;

    // Curl-perturbed detail erosion at the cloud boundary. Only erode
    // where there is already some density (saves the per-sample texture
    // fetches in empty space).
    if (cloud > 0.0) {
        let curl_uv = p.xz / max(params.detail_scale_m, 1.0);
        let curl = textureSampleLevel(noise_curl, noise_sampler, curl_uv, 0.0).rg * 2.0
                 - vec2<f32>(1.0);
        let detail_p = p + vec3<f32>(curl.x, 0.0, curl.y) * params.curl_strength
                                     * params.detail_scale_m;
        let detail_uv = detail_p / max(params.detail_scale_m, 1.0);
        let detail = textureSampleLevel(noise_detail, noise_sampler, detail_uv, 0.0);
        let hf_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
        // Schneider 2015: detail erosion polarity flips with height -- wispy
        // top, fluffy base.
        let detail_mod = mix(hf_fbm, 1.0 - hf_fbm, clamp(h * 10.0, 0.0, 1.0));
        cloud = remap(cloud, detail_mod * params.detail_strength, 1.0, 0.0, 1.0);
    }

    return clamp(cloud, 0.0, 1.0) * layer.density_scale;
}

// ---------------------------------------------------------------------------
// Light march (optical depth from sample to sun)
// ---------------------------------------------------------------------------

fn march_to_light(p: vec3<f32>, p_alt: f32, sun_dir: vec3<f32>, layer: CloudLayerGpu) -> f32 {
    let cos_sun = max(sun_dir.y, 0.05);
    let dist_to_top = max((layer.top_m - p_alt) / cos_sun, 1.0);
    let step = dist_to_top / f32(params.light_steps);

    var od = 0.0;
    var pos = p;
    let centre = planet_centre();
    let r0 = length(pos - centre);
    let cos_view = dot((pos - centre) / max(r0, 1.0), sun_dir);
    let h0 = r0 - world.planet_radius_m;

    for (var i = 0u; i < params.light_steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * step;
        let pi = p + sun_dir * t;
        let alt = altitude_from_entry(pi, r0, cos_view, h0, t);
        let weather = textureSampleLevel(
            weather_map, noise_sampler,
            world_to_weather_uv(pi.xz), 0.0,
        );
        let local_density = sample_density(pi, alt, layer, weather);
        od = od + local_density * step;
    }
    return od;
}

// ---------------------------------------------------------------------------
// Sky-view ambient sample (reads Phase 5 sky-view LUT at p's local-up)
// ---------------------------------------------------------------------------

fn skyview_uv_local(view_dir: vec3<f32>, sun_dir: vec3<f32>, cam_r: f32) -> vec2<f32> {
    let cos_v = clamp(view_dir.y, -1.0, 1.0);
    let vza = acos(cos_v);
    let sin_horizon = clamp(world.planet_radius_m / cam_r, 0.0, 1.0);
    let zenith_horizon_angle = HG_PI - asin(sin_horizon);
    var v: f32;
    if (vza < zenith_horizon_angle) {
        let coord = sqrt(clamp(vza / max(zenith_horizon_angle, 1e-6), 0.0, 1.0));
        v = coord * 0.5;
    } else {
        let below_range = max(HG_PI - zenith_horizon_angle, 1e-6);
        let coord = sqrt(clamp((vza - zenith_horizon_angle) / below_range, 0.0, 1.0));
        v = 0.5 + coord * 0.5;
    }
    let view_az = atan2(view_dir.x, view_dir.z);
    let sun_az = atan2(sun_dir.x, sun_dir.z);
    var du = (view_az - sun_az) / (2.0 * HG_PI);
    du = du - floor(du);
    return vec2<f32>(du, v);
}

fn sample_sky_ambient(p: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let centre = planet_centre();
    let local_up = normalize(p - centre);
    let cam_r = length(p - centre);
    let uv = skyview_uv_local(local_up, sun_dir, cam_r);
    return textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb
         * frame.sun_illuminance.rgb;
}

// ---------------------------------------------------------------------------
// Main fragment shader
// ---------------------------------------------------------------------------

struct LayerHit {
    idx: u32,
    t0: f32,
    t1: f32,
    hit: u32, // 0/1 flag (WGSL switch types want u32)
};

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ray = compute_view_ray(in.pos.xy);
    let cos_theta = dot(ray.dir, frame.sun_direction.xyz);

    // Spatial blue-noise jitter (frame-deterministic — no time component).
    let jitter_xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    let jitter = textureLoad(blue_noise, jitter_xy & vec2<i32>(63), 0).r;

    // Per-layer intervals.
    var hits: array<LayerHit, 8>;
    let layer_count = min(params.cloud_layer_count, MAX_CLOUD_LAYERS);
    for (var l = 0u; l < layer_count; l = l + 1u) {
        var t0: f32;
        var t1: f32;
        let ok = ray_layer_intersect(ray.origin, ray.dir,
                                     cloud_layers.layers[l], &t0, &t1);
        var flag: u32 = 0u;
        if (ok) { flag = 1u; }
        hits[l] = LayerHit(l, t0, t1, flag);
    }

    // Insertion sort by t0 ascending (small N; the 8-entry sort is fine).
    for (var i = 1u; i < layer_count; i = i + 1u) {
        var j = i;
        loop {
            if (j == 0u) { break; }
            if (hits[j - 1u].t0 <= hits[j].t0) { break; }
            let tmp = hits[j];
            hits[j] = hits[j - 1u];
            hits[j - 1u] = tmp;
            j = j - 1u;
        }
    }

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);
    let sigma_t = params.sigma_s + params.sigma_a;
    let sun_dir = frame.sun_direction.xyz;

    // Track the luminance-weighted distance along the ray. Used after the
    // march to sample the aerial-perspective LUT at the cloud's perceived
    // depth (plan §6.7). Falls back to the segment midpoint when the
    // weighted sum is degenerate.
    var t_weight: f32 = 0.0;
    var t_weight_norm: f32 = 0.0;

    for (var i = 0u; i < layer_count; i = i + 1u) {
        let lh = hits[i];
        if (lh.hit == 0u) { continue; }
        let layer = cloud_layers.layers[lh.idx];

        let segment = lh.t1 - lh.t0;
        if (segment <= 0.0) { continue; }
        let step = segment / f32(params.cloud_steps);
        var t = lh.t0 + jitter * step;

        // Pre-compute geometry for the stable altitude formula.
        let centre = planet_centre();
        let r0 = length(ray.origin - centre);
        let cos_view = dot((ray.origin - centre) / max(r0, 1.0), ray.dir);
        let h0 = r0 - world.planet_radius_m;

        for (var s = 0u; s < params.cloud_steps; s = s + 1u) {
            let p = ray.origin + ray.dir * t;
            let alt = altitude_from_entry(p, r0, cos_view, h0, t);
            let weather_sample = textureSampleLevel(
                weather_map, noise_sampler,
                world_to_weather_uv(p.xz), 0.0,
            );
            let density = sample_density(p, alt, layer, weather_sample);

            if (density > 1e-3) {
                let od_to_sun = march_to_light(p, alt, sun_dir, layer);

                // Schneider 2015 Beer-Powder (canonical form).
                let beer        = exp(-sigma_t * od_to_sun);
                let powder      = 1.0 - exp(-2.0 * sigma_t * od_to_sun);
                let beer_powder = beer * powder * 2.0;
                let energy = mix(beer, beer_powder, params.powder_strength);

                // Hillaire 2016 multi-octave multiple-scattering: each octave
                // scales energy×a, optical-depth×b, anisotropy×c. cos_theta
                // is geometric and never scaled; only g is.
                var sun_in = vec3<f32>(0.0);
                var a = 1.0;
                var b = 1.0;
                var c = 1.0;
                for (var n = 0u; n < params.multi_scatter_octaves; n = n + 1u) {
                    let phase = dual_lobe_hg_with_g_scale(cos_theta, c);
                    sun_in = sun_in + a * frame.sun_illuminance.rgb
                                    * phase
                                    * exp(-sigma_t * od_to_sun * b);
                    a = a * params.multi_scatter_a;
                    b = b * params.multi_scatter_b;
                    c = c * params.multi_scatter_c;
                }

                let h_norm = clamp((alt - layer.base_m)
                                   / max(layer.top_m - layer.base_m, 1.0), 0.0, 1.0);
                let ambient = sample_sky_ambient(p, sun_dir)
                            * params.ambient_strength
                            * ambient_height_gradient(h_norm);

                let in_scatter = density * params.sigma_s
                                * (sun_in * energy + ambient);

                // Energy-conserving step integration.
                let local_sigma = max(density * sigma_t, 1e-6);
                let sample_t = exp(-vec3<f32>(local_sigma) * step);
                let s_int = (in_scatter - in_scatter * sample_t) / vec3<f32>(local_sigma);
                let step_lum = transmittance * s_int;
                let step_weight = dot(step_lum, vec3<f32>(0.2126, 0.7152, 0.0722));
                luminance = luminance + step_lum;
                t_weight = t_weight + step_weight * t;
                t_weight_norm = t_weight_norm + step_weight;
                transmittance = transmittance * sample_t;
            }

            if (max3(transmittance) < 0.01) { break; }
            t = t + step;
        }
    }

    // Composition target: premultiplied luminance + scalar opacity.
    let t_lum = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    let cloud_alpha = 1.0 - t_lum;

    // Aerial perspective on the cloud luminance (plan §6.7). Sample the
    // AP LUT at the luminance-weighted depth so the haze is applied where
    // the cloud's light actually came from. AP_FAR_M matches the bake
    // (plan §5.2.4 — 32 km linear); samples beyond that clamp.
    let t_cloud = t_weight / max(t_weight_norm, 1e-6);
    if (cloud_alpha > 1e-4) {
        let viewport = frame.viewport_size.xy;
        let ndc_xy = vec2<f32>(
            (in.pos.x / viewport.x) * 2.0 - 1.0,
            1.0 - (in.pos.y / viewport.y) * 2.0,
        );
        let ap_uvw = vec3<f32>(
            ndc_xy.x * 0.5 + 0.5,
            ndc_xy.y * 0.5 + 0.5,
            clamp(t_cloud / 32000.0, 0.0, 1.0),
        );
        let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);
        // Premultiplied form: cloud_lum is already pre-multiplied by
        // cloud_alpha along the ray. Attenuate by AP transmittance, then
        // add AP inscatter scaled by cloud_alpha so the haze only affects
        // the opaque parts of the cloud (the transparent parts let the
        // sky through, and the sky pass already has its own atmospheric
        // contribution baked in).
        return vec4<f32>(luminance * ap.a + ap.rgb * cloud_alpha, cloud_alpha);
    }
    return vec4<f32>(luminance, cloud_alpha);
}
