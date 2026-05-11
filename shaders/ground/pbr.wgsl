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
// Phase 12.6 — cloud-modulated overcast diffuse: under thick stratus
// the sky-view LUT (which knows about atmosphere only) reports the
// blue zenith brightness even though physically a viewer on the
// ground would see a uniform white-grey overcast hemisphere. Sample
// the synthesised top-down cloud density at the surface point's XZ;
// where it indicates substantial cloud cover, blend in a white
// diffuse irradiance proportional to the cloud-attenuated solar
// illuminance.
//
// Bindings (the pipeline layout in ps-ground wires these explicitly):
//   group 0 binding 0      FrameUniforms (frame)
//   group 1 binding 0      WorldUniforms (world)
//   group 2 binding 0      SurfaceParamsGpu (surface)
//   group 2 binding 1      top_down_density_mask (R8Unorm 2D)
//   group 2 binding 2      density_mask_sampler (linear-clamp)
//   group 3 binding {0..4} atmosphere LUTs

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(0) var<uniform> surface: SurfaceParams;
@group(2) @binding(1) var top_down_density_mask: texture_2d<f32>;
@group(2) @binding(2) var density_mask_sampler: sampler;

@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut:  texture_2d<f32>;
@group(3) @binding(2) var skyview_lut:       texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler:       sampler;

const GR_PI: f32 = 3.14159265358979;
const AP_FAR_M: f32 = 32000.0;

/// The top-down cloud density mask covers a 32 km × 32 km square
/// centred on the world origin, matching the weather-map extent
/// (synthesis §3.2.5). Beyond this extent the loader's clamp-to-edge
/// sampler returns the edge value, which is fine for a v1 stationary
/// camera that doesn't fly off the grid.
const DENSITY_MASK_EXTENT_M: f32 = 32000.0;

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
// Phase 8.5 surface ripples
//
// Procedural ripple field driven by simulated_seconds. Each spatial
// "spawn cell" (1 m × 1 m) chooses a Halton-sequenced spawn time per
// second of simulation; the ripple wave is a damped sin centred on the
// cell, expanding at ~0.5 m/s with a 0.6 s lifetime. Multiple spawns
// per second per cell stack via XOR-rotated cell hashes.
//
// The function returns a perturbation to the surface normal in the
// XZ plane (the water surface tangent space). The caller composes this
// with the base normal.
// ---------------------------------------------------------------------------

const RIPPLE_LIFETIME_S: f32 = 0.6;
const RIPPLE_SPEED_MPS: f32 = 0.5;
const RIPPLE_AMPLITUDE: f32 = 0.4;
const RIPPLE_FREQ_HZ: f32 = 6.0;

fn halton2(i: u32, base: u32) -> f32 {
    var f = 1.0;
    var r = 0.0;
    var ii = i;
    loop {
        if (ii == 0u) { break; }
        f = f / f32(base);
        r = r + f * f32(ii % base);
        ii = ii / base;
    }
    return r;
}

/// Compute the ripple-induced surface normal perturbation at world
/// position `p_xz`. `intensity` ∈ [0, 1] scales the wave amplitude
/// (for fading in / out as precipitation starts).
fn ripple_normal_offset(p_xz: vec2<f32>, time_s: f32, intensity: f32) -> vec2<f32> {
    if (intensity <= 0.0) { return vec2<f32>(0.0); }
    let cell_size = 1.0;
    let cell = vec2<i32>(floor(p_xz / cell_size));
    var dn = vec2<f32>(0.0);
    // Examine the 3x3 neighbourhood; ripples that started in nearby
    // cells can have travelled into this one.
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let c = cell + vec2<i32>(dx, dy);
            let h = hash21(c);
            // Two spawns per cell per second of sim time. Use the cell
            // hash to pick offsets within the second.
            for (var spawn = 0u; spawn < 2u; spawn = spawn + 1u) {
                let phase = halton2(h ^ (spawn * 0x9e3779b9u), 2u);
                // Time within current 1 s slot when this spawn fires.
                let t_in_second = phase;
                let t_floor = floor(time_s);
                var spawn_t = t_floor + t_in_second;
                if (spawn_t > time_s) { spawn_t = spawn_t - 1.0; }
                let age = time_s - spawn_t;
                if (age < 0.0 || age > RIPPLE_LIFETIME_S) { continue; }

                // Centre the ripple inside the cell using a second
                // Halton dimension.
                let cx = halton2(h, 3u);
                let cz = halton2(h ^ 0xa5a5a5a5u, 5u);
                let centre = vec2<f32>(f32(c.x) + cx, f32(c.y) + cz) * cell_size;
                let to_p = p_xz - centre;
                let r = length(to_p);
                if (r < 1e-4) { continue; }
                // Ripple expands at RIPPLE_SPEED_MPS; the wavefront sits
                // at radius age * speed.
                let front = age * RIPPLE_SPEED_MPS;
                let dr = r - front;
                if (dr > 0.0 || dr < -2.0 * RIPPLE_SPEED_MPS * RIPPLE_LIFETIME_S) {
                    // Outside the visible wavefront window.
                    continue;
                }
                // Sin wave behind the front, damped by age.
                let damp = clamp(1.0 - age / RIPPLE_LIFETIME_S, 0.0, 1.0);
                let amp = damp * damp * RIPPLE_AMPLITUDE * intensity;
                let phase_rad = dr * RIPPLE_FREQ_HZ * 2.0 * GR_PI / max(RIPPLE_SPEED_MPS, 1e-3);
                let dh = sin(phase_rad) * amp;
                // Normal offset in XZ direction toward the centre,
                // proportional to dh.
                let dir = to_p / r;
                dn = dn + dir * dh;
            }
        }
    }
    return dn;
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

/// Map a world XZ position to a UV in the top-down density mask. The
/// mask is centred on the world origin with a fixed extent
/// (`DENSITY_MASK_EXTENT_M`); positions outside the extent clamp to
/// the edge value (the sampler is clamp-to-edge).
fn mask_uv_from_world(p_xz: vec2<f32>) -> vec2<f32> {
    return p_xz / DENSITY_MASK_EXTENT_M + vec2<f32>(0.5);
}

/// Phase 12.6 — sample the top-down cloud column at a surface
/// position and produce a [0, 1] "overcast blocking" scalar:
/// 0 = clear sky overhead, 1 = thick overcast.
///
/// The mask is a saturated column-density measure where 1.0 means
/// "10 km of full-density cloud" (synthesis §3.2.5 — sized for the
/// Phase 8 precipitation cloud-occlusion gate, which needs to
/// distinguish "any cloud" from "thunderstorm tower"). For the
/// overcast-diffuse modulation we want sensitivity at *much* lower
/// mask values: a typical low stratus deck reads ~0.03 in mask
/// units but should produce near-full overcast blocking
/// (T_through ≈ 0.05). The exponent constant below converts the
/// precip-friendly mask into an overcast-friendly transmittance:
///
///   Stratus 600 m   coverage 1.0 → mask ≈ 0.028 → blocking ≈ 0.67
///   Stratocumulus   coverage 0.9 → mask ≈ 0.041 → blocking ≈ 0.81
///   Cumulus 800 m   coverage 0.5 → mask ≈ 0.024 → blocking ≈ 0.62
///   Thunderstorm Cb coverage 0.9 → mask ≈ 0.444 → blocking ≈ 1.000
fn overcast_blocking(p_xz: vec2<f32>) -> f32 {
    let uv = mask_uv_from_world(p_xz);
    let mask = textureSampleLevel(
        top_down_density_mask, density_mask_sampler, uv, 0.0,
    ).r;
    let transmittance = exp(-40.0 * mask);
    return 1.0 - transmittance;
}

/// White diffuse irradiance produced by an overcast hemisphere.
/// Real-world references:
///   midday overcast, sun ~60° altitude:   8,000–15,000 cd/m² zenith
///   medium overcast, sun ~30° altitude:   3,000–6,000 cd/m²
///   thin overcast, sun ~15° altitude:     1,000–2,000 cd/m²
///   dim overcast at sunset, sun ~5°:      300–500 cd/m²
///
/// Model: a fraction of the TOA solar illuminance is forward-
/// scattered through the cloud layer and emerges as a roughly
/// hemispheric diffuse glow. The fraction depends on cloud optical
/// thickness (low fraction for thick storm cells, higher for thin
/// stratus); we use a single empirically-tuned constant.
///
/// The sun elevation curve uses sqrt(max(sun.y, 0.05)) rather than
/// linear sun.y because real overcast attenuates *less* steeply with
/// sun angle than a clear-sky cosine — even at low sun the cloud
/// layer catches some forward-scattered light from below the horizon
/// and redistributes it isotropically. The 0.05 floor keeps a small
/// glow at twilight when the sun is barely below the horizon.
///
/// Spectrum is white (per-channel equal) — overcast scattering
/// flattens the spectrum, unlike the blue-shifted clear sky.
fn overcast_diffuse_irradiance(sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_alt_factor = sqrt(clamp(sun_dir.y, 0.05, 1.0));
    // 25 % forward-scattered fraction. Tuned so winter noon overcast
    // (toa=131000 lx, sun_dir.y≈0.21) gives ~15,000 lx ground
    // illuminance, matching photographic references for snowy
    // overcast scenes; OVC summer noon (toa=128000 lx, sun_dir.y≈0.85)
    // gives ~30,000 lx — still reading as overcast (vs ~120,000 lx
    // direct sun), distinct from clear-sky cases.
    let overcast_scatter_fraction = 0.25;
    let toa = frame.sun_illuminance.w;  // lux at TOA, per FrameUniforms
    let irradiance = toa * sun_alt_factor * overcast_scatter_fraction;
    return vec3<f32>(irradiance);
}

/// Cloud-modulated sky irradiance for ambient lighting on a surface
/// at `p_world` looking in direction `up`. `cloud_blocking` is the
/// pre-computed (1 − T_overcast) at this fragment, supplied by the
/// caller so we don't double-sample the density mask.
fn sample_sky_at(
    p_world: vec3<f32>,
    up: vec3<f32>,
    sun_dir: vec3<f32>,
    cloud_blocking: f32,
) -> vec3<f32> {
    let centre = vec3<f32>(0.0, -world.planet_radius_m, 0.0);
    let cam_r = length(p_world - centre);
    let uv = skyview_uv_local(up, sun_dir, cam_r);
    let clear_sky = textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb
                  * frame.sun_illuminance.rgb;
    // Phase 12.6 — under overcast, replace the clear-sky LUT (which
    // doesn't know about clouds) with a white diffuse irradiance
    // term representing hemispheric cloud scattering. Smooth blend
    // by `cloud_blocking` so partial cloud transitions cleanly.
    let overcast = overcast_diffuse_irradiance(sun_dir);
    return mix(clear_sky, overcast, cloud_blocking);
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
    // The palette entries average ~0.18 (a generic ground albedo).
    // world.ground_albedo acts as a tint: with the default (0.18,0.18,0.18)
    // it leaves the palette unchanged; brighter values scale up uniformly,
    // and per-channel values let the user push the ground warmer or cooler.
    let base_albedo = palette * (world.ground_albedo.rgb / 0.18);
    let base_roughness = 0.85;
    let dielectric_f0 = vec3<f32>(0.04);

    let wetness = clamp(surface.ground_wetness, 0.0, 1.0);
    let wm = wet_material(base_albedo, base_roughness, wetness);

    // Phase 12.6 — top-down cloud cover at this surface position.
    // Used to (a) attenuate the direct-sun path (the cloud column
    // blocks the sun) and (b) replace the clear-sky LUT ambient
    // with a white overcast-diffuse term inside `sample_sky_at`.
    let cloud_blocking = overcast_blocking(p.xz);
    let cloud_through = 1.0 - cloud_blocking;

    // Compute the dry/wet diffuse-and-specular ground (before puddles + snow).
    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let sun_t = sun_visibility(p, sun_dir);
    // Direct sunlight is attenuated by cloud cover. Under thick
    // overcast the direct path goes to zero and only the white
    // overcast diffuse (added below as ambient) lights the ground.
    let direct_E = frame.sun_illuminance.rgb * sun_t * n_dot_l * cloud_through;
    let f_dry = brdf(n, v, sun_dir, wm.albedo, wm.roughness, dielectric_f0);
    let direct_lit = f_dry * direct_E;

    // Ambient: sky-view at local up, weighted by Lambertian diffuse only
    // (cheap stand-in for proper diffuse irradiance from the upper
    // hemisphere; Hillaire 2020 §6 quotes this approximation).
    let sky_irr = sample_sky_at(p, n, sun_dir, cloud_blocking);
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

            // Phase 8.5 ripples: only when wetness > 0.5 AND precip > 0.
            // The check matches plan §8.5 verbatim.
            var n_water = vec3<f32>(0.0, 1.0, 0.0);
            if (wetness > 0.5 && surface.precip_intensity_mm_per_h > 0.0) {
                let ripple_intensity = clamp(
                    surface.precip_intensity_mm_per_h / 25.0, 0.0, 1.0,
                );
                let dn = ripple_normal_offset(p.xz, frame.simulated_seconds, ripple_intensity);
                n_water = normalize(vec3<f32>(-dn.x, 1.0, -dn.y));
            }

            let f_water = brdf(n_water, v, sun_dir, vec3<f32>(0.0), water_roughness, water_f0);
            let water_direct = f_water * direct_E;
            // Specular reflection of sky off the water surface; sample
            // sky-view LUT at the reflected view direction (with the
            // ripple-perturbed normal).
            let r = reflect(-v, n_water);
            let sky_refl = sample_sky_at(p, r, sun_dir, cloud_blocking);
            let n_dot_v = max(dot(n_water, v), 1e-4);
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
