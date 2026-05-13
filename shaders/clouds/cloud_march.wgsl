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
// Phase 9.1: per-pixel HDR depth used as the cloud march t_max so
// clouds correctly clip behind opaque geometry.
@group(2) @binding(9) var scene_depth: texture_depth_2d;
// Phase 12.1: per-pixel cloud-type override grid (R8Uint). Sampled
// via textureLoad with integer coords (no sampler — interpolating a
// type index would be meaningless). Value 255 is the sentinel "use
// the layer's cloud_type instead"; values 0..7 select a specific
// cloud type for this pixel regardless of the layer's default.
@group(2) @binding(10) var cloud_type_grid: texture_2d<u32>;

// Phase 14.C: 3D wind-field volume synthesised by ps-synthesis. The
// RGBA16Float texture covers the same 32 km × 32 km horizontal
// footprint as the weather map and spans 0..WIND_FIELD_TOP_M
// vertically. Channels: (u east m/s, v south m/s, w up m/s,
// turbulence). The march samples this at each step to advect the
// cloud noise lookup by `wind(altitude) * simulated_seconds *
// wind_drift_strength`. The vertical and turbulence channels are
// reserved for follow-up phases.
@group(2) @binding(11) var wind_field: texture_3d<f32>;

/// Spatial extent (metres) the wind field covers — matches the
/// weather map's 32 km tile. Mirrors `EXTENT_M` in
/// `crates/ps-synthesis/src/wind_field.rs`.
const WIND_FIELD_EXTENT_M: f32 = 32000.0;
/// Top altitude (m AGL) the wind field spans on its Y axis — mirrors
/// `TOP_M` in `crates/ps-synthesis/src/wind_field.rs`. Samples at
/// altitudes above this clamp to the topmost voxel.
const WIND_FIELD_TOP_M: f32 = 12000.0;

/// Sentinel value in the cloud_type_grid meaning "use the per-layer
/// cloud_type rather than a per-pixel override". Mirrors the Rust
/// constant `ps_synthesis::cloud_type_grid::SENTINEL`.
const CLOUD_TYPE_SENTINEL: u32 = 255u;
/// Spatial extent (metres) the cloud_type_grid covers — matches the
/// weather map's extent (synthesis uploads at 128×128 over the same
/// 32 km square centred on world origin).
const CLOUD_TYPE_GRID_EXTENT_M: f32 = 32000.0;
/// Resolution of the cloud_type_grid texture (square; matches
/// `ps_synthesis::cloud_type_grid::SIZE`).
const CLOUD_TYPE_GRID_SIZE: i32 = 128;

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
// Chromatic Mie modulation
// ---------------------------------------------------------------------------

/// Per-channel chromatic scaling of the cloud's scattering cross
/// section, derived from the droplet effective diameter `d` (µm).
///
/// The Mie size parameter is `x = π d / λ`. For everyday cloud
/// droplets (`d ≥ ~5 µm` at visible λ ≈ 0.5 µm we get `x ≈ 30`),
/// scattering is in the geometric-optics regime and is essentially
/// wavelength-flat — this is why most clouds look white. For
/// sub-micron droplet populations (fresh fog tops, thin cirrus
/// edges, cumulus updraft tops) `x` drops toward unity and
/// scattering picks up a wavelength-selective component that
/// produces the warm fringes visible in real photography at
/// sunset.
///
/// The function returns an RGB multiplier centred on 1.0 at large
/// `d`. Small `d` boosts the blue end (more Rayleigh-like blue
/// scattering means blue light gets scattered out of the
/// transmitted path, leaving warmer remaining light). The transition
/// is centred at `d ≈ 8 µm` and is fully wavelength-flat by
/// `d ≈ 20 µm`. The strength is capped so the chromatic shift
/// stays visually subtle — this is a fringe-effect knob, not a
/// hero feature.
///
/// Wavelengths (centre of each sRGB primary's band): R ≈ 620 nm,
/// G ≈ 550 nm, B ≈ 470 nm. `d` is in µm, λ in nm; the size
/// parameter `x = π · d / (λ · 1e-3)` is unitless.
fn chromatic_mie_modulation(d_um: f32) -> vec3<f32> {
    // Wavelength-flat above this diameter (geometric optics).
    let d_flat: f32 = 20.0;
    // Maximum departure from grey at d → 0. Capped so the effect
    // stays a fringe modulation rather than dominating the cloud
    // appearance.
    let max_strength: f32 = 0.25;

    let t = clamp(1.0 - d_um / d_flat, 0.0, 1.0);
    if (t <= 0.0) {
        return vec3<f32>(1.0);
    }
    let strength = max_strength * t * t;

    // Per-channel boost on the short-wavelength end. The blue
    // channel scatters more than red as droplets shrink toward the
    // Mie/Rayleigh boundary; reverse-modulate the channels around
    // a unity midpoint so the chromatic shift averages to zero.
    return vec3<f32>(
        1.0 - 0.5 * strength,   // R: less scatter
        1.0,                    // G: pivot
        1.0 + strength,         // B: more scatter
    );
}

// ---------------------------------------------------------------------------
// Phase functions
// ---------------------------------------------------------------------------

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * HG_PI * pow(max(denom, 1e-4), 1.5));
}

/// Draine phase function (Draine 2003), normalised to integrate to
/// 1 over the sphere. Generalises HG: at α=0 it reduces to HG; at
/// α=1, g=0 it is Rayleigh; at α=1 it is Cornette–Shanks. Used as
/// the "bulk" lobe of the Jendersie & d'Eon 2023 HG+Draine fit.
fn draine_phase(cos_theta: f32, g: f32, alpha: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    let num = (1.0 - g2) * (1.0 + alpha * cos_theta * cos_theta);
    let den = 4.0 * (1.0 + alpha * (1.0 + 2.0 * g2) / 3.0) * HG_PI
            * pow(max(denom, 1e-4), 1.5);
    return num / max(den, 1e-6);
}

/// Cloud march phase function — Jendersie & d'Eon 2023 "An
/// Approximate Mie Scattering Function for Fog and Cloud Rendering"
/// (SIGGRAPH 2023). A single droplet effective diameter `d` (µm)
/// parametrises the fit; the closed-form expressions (paper Eqs.
/// 4–7) derive (g_HG, g_D, α, w_D) which evaluate as a blend of an
/// HG forward peak and a Draine bulk lobe (paper Eq. 3):
///
///     p_fog(θ) = (1 - w_D) · HG(g_HG; θ) + w_D · Draine(g_D, α; θ).
///
/// The fit is valid for 5 ≤ d ≤ 50 µm (water-droplet and ice-crystal
/// effective diameters typical of fog, cumulus, stratus, and cirrus).
/// Outside that range the expressions extrapolate; clamp before
/// evaluation. Wavelength-independent (paper averages over 400–700
/// nm).
///
/// `g_scale` is the Hillaire multi-octave multiple-scattering
/// attenuation factor — applied to both HG and Draine `g` values so
/// each successive octave broadens the phase function the same way.
/// `params.droplet_diameter_bias` multiplies the layer diameter so
/// the user can dial droplet size from the Clouds UI panel.
///
/// Cumulonimbus is mixed-phase (water droplets in the convective
/// core, ice crystals in the anvil). When the cloud type is 7
/// (Cumulonimbus) the effective diameter blends from the layer's
/// own value (water core) toward 50 µm (ice anvil) across
/// h_norm ∈ [0.6, 0.85]. The transition pre-empts the anvil's NDF
/// rise (which starts at 0.7) so the phase shift is visible before
/// the anvil mass dominates.
const APPROX_MIE_D_MIN: f32 = 5.0;
const APPROX_MIE_D_MAX: f32 = 50.0;
const CB_ANVIL_DROPLET_DIAMETER_UM: f32 = 50.0;

/// Ice halo features at 22° and 46° from the sun direction. Real
/// hexagonal ice crystals refract light through 60° prism faces,
/// producing concentrated angular peaks at these two deflection
/// angles. The 22° halo (single refraction) is the headline
/// feature — most photographs of cirrus around the sun show a
/// faint circle of brightness at this angle; the 46° halo (double
/// refraction) is fainter and rarer.
///
/// No HG/Draine-class approximation captures these features (they
/// arise from coherent geometric optics through ordered crystal
/// shapes, not from probabilistic single-scatter integrals). We add
/// them as two narrow Gaussian-like lobes on top of the Approximate
/// Mie evaluation, gated by an ice fraction smoothly ramped from
/// the water/ice boundary at d ≈ 35 µm.
///
/// Strength is intentionally subtle (peak 8% amplitude relative to
/// the Mie evaluation). Width is 0.5° (Gaussian σ in angle space)
/// — narrow enough to read as a halo, wide enough to survive a
/// modest march step count.
const ICE_HALO_22_COS:    f32 = 0.92718; // cos(22°)
const ICE_HALO_46_COS:    f32 = 0.69466; // cos(46°)
const ICE_HALO_WIDTH_DEG: f32 = 0.5;
const ICE_HALO_22_PEAK:   f32 = 0.08;
const ICE_HALO_46_PEAK:   f32 = 0.03;
const ICE_DIAMETER_LOW:   f32 = 35.0;
const ICE_DIAMETER_HIGH:  f32 = 50.0;

fn ice_halo_lobe(cos_theta: f32, peak_cos: f32, width_rad: f32) -> f32 {
    // Approximate the angular Gaussian via a Gaussian in cos-space.
    // For small width, cos(α + δ) ≈ cos(α) - sin(α)·δ, so a
    // width-σ in degrees becomes width-σ·sin(α) in cos-space.
    let sigma_cos = width_rad * sqrt(max(1.0 - peak_cos * peak_cos, 1e-4));
    let dx = (cos_theta - peak_cos) / max(sigma_cos, 1e-4);
    return exp(-0.5 * dx * dx);
}

fn cloud_phase(
    cos_theta: f32, layer: CloudLayerGpu, h_norm: f32, g_scale: f32,
) -> f32 {
    // Layer diameter, with the Cb mixed-phase transition.
    var d = layer.droplet_diameter_um;
    if (layer.cloud_type == 7u) {
        let mix_t = smoothstep(0.60, 0.85, clamp(h_norm, 0.0, 1.0));
        d = mix(layer.droplet_diameter_um, CB_ANVIL_DROPLET_DIAMETER_UM, mix_t);
    }
    d = clamp(d * params.droplet_diameter_bias, APPROX_MIE_D_MIN, APPROX_MIE_D_MAX);

    // Approximate Mie fit (Jendersie & d'Eon 2023, Eqs. 4–7).
    let g_hg    = exp(-0.0990567 / (d - 1.67154));
    let g_d     = exp(-2.20679 / (d + 3.91029)) - 0.428934;
    let alpha   = exp(3.62489 - 8.29288 / (d + 5.52825));
    let w_d     = exp(-0.599085 / (d - 0.641583) - 0.665888);

    // Apply Hillaire multi-octave `g_scale` to both `g` values; α
    // is fit-dependent and untouched. cos_theta is geometric.
    let p_hg = henyey_greenstein(cos_theta, g_hg * g_scale);
    let p_d  = draine_phase(cos_theta, g_d * g_scale, alpha);
    var p_mie = (1.0 - w_d) * p_hg + w_d * p_d;

    // Ice halo lobes — ramp in from the water/ice boundary. The
    // multi-octave g_scale also broadens the halo across successive
    // octaves (physically: forward-multiply-scattered light loses
    // angular precision), which keeps the halo from "punching
    // through" the diffuse multi-scatter background.
    let ice_fraction = smoothstep(ICE_DIAMETER_LOW, ICE_DIAMETER_HIGH, d);
    if (ice_fraction > 0.0) {
        let width = radians(ICE_HALO_WIDTH_DEG) / max(g_scale, 0.1);
        let h22 = ice_halo_lobe(cos_theta, ICE_HALO_22_COS, width);
        let h46 = ice_halo_lobe(cos_theta, ICE_HALO_46_COS, width);
        let halo = ICE_HALO_22_PEAK * h22 + ICE_HALO_46_PEAK * h46;
        p_mie = p_mie + ice_fraction * halo;
    }
    return p_mie;
}

// ---------------------------------------------------------------------------
// Density-height profile (NDF, 8 cloud types)
// ---------------------------------------------------------------------------

fn ndf(h: f32, t: u32, anvil_bias: f32) -> f32 {
    // All NDF profiles target a peak of ~0.78 so that scene-side
    // density_scale=1.0 produces comparable optical thickness across
    // cloud types. Per-type density character (cirrus = thin, cumulus
    // = puffy) is conveyed through the *shape* of the profile and the
    // scene's density_scale slider, not via per-type peak attenuation.
    // Followup #64.
    switch t {
        case 0u: { // Cumulus: bell, peak ~0.78
            return smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.2, h);
        }
        case 1u: { // Stratus: top-heavy, low and thin
            return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.6, 1.0, h));
        }
        case 2u: { // Stratocumulus: mid-heavy
            return smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.4, h);
        }
        case 3u: { // Altocumulus: mid-heavy
            return smoothstep(0.0, 0.20, h) * smoothstep(1.0, 0.3, h);
        }
        case 4u: { // Altostratus: top-heavy sheet
            return smoothstep(0.0, 0.30, h) * (1.0 - smoothstep(0.7, 1.0, h));
        }
        case 5u: { // Cirrus: thin top-heavy wisp profile
            return smoothstep(0.0, 0.40, h) * (1.0 - smoothstep(0.6, 1.0, h));
        }
        case 6u: { // Cirrostratus: very thin sheet, highest
            return smoothstep(0.0, 0.50, h) * (1.0 - smoothstep(0.8, 1.0, h));
        }
        case 7u: { // Cumulonimbus: bottom-heavy + anvil top
            // Phase 13 follow-up — anvil strength scales with the
            // per-layer `anvil_bias`. Default value 1.0 reproduces
            // the v1 hard-coded look; raise toward 2 for a heavier
            // anvil spread, lower toward 0 to suppress it entirely.
            let base  = smoothstep(0.0, 0.05, h);
            let mid   = smoothstep(0.95, 0.5, h);
            let anvil = smoothstep(0.7, 0.9, h) * 1.5 * clamp(anvil_bias, 0.0, 2.0);
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

/// Phase 14.C — horizontal advection offset (metres) for the cloud
/// noise lookup at world position `p` with precomputed altitude
/// `p_alt`. Samples the synthesised wind_field volume (which already
/// embeds the boundary-layer power law and either the ingested
/// pressure-level winds or the procedural Ekman veer, depending on
/// the scene) and scales by `simulated_seconds * wind_drift_strength`.
/// Returns a `vec3` where `.y = 0` for the v1 horizontal-only
/// advection — vertical advection is reserved for a follow-up phase
/// once we settle on the cloud-build semantics.
///
/// The returned offset is **subtracted** from the world-space sample
/// position before noise lookups: a cloud structure that physically
/// moves with the wind requires the texture coordinate to step
/// backwards by the same amount, so the cloud appears anchored to
/// the moving body rather than the fixed world.
fn cloud_wind_offset(p: vec3<f32>, p_alt: f32) -> vec3<f32> {
    let strength = params.wind_drift_strength * frame.simulated_seconds;
    if (strength == 0.0) {
        return vec3<f32>(0.0);
    }
    let half = WIND_FIELD_EXTENT_M * 0.5;
    let u_x = (p.x + half) / WIND_FIELD_EXTENT_M;
    let u_z = (p.z + half) / WIND_FIELD_EXTENT_M;
    // Map the precomputed altitude to the wind field's Y axis. Clamp
    // so samples above the top voxel hold the topmost wind rather
    // than wrapping into the boundary layer.
    let u_y = clamp(p_alt / WIND_FIELD_TOP_M, 0.0, 0.9999);
    // Wind field uses (u east, v south, w up, turb). The U/V channels
    // are world-space horizontal m/s.
    let w_sample = textureSampleLevel(wind_field, noise_sampler, vec3<f32>(u_x, u_y, u_z), 0.0);
    return vec3<f32>(w_sample.x * strength, 0.0, w_sample.y * strength);
}

/// Look up the effective cloud-type index for a sample position.
/// Phase 12.1 — when the per-pixel grid has a non-sentinel value at
/// this XZ, use it; otherwise fall back to the layer's default
/// `cloud_type`. The grid is sampled with `textureLoad` because
/// interpolating type indices is meaningless (you can't be "halfway
/// between cumulus and stratus" — you need a discrete value).
fn effective_cloud_type(p_xz: vec2<f32>, layer: CloudLayerGpu) -> u32 {
    // World XZ → texel coordinate in the 128×128 grid covering
    // [-CLOUD_TYPE_GRID_EXTENT_M/2, +/2] on each axis.
    let half = CLOUD_TYPE_GRID_EXTENT_M * 0.5;
    let u = (p_xz.x + half) / CLOUD_TYPE_GRID_EXTENT_M;
    let v = (p_xz.y + half) / CLOUD_TYPE_GRID_EXTENT_M;
    // Clamp to texel range so out-of-extent samples read the edge.
    let tx = clamp(i32(u * f32(CLOUD_TYPE_GRID_SIZE)),
                   0, CLOUD_TYPE_GRID_SIZE - 1);
    let ty = clamp(i32(v * f32(CLOUD_TYPE_GRID_SIZE)),
                   0, CLOUD_TYPE_GRID_SIZE - 1);
    let t = textureLoad(cloud_type_grid, vec2<i32>(tx, ty), 0).r;
    if (t == CLOUD_TYPE_SENTINEL) {
        return layer.cloud_type;
    }
    return t;
}

/// Schneider 2015/2017 cloud density at world position `p`.
/// `p_alt` is the precomputed altitude (m) at `p`.
fn sample_density(
    p: vec3<f32>, p_alt: f32, layer: CloudLayerGpu, weather: vec4<f32>,
) -> f32 {
    let h = (p_alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
    if (h < 0.0 || h > 1.0) { return 0.0; }

    // Phase 14.C — subtract the wind advection offset so the noise
    // lookups effectively trail the cloud body. Per-altitude wind
    // shear emerges automatically because the offset depends on
    // `p_alt`: cumulus tops at 3 km may see a different vector to
    // their base at 1.5 km, leaning the cloud downwind exactly as
    // physical clouds do. The coverage / cloud-type grids stay
    // anchored to the original `p.xz` (advection of those textures
    // is Phase 14.D).
    let p_advected = p - cloud_wind_offset(p, p_alt);

    let base_uv = p_advected / max(params.base_scale_m, 1.0);
    let base = textureSampleLevel(noise_base, noise_sampler, base_uv, 0.0);
    // Schneider remap: low-frequency Worley FBM erodes the Perlin-Worley.
    // Phase 13 follow-up — per-layer `shape_bias` shifts the LF FBM
    // sum (clamped to a small range so the remap stays well-formed).
    // Positive → more Worley FBM dominance → wispier cauliflower
    // structure on the base shape; negative → smoother bulbous body.
    let lf_fbm = clamp(
        base.g * 0.625 + base.b * 0.25 + base.a * 0.125 + layer.shape_bias * 0.5,
        0.0, 1.0,
    );
    let base_cloud = remap(base.r, -(1.0 - lf_fbm), 1.0, 0.0, 1.0);

    // Phase 12.1: per-pixel cloud type override (or layer default).
    let cloud_type = effective_cloud_type(p.xz, layer);
    let profile = ndf(h, cloud_type, layer.anvil_bias);
    var cloud = base_cloud * profile;

    // Coverage from synthesised weather × per-layer scalar.
    let coverage = clamp(weather.r * layer.coverage, 0.0, 1.0);
    cloud = remap(cloud, 1.0 - coverage, 1.0, 0.0, 1.0);
    cloud = cloud * coverage;

    // Curl-perturbed detail erosion at the cloud boundary. Only erode
    // where there is already some density (saves the per-sample texture
    // fetches in empty space). Phase 14.C — use the same advected
    // position as the base lookup so the curl pattern drifts with
    // the cloud body rather than the wind itself; otherwise the
    // detail wisps would slide across a stationary cloud.
    if (cloud > 0.0) {
        let curl_uv = p_advected.xz / max(params.detail_scale_m, 1.0);
        let curl = textureSampleLevel(noise_curl, noise_sampler, curl_uv, 0.0).rg * 2.0
                 - vec2<f32>(1.0);
        let detail_p = p_advected + vec3<f32>(curl.x, 0.0, curl.y) * params.curl_strength
                                     * params.detail_scale_m;
        let detail_uv = detail_p / max(params.detail_scale_m, 1.0);
        let detail = textureSampleLevel(noise_detail, noise_sampler, detail_uv, 0.0);
        // Phase 13 follow-up — per-layer `detail_bias` shifts the HF
        // erosion strength. Combined with the global
        // `params.detail_strength`, it lets each layer have its own
        // edge character (cirrus very wispy, cumulonimbus aggressively
        // eroded, stratus fairly smooth). Clamped to [0, 1] so the
        // remap stays well-formed.
        let hf_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
        // Schneider 2015: detail erosion polarity flips with height -- wispy
        // top, fluffy base.
        let detail_mod = mix(hf_fbm, 1.0 - hf_fbm, clamp(h * 10.0, 0.0, 1.0));
        let layer_detail = clamp(params.detail_strength + layer.detail_bias, 0.0, 1.0);
        cloud = remap(cloud, detail_mod * layer_detail, 1.0, 0.0, 1.0);
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

/// Phase 12.2 — fragment output. Two MRT attachments:
///   - `luminance`  : premultiplied RGB radiance from the cloud
///                    column, with aerial-perspective applied. The
///                    composite pass adds this to the destination
///                    HDR target unattenuated.
///   - `transmittance`: per-channel atmospheric transmittance
///                    through the entire cloud column. The composite
///                    pass multiplies the destination HDR target by
///                    this per channel before adding `luminance`.
///                    For a pixel with no cloud the cloud_march
///                    pass is cleared to (1, 1, 1, 1) and the
///                    composite leaves the dst unchanged.
struct CloudOut {
    @location(0) luminance: vec4<f32>,
    @location(1) transmittance: vec4<f32>,
};

struct LayerHit {
    idx: u32,
    t0: f32,
    t1: f32,
    hit: u32, // 0/1 flag (WGSL switch types want u32)
};

@fragment
fn fs_main(in: VsOut) -> CloudOut {
    let ray = compute_view_ray(in.pos.xy);
    let cos_theta = dot(ray.dir, frame.sun_direction.xyz);

    // Spatial blue-noise jitter (frame-deterministic — no time
    // component). Phase 13.9 adds an optional 16-frame rotation
    // (`params.temporal_jitter`) by XORing the lookup coords with a
    // frame-index-derived offset. The shifts pick deterministic
    // 6-bit offsets that don't preserve the spatial dither structure
    // across the cycle — exactly what a TAA accumulator wants.
    let jitter_xy = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    var sample_xy = jitter_xy & vec2<i32>(63);
    if (params.temporal_jitter != 0u) {
        let phase = frame.frame_index & 15u;
        let off_x = i32(((phase * 11u + 7u) >> 0u) & 63u);
        let off_y = i32(((phase * 23u + 13u) >> 0u) & 63u);
        sample_xy = vec2<i32>(
            (sample_xy.x ^ off_x) & 63,
            (sample_xy.y ^ off_y) & 63,
        );
    }
    let jitter = textureLoad(blue_noise, sample_xy, 0).r;

    // Phase 9.1 depth-aware termination: read the HDR depth at this
    // pixel and convert to a world-space distance along the view ray.
    // depth_ndc == 0 means the pixel reached the far plane (sky); the
    // march should run to its full per-layer interval. Otherwise t_max
    // is the distance from the camera to the opaque hit so clouds clip
    // behind ground / future opaque geometry.
    let depth_ndc = textureLoad(scene_depth, jitter_xy, 0);
    var depth_t_max: f32 = 1.0e30;
    if (depth_ndc > 0.0) {
        let viewport = frame.viewport_size.xy;
        let ndc = vec2<f32>(
            (in.pos.x / viewport.x) * 2.0 - 1.0,
            1.0 - (in.pos.y / viewport.y) * 2.0,
        );
        let h = frame.inv_view_proj * vec4<f32>(ndc, depth_ndc, 1.0);
        let world = h.xyz / h.w;
        depth_t_max = max(length(world - ray.origin), 0.0);
    }

    // Per-layer intervals.
    var hits: array<LayerHit, 8>;
    let layer_count = min(params.cloud_layer_count, MAX_CLOUD_LAYERS);
    for (var l = 0u; l < layer_count; l = l + 1u) {
        var t0: f32;
        var t1: f32;
        let ok = ray_layer_intersect(ray.origin, ray.dir,
                                     cloud_layers.layers[l], &t0, &t1);
        // Clip the layer interval against the opaque hit. If the entire
        // interval is behind the depth buffer, mark the layer as not hit.
        var flag: u32 = 0u;
        if (ok) {
            t1 = min(t1, depth_t_max);
            if (t1 > t0) { flag = 1u; }
        }
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
    // Phase 12.2b — per-channel extinction. sigma_s and sigma_a are
    // engine-wide grey baselines (vec3 only because of historical
    // chromaticity tuning); the per-cloud-type chromatic shift now
    // comes from `chromatic_mie_modulation(layer.droplet_diameter_um)`
    // applied below inside the layer loop.
    let sigma_t_baseline = params.sigma_s + params.sigma_a;
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

        // Per-layer chromatic extinction. Mie scattering is
        // wavelength-flat for typical cloud droplets (d ≥ ~20 µm),
        // but smaller droplet populations exhibit a Rayleigh-like
        // blue boost — visible as warm fringes on fog tops, thin
        // cirrus, and cumulus updraft tops at sunset. The
        // chromatic modulation centres on 1.0 at the median
        // wavelength so the achromatic optical thickness is
        // preserved.
        let sigma_t = sigma_t_baseline
                    * chromatic_mie_modulation(layer.droplet_diameter_um);

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

                // Sample's normalised height inside the layer. Used
                // by the Cumulonimbus mixed-phase branch in
                // `cloud_phase` to blend the droplet diameter from
                // water (core) to ice (anvil); other layers ignore it.
                let phase_h = clamp(
                    (alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0),
                    0.0, 1.0,
                );

                // Hillaire 2016 multi-octave multiple-scattering: each octave
                // scales energy×a, optical-depth×b, anisotropy×c. cos_theta
                // is geometric and never scaled; only g is.
                var sun_in = vec3<f32>(0.0);
                var a = 1.0;
                var b = 1.0;
                var c = 1.0;
                for (var n = 0u; n < params.multi_scatter_octaves; n = n + 1u) {
                    let phase = cloud_phase(cos_theta, layer, phase_h, c);
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

                // Phase 12.3 — lightning in-scatter contribution.
                // The aggregated emission is in
                // `frame.lightning_illuminance.rgb`; localise by
                // horizontal distance from the strongest active
                // strike's origin (`frame.lightning_origin_world.xyz`)
                // with a smooth falloff out to the radius packed in
                // `lightning_illuminance.w`. Lightning illumination
                // is broadly isotropic (it's volume emission inside
                // the cloud, not directional like the sun), so we
                // skip the phase function and feed the term directly
                // alongside ambient.
                var lightning = vec3<f32>(0.0);
                let r = frame.lightning_illuminance.w;
                if (r > 0.0) {
                    let d = length(p.xz - frame.lightning_origin_world.xz);
                    let falloff = clamp(1.0 - d / r, 0.0, 1.0);
                    // Cubic smoothstep — soft visible edge to the
                    // illuminated zone.
                    let weight = falloff * falloff * (3.0 - 2.0 * falloff);
                    lightning = frame.lightning_illuminance.rgb * weight;
                }

                let in_scatter = density * params.sigma_s
                                * (sun_in * energy + ambient + lightning);

                // Energy-conserving step integration. local_sigma is
                // per-channel after Phase 12.2b; the lower clamp uses
                // vec3 arithmetic.
                let local_sigma = max(density * sigma_t, vec3<f32>(1e-6));
                let sample_t = exp(-local_sigma * step);
                let s_int = (in_scatter - in_scatter * sample_t) / local_sigma;
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

    // Phase 12.2 — RGB transmittance compositing.
    //
    // attachment 0 = luminance (premultiplied along the ray;
    //                AP-applied below)
    // attachment 1 = transmittance (per-channel atmospheric
    //                transmittance through the cloud column,
    //                bypasses AP because AP attenuates the cloud's
    //                own emission, not the destination HDR pixel
    //                visible through the cloud — that pixel was
    //                already AP-attenuated in the sky/ground pass).
    //
    // A scalar opacity proxy `cloud_opacity` is still useful for
    // weighting AP inscatter contribution (a fully transparent
    // pixel should contribute no AP, since "AP from inside the
    // cloud" only makes physical sense where there's cloud).
    let t_lum = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    let cloud_opacity = 1.0 - t_lum;

    // Aerial perspective on the cloud luminance (plan §6.7). Sample
    // the AP LUT at the luminance-weighted depth so the haze is
    // applied where the cloud's light actually came from. Phase 13.1
    // — depth coordinate uses the exponential mapping
    // `ap_depth_uv` (50 m → 100 km).
    let t_cloud = t_weight / max(t_weight_norm, 1e-6);
    var luminance_out = luminance;
    if (cloud_opacity > 1e-4) {
        // The AP LUT is baked with gid.y=0 at screen top (the bake
        // negates ndc.y when reconstructing view_dir from
        // inv_view_proj). Sample with raw screen-fractional UV —
        // frag.y/h is 0 at top — so v=0 reads the row whose stored
        // ray matches this fragment.
        let viewport = frame.viewport_size.xy;
        // Phase 13.1 — exponential AP depth mapping; mirrors the
        // bake's spacing (50 m → 100 km). Inlined here because
        // cloud_march doesn't compose in the atmosphere LUT helpers.
        let d_safe = max(t_cloud, 50.0);
        let ap_z = clamp(log(d_safe / 50.0) / log(100000.0 / 50.0), 0.0, 1.0);
        let ap_uvw = vec3<f32>(
            in.pos.x / viewport.x,
            in.pos.y / viewport.y,
            ap_z,
        );
        let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);
        // Cloud's own light is dimmed by AP transmittance; AP
        // inscatter is added scaled by cloud_opacity so haze only
        // shows where the cloud is solid enough to see it against.
        luminance_out = luminance * ap.a + ap.rgb * cloud_opacity;
    }

    var out: CloudOut;
    out.luminance = vec4<f32>(luminance_out, 1.0);
    out.transmittance = vec4<f32>(transmittance, 1.0);
    return out;
}
