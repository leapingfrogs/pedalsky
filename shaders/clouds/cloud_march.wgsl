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

/// Loop-invariant outputs of the Mie fit, computed once per dense
/// sample and reused across each multi-scatter octave.
///
/// The Jendersie & d'Eon 2023 fit (Eqs. 4–7) feeds layer droplet
/// diameter through four `exp()` calls — none of which depend on
/// `cos_theta` or the octave's `g_scale`. Originally the full
/// `cloud_phase` was called once per octave (typically 4×), so each
/// dense sample paid for 16 `exp()` plus the ice smoothstep. Hoisting
/// these out (perf §2.2 of the audit) leaves the per-octave path with
/// just the two phase evaluations and an optional halo blend.
struct CloudPhaseConstants {
    g_hg: f32,
    g_d: f32,
    alpha: f32,
    w_d: f32,
    ice_fraction: f32,
}

fn cloud_phase_constants(layer: CloudLayerGpu, h_norm: f32) -> CloudPhaseConstants {
    // Layer diameter, with the Cb mixed-phase transition.
    var d = layer.droplet_diameter_um;
    if (layer.cloud_type == 7u) {
        let mix_t = smoothstep(0.60, 0.85, clamp(h_norm, 0.0, 1.0));
        d = mix(layer.droplet_diameter_um, CB_ANVIL_DROPLET_DIAMETER_UM, mix_t);
    }
    d = clamp(d * params.droplet_diameter_bias, APPROX_MIE_D_MIN, APPROX_MIE_D_MAX);

    var c: CloudPhaseConstants;
    c.g_hg         = exp(-0.0990567 / (d - 1.67154));
    c.g_d          = exp(-2.20679 / (d + 3.91029)) - 0.428934;
    c.alpha        = exp(3.62489 - 8.29288 / (d + 5.52825));
    c.w_d          = exp(-0.599085 / (d - 0.641583) - 0.665888);
    c.ice_fraction = smoothstep(ICE_DIAMETER_LOW, ICE_DIAMETER_HIGH, d);
    return c;
}

/// Per-octave phase evaluator. `g_scale` is Hillaire's multi-octave
/// anisotropy scale (1.0 on octave 0, `c^n` thereafter). Applies to
/// both HG and Draine `g` values; α is fit-dependent and untouched.
/// cos_theta is geometric and never scaled.
fn cloud_phase_with(cos_theta: f32, g_scale: f32, c: CloudPhaseConstants) -> f32 {
    let p_hg = henyey_greenstein(cos_theta, c.g_hg * g_scale);
    let p_d  = draine_phase(cos_theta, c.g_d * g_scale, c.alpha);
    var p_mie = (1.0 - c.w_d) * p_hg + c.w_d * p_d;

    // Ice halo lobes — ramp in from the water/ice boundary. The
    // multi-octave g_scale also broadens the halo across successive
    // octaves (physically: forward-multiply-scattered light loses
    // angular precision), which keeps the halo from "punching
    // through" the diffuse multi-scatter background.
    if (c.ice_fraction > 0.0) {
        let width = radians(ICE_HALO_WIDTH_DEG) / max(g_scale, 0.1);
        let h22 = ice_halo_lobe(cos_theta, ICE_HALO_22_COS, width);
        let h46 = ice_halo_lobe(cos_theta, ICE_HALO_46_COS, width);
        let halo = ICE_HALO_22_PEAK * h22 + ICE_HALO_46_PEAK * h46;
        p_mie = p_mie + c.ice_fraction * halo;
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

/// Phase 14.C/F — per-layer horizontal advection offset (metres XZ)
/// for both the cloud noise lookups and the coverage / cloud-type
/// grid lookups. Samples the wind field at the layer's mid-altitude
/// over a fixed XZ position (world origin) — the wind field is
/// approximately spatially uniform at the 32 km extent for a given
/// altitude, so one sample per layer captures the bulk drift of
/// that altitude sheet. This makes the cumulus layer drift at
/// ~700 hPa wind while a cirrus layer above it drifts at ~300 hPa
/// wind, producing the physical "cells move at the wind speed of
/// their altitude band" behaviour.
///
/// **Per-layer, not per-sample.** Phase 14.C originally used a
/// per-sample offset (sampling the wind at each ray step's `p_alt`),
/// which produced visible streaking: the wind at a cloud's top is
/// different from the wind at its base, so the noise lookup at the
/// top read from a different XZ tile than the noise at the base.
/// Over a few minutes the vertical XZ mismatch sheared cloud cells
/// horizontally and the rendered density looked like long streaks
/// instead of fluffy cells. Switching to one offset per layer keeps
/// each cloud's vertical column coherent while preserving the
/// inter-layer altitude shear. A separate "skew with height"
/// technique (Schneider Nubis 2017) can layer on top of this when
/// we want the visual lean of cumulus tops downwind of bases.
///
/// The advected coverage / cloud-type readouts wrap modulo the
/// texture extent (Repeat sampler / rem_euclid) so multi-hour
/// `simulated_seconds` doesn't run off the end of the tile.
/// Perf S.G3 — single 3D-texture sample of the wind field at the
/// layer's mid-altitude, returning everything the per-sample density
/// kernels need: the time-scaled drift offset and the unit-direction +
/// thickness-factor that power the height-scaled skew. Previously
/// `layer_wind_offset_m` and `layer_skew_xz` each fired a 3D
/// `textureSampleLevel`, and `layer_skew_xz` was called once per dense
/// step — so the wind field was being re-sampled ~10× per pixel for a
/// per-layer constant. Caching it here drops that to one sample per
/// layer per pixel.
struct LayerWindCache {
    /// `wind_xz * params.wind_drift_strength * frame.simulated_seconds`
    /// — the per-layer XZ noise-offset shared across every sample in
    /// the march. Zero when `freeze_time` / `wind_drift_strength = 0`.
    wind_offset: vec2<f32>,
    /// Unit-vector skew direction (zero when wind is calm or
    /// `wind_skew_strength = 0`). Final per-sample skew at height
    /// `h` is `skew_dir * (h * skew_thickness)`.
    skew_dir: vec2<f32>,
    /// `(layer.top_m - layer.base_m) * params.wind_skew_strength`.
    /// Zero when skew is disabled or wind is calm.
    skew_thickness: f32,
}

/// Phase 14.H — raw wind vector (m/s, world XZ) at the layer's
/// mid-altitude. The single 3D `textureSampleLevel` source feeding
/// both the time-drift offset and the skew direction.
fn layer_wind_mps(layer: CloudLayerGpu) -> vec2<f32> {
    let mid_alt = (layer.base_m + layer.top_m) * 0.5;
    let u_y = clamp(mid_alt / WIND_FIELD_TOP_M, 0.0, 0.9999);
    let w_sample = textureSampleLevel(
        wind_field, noise_sampler, vec3<f32>(0.5, u_y, 0.5), 0.0,
    );
    return vec2<f32>(w_sample.x, w_sample.y);
}

fn layer_wind_cache(layer: CloudLayerGpu) -> LayerWindCache {
    let wind_xz = layer_wind_mps(layer);
    let drift_strength = params.wind_drift_strength * frame.simulated_seconds;
    let wind_offset = wind_xz * drift_strength;

    let skew_strength = params.wind_skew_strength;
    var skew_dir = vec2<f32>(0.0);
    var skew_thickness = 0.0;
    // Skew uses only the wind **direction** (unit vector), not the
    // magnitude — leaning a cloud further than its own thickness when
    // the wind is strong is unrealistic (real clouds get shredded by
    // detail erosion instead). Calm wind ⇒ zero skew.
    if (skew_strength != 0.0) {
        let speed = length(wind_xz);
        if (speed >= 0.1) {
            skew_dir = wind_xz / speed;
            skew_thickness = max(layer.top_m - layer.base_m, 1.0) * skew_strength;
        }
    }
    return LayerWindCache(wind_offset, skew_dir, skew_thickness);
}

/// Phase 18 — diurnal modulation factor for the per-layer
/// `shape_bias` / `detail_bias`. Returns `[0, 1] * diurnal_strength`,
/// peaking around midday for **convective** cloud types and zero
/// for non-convective types (which have no diurnal signal — stratus
/// is smooth all day, cirrus is jet-driven not turbulence-driven).
///
/// Rationale: real cumulus character grows through the day as the
/// boundary layer warms and turbulence builds (peaks ~3 h after
/// solar noon in reality, though we use a symmetric bell here for
/// simplicity — asymmetric lag is a Phase 19 polish item if it
/// turns out to matter visually). Phase 17 set the per-fetch
/// baselines from CAPE / cover / shear; this scalar then animates
/// them over the simulated day.
///
/// `sun_direction.y = sin(altitude)`. The smoothstep ramps from
/// just below horizon (`-0.1`) to a moderately high sun (`0.4`,
/// ≈ 24° altitude) and plateaus above. Dawn and dusk produce
/// near-zero modulation; mid-morning to mid-afternoon produce the
/// full Phase 17 character.
fn diurnal_modulation(cloud_type: u32) -> f32 {
    let strength = params.diurnal_strength;
    if (strength == 0.0) {
        return 0.0;
    }
    // CloudType ordinals: 0=Cu, 1=St, 2=Sc, 3=Ac, 4=As, 5=Ci, 6=Cs, 7=Cb.
    // Match the convective set (Cu, Sc, Ac, Cb).
    let is_convective =
        cloud_type == 0u || cloud_type == 2u || cloud_type == 3u || cloud_type == 7u;
    if (!is_convective) {
        return 0.0;
    }
    let sin_alt = frame.sun_direction.y;
    let bell = smoothstep(-0.1, 0.4, sin_alt);
    return bell * strength;
}

/// Look up the effective cloud-type index for a sample position.
/// Phase 12.1 — when the per-pixel grid has a non-sentinel value at
/// this XZ, use it; otherwise fall back to the layer's default
/// `cloud_type`. The grid is sampled with `textureLoad` because
/// interpolating type indices is meaningless (you can't be "halfway
/// between cumulus and stratus" — you need a discrete value).
///
/// Phase 14.D — `p_xz` is expected to be **already offset** by the
/// caller via `layer_coverage_offset_m(layer)`, so advected cloud
/// cells migrate downwind. The texel index wraps modulo grid size
/// so samples that drift off the tile re-enter from the opposite
/// edge (cells aren't lost from the upwind boundary).
fn effective_cloud_type(p_xz: vec2<f32>, layer: CloudLayerGpu) -> u32 {
    // World XZ → texel coordinate in the 128×128 grid covering
    // [-CLOUD_TYPE_GRID_EXTENT_M/2, +/2] on each axis.
    let half = CLOUD_TYPE_GRID_EXTENT_M * 0.5;
    let u = (p_xz.x + half) / CLOUD_TYPE_GRID_EXTENT_M;
    let v = (p_xz.y + half) / CLOUD_TYPE_GRID_EXTENT_M;
    // Wrap into [0, SIZE). The double-mod handles negative inputs
    // (WGSL `%` follows truncated-division semantics, so a naive
    // single `%` leaves -1 as -1, not SIZE-1).
    let tx_raw = i32(floor(u * f32(CLOUD_TYPE_GRID_SIZE)));
    let ty_raw = i32(floor(v * f32(CLOUD_TYPE_GRID_SIZE)));
    let tx = ((tx_raw % CLOUD_TYPE_GRID_SIZE) + CLOUD_TYPE_GRID_SIZE) % CLOUD_TYPE_GRID_SIZE;
    let ty = ((ty_raw % CLOUD_TYPE_GRID_SIZE) + CLOUD_TYPE_GRID_SIZE) % CLOUD_TYPE_GRID_SIZE;
    let t = textureLoad(cloud_type_grid, vec2<i32>(tx, ty), 0).r;
    if (t == CLOUD_TYPE_SENTINEL) {
        return layer.cloud_type;
    }
    return t;
}

/// Schneider 2015/2017 cloud density at world position `p`.
/// `p_alt` is the precomputed altitude (m) at `p`.
///
/// Phase 14.F — takes the per-layer wind advection offset (computed
/// once by the caller at the layer's mid-altitude) and uses it for
/// both the noise lookups and the cloud-type grid lookup. Using one
/// offset per layer keeps a cloud cell's vertical column coherent;
/// the earlier per-sample variant (14.C) produced visible streaks
/// because the wind at a cell's top differs from the wind at its
/// base, shearing the noise pattern horizontally over time.
fn sample_density(
    p: vec3<f32>,
    p_alt: f32,
    layer: CloudLayerGpu,
    weather: vec4<f32>,
    wind: LayerWindCache,
    cloud_type: u32,
) -> f32 {
    let h = (p_alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
    if (h < 0.0 || h > 1.0) { return 0.0; }

    // Effective coverage gate. The Schneider remap further down
    // (`cloud = remap(cloud, 1 - coverage, 1, 0, 1) * coverage`)
    // silently produces near-zero density when `weather.r *
    // layer.coverage` is below the Schneider visible threshold
    // (≈ 0.40 at typical cloud profiles). Short-circuit here so
    // clear-sky pixels skip the three noise-texture fetches +
    // optional curl + detail taps on every primary march step.
    let effective_cov = clamp(weather.r * layer.coverage, 0.0, 1.0);
    if (effective_cov < 0.01) { return 0.0; }

    // Phase 14.F — subtract the per-layer wind offset so the noise
    // lookups effectively trail the cloud body. Phase 14.H — additional
    // height-scaled skew. Both pre-computed in `LayerWindCache` once
    // per layer (see audit S.G3) so the per-sample evaluation is now a
    // vector multiply rather than a 3D texture sample.
    let skew = wind.skew_dir * (h * wind.skew_thickness);
    let p_advected = p
        - vec3<f32>(wind.wind_offset.x, 0.0, wind.wind_offset.y)
        - vec3<f32>(skew.x, 0.0, skew.y);

    let base_uv = p_advected / max(params.base_scale_m, 1.0);
    let base = textureSampleLevel(noise_base, noise_sampler, base_uv, 0.0);
    // Phase 18 — diurnal modulation: convective types' bias scales
    // up with solar altitude (peaks at midday, vanishes at night).
    // Non-convective types ignore this. Loop-invariant within a
    // single sample_density call; the compiler hoists it.
    let diurnal = diurnal_modulation(layer.cloud_type);
    let effective_shape_bias = layer.shape_bias * diurnal;
    // Schneider remap: low-frequency Worley FBM erodes the Perlin-Worley.
    // Phase 13 follow-up — per-layer `shape_bias` shifts the LF FBM
    // sum (clamped to a small range so the remap stays well-formed).
    // Positive → more Worley FBM dominance → wispier cauliflower
    // structure on the base shape; negative → smoother bulbous body.
    let lf_fbm = clamp(
        base.g * 0.625 + base.b * 0.25 + base.a * 0.125 + effective_shape_bias * 0.5,
        0.0, 1.0,
    );
    let base_cloud = remap(base.r, -(1.0 - lf_fbm), 1.0, 0.0, 1.0);

    // Phase 12.1: per-pixel cloud type override (or layer default).
    // Perf S.G2 — `cloud_type` is now caller-provided (the primary
    // march computes it once per dense sample and threads it through
    // both `sample_density` and the light-march variants). Saves one
    // `effective_cloud_type` call per dense sample plus six per cone
    // tap in the light path.
    let profile = ndf(h, cloud_type, layer.anvil_bias);
    var cloud = base_cloud * profile;

    // Coverage from synthesised weather × per-layer scalar.
    let coverage = clamp(weather.r * layer.coverage, 0.0, 1.0);
    cloud = remap(cloud, 1.0 - coverage, 1.0, 0.0, 1.0);
    cloud = cloud * coverage;

    // Curl-perturbed detail erosion at the cloud boundary. Only erode
    // where there is already some density (saves the per-sample texture
    // fetches in empty space). Use the same advected position as the
    // base lookup so the curl pattern drifts with the cloud body
    // rather than the wind itself; otherwise the detail wisps would
    // slide across a stationary cloud.
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
        // Phase 18 — `detail_bias` takes the same diurnal modulation
        // as `shape_bias` so cumulus edges erode more aggressively
        // in the afternoon convective peak and revert to the smoother
        // baseline at night.
        let effective_detail_bias = layer.detail_bias * diurnal;
        let layer_detail = clamp(params.detail_strength + effective_detail_bias, 0.0, 1.0);
        cloud = remap(cloud, detail_mod * layer_detail, 1.0, 0.0, 1.0);
    }

    return clamp(cloud, 0.0, 1.0) * layer.density_scale;
}

/// Cheap density variant for the light march (perf §2.3 of the audit).
///
/// Same base-shape + coverage remap as [`sample_density`] but skips the
/// curl + detail noise taps. The light march integrates optical depth
/// across 6 cone samples toward the sun; high-frequency edge erosion
/// changes the per-sample value by <5% but costs three texture fetches
/// per step. Empirically the visual difference in shadow softness is
/// imperceptible at typical light-step counts because the detail tap's
/// per-sample variance averages out across the cone.
///
/// `weather`, `wind`, and `cloud_type` semantics match the full
/// variant — the primary march pre-computes them once per dense
/// sample and threads them through.
fn sample_density_light(
    p: vec3<f32>,
    p_alt: f32,
    layer: CloudLayerGpu,
    weather: vec4<f32>,
    wind: LayerWindCache,
    cloud_type: u32,
) -> f32 {
    let h = (p_alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
    if (h < 0.0 || h > 1.0) { return 0.0; }

    let effective_cov = clamp(weather.r * layer.coverage, 0.0, 1.0);
    if (effective_cov < 0.01) { return 0.0; }

    // Perf S.G3 — wind cache replaces the per-step `layer_skew_xz`
    // call (which used to fire a 3D `textureSampleLevel`).
    let skew = wind.skew_dir * (h * wind.skew_thickness);
    let p_advected = p
        - vec3<f32>(wind.wind_offset.x, 0.0, wind.wind_offset.y)
        - vec3<f32>(skew.x, 0.0, skew.y);

    let base_uv = p_advected / max(params.base_scale_m, 1.0);
    let base = textureSampleLevel(noise_base, noise_sampler, base_uv, 0.0);
    let diurnal = diurnal_modulation(layer.cloud_type);
    let effective_shape_bias = layer.shape_bias * diurnal;
    let lf_fbm = clamp(
        base.g * 0.625 + base.b * 0.25 + base.a * 0.125 + effective_shape_bias * 0.5,
        0.0, 1.0,
    );
    let base_cloud = remap(base.r, -(1.0 - lf_fbm), 1.0, 0.0, 1.0);

    let profile = ndf(h, cloud_type, layer.anvil_bias);
    var cloud = base_cloud * profile;

    let coverage = effective_cov;
    cloud = remap(cloud, 1.0 - coverage, 1.0, 0.0, 1.0);
    cloud = cloud * coverage;

    return clamp(cloud, 0.0, 1.0) * layer.density_scale;
}

// ---------------------------------------------------------------------------
// Light march (optical depth from sample to sun)
// ---------------------------------------------------------------------------

/// Light-march: optical depth from the primary sample `p` toward the
/// sun. `weather_sample` is the primary ray's already-fetched
/// `weather_map` value at `p.xz - wind_offset`; the light-march reuses
/// it for every step rather than re-fetching. The weather map varies
/// on a 32 km tile; a 6-step march toward the sun spans at most a
/// couple of km, so the weather value is effectively constant across
/// the cone (perf §2.5 of the audit).
fn march_to_light(
    p: vec3<f32>,
    p_alt: f32,
    sun_dir: vec3<f32>,
    layer: CloudLayerGpu,
    weather_sample: vec4<f32>,
    wind: LayerWindCache,
    cloud_type: u32,
) -> f32 {
    if (params.cone_light_sampling != 0u) {
        return march_to_light_cone(p, p_alt, sun_dir, layer, weather_sample, wind, cloud_type);
    }
    let cos_sun = max(sun_dir.y, 0.05);
    let dist_to_top = max((layer.top_m - p_alt) / cos_sun, 1.0);
    let step = dist_to_top / f32(params.light_steps);

    var od = 0.0;
    let centre = planet_centre();
    // Perf S.G1 — caller already has the precise altitude in `p_alt`
    // (computed via `altitude_from_entry` in the primary march), so
    // `r0 = planet_radius_m + p_alt` and `h0 = p_alt` directly. Saves
    // one `length()` per light-march invocation vs the previous
    // `length(p - centre)`.
    let r0 = world.planet_radius_m + p_alt;
    let cos_view = dot((p - centre) / max(r0, 1.0), sun_dir);
    let h0 = p_alt;

    for (var i = 0u; i < params.light_steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * step;
        let pi = p + sun_dir * t;
        let alt = altitude_from_entry(pi, r0, cos_view, h0, t);
        let local_density = sample_density_light(pi, alt, layer, weather_sample, wind, cloud_type);
        od = od + local_density * step;
    }
    return od;
}

/// Schneider / Nubis 2017 cone-tap light sampling.
///
/// Six samples toward the sun: five forward-stepped on a cone whose
/// half-angle widens with sample index, plus a sixth long-distance
/// tap (3× the layer crossing) that catches occlusion from the
/// distant cloud body. The widening cone gives the silver-lining
/// effect — thin material near a cumulus edge reads bright in
/// forward-scatter because the off-axis taps miss the cloud's
/// optical mass entirely, returning near-zero density and producing
/// a low integrated optical depth toward the sun.
///
/// The kernel is six pre-normalised unit-sphere offsets, hardcoded
/// in the original HZD Nubis implementation. Same offsets are used
/// for both forward and far taps; the radius scaling makes the cone
/// shape work out.
///
/// Step count is fixed at 6 (5 + 1); `params.light_steps` is
/// ignored when this path runs. That mirrors Schneider's recipe —
/// the cone count is part of the tuned kernel, not a slider.
fn cone_kernel_offset(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return vec3<f32>( 0.38051305,  0.92453449, -0.02111345); }
        case 1u: { return vec3<f32>(-0.50625799, -0.03590792, -0.86163418); }
        case 2u: { return vec3<f32>(-0.32509218, -0.94557439,  0.01428793); }
        case 3u: { return vec3<f32>( 0.09026238, -0.27376545,  0.95755165); }
        case 4u: { return vec3<f32>( 0.28128598,  0.42443639, -0.86065785); }
        default: { return vec3<f32>(-0.16852403,  0.14748697,  0.97460106); }
    }
}

/// Cone-tap variant of [`march_to_light`]. Reuses the primary ray's
/// `weather_sample` across all six taps — the cone spans at most a
/// few km, well below the weather map's 32 km tile, so the variation
/// is sub-texel.
fn march_to_light_cone(
    p: vec3<f32>,
    p_alt: f32,
    sun_dir: vec3<f32>,
    layer: CloudLayerGpu,
    weather_sample: vec4<f32>,
    wind: LayerWindCache,
    cloud_type: u32,
) -> f32 {
    let cos_sun = max(sun_dir.y, 0.05);
    let dist_to_top = max((layer.top_m - p_alt) / cos_sun, 1.0);
    // 5 forward samples evenly partitioning the path to the top of
    // the layer along the sun direction.
    let n_forward: u32 = 5u;
    let step_size = dist_to_top / f32(n_forward);

    let centre = planet_centre();
    // Perf S.G1 — see march_to_light for the rationale.
    let r0 = world.planet_radius_m + p_alt;
    let cos_view = dot((p - centre) / max(r0, 1.0), sun_dir);
    let h0 = p_alt;

    var od = 0.0;
    // Forward cone samples. The cone half-radius at sample i is
    // proportional to `(i + 1) / n_forward`, so the first sample sits
    // close to the ray (small offset, conservative shadow estimate)
    // and the last is at the widest cone angle (catches off-axis
    // structure that contributes to the silver lining).
    for (var i = 0u; i < n_forward; i = i + 1u) {
        let t = (f32(i) + 0.5) * step_size;
        let cone_off = cone_kernel_offset(i)
                     * step_size * (f32(i) + 1.0) / f32(n_forward);
        let pi = p + sun_dir * t + cone_off;
        let alt = altitude_from_entry(pi, r0, cos_view, h0, t);
        let local_density = sample_density_light(pi, alt, layer, weather_sample, wind, cloud_type);
        od = od + local_density * step_size;
    }

    // Long-distance "anti-shadow" tap (Schneider's 6th sample). Sits
    // 3× the layer-crossing distance further out and captures the
    // bulk of a distant cloud cell that still shadows the current
    // sample. Its effective integration length is wider than a step
    // so we weight it with 3× step_size.
    let t_far = dist_to_top * 3.0;
    let cone_off_far = cone_kernel_offset(5u) * step_size * 3.0;
    let pi_far = p + sun_dir * t_far + cone_off_far;
    let alt_far = altitude_from_entry(pi_far, r0, cos_view, h0, t_far);
    let local_density_far =
        sample_density_light(pi_far, alt_far, layer, weather_sample, wind, cloud_type);
    od = od + local_density_far * step_size * 3.0;

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
    // Perf §2.2 — forward-bias directional factor is per-pixel
    // (cos_theta = ray·sun, fixed across the march). Hoist out of the
    // per-sample hot loop.
    let forward_bias_dir_factor = smoothstep(0.0, 1.0, max(cos_theta, 0.0));

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
        // Outer early-exit: stacked-layer scenes (real weather often
        // hits 3–4 layers — boundary cumulus + altostratus + cirrus)
        // pay ~N× the single-layer cost. Once the nearer layers have
        // driven transmittance below 1% the back layers contribute
        // imperceptibly per pixel, so skip them rather than running
        // a fresh 192-step march that mostly accumulates rounding
        // error. Mirrors the inner-loop break threshold (0.01).
        if (max3(transmittance) < 0.01) { break; }
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

        // Phase 14.D/F + perf S.G3 — single wind-field 3D-texture
        // sample per layer, packaged into a `LayerWindCache` that
        // carries the time-scaled drift offset plus the precomputed
        // skew direction and thickness factor. Loop-invariant within
        // this layer's march; threaded into `sample_density`,
        // `sample_density_light`, and the light-march helpers so they
        // never re-sample the wind field per step.
        let wind = layer_wind_cache(layer);

        for (var s = 0u; s < params.cloud_steps; s = s + 1u) {
            let p = ray.origin + ray.dir * t;
            let alt = altitude_from_entry(p, r0, cos_view, h0, t);
            let weather_sample = textureSampleLevel(
                weather_map, noise_sampler,
                world_to_weather_uv(p.xz - wind.wind_offset), 0.0,
            );
            // Perf S.G2 — `effective_cloud_type` is computed once per
            // dense sample and reused across `sample_density` plus all
            // six cone taps in `sample_density_light`. The 6-tap cone
            // spans a few km vs the cloud-type grid's 250 m / texel,
            // so reusing the primary sample's lookup is sub-texel
            // accurate.
            let cloud_type = effective_cloud_type(p.xz - wind.wind_offset, layer);
            let density = sample_density(p, alt, layer, weather_sample, wind, cloud_type);

            if (density > 1e-3) {
                let od_to_sun = march_to_light(p, alt, sun_dir, layer, weather_sample, wind, cloud_type);

                // Schneider 2015 Beer-Powder (canonical form).
                let beer        = exp(-sigma_t * od_to_sun);
                let powder      = 1.0 - exp(-2.0 * sigma_t * od_to_sun);
                let beer_powder = beer * powder * 2.0;
                let energy = mix(beer, beer_powder, params.powder_strength);

                // Sample's normalised height inside the layer. Used
                // by the Cumulonimbus mixed-phase branch in
                // `cloud_phase_constants` to blend the droplet
                // diameter from water (core) to ice (anvil); other
                // layers ignore it.
                let phase_h = clamp(
                    (alt - layer.base_m) / max(layer.top_m - layer.base_m, 1.0),
                    0.0, 1.0,
                );
                // Perf §2.2 — precompute the Mie fit's loop-invariant
                // exp()s once per dense sample. The per-octave loop
                // below only needs the cheap HG + Draine evals.
                let phase_consts = cloud_phase_constants(layer, phase_h);

                // Hillaire 2016 multi-octave multiple-scattering: each octave
                // scales energy×a, optical-depth×b, anisotropy×c. cos_theta
                // is geometric and never scaled; only g is.
                var sun_in = vec3<f32>(0.0);
                var a = 1.0;
                var b = 1.0;
                var c = 1.0;
                for (var n = 0u; n < params.multi_scatter_octaves; n = n + 1u) {
                    let phase = cloud_phase_with(cos_theta, c, phase_consts);
                    sun_in = sun_in + a * frame.sun_illuminance.rgb
                                    * phase
                                    * exp(-sigma_t * od_to_sun * b);
                    a = a * params.multi_scatter_a;
                    b = b * params.multi_scatter_b;
                    c = c * params.multi_scatter_c;
                }

                // Forward bias: directional gain on the multi-scatter
                // sum that amplifies in-cloud forward-scatter when the
                // camera looks roughly toward the sun. Falls off
                // smoothly from full effect at the sun (cos_theta = 1)
                // to zero effect perpendicular to it (cos_theta = 0).
                // 0.0 reproduces the unbiased Hillaire model.
                //
                // Why this is structured as a post-multiply rather
                // than a per-octave `a` boost: the natural HG `g` for
                // water droplets (~0.99) gives octave 0 an extremely
                // narrow forward peak — almost all pixels off-axis
                // see zero from it, so scaling octave 0's energy is
                // visually invisible at any camera angle except the
                // sun disk. The directional-gain formulation lifts
                // the entire wide-angle "sun glow through cloud" band
                // that the user actually perceives as in-cloud sun
                // shafts (Schneider's HZD paper uses a similar
                // post-multiplier for the same reason).
                if (params.forward_bias > 0.0) {
                    sun_in = sun_in * (1.0 + params.forward_bias * forward_bias_dir_factor);
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
