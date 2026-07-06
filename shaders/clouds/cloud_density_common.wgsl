// Cloud density kernel shared between the in-crate cloud march
// (`cloud_march.wgsl`) and host-owned shadow projections (e.g. a
// terrain cloud-shadow pass that projects cloud optical depth onto
// the ground). Factored out of `cloud_march.wgsl` verbatim — any
// change here alters both the rendered clouds and their shadows, so
// the two can never drift apart.
//
// This file declares no bindings of its own. Consumers must declare
// module-scope bindings with these exact names (WGSL module-scope
// declarations are order-independent, so this include can precede or
// follow them in the composed source):
//
//   frame            var<uniform> FrameUniforms       (common/uniforms.wgsl)
//   params           var<uniform> CloudParams          (clouds/cloud_uniforms.wgsl)
//   cloud_layers     var<storage, read> CloudLayerArray
//   wind_field       texture_3d<f32>
//   noise_base       texture_3d<f32>
//   noise_sampler    sampler (linear-repeat)
//   cloud_type_grid  texture_2d<u32>
//
// It also calls `remap` from `common/math.wgsl` and uses the
// `CloudParams` / `CloudLayerGpu` structs from
// `clouds/cloud_uniforms.wgsl`, so compose with
// `ps_core::shaders::{COMMON_UNIFORMS_WGSL, COMMON_MATH_WGSL}` and
// `ps_clouds::pipeline::cloud_uniforms_wgsl()` alongside this file
// (see `ps_clouds::pipeline::density_common_wgsl`).

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
