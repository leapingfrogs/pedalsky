// Phase 6 cloud march uniforms.
//
// CloudParams is owned by ps-clouds and uploaded once per frame. The
// per-layer envelope CloudLayerGpu mirrors the ps-core::weather struct
// (32 bytes); MAX_CLOUD_LAYERS matches the array length agreed with the
// Phase 3 synthesis pipeline.

const MAX_CLOUD_LAYERS: u32 = 8u;

// Phase 12.2b — per-channel sigma_s / sigma_a (Vec3) so the cloud
// march can integrate per-channel transmittance with chromatic Mie
// behaviour. WGSL packs `vec3<f32>` as 16-byte aligned but only 12
// bytes used; a following scalar fits into the trailing 4 bytes.
// `hg_forward_bias` deliberately follows `sigma_a` so it lives in
// those 4 trailing bytes — the Rust struct in `params.rs` mirrors
// this exact packing. The std140 linter in
// `crates/ps-app/tests/wgsl_layout.rs` enforces it per-field.
//
// Phase 13 follow-up B — the HG anisotropy moved per-layer
// (`CloudLayerGpu.g_forward / g_backward / g_blend`). What lives
// here is a global multiplier the user can dial as an artistic
// bias; default 1.0 reproduces the synthesised per-layer values.
struct CloudParams {
    sigma_s: vec3<f32>,           // per-channel scattering /m
    _pad_after_sigma_s: f32,      // forces sigma_a onto a fresh 16 B chunk
    sigma_a: vec3<f32>,           // per-channel absorption /m
    droplet_diameter_bias: f32,   // multiplier on layer.droplet_diameter_um; packs into sigma_a's trailing 4 bytes

    _pad_after_droplet_bias_0: f32, // retired hg_backward_bias slot
    _pad_after_droplet_bias_1: f32, // retired hg_blend_bias slot
    detail_strength: f32,         // 0.35
    curl_strength: f32,           // 0.1

    powder_strength: f32,         // 1.0
    multi_scatter_a: f32,         // 0.5
    multi_scatter_b: f32,         // 0.5
    multi_scatter_c: f32,         // 0.5

    ambient_strength: f32,        // 1.0
    base_scale_m: f32,            // 4500
    detail_scale_m: f32,          // 800
    weather_scale_m: f32,         // 32000

    light_steps: u32,             // 6
    cloud_steps: u32,             // 192
    multi_scatter_octaves: u32,   // 4
    cloud_layer_count: u32,

    // Phase 13.9 — optional per-frame temporal rotation of the
    // spatial blue-noise lookup (16-frame cycle). 0 = off, 1 = on.
    temporal_jitter: u32,
    // Phase 14.C — global scalar that multiplies the wind-driven
    // cloud advection offset (the cloud march samples wind_field at
    // each step and offsets the noise lookup by
    // `wind(altitude) * frame.simulated_seconds * wind_drift_strength`).
    // 1.0 = full physical drift; 0.0 = stationary clouds (golden
    // bless / paused screenshot path). Lives in the slot that used
    // to be `_pad_temporal_jitter_0`; struct size unchanged.
    wind_drift_strength: f32,
    // Phase 14.H — Schneider Nubis 2017 "skew with height". Offsets
    // the noise lookup by `h * layer_thickness * wind_dir *
    // wind_skew_strength` so cloud tops read from a position
    // downwind of their bases, producing visible cumulus lean and
    // anvil tilt under directional shear. 0.0 disables.
    wind_skew_strength: f32,
    // Phase 18 — diurnal modulation of `shape_bias` / `detail_bias`
    // for convective cloud types (Cu / Sc / Ac / Cb). Multiplied by
    // a smoothstep on solar altitude so biases peak at noon and go
    // to zero at night. 0.0 disables (biases lock at their Phase 17
    // baseline). See `diurnal_modulation` helper in cloud_march.wgsl.
    diurnal_strength: f32,
};

struct CloudLayerGpu {
    base_m: f32,
    top_m: f32,
    coverage: f32,
    density_scale: f32,
    cloud_type: u32,
    shape_bias: f32,
    detail_bias: f32,
    anvil_bias: f32,
    // Approximate Mie (Jendersie & d'Eon 2023) droplet effective
    // diameter in micrometres. Water clouds ~14–20 µm, ice clouds
    // ~50 µm. The cloud march evaluates Eqs. 4–7 of the paper at
    // runtime to derive (g_HG, g_D, α, w_D), then evaluates the
    // HG+Draine blend (Eq. 3). The global
    // `CloudParams.droplet_diameter_bias` multiplier scales the
    // diameter at render time so users can dial the apparent
    // droplet size without re-synthesising.
    droplet_diameter_um: f32,
    _pad_after_droplets_0: f32,
    _pad_after_droplets_1: f32,
    _pad_after_droplets_2: f32,
};

struct CloudLayerArray {
    layers: array<CloudLayerGpu, MAX_CLOUD_LAYERS>,
};
