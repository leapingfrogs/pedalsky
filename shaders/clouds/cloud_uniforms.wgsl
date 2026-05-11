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
// `g_forward` deliberately follows `sigma_a` so it lives in those
// 4 trailing bytes — the Rust struct in `params.rs` mirrors this
// exact packing. The std140 linter in
// `crates/ps-app/tests/wgsl_layout.rs` enforces it per-field.
struct CloudParams {
    sigma_s: vec3<f32>,           // per-channel scattering /m
    _pad_after_sigma_s: f32,      // forces sigma_a onto a fresh 16 B chunk
    sigma_a: vec3<f32>,           // per-channel absorption /m
    g_forward: f32,               // packs into sigma_a's trailing 4 bytes

    g_backward: f32,              // -0.3
    g_blend: f32,                 // 0.5
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
    // The next three slots are std140 padding so the struct keeps
    // its 16-byte tail alignment.
    temporal_jitter: u32,
    _pad_temporal_jitter_0: u32,
    _pad_temporal_jitter_1: u32,
    _pad_temporal_jitter_2: u32,
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
};

struct CloudLayerArray {
    layers: array<CloudLayerGpu, MAX_CLOUD_LAYERS>,
};
