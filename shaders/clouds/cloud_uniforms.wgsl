// Phase 6 cloud march uniforms.
//
// CloudParams is owned by ps-clouds and uploaded once per frame. The
// per-layer envelope CloudLayerGpu mirrors the ps-core::weather struct
// (32 bytes); MAX_CLOUD_LAYERS matches the array length agreed with the
// Phase 3 synthesis pipeline.

const MAX_CLOUD_LAYERS: u32 = 8u;

struct CloudParams {
    sigma_s: f32,                 // 0.04 — scattering coefficient
    sigma_a: f32,                 // 0.0  — absorption
    g_forward: f32,               // 0.8
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
    _pad: u32,
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
