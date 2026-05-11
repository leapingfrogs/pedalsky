//! Plan §4.1 std140 cross-check, extended.
//!
//! ps-core already validates `FrameUniforms`, `WorldUniforms`, and
//! `SurfaceParams` (`crates/ps-core/tests/uniform_layout.rs`). This
//! integration test extends coverage to the per-subsystem uniform
//! mirrors that ps-core can't see:
//!
//! - `CloudParams`     ↔ `ps_clouds::CloudParamsGpu`
//! - `CloudLayerGpu`   ↔ `ps_core::CloudLayerGpu`
//! - `PrecipUniforms`  ↔ `ps_precip::uniforms::PrecipUniformsGpu`
//!
//! Each test parses the WGSL via naga, locates the named struct, and
//! asserts that:
//!   1. Total stride/span matches the Rust `size_of`.
//!   2. Each member's WGSL byte offset matches the Rust `offset_of!`.
//!
//! When a layout mismatch arises (e.g. a field added without a
//! matching `_pad`), this fires a focused error pointing at the
//! offending member rather than a wgpu validation panic at
//! pipeline-creation time.

use naga::front::wgsl::Frontend;

const CLOUD_UNIFORMS_WGSL: &str =
    include_str!("../../../shaders/clouds/cloud_uniforms.wgsl");
const PRECIP_ADVANCE_WGSL: &str =
    include_str!("../../../shaders/precip/particle_advance.comp.wgsl");

fn parse_module(src: &str) -> naga::Module {
    Frontend::new()
        .parse(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed: {e:#?}"))
}

/// Members of a named WGSL struct as `(field_name, byte_offset)` pairs.
fn struct_members(module: &naga::Module, name: &str) -> Vec<(String, u32)> {
    for (_, ty) in module.types.iter() {
        if let Some(ref n) = ty.name {
            if n == name {
                if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                    return members
                        .iter()
                        .map(|m| {
                            (
                                m.name.clone().unwrap_or_else(|| "<unnamed>".to_string()),
                                m.offset,
                            )
                        })
                        .collect();
                }
            }
        }
    }
    panic!("WGSL struct `{name}` not found in module");
}

fn struct_size(module: &naga::Module, name: &str) -> u32 {
    for (_, ty) in module.types.iter() {
        if let Some(ref n) = ty.name {
            if n == name {
                if let naga::TypeInner::Struct { span, .. } = &ty.inner {
                    return *span;
                }
            }
        }
    }
    panic!("WGSL struct `{name}` not found in module");
}

/// Assert WGSL field offsets equal the corresponding Rust offsets.
/// `expected` is `(rust_field_name, rust_offset)`. The WGSL field name
/// must match the Rust name verbatim.
fn assert_field_offsets(
    struct_label: &str,
    wgsl_members: &[(String, u32)],
    expected: &[(&str, u32)],
) {
    for (rust_name, rust_off) in expected {
        let Some((_, wgsl_off)) =
            wgsl_members.iter().find(|(name, _)| name == rust_name)
        else {
            panic!(
                "{struct_label}: WGSL has no member named `{rust_name}` \
                 (members: {:?})",
                wgsl_members.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
            );
        };
        assert_eq!(
            *wgsl_off, *rust_off,
            "{struct_label}::{rust_name} offset: WGSL = {wgsl_off}, Rust = {rust_off}"
        );
    }
}

// ---------------------------------------------------------------------------
// Clouds
// ---------------------------------------------------------------------------

#[test]
fn cloud_params_layout_matches_wgsl() {
    use ps_clouds::CloudParamsGpu;
    let module = parse_module(CLOUD_UNIFORMS_WGSL);

    let wgsl_size = struct_size(&module, "CloudParams");
    let rust_size = std::mem::size_of::<CloudParamsGpu>() as u32;
    assert_eq!(
        wgsl_size, rust_size,
        "WGSL CloudParams = {wgsl_size} B, Rust CloudParamsGpu = {rust_size} B"
    );

    let members = struct_members(&module, "CloudParams");
    assert_field_offsets(
        "CloudParams",
        &members,
        &[
            ("sigma_s", std::mem::offset_of!(CloudParamsGpu, sigma_s) as u32),
            ("sigma_a", std::mem::offset_of!(CloudParamsGpu, sigma_a) as u32),
            ("g_forward", std::mem::offset_of!(CloudParamsGpu, g_forward) as u32),
            ("g_backward", std::mem::offset_of!(CloudParamsGpu, g_backward) as u32),
            ("g_blend", std::mem::offset_of!(CloudParamsGpu, g_blend) as u32),
            ("detail_strength", std::mem::offset_of!(CloudParamsGpu, detail_strength) as u32),
            ("curl_strength", std::mem::offset_of!(CloudParamsGpu, curl_strength) as u32),
            ("powder_strength", std::mem::offset_of!(CloudParamsGpu, powder_strength) as u32),
            ("multi_scatter_a", std::mem::offset_of!(CloudParamsGpu, multi_scatter_a) as u32),
            ("multi_scatter_b", std::mem::offset_of!(CloudParamsGpu, multi_scatter_b) as u32),
            ("multi_scatter_c", std::mem::offset_of!(CloudParamsGpu, multi_scatter_c) as u32),
            ("ambient_strength", std::mem::offset_of!(CloudParamsGpu, ambient_strength) as u32),
            ("base_scale_m", std::mem::offset_of!(CloudParamsGpu, base_scale_m) as u32),
            ("detail_scale_m", std::mem::offset_of!(CloudParamsGpu, detail_scale_m) as u32),
            ("weather_scale_m", std::mem::offset_of!(CloudParamsGpu, weather_scale_m) as u32),
            ("light_steps", std::mem::offset_of!(CloudParamsGpu, light_steps) as u32),
            ("cloud_steps", std::mem::offset_of!(CloudParamsGpu, cloud_steps) as u32),
            ("multi_scatter_octaves", std::mem::offset_of!(CloudParamsGpu, multi_scatter_octaves) as u32),
            ("cloud_layer_count", std::mem::offset_of!(CloudParamsGpu, cloud_layer_count) as u32),
            ("temporal_jitter", std::mem::offset_of!(CloudParamsGpu, temporal_jitter) as u32),
        ],
    );
}

#[test]
fn cloud_layer_gpu_layout_matches_wgsl() {
    use ps_core::CloudLayerGpu;
    let module = parse_module(CLOUD_UNIFORMS_WGSL);

    let wgsl_size = struct_size(&module, "CloudLayerGpu");
    let rust_size = std::mem::size_of::<CloudLayerGpu>() as u32;
    assert_eq!(
        wgsl_size, rust_size,
        "WGSL CloudLayerGpu = {wgsl_size} B, Rust CloudLayerGpu = {rust_size} B"
    );

    let members = struct_members(&module, "CloudLayerGpu");
    assert_field_offsets(
        "CloudLayerGpu",
        &members,
        &[
            ("base_m", std::mem::offset_of!(CloudLayerGpu, base_m) as u32),
            ("top_m", std::mem::offset_of!(CloudLayerGpu, top_m) as u32),
            ("coverage", std::mem::offset_of!(CloudLayerGpu, coverage) as u32),
            ("density_scale", std::mem::offset_of!(CloudLayerGpu, density_scale) as u32),
            ("cloud_type", std::mem::offset_of!(CloudLayerGpu, cloud_type) as u32),
            ("shape_bias", std::mem::offset_of!(CloudLayerGpu, shape_bias) as u32),
            ("detail_bias", std::mem::offset_of!(CloudLayerGpu, detail_bias) as u32),
            ("anvil_bias", std::mem::offset_of!(CloudLayerGpu, anvil_bias) as u32),
        ],
    );
}

// ---------------------------------------------------------------------------
// Precipitation
// ---------------------------------------------------------------------------

#[test]
fn precip_uniforms_layout_matches_wgsl() {
    use ps_precip::uniforms::PrecipUniformsGpu;
    let module = parse_module(PRECIP_ADVANCE_WGSL);

    let wgsl_size = struct_size(&module, "PrecipUniforms");
    let rust_size = std::mem::size_of::<PrecipUniformsGpu>() as u32;
    assert_eq!(
        wgsl_size, rust_size,
        "WGSL PrecipUniforms = {wgsl_size} B, Rust PrecipUniformsGpu = {rust_size} B"
    );

    let members = struct_members(&module, "PrecipUniforms");
    assert_field_offsets(
        "PrecipUniforms",
        &members,
        &[
            ("camera_position", std::mem::offset_of!(PrecipUniformsGpu, camera_position) as u32),
            ("wind_velocity", std::mem::offset_of!(PrecipUniformsGpu, wind_velocity) as u32),
            ("intensity_mm_per_h", std::mem::offset_of!(PrecipUniformsGpu, intensity_mm_per_h) as u32),
            ("dt_seconds", std::mem::offset_of!(PrecipUniformsGpu, dt_seconds) as u32),
            ("simulated_seconds", std::mem::offset_of!(PrecipUniformsGpu, simulated_seconds) as u32),
            ("kind", std::mem::offset_of!(PrecipUniformsGpu, kind) as u32),
            ("particle_count", std::mem::offset_of!(PrecipUniformsGpu, particle_count) as u32),
            ("spawn_radius_m", std::mem::offset_of!(PrecipUniformsGpu, spawn_radius_m) as u32),
            ("spawn_top_m", std::mem::offset_of!(PrecipUniformsGpu, spawn_top_m) as u32),
            ("fall_speed_mps", std::mem::offset_of!(PrecipUniformsGpu, fall_speed_mps) as u32),
            ("user_seed", std::mem::offset_of!(PrecipUniformsGpu, user_seed) as u32),
        ],
    );
}
