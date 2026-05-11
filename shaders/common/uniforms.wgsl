// Canonical WGSL declarations for the engine-wide uniform bind groups.
// This file is the single source of truth for the std140 layout that
// `ps-core::frame_uniforms::FrameUniformsGpu` and
// `ps-core::weather::AtmosphereParams` (`WorldUniformsGpu`) mirror on the
// CPU side. The Phase 4 cross-check test in
// `crates/ps-core/tests/uniform_layout.rs` parses this file with `naga`
// and verifies every struct's size matches the Rust `repr(C)` equivalent.
//
// Subsystem shaders include this file by string-prepend at build time.

struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_position_world: vec4<f32>,
    camera_velocity_world: vec4<f32>,// xyz=m/s, w unused (Phase 8.2)
    sun_direction: vec4<f32>,        // xyz=dir, w=angular_radius_rad
    sun_illuminance: vec4<f32>,      // rgb=cd/m²·sr, w=lux at TOA
    lightning_illuminance: vec4<f32>,// rgb=lightning emission, w=falloff radius (m) (Phase 12.3)
    lightning_origin_world: vec4<f32>,// xyz=strongest active strike origin, w unused (Phase 12.3)
    viewport_size: vec4<f32>,        // w, h, 1/w, 1/h
    time_seconds: f32,
    simulated_seconds: f32,
    frame_index: u32,
    ev100: f32,
};

// Note: std140 vec3 alignment is 16 (size 16). The Rust mirror uses
// `[f32; 3]` (12 bytes contiguous) for `_pad_after_mie_g`, so the WGSL
// version expresses the same as three scalar f32s — std140 scalar
// alignment is 4, matching the Rust packing exactly.
struct WorldUniforms {
    planet_radius_m: f32,
    atmosphere_top_m: f32,
    rayleigh_scale_height_m: f32,
    mie_scale_height_m: f32,
    rayleigh_scattering: vec4<f32>,
    mie_scattering: vec4<f32>,
    mie_absorption: vec4<f32>,
    mie_g: f32,
    _pad_after_mie_g_0: f32,
    _pad_after_mie_g_1: f32,
    _pad_after_mie_g_2: f32,
    ozone_absorption: vec4<f32>,
    ozone_center_m: f32,
    ozone_thickness_m: f32,
    _pad_after_ozone_0: f32,
    _pad_after_ozone_1: f32,
    ground_albedo: vec4<f32>,
    haze_extinction_per_m: vec4<f32>,
};

// Mirrors `ps-core::weather::SurfaceParams`. 64 bytes (Phase 13.4
// added `material` + 3 pad scalars to reach the next vec4 boundary).
struct SurfaceParams {
    visibility_m: f32,
    temperature_c: f32,
    dewpoint_c: f32,
    pressure_hpa: f32,
    wind_dir_deg: f32,
    wind_speed_mps: f32,
    ground_wetness: f32,
    puddle_coverage: f32,
    snow_depth_m: f32,
    puddle_start: f32,
    precip_intensity_mm_per_h: f32,
    precip_kind: f32,
    material: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
