// Phase 0 placeholder ground shader: a 200×200 km quad in the XZ plane,
// shaded with a procedural checker. Phase 7 replaces this with a real PBR
// ground + wet surface; this exists to give the camera something to fly
// over while the rest of the engine comes online.

struct FrameUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    camera_position_world: vec4<f32>,
    viewport_size: vec4<f32>,
    time_seconds: f32,
    simulated_seconds: f32,
    frame_index: u32,
    ev100: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // 5 m squares — large enough to read while hovering at ~2 m AGL, small
    // enough to give shadows local contrast at distance.
    let cell = 5.0;
    let g = floor(in.world_pos.xz / cell);
    let parity = u32(abs(g.x + g.y)) & 1u;
    let dark  = vec3<f32>(0.05, 0.06, 0.07);
    let light = vec3<f32>(0.18, 0.18, 0.20);
    var albedo = select(light, dark, parity == 1u);

    // Distance fade towards a flat sky-grey so the 200 km quad fades out
    // smoothly rather than aliasing into a horizon line.
    let camera_xz = frame.camera_position_world.xz;
    let d = length(in.world_pos.xz - camera_xz);
    let fade = clamp(d / 5000.0, 0.0, 1.0);
    let horizon = vec3<f32>(0.10, 0.12, 0.14);
    albedo = mix(albedo, horizon, fade * 0.7);

    // Internal HDR is in cd/m² and very far from 0..1; multiply by a sky
    // illumination proxy until Phase 5's atmosphere LUTs land.
    let scene_luminance = 8000.0;
    return vec4<f32>(albedo * scene_luminance, 1.0);
}
