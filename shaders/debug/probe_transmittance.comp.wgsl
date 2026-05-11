// Phase 10.A4 — probe-pixel transmittance readback.
//
// One thread runs per dispatch. Reconstructs the world-space view ray
// at the probe pixel (via inv_view_proj on the near plane), shifts the
// camera into atmosphere-local coordinates, and samples the
// transmittance LUT toward the view direction. Writes RGB transmittance
// to a single-element storage buffer the host reads back.
//
// Bindings:
//   group 0 binding 0 — FrameUniforms (frame)
//   group 1 binding 0 — WorldUniforms (world)
//   group 2 binding 0 — ProbeUniforms (probe pixel xy)
//   group 2 binding 1 — output storage buffer (vec4<f32>)
//   group 3 binding 0 — transmittance LUT
//   group 3 binding 4 — lut_sampler

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;

struct ProbeUniforms {
    pixel: vec2<u32>,
    _pad: vec2<u32>,
};
@group(2) @binding(0) var<uniform> probe: ProbeUniforms;
@group(2) @binding(1) var<storage, read_write> output: vec4<f32>;

@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;

@compute @workgroup_size(1)
fn cs_main() {
    let viewport = frame.viewport_size.xy;
    let px = vec2<f32>(f32(probe.pixel.x), f32(probe.pixel.y)) + vec2<f32>(0.5);
    let ndc = vec2<f32>(
        (px.x / viewport.x) * 2.0 - 1.0,
        1.0 - (px.y / viewport.y) * 2.0,
    );
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let view_dir = normalize(near_p - frame.camera_position_world.xyz);
    let cam_atm = world_to_atmosphere_pos(frame.camera_position_world.xyz);
    let t = sample_transmittance_lut(cam_atm, view_dir);
    output = vec4<f32>(t, 1.0);
}
