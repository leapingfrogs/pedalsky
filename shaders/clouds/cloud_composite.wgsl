// Phase 6.7 — premultiplied-alpha cloud composite.
//
// Reads the cloud RT (premultiplied luminance, scalar opacity) and blits
// it over the HDR target with the standard `One, OneMinusSrcAlpha` blend
// state. The blend state is configured on the pipeline; this fragment
// just emits the cloud sample unchanged.

@group(0) @binding(0) var cloud_target: texture_2d<f32>;
@group(0) @binding(1) var cloud_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(cloud_target, cloud_sampler, in.uv, 0.0);
}
