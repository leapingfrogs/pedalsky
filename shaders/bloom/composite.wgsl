// Phase 13.3 — bloom composite.
//
// Reads the fully-accumulated half-res bloom pyramid (level-0 after
// the upsample chain has folded the smaller levels in) and
// additively blends into the HDR target via the host-side blend
// state. `params.config.x` scales the contribution.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct Params { config: vec4<f32> };
@group(0) @binding(2) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let intensity = params.config.x;
    let bloom = textureSampleLevel(src, samp, in.uv, 0.0).rgb;
    return vec4<f32>(bloom * intensity, 1.0);
}
