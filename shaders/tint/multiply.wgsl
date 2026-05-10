// Phase 1 demo TintSubsystem: per-fragment RGB multiply.
//
// Reads a scratch copy of the HDR target (sampled), multiplies by a
// uniform RGB factor, writes the HDR target. The scratch copy step
// exists because wgpu forbids sampling and render-attaching the same
// texture in a single pass.

struct TintUniforms {
    multiplier: vec4<f32>,
};

@group(0) @binding(0) var src_tex:  texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> tint: TintUniforms;

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
    out.uv  = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSample(src_tex, src_samp, in.uv).rgb;
    return vec4<f32>(src * tint.multiplier.rgb, 1.0);
}
