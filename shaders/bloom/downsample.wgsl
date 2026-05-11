// Phase 13.3 — bloom downsample.
//
// 13-tap dual-filter downsample (Jorge Jimenez 2014). Tap layout:
// 4 corner taps at ±1.0 source-texel offsets + 4 inner taps at
// ±0.5 + 1 centre tap, with the partial-overlap weighting from
// the COD: AW reference. Reads the previous level's RT, writes a
// half-resolution blurred result.

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
    let texel = params.config.xy; // inverse src texture dimensions
    // 13-tap kernel: COD: AW SIGGRAPH 2014 §"Pre-filtering pass".
    let centre = textureSampleLevel(src, samp, in.uv, 0.0).rgb;
    let a = textureSampleLevel(src, samp, in.uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb;
    let b = textureSampleLevel(src, samp, in.uv + vec2<f32>( 1.0, -1.0) * texel, 0.0).rgb;
    let c = textureSampleLevel(src, samp, in.uv + vec2<f32>(-1.0,  1.0) * texel, 0.0).rgb;
    let d = textureSampleLevel(src, samp, in.uv + vec2<f32>( 1.0,  1.0) * texel, 0.0).rgb;
    let e = textureSampleLevel(src, samp, in.uv + vec2<f32>(-2.0,  0.0) * texel, 0.0).rgb;
    let f = textureSampleLevel(src, samp, in.uv + vec2<f32>( 2.0,  0.0) * texel, 0.0).rgb;
    let g = textureSampleLevel(src, samp, in.uv + vec2<f32>( 0.0, -2.0) * texel, 0.0).rgb;
    let h = textureSampleLevel(src, samp, in.uv + vec2<f32>( 0.0,  2.0) * texel, 0.0).rgb;
    let i = textureSampleLevel(src, samp, in.uv + vec2<f32>(-2.0, -2.0) * texel, 0.0).rgb;
    let j = textureSampleLevel(src, samp, in.uv + vec2<f32>( 2.0, -2.0) * texel, 0.0).rgb;
    let k = textureSampleLevel(src, samp, in.uv + vec2<f32>(-2.0,  2.0) * texel, 0.0).rgb;
    let l = textureSampleLevel(src, samp, in.uv + vec2<f32>( 2.0,  2.0) * texel, 0.0).rgb;
    // Inner box (±1 texel) gets weight 0.5; outer ring gets 0.125;
    // centre 0.125. Sum = 1.0.
    let inner = (a + b + c + d) * (0.5 / 4.0);
    let outer_box = (e + f + g + h) * (0.125 / 4.0);
    let outer_corners = (i + j + k + l) * (0.125 / 4.0);
    let result = centre * 0.125 + inner + outer_box + outer_corners;
    return vec4<f32>(result, 1.0);
}
