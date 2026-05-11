// Phase 13.3 — bloom upsample.
//
// 9-tap tent filter (Karis 2013) at half-source-pixel offsets.
// Reads the smaller pyramid level and writes its filtered upsample
// into the next-larger level via additive blending (set up on the
// host side by the bloom subsystem's pipeline state).

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
    let texel = params.config.xy; // inverse smaller-source dimensions
    let scale = params.config.z;  // additive scale per upsample step

    let a = textureSampleLevel(src, samp, in.uv + vec2<f32>(-1.0, -1.0) * texel, 0.0).rgb;
    let b = textureSampleLevel(src, samp, in.uv + vec2<f32>( 0.0, -1.0) * texel, 0.0).rgb;
    let c = textureSampleLevel(src, samp, in.uv + vec2<f32>( 1.0, -1.0) * texel, 0.0).rgb;
    let d = textureSampleLevel(src, samp, in.uv + vec2<f32>(-1.0,  0.0) * texel, 0.0).rgb;
    let e = textureSampleLevel(src, samp, in.uv,                                  0.0).rgb;
    let f = textureSampleLevel(src, samp, in.uv + vec2<f32>( 1.0,  0.0) * texel, 0.0).rgb;
    let g = textureSampleLevel(src, samp, in.uv + vec2<f32>(-1.0,  1.0) * texel, 0.0).rgb;
    let h = textureSampleLevel(src, samp, in.uv + vec2<f32>( 0.0,  1.0) * texel, 0.0).rgb;
    let i = textureSampleLevel(src, samp, in.uv + vec2<f32>( 1.0,  1.0) * texel, 0.0).rgb;

    // 9-tap tent: 1/16 corners, 2/16 edges, 4/16 centre.
    let result =
        (a + c + g + i) * (1.0 / 16.0) +
        (b + d + f + h) * (2.0 / 16.0) +
        e               * (4.0 / 16.0);
    return vec4<f32>(result * scale, 1.0);
}
