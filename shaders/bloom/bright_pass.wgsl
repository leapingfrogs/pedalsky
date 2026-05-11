// Phase 13.3 — bloom bright pass.
//
// Reads the HDR copy (full-res, sampled at half-res output via the
// linear sampler), thresholds against `config.x` with a soft knee
// of width `config.y`, and writes the isolated highlights to the
// half-res scratch.

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
    let rgb = textureSampleLevel(src, samp, in.uv, 0.0).rgb;
    let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let threshold = params.config.x;
    let knee = params.config.y;
    // Soft-knee bright pass: pixels at `lum >= threshold` pass full;
    // pixels in `[threshold - knee, threshold]` ramp from 0 → 1
    // smoothly so the bloom mask doesn't have a hard edge.
    let lo = max(threshold - knee, 1.0e-6);
    let weight = smoothstep(lo, threshold, lum);
    // Pre-multiply the bright pixels so the downstream blur runs
    // on already-isolated highlights rather than on a masked +
    // weighted blend. This matches the standard bloom recipe
    // (Karis 2013).
    return vec4<f32>(rgb * weight, weight);
}
