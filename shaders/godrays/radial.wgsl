// Phase 12.4 — godrays radial accumulation pass.
//
// Reads the post-cloud HDR target and the sun's screen-space NDC
// position; for each output pixel marches N samples along the ray
// from the pixel toward the sun, accumulating brightness with
// exponential decay. Writes the result to a half-res scratch RT;
// the composite pass then additively blends it back into the HDR.
//
// Bindings:
//   group 0 binding 0     FrameUniforms (frame)
//   group 1 binding 0     hdr_input (texture_2d<f32>)
//   group 1 binding 1     hdr_sampler
//   group 1 binding 2     GodraysParams uniform

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var hdr_input: texture_2d<f32>;
@group(1) @binding(1) var hdr_sampler: sampler;
@group(1) @binding(2) var<uniform> params: GodraysParams;

struct GodraysParams {
    /// Sun position in screen-space NDC (xy). `z` = 1.0 if the
    /// sun is on-screen and above the horizon, 0.0 otherwise.
    sun_ndc: vec4<f32>,
    /// Tunables packed:
    ///   .x = samples (cast to int)
    ///   .y = decay (per-sample multiplier in [0.9, 1.0])
    ///   .z = intensity (final additive scalar)
    ///   .w = bright_threshold (cd/m²)
    tunables: vec4<f32>,
};

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

/// Soft bright-pass: pixels above `bright_threshold` contribute
/// linearly; dim pixels are zeroed. Returns RGB (per-channel
/// thresholded, so chromatic content survives).
fn bright_pass(rgb: vec3<f32>, threshold: f32) -> vec3<f32> {
    let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mask = clamp((lum - threshold) / max(threshold, 1.0), 0.0, 1.0);
    return rgb * mask;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // If the sun isn't on-screen, contribute nothing.
    if (params.sun_ndc.z < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let n_samples = i32(params.tunables.x);
    let decay = params.tunables.y;
    let bright_threshold = params.tunables.w;

    // Sun position in UV space (top-left origin, [0, 1]).
    // sun_ndc.x in [-1, 1] mapped to [0, 1] horizontally;
    // sun_ndc.y is +1 at top of NDC -> 0 in UV.
    let sun_uv = vec2<f32>(
        params.sun_ndc.x * 0.5 + 0.5,
        1.0 - (params.sun_ndc.y * 0.5 + 0.5),
    );

    // Step from this pixel toward the sun, in UV space.
    let delta = (sun_uv - in.uv) / f32(n_samples);

    var accum = vec3<f32>(0.0);
    var weight = 1.0;
    var uv = in.uv;
    for (var i = 0; i < n_samples; i = i + 1) {
        let sample = textureSampleLevel(hdr_input, hdr_sampler, uv, 0.0).rgb;
        accum = accum + bright_pass(sample, bright_threshold) * weight;
        weight = weight * decay;
        uv = uv + delta;
    }
    // Average across samples to keep the integral roughly
    // independent of sample count.
    accum = accum / f32(n_samples);
    return vec4<f32>(accum, 1.0);
}
