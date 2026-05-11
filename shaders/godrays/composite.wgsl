// Phase 12.4 — godrays composite pass.
//
// Reads the half-res radial-accumulation RT (built by radial.wgsl)
// and adds it (scaled by intensity) to the HDR target via standard
// additive blending: `dst = dst + src.rgb * intensity`.
//
// Bindings:
//   group 0 binding 0     godrays_rt (texture_2d<f32>)
//   group 0 binding 1     godrays_sampler (linear, for upscale)
//   group 0 binding 2     GodraysCompositeParams uniform

@group(0) @binding(0) var godrays_rt: texture_2d<f32>;
@group(0) @binding(1) var godrays_sampler: sampler;
@group(0) @binding(2) var<uniform> params: GodraysCompositeParams;

struct GodraysCompositeParams {
    /// .x = intensity scalar
    /// .y = enabled (1 = blend in, 0 = no-op)
    /// .zw = pad
    config: vec4<f32>,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (params.config.y < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let rays = textureSampleLevel(godrays_rt, godrays_sampler, in.uv, 0.0).rgb;
    return vec4<f32>(rays * params.config.x, 0.0);
}
