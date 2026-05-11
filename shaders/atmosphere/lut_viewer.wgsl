// Phase 10.2 — fullscreen LUT viewer.
//
// Reads one of the four atmosphere LUTs (selected by `mode`) and
// draws it stretched to fill the swapchain.
//   mode 1 → transmittance LUT (256x64 RGB transmittance)
//   mode 2 → multi-scatter LUT (32x32 per-unit-illuminance radiance)
//   mode 3 → sky-view LUT (192x108 per-unit-illuminance radiance)
//   mode 4 → aerial-perspective LUT depth slice (32x32 per Z, where
//            Z = depth_slice in [0, 1])
//
// `scale` is a multiplier applied to the sampled value before output.

struct ViewerUniforms {
    mode: u32,
    _pad: u32,
    depth_slice: f32,
    scale: f32,
};

@group(0) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(1) var multiscatter_lut:  texture_2d<f32>;
@group(0) @binding(2) var skyview_lut:       texture_2d<f32>;
@group(0) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler:       sampler;
@group(0) @binding(5) var<uniform> viewer:   ViewerUniforms;

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
    var col = vec3<f32>(0.0, 0.0, 0.0);
    if (viewer.mode == 1u) {
        col = textureSampleLevel(transmittance_lut, lut_sampler, in.uv, 0.0).rgb;
    } else if (viewer.mode == 2u) {
        col = textureSampleLevel(multiscatter_lut, lut_sampler, in.uv, 0.0).rgb;
    } else if (viewer.mode == 3u) {
        col = textureSampleLevel(skyview_lut, lut_sampler, in.uv, 0.0).rgb;
    } else if (viewer.mode == 4u) {
        col = textureSampleLevel(
            aerial_perspective_lut, lut_sampler,
            vec3<f32>(in.uv, viewer.depth_slice), 0.0,
        ).rgb;
    } else {
        // mode == 0u: viewer disabled. Fragment should never run because
        // the host gates the draw call, but emit black for safety.
        col = vec3<f32>(0.0);
    }
    return vec4<f32>(col * viewer.scale, 1.0);
}
