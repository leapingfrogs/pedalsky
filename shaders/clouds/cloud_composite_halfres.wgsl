// Cloud composite — half-res Catmull-Rom upsample variant.
//
// Identical to `cloud_composite.wgsl` except the two texture reads
// of the cloud RT (luminance + transmittance) are replaced by a
// 9-tap Catmull-Rom bicubic reconstruction (Sigvardsson, "Bicubic
// Filtering in Fewer Taps", https://vec3.ca/bicubic-filtering-in-fewer-taps/).
//
// The 9-tap variant collapses the 16-nearest-tap Catmull-Rom kernel
// into 9 bilinear-sampled taps positioned at fractional offsets so
// the hardware bilinear filter produces the correct Catmull-Rom
// blend. Output is mathematically identical to the naive 16-tap
// kernel at roughly 2× the speed. This is the canonical
// reconstruction kernel for volumetric clouds (Frostbite, Decima,
// Horizon Zero Dawn).
//
// This file is compiled into its own pipeline (`march_halfres`'s
// composite counterpart). The full-res pipeline keeps using the
// unmodified `cloud_composite.wgsl` so its output is byte-identical
// to the pre-toggle baseline.

enable dual_source_blending;

@group(0) @binding(0) var cloud_luminance: texture_2d<f32>;
@group(0) @binding(1) var cloud_transmittance: texture_2d<f32>;
@group(0) @binding(2) var cloud_sampler: sampler;
@group(0) @binding(3) var<uniform> params: CloudCompositeParams;

/// `(w, h, 1/w, 1/h)` of the cloud RT — the half-res pixel grid the
/// Catmull-Rom kernel reconstructs from. Written by the cloud
/// subsystem each frame from the live cloud RT dimensions.
struct CloudCompositeParams {
    cloud_rt_size: vec4<f32>,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

/// 9-tap Catmull-Rom bicubic reconstruction. `uv ∈ [0, 1]` over the
/// source texture; `tex_size` is the source texture's pixel
/// dimensions. The sampler bound to `tex` must be Linear /
/// clamp-to-edge — the kernel's offset-bilinear trick relies on the
/// hardware filter weighting two adjacent texels per axis.
fn catmull_rom_9_tap(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    tex_size: vec2<f32>,
) -> vec4<f32> {
    let sample_pos = uv * tex_size;
    let tex_pos_1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos_1;

    // Catmull-Rom kernel weights (B = 0, C = 0.5).
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    // Collapse the middle two weights into a single offset tap so the
    // hardware bilinear sample reproduces their combined contribution.
    let w12 = w1 + w2;
    let offset_12 = w2 / max(w12, vec2<f32>(1e-6));

    let inv_size = 1.0 / tex_size;
    let pos_0  = (tex_pos_1 - 1.0)       * inv_size;
    let pos_3  = (tex_pos_1 + 2.0)       * inv_size;
    let pos_12 = (tex_pos_1 + offset_12) * inv_size;

    var result = vec4<f32>(0.0);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_0.x,  pos_0.y),  0.0) * (w0.x  * w0.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_12.x, pos_0.y),  0.0) * (w12.x * w0.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_3.x,  pos_0.y),  0.0) * (w3.x  * w0.y);

    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_0.x,  pos_12.y), 0.0) * (w0.x  * w12.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_12.x, pos_12.y), 0.0) * (w12.x * w12.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_3.x,  pos_12.y), 0.0) * (w3.x  * w12.y);

    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_0.x,  pos_3.y),  0.0) * (w0.x  * w3.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_12.x, pos_3.y),  0.0) * (w12.x * w3.y);
    result = result + textureSampleLevel(tex, samp, vec2<f32>(pos_3.x,  pos_3.y),  0.0) * (w3.x  * w3.y);

    return result;
}

struct CompositeOut {
    /// src0: premultiplied luminance, added unattenuated to dst.
    @location(0) @blend_src(0) luminance: vec4<f32>,
    /// src1: per-channel transmittance; multiplies dst before add.
    @location(0) @blend_src(1) transmittance: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> CompositeOut {
    let tex_size = params.cloud_rt_size.xy;
    var luminance =
        catmull_rom_9_tap(cloud_luminance, cloud_sampler, in.uv, tex_size);
    var transmittance =
        catmull_rom_9_tap(cloud_transmittance, cloud_sampler, in.uv, tex_size);
    // Catmull-Rom rings negative on sharp transitions. Keep the
    // premultiplied luminance non-negative for the additive blend
    // and clamp per-channel transmittance into `[0, 1]` so the
    // dual-source `dst * src1` term can't amplify the destination.
    luminance = max(luminance, vec4<f32>(0.0));
    transmittance = clamp(transmittance, vec4<f32>(0.0), vec4<f32>(1.0));
    var out: CompositeOut;
    out.luminance = luminance;
    out.transmittance = transmittance;
    return out;
}
