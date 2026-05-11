// Phase 6.7 — cloud composite into the HDR target.
//
// Phase 12.2 — RGB transmittance via dual-source blending.

enable dual_source_blending;
//
// The cloud march writes two MRT attachments:
//   binding 0: premultiplied luminance from the cloud column
//              (AP-applied; ready to add to dst)
//   binding 1: per-channel atmospheric transmittance through the
//              cloud column (1.0 where there's no cloud)
//
// This pass emits both as dual-source blend outputs at @location(0).
// The pipeline blend state is configured as
//   final.rgb = src0.rgb * One  +  dst.rgb * src1.rgb
// which is exactly the per-channel compositing equation: dst sky
// (atmosphere or ground) is dimmed by the cloud's per-channel
// transmittance, and the cloud's own light is added unattenuated.

@group(0) @binding(0) var cloud_luminance: texture_2d<f32>;
@group(0) @binding(1) var cloud_transmittance: texture_2d<f32>;
@group(0) @binding(2) var cloud_sampler: sampler;

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

struct CompositeOut {
    /// src0: premultiplied luminance, added unattenuated to dst.
    @location(0) @blend_src(0) luminance: vec4<f32>,
    /// src1: per-channel transmittance; multiplies dst before add.
    @location(0) @blend_src(1) transmittance: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> CompositeOut {
    let luminance =
        textureSampleLevel(cloud_luminance, cloud_sampler, in.uv, 0.0);
    let transmittance =
        textureSampleLevel(cloud_transmittance, cloud_sampler, in.uv, 0.0);
    var out: CompositeOut;
    out.luminance = luminance;
    out.transmittance = transmittance;
    return out;
}
