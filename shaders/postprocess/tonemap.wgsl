// Phase 0 tone-map pass.
//
// Fullscreen-triangle pass (no vertex buffer) reading the HDR target and
// writing the swapchain. ACES Filmic (Narkowicz fit) and Passthrough/clamp
// modes. Exposure as EV100; HDR luminance scaled by 1 / (1.2 * 2^EV100).

struct TonemapUniforms {
    ev100: f32,
    /// 0 = ACES Filmic, 1 = Passthrough/clamp.
    mode: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var hdr_tex:  texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@group(0) @binding(2) var<uniform> tm: TonemapUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // Larger-than-screen triangle so a single draw covers [0,1]² UV with no
    // gaps. UVs are flipped vertically since wgpu's clip-Y is +up but UV
    // origin is top-left.
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_samp, in.uv).rgb;
    let exposure = 1.0 / (1.2 * pow(2.0, tm.ev100));
    let scaled = hdr * exposure;
    var mapped: vec3<f32>;
    if (tm.mode == 0u) {
        mapped = aces_filmic(scaled);
    } else {
        mapped = clamp(scaled, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    // The swapchain is *Srgb, so we write linear and the GPU encodes.
    return vec4<f32>(mapped, 1.0);
}
