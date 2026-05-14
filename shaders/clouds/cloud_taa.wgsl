// Cloud temporal anti-aliasing (TAA) pass.
//
// Runs between the cloud march and the cloud composite. Reads the
// current frame's raw march output (luminance + transmittance), the
// previous frame's resolved output via UV reprojection, and emits
// the blended "resolved" pair on two MRT attachments that the
// composite then reads in place of the raw march.
//
// Algorithm (Karis 2014 / Frostbite 2014 / Hillaire 2020 §6, simplified):
//
//   1. Reproject this fragment's screen position into the previous
//      frame using `frame.prev_view_proj` applied to the view ray
//      reconstructed from the current frame's inv_view_proj. Cloud
//      depth is treated as infinity (camera translation between
//      frames is ignored — for ground-based observers at typical
//      fly-camera speeds the parallax against clouds 1–15 km away
//      is sub-pixel per frame, so the simpler model holds).
//
//   2. Sample the history at the reprojected UV (bilinear).
//
//   3. Compute the min/max neighbourhood AABB of the current
//      frame's luminance + transmittance over a 3×3 window. Clamp
//      the history sample into that AABB — this is the standard
//      ghosting-suppression mechanism (Karis 2014); when the
//      camera rotates and previously-occluded pixels disocclude,
//      the history sample lies outside the local neighbourhood
//      and gets pulled toward valid current values.
//
//   4. Blend: resolved = mix(history_clamped, current, blend_weight).
//      `blend_weight = 1.0 / 8.0` gives effective 8-frame
//      accumulation. The CPU passes `blend_weight = 1.0` on the
//      very first frame (or after a size change / TAA toggle), so
//      the output equals the current sample.
//
// History validity is the responsibility of the host: when the
// cloud RT resizes or TAA is first enabled, the host writes the
// uniform with `history_valid = 0u`, and the shader treats it as
// "no history" — output = current sample, no reprojection.

@group(0) @binding(0) var current_luminance: texture_2d<f32>;
@group(0) @binding(1) var current_transmittance: texture_2d<f32>;
@group(0) @binding(2) var history_luminance: texture_2d<f32>;
@group(0) @binding(3) var history_transmittance: texture_2d<f32>;
@group(0) @binding(4) var taa_sampler: sampler;
@group(0) @binding(5) var<uniform> taa_params: CloudTaaParams;

/// Per-frame TAA tunables and validity flags.
///
/// Packed so the struct is a single vec4 slot followed by another:
///   `.x` = blend weight (current sample weight; history weight = 1 - w).
///         Host writes 1.0 on the first-valid-history frame to bootstrap.
///   `.y` = history_valid: 1.0 if reprojection is meaningful, 0.0 if not
///         (first frame, post-resize, TAA freshly enabled). The shader
///         treats 0.0 as "use the current sample directly".
///   `.z`, `.w` = reserved (alignment pad).
struct CloudTaaParams {
    config: vec4<f32>,
};

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

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

/// Reproject the current fragment's UV into the previous frame's UV.
/// Returns the prev UV in `.xy`; `.z` is 1.0 if the reprojection
/// landed on-screen and behind the camera, 0.0 otherwise.
fn reproject_uv(curr_uv: vec2<f32>) -> vec3<f32> {
    // Current frame: UV → NDC → world-space view direction.
    // The Y flip matches the convention elsewhere in the engine
    // (screen-y points down, NDC-y points up).
    let curr_ndc = vec2<f32>(
        curr_uv.x * 2.0 - 1.0,
        1.0 - curr_uv.y * 2.0,
    );
    let near_h = frame.inv_view_proj * vec4<f32>(curr_ndc, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let view_dir = normalize(near_p - frame.camera_position_world.xyz);

    // Project that direction (w=0 to ignore camera translation —
    // infinity-depth assumption) into the previous frame's clip space.
    let prev_clip = frame.prev_view_proj * vec4<f32>(view_dir, 0.0);
    if (prev_clip.w <= 1e-4) {
        // Direction points behind the previous camera (or numerically
        // degenerate); reject the reprojection.
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let prev_uv = vec2<f32>(
        prev_ndc.x * 0.5 + 0.5,
        1.0 - (prev_ndc.y * 0.5 + 0.5),
    );
    // Validity: inside [0, 1] square.
    let inside = step(0.0, prev_uv.x) * step(0.0, prev_uv.y)
               * step(prev_uv.x, 1.0) * step(prev_uv.y, 1.0);
    return vec3<f32>(prev_uv, inside);
}

/// 3×3 neighbourhood AABB clamp (Karis 2014). Returns the history
/// sample clamped per-component into the min/max range of the 3×3
/// neighbourhood around `centre_pixel` in the current frame.
fn clamp_to_neighbourhood(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    tex_dims: vec2<f32>,
    sample: vec4<f32>,
) -> vec4<f32> {
    let texel = 1.0 / tex_dims;
    var aabb_min = vec4<f32>(1e10);
    var aabb_max = vec4<f32>(-1e10);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            let s = textureSampleLevel(tex, samp, uv + offset, 0.0);
            aabb_min = min(aabb_min, s);
            aabb_max = max(aabb_max, s);
        }
    }
    return clamp(sample, aabb_min, aabb_max);
}

struct TaaOut {
    @location(0) luminance: vec4<f32>,
    @location(1) transmittance: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> TaaOut {
    let curr_uv = in.uv;
    let curr_lum =
        textureSampleLevel(current_luminance, taa_sampler, curr_uv, 0.0);
    let curr_trans =
        textureSampleLevel(current_transmittance, taa_sampler, curr_uv, 0.0);

    let blend_weight = clamp(taa_params.config.x, 0.0, 1.0);
    let history_valid = taa_params.config.y;
    if (history_valid < 0.5 || blend_weight >= 1.0) {
        // First frame after enable / post-resize. Use current only.
        var out: TaaOut;
        out.luminance = curr_lum;
        out.transmittance = curr_trans;
        return out;
    }

    // Reproject; reject if off-screen.
    let reproj = reproject_uv(curr_uv);
    let prev_uv = reproj.xy;
    let reproj_inside = reproj.z;

    let hist_lum_raw =
        textureSampleLevel(history_luminance, taa_sampler, prev_uv, 0.0);
    let hist_trans_raw =
        textureSampleLevel(history_transmittance, taa_sampler, prev_uv, 0.0);

    // Sample dimensions for the neighbourhood clamp. The current and
    // history RTs share the same size (the host invalidates history
    // on resize), so we only need one set of dims.
    let dims = vec2<f32>(textureDimensions(current_luminance, 0));
    let hist_lum =
        clamp_to_neighbourhood(current_luminance, taa_sampler, curr_uv, dims, hist_lum_raw);
    let hist_trans =
        clamp_to_neighbourhood(current_transmittance, taa_sampler, curr_uv, dims, hist_trans_raw);

    // Disocclusion / off-screen handling: if the reprojection landed
    // outside the previous frame, fall back to current. We do this
    // by ramping the blend weight up to 1.0 (= no history) when the
    // reprojection is invalid.
    let effective_blend = mix(1.0, blend_weight, reproj_inside);

    var resolved_lum = mix(hist_lum, curr_lum, effective_blend);
    var resolved_trans = mix(hist_trans, curr_trans, effective_blend);
    // Premultiplied luminance must stay non-negative; per-channel
    // transmittance must stay in [0, 1] for the dual-source blend
    // equation to behave (the composite computes `dst * src1`, and
    // values outside [0, 1] would amplify rather than attenuate).
    resolved_lum = max(resolved_lum, vec4<f32>(0.0));
    resolved_trans = clamp(resolved_trans, vec4<f32>(0.0), vec4<f32>(1.0));

    var out: TaaOut;
    out.luminance = resolved_lum;
    out.transmittance = resolved_trans;
    return out;
}
