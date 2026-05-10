// Phase 0 placeholder ground shader: a 200×200 km quad in the XZ plane,
// shaded with a procedural checker. Phase 7 replaces this with a real PBR
// ground + wet surface; this exists to give the camera something to fly
// over while the rest of the engine comes online.
//
// Phase 4 §4.2: uses the engine-wide group-0 FrameUniforms binding.
// Phase 5 §5.4: applies aerial perspective from the AP LUT (group 3).

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;

const AP_FAR_M: f32 = 32000.0;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.world_pos = pos;
    out.clip_pos = frame.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // 5 m squares — large enough to read while hovering at ~2 m AGL, small
    // enough to give shadows local contrast at distance.
    let cell = 5.0;
    let g = floor(in.world_pos.xz / cell);
    let parity = u32(abs(g.x + g.y)) & 1u;
    let dark = vec3<f32>(0.05, 0.06, 0.07);
    let light = vec3<f32>(0.18, 0.18, 0.20);
    let albedo = select(light, dark, parity == 1u);

    // Direct sun illumination (Lambertian). Ground normal is +Y.
    let sun_dir = frame.sun_direction.xyz;
    let n_dot_l = max(sun_dir.y, 0.0);
    // Use the TOA illuminance carried in sun_illuminance.w as a proxy for
    // direct sun lighting before Phase 5's transmittance is applied via AP.
    let direct = albedo * frame.sun_illuminance.rgb * n_dot_l / 3.14159265;

    // Aerial perspective application (plan §5.4).
    //   final = lit · ap.a + ap.rgb
    // The AP LUT is camera-relative, indexed by NDC.xy + linear-depth slice.
    let ndc_xy = (in.clip_pos.xy / frame.viewport_size.xy) * 2.0 - 1.0;
    // Distance from camera to fragment.
    let d_world = length(in.world_pos - frame.camera_position_world.xyz);
    let ap_uvw = vec3<f32>(
        ndc_xy.x * 0.5 + 0.5,
        ndc_xy.y * 0.5 + 0.5,
        clamp(d_world / AP_FAR_M, 0.0, 1.0),
    );
    let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);

    let final_color = direct * ap.a + ap.rgb;
    return vec4<f32>(final_color, 1.0);
}
