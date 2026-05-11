// Phase 13.6 — wind sock geometry.
//
// Draws a small cone billboard in world space, anchored at a
// camera-relative offset, oriented along the downwind direction with a
// gravity-induced droop that increases as wind speed falls. Depth
// tested against the live framebuffer; aerial perspective applied
// in-shader so the sock fades cleanly into haze at the (uncommon)
// long viewing distance.
//
// Bindings:
//   group 0 binding 0      FrameUniforms (frame)
//   group 1 binding 0      WindsockParams (per-frame upload from host)
//   group 2 binding {0..4} atmosphere LUTs (lut_sampler at slot 4)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct WindsockParams {
    // Column-major model matrix mapping local sock space → world.
    model: mat4x4<f32>,
    // rgb = nominal albedo (used by the day-time direct + ambient path)
    // a   = stripe parameter t along the cone axis where the albedo
    //       switches from band A to band B. The shader picks A/B
    //       deterministically from the world-space ring position.
    albedo: vec4<f32>,
    // rgb = secondary stripe colour (band B).
    // a   = unused.
    stripe: vec4<f32>,
};

@group(1) @binding(0) var<uniform> ws: WindsockParams;

@group(2) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(1) var multiscatter_lut:  texture_2d<f32>;
@group(2) @binding(2) var skyview_lut:       texture_2d<f32>;
@group(2) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(2) @binding(4) var lut_sampler:       sampler;

struct VsIn {
    @location(0) position_local: vec3<f32>,
    @location(1) normal_local:   vec3<f32>,
    // t along the axis ∈ [0,1] from base ring (0) to apex (1).
    @location(2) axial_t:        f32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:  vec3<f32>,
    @location(1) world_norm: vec3<f32>,
    @location(2) axial_t:    f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let world4 = ws.model * vec4<f32>(in.position_local, 1.0);
    out.world_pos = world4.xyz;
    // For pure rotation + uniform scale the inverse-transpose equals
    // the upper-left 3×3 of `model`; the host builds the matrix from
    // an orthonormal basis × scale so this is correct.
    let n4 = ws.model * vec4<f32>(in.normal_local, 0.0);
    out.world_norm = normalize(n4.xyz);
    out.axial_t = in.axial_t;
    out.clip_pos = frame.view_proj * world4;
    return out;
}

fn ap_depth_uv(d_world: f32) -> f32 {
    let d_safe = max(d_world, 50.0);
    let z_norm = log(d_safe / 50.0) / log(100000.0 / 50.0);
    return clamp(z_norm, 0.0, 1.0);
}

const WS_PI: f32 = 3.14159265358979;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n_front = normalize(in.world_norm);
    let sun_dir = normalize(frame.sun_direction.xyz);

    // Lambert diffuse with both faces lit. Backface culling is off
    // so the cone's inside-facing fragments need a sensible colour
    // too. Take the strongest of front/back n·L.
    let n_dot_l = max(dot(n_front, sun_dir), 0.0);
    let n_dot_l_back = max(dot(-n_front, sun_dir), 0.0);
    let lambert = max(n_dot_l, n_dot_l_back);

    // Pick albedo / stripe based on axial position. The base 30 % of
    // the cone uses the secondary (white) stripe colour so the sock
    // reads as a banded windsock.
    var base = ws.albedo.rgb;
    if (in.axial_t < ws.albedo.w) {
        base = ws.stripe.rgb;
    }

    // Direct diffuse: matches the ground PBR convention (Lambert
    // diffuse = albedo · n·L · E_sun / π). `frame.sun_illuminance.rgb`
    // is the cd/m²·sr proxy already scaled to the engine's HDR space.
    let direct = base * lambert * frame.sun_illuminance.rgb / WS_PI;

    // Cheap sky-ambient: a simple top-hemisphere bias so the shaded
    // side never goes black. Picked to land roughly 1/8 of the direct
    // term at mid-day exposure.
    let n_for_ambient = select(n_front, -n_front, n_dot_l_back > n_dot_l);
    let up_bias = clamp(0.5 + 0.5 * n_for_ambient.y, 0.0, 1.0);
    let ambient = base * up_bias * frame.sun_illuminance.rgb * 0.04;

    var lit = direct + ambient;

    // Aerial perspective composite (plan §5.4 / §7.4): camera-relative
    // AP LUT lookup.
    let ndc_xy = (in.clip_pos.xy / frame.viewport_size.xy) * 2.0 - 1.0;
    let d_world = length(in.world_pos - frame.camera_position_world.xyz);
    let ap_uvw = vec3<f32>(
        ndc_xy.x * 0.5 + 0.5,
        ndc_xy.y * 0.5 + 0.5,
        ap_depth_uv(d_world),
    );
    let ap = textureSampleLevel(aerial_perspective_lut, lut_sampler, ap_uvw, 0.0);
    let final_color = lit * ap.a + ap.rgb;
    return vec4<f32>(final_color, 1.0);
}
