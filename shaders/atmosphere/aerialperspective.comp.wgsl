// Phase 5.2.4 — Aerial-perspective LUT bake.
//
// Output: 32 × 32 × 32 Rgba16Float. Camera-relative froxel volume.
//
//   X, Y → screen (NDC) extents
//   Z   → depth slice, linear in [0, AP_FAR_M]
//
// Each voxel stores `(r_inscatter, g_inscatter, b_inscatter, transmittance_mono)`
// for a ray from the camera through the (x,y) NDC at depth z. The ground
// shader composites with `final = lit * ap.a + ap.rgb`.
//
// The far slice covers AP_FAR_M = 32 km along the view ray (plan §5.2.4).
// Limitation: insufficient for high-altitude views; documented for v2.
//
// Bindings:
//   group 0 binding 0 — FrameUniforms (provides view/proj/inv_view_proj/sun)
//   group 1 binding 0 — WorldUniforms
//   group 3 binding 0 — transmittance LUT
//   group 3 binding 1 — multi-scatter LUT
//   group 3 binding 4 — sampler
//   group 2 binding 0 — output AP LUT (storage 3D)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;
@group(2) @binding(0) var output: texture_storage_3d<rgba16float, write>;

const SIZE: vec3<u32> = vec3<u32>(32u, 32u, 32u);
const AP_FAR_M: f32 = 32000.0;

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE.x || gid.y >= SIZE.y || gid.z >= SIZE.z) { return; }

    let uv01 = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
             / vec2<f32>(f32(SIZE.x), f32(SIZE.y));
    let ndc = uv01 * 2.0 - vec2<f32>(1.0);
    // Reverse-Z: depth=1 = near, depth=0 = far. We pick depth=0 to get
    // the world-space far direction, then march along it.
    let near_h = frame.inv_view_proj * vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p = far_h.xyz / far_h.w;
    let view_dir = normalize(far_p - near_p);

    let z_norm = (f32(gid.z) + 0.5) / f32(SIZE.z);
    let t_target = z_norm * AP_FAR_M;

    // Camera in atmosphere frame.
    let p0 = world_to_atmosphere_pos(frame.camera_position_world.xyz);
    let sun_dir = frame.sun_direction.xyz;
    let cos_theta = dot(view_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, world.mie_g);

    // March from camera to t_target. Use a coarse step count proportional
    // to the slice index so close slices integrate finely and far slices
    // amortise the work.
    let n_steps = max(1u, gid.z);
    let dt = t_target / f32(n_steps);

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);

    for (var s = 0u; s < n_steps; s = s + 1u) {
        let t = (f32(s) + 0.5) * dt;
        let pi = p0 + view_dir * t;
        let h = length(pi) - world.planet_radius_m;
        if (h < 0.0) { break; }
        let sigma_t = extinction_at(h);
        let sample_transmit = exp(-sigma_t * dt);

        let sun_vis = sample_transmittance_lut(pi, sun_dir);
        let scat = scattering_pair(h);
        let sun_inscatter = (scat.rayleigh * phase_r + scat.mie * phase_m) * sun_vis;

        let r_p = length(pi);
        let h_norm = clamp((r_p - world.planet_radius_m)
            / max(world.atmosphere_top_m - world.planet_radius_m, 1.0), 0.0, 1.0);
        let sun_cos = clamp(dot(pi / max(r_p, 1.0), sun_dir), -1.0, 1.0);
        let ms_uv = vec2<f32>(sun_cos * 0.5 + 0.5, h_norm);
        let l_ms = textureSampleLevel(multiscatter_lut, lut_sampler, ms_uv, 0.0).rgb;
        let multi = (scat.rayleigh + scat.mie) * l_ms;

        let in_scatter = (sun_inscatter + multi) * frame.sun_illuminance.rgb;
        let safe_sigma_t = max(sigma_t, vec3<f32>(1e-7));
        let s_int = (in_scatter - in_scatter * sample_transmit) / safe_sigma_t;
        luminance = luminance + transmittance * s_int;
        transmittance = transmittance * sample_transmit;
    }

    // Transmittance stored as luminance-weighted scalar so the ground
    // shader can do `lit * ap.a + ap.rgb` with a single channel.
    let t_lum = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    textureStore(output,
                 vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
                 vec4<f32>(luminance, t_lum));
}
