// Phase 5.2.4 — Aerial-perspective LUT bake.
// Phase 13.1 — depth bumped 32→64 slices, spacing switched from
//              quadratic (covering 32 km) to exponential (covering
//              50 m → 100 km) for high-altitude / panoramic scenes.
//
// Output: 32 × 32 × 64 Rgba16Float. Camera-relative froxel volume.
//
//   X, Y → screen (NDC) extents
//   Z   → depth slice, exponentially spaced in [AP_NEAR_M, AP_FAR_M]
//
// Each voxel stores `(r_inscatter, g_inscatter, b_inscatter, transmittance_mono)`
// for a ray from the camera through the (x,y) NDC at depth z. The ground
// shader composites with `final = lit * ap.a + ap.rgb`.
//
// Sampling formula (consumer side):
//   z_norm = log(d_world / AP_NEAR_M) / log(AP_FAR_M / AP_NEAR_M)
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

const SIZE: vec3<u32> = vec3<u32>(32u, 32u, 64u);
const AP_NEAR_M: f32 = 50.0;
const AP_FAR_M: f32 = 100000.0;
// March step count per froxel.  Hillaire's reference uses fewer steps
// (~10) because the AP volume is shallow; keep a moderate count to
// reduce banding while remaining cheap relative to the sky-view bake.
const AP_STEPS: u32 = 16u;

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE.x || gid.y >= SIZE.y || gid.z >= SIZE.z) { return; }

    let uv01 = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
             / vec2<f32>(f32(SIZE.x), f32(SIZE.y));
    let ndc = uv01 * 2.0 - vec2<f32>(1.0);
    // Reconstruct world-space view direction.  For an infinite-far
    // perspective matrix, NDC z=0 maps to w=0; using only the near
    // plane and subtracting the camera position avoids that NaN.  The
    // y is flipped because Y on the screen runs top-to-bottom (gid.y=0
    // is the top of the framebuffer) whereas NDC +Y points up.
    let near_h = frame.inv_view_proj * vec4<f32>(ndc.x, -ndc.y, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let view_dir = normalize(near_p - frame.camera_position_world.xyz);

    // Exponential Z spacing (Phase 13.1): t = AP_NEAR_M · (AP_FAR_M /
    // AP_NEAR_M)^z_norm. With AP_NEAR_M = 50 m and AP_FAR_M =
    // 100 km, the slice nearest the camera ends at ~50 m (vs the
    // old quadratic spacing's ~31 m at slice 0 / 1 km at slice 1)
    // and the far slice reaches 100 km — enough to cover horizon
    // mountain ranges and high-altitude panoramic views.
    let z_norm = (f32(gid.z) + 0.5) / f32(SIZE.z);
    let t_target = AP_NEAR_M * pow(AP_FAR_M / AP_NEAR_M, z_norm);

    // Camera in atmosphere frame.
    let p0 = world_to_atmosphere_pos(frame.camera_position_world.xyz);
    let r0 = length(p0);
    let cos_view = dot(p0 / max(r0, 1.0), view_dir);
    let sun_dir = frame.sun_direction.xyz;
    let cos_theta = dot(view_dir, sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, world.mie_g);

    // Fixed step count: every slice gets the same march resolution.
    // Earlier code used `n_steps = max(1, gid.z)` which made slice 0 a
    // single 0-length step and produced NaN through divide-by-zero in
    // the energy-conserving integral (sigma_t · dt → 0).
    let n_steps = AP_STEPS;
    let dt = t_target / f32(n_steps);

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);

    for (var s = 0u; s < n_steps; s = s + 1u) {
        let t = (f32(s) + 0.5) * dt;
        let pi = p0 + view_dir * t;
        // Numerically-stable height calculation. At planet scale,
        // `length(pi) - planet_radius_m` suffers catastrophic
        // cancellation: |pi|² ≈ R² (huge) and the small height delta
        // disappears in the squared term.  Instead, use the algebraic
        // identity: |pi|² = r0² + 2t·r0·cos_view + t², so
        // |pi| − r0 = (2t·r0·cos_view + t²) / (r0 + |pi|).
        let r0_safe = max(r0, 1.0);
        let r_delta_num = 2.0 * t * r0 * cos_view + t * t;
        let r_delta = r_delta_num / (r0_safe + length(pi));
        let h = (r0 - world.planet_radius_m) + r_delta;
        if (h < 0.0) { break; }
        let sigma_t = extinction_at(h);
        let sample_transmit = exp(-sigma_t * dt);

        let sun_vis = sample_transmittance_lut(pi, sun_dir);
        let scat = scattering_pair(h);
        let sun_inscatter = (scat.rayleigh * phase_r + scat.mie * phase_m) * sun_vis;

        let r_p = r0 + r_delta;
        let h_norm = clamp(h
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
