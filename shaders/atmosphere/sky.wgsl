// Phase 5.3 — sky raymarch fragment shader.
//
// Fullscreen-triangle pass at PassStage::SkyBackdrop. For each pixel:
// derive the world-space view direction from inv_view_proj, look up the
// pre-baked sky-view LUT, composite an analytic sun disk with Hillaire
// limb darkening, and write at the reverse-Z far depth (0.0).
//
// Bindings:
//   group 0 — FrameUniforms (frame)
//   group 1 — WorldUniforms (world)
//   group 3 — atmosphere LUTs (transmittance, multiscatter, skyview, ap, sampler)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(3) @binding(2) var skyview_lut: texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;

// PI comes from common/atmosphere.wgsl.
const SUN_DISK_LIMB_DARKEN: f32 = 0.6;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.pos = vec4<f32>(p, 0.0, 1.0); // depth = 0 → reverse-Z far
    out.ndc = p;
    return out;
}

/// Map a world-space view direction to (u, v) in the sky-view LUT.
fn skyview_lut_uv(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec2<f32> {
    // Latitude = angle above the horizon plane.
    let sin_lat = clamp(view_dir.y, -1.0, 1.0);
    let cos_lat = sqrt(max(1.0 - sin_lat * sin_lat, 0.0));
    let lat = asin(sin_lat);
    let v_centred = sign(lat) * sqrt(min(abs(lat) / (PI * 0.5), 1.0));
    let v = v_centred * 0.5 + 0.5;

    // Azimuth: angle around the up-axis. Compute relative to sun's
    // azimuth so the LUT u channel can be wrapped.
    let view_az = atan2(view_dir.x, view_dir.z);
    let sun_az = atan2(sun_dir.x, sun_dir.z);
    var du = (view_az - sun_az) / (2.0 * PI);
    du = du - floor(du);
    return vec2<f32>(du, v);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let near_h = frame.inv_view_proj * vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(in.ndc.x, in.ndc.y, 0.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p = far_h.xyz / far_h.w;
    let view_dir = normalize(far_p - near_p);

    let sun_dir = frame.sun_direction.xyz;

    let uv = skyview_lut_uv(view_dir, sun_dir);
    var luminance = textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb;

    // Analytic sun disk with Hillaire limb darkening. Only when the view
    // ray points within the configured angular radius and the planet
    // doesn't occlude.
    let cos_disk = cos(frame.sun_direction.w); // angular radius
    let cos_view_sun = dot(view_dir, sun_dir);
    if (cos_view_sun > cos_disk) {
        // Distance from sample to ray-disk centre, normalised.
        let theta = acos(clamp(cos_view_sun, -1.0, 1.0));
        let theta_norm = theta / max(frame.sun_direction.w, 1e-6);
        let limb = 1.0 - SUN_DISK_LIMB_DARKEN * (1.0 - sqrt(max(1.0 - theta_norm * theta_norm, 0.0)));
        // Sun disk is the TOA solar illuminance distributed over the
        // disk's solid angle. Solid angle ≈ π · (angular_radius)².
        let omega = PI * frame.sun_direction.w * frame.sun_direction.w;
        let l_disk = frame.sun_illuminance.rgb * limb / max(omega, 1e-6);

        // Multiply by transmittance from camera through atmosphere along
        // the sun direction (sun visibility from the camera position).
        let cam_atm = world_to_atmosphere_pos(frame.camera_position_world.xyz);
        let sun_visibility = sample_transmittance_lut(cam_atm, sun_dir);
        luminance = luminance + l_disk * sun_visibility;
    }

    return vec4<f32>(max(luminance, vec3<f32>(0.0)), 1.0);
}
