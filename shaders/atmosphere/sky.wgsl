// Phase 5.3 — sky raymarch fragment shader.
//
// Fullscreen-triangle pass at PassStage::SkyBackdrop. For each pixel:
// derive the world-space view direction from inv_view_proj, look up the
// pre-baked sky-view LUT, composite an analytic sun disk with Hillaire
// limb darkening, and write at the reverse-Z far depth (0.0).
//
// Phase 12.6b / followup #74 — the sky-view LUT is built from the
// atmosphere model only, no clouds. Under thick overcast, the sky
// reads as clear blue between visible cloud puffs (where the cloud
// march finds no sample, the sky pass shows through). To match
// real overcast skies — which are a uniform white/grey hemisphere —
// each sky pixel projects its view ray onto a reference cloud
// altitude, samples the synthesised top-down density mask there,
// and mixes the LUT-sky toward an overcast-grey term.
//
// Bindings:
//   group 0 — FrameUniforms (frame)
//   group 1 — WorldUniforms (world)
//   group 2 — Phase 12.6b cloud overcast modulation (mask + sampler)
//   group 3 — atmosphere LUTs (transmittance, multiscatter, skyview, ap, sampler)

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(0) var overcast_field: texture_2d<f32>;
@group(2) @binding(1) var density_mask_sampler: sampler;
@group(3) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(3) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(3) @binding(2) var skyview_lut: texture_2d<f32>;
@group(3) @binding(3) var aerial_perspective_lut: texture_3d<f32>;
@group(3) @binding(4) var lut_sampler: sampler;

/// Reference cloud altitude in metres. Sky pixels project their
/// view ray to this altitude and sample the top-down density mask
/// there. Picked to roughly match the typical low-cloud base in the
/// scene library (cumulus 1500m, stratus 800m, stratocumulus 1000m).
/// Below this altitude (looking down) the projection clamps to
/// "directly above the camera" so we never sample below the cloud
/// layer.
const CLOUD_REFERENCE_ALT_M: f32 = 1500.0;

/// Spatial extent (m) the density mask covers — matches the
/// weather-map and the synthesis output (32 km square centred on
/// world origin).
const DENSITY_MASK_EXTENT_M: f32 = 32000.0;

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
/// Must match the inverse parametrisation used by skyview.comp.wgsl.
///
/// `view_dir` is normalised, in atmosphere-local (= world translation
/// only) basis: +Y is local up.  `cam_r` is the camera's radius in the
/// atmosphere-local frame; passing it in (rather than recomputing it
/// from frame uniforms) keeps this function pure.
fn skyview_lut_uv(view_dir: vec3<f32>, sun_dir: vec3<f32>, cam_r: f32) -> vec2<f32> {
    // Zenith angle of the view ray (0 = straight up, π = straight down).
    let cos_v = clamp(view_dir.y, -1.0, 1.0);
    let vza = acos(cos_v);

    // Geometric horizon as seen from radius cam_r: angle from local
    // zenith to the tangent ray.  For r outside the planet of radius R
    // this is π − asin(R/r); at the surface (r=R) it collapses to π/2.
    let sin_horizon = clamp(world.planet_radius_m / cam_r, 0.0, 1.0);
    let zenith_horizon_angle = PI - asin(sin_horizon);

    var v: f32;
    if (vza < zenith_horizon_angle) {
        // Above-horizon: coord = sqrt(vza / zenith_horizon_angle), uv.y = coord * 0.5.
        let coord = sqrt(clamp(vza / max(zenith_horizon_angle, 1e-6), 0.0, 1.0));
        v = coord * 0.5;
    } else {
        let below_range = max(PI - zenith_horizon_angle, 1e-6);
        let coord = sqrt(clamp((vza - zenith_horizon_angle) / below_range, 0.0, 1.0));
        v = 0.5 + coord * 0.5;
    }

    // Azimuth: angle around the up-axis. Compute relative to sun's
    // azimuth so the LUT u channel can be wrapped.
    let view_az = atan2(view_dir.x, view_dir.z);
    let sun_az = atan2(sun_dir.x, sun_dir.z);
    var du = (view_az - sun_az) / (2.0 * PI);
    du = du - floor(du);
    return vec2<f32>(du, v);
}

/// Map a world XZ position to a UV in the top-down density mask. The
/// mask covers a `DENSITY_MASK_EXTENT_M` square centred on the world
/// origin; world (-extent/2, +extent/2) → UV (0, 1) on each axis.
fn mask_uv_from_world(p_xz: vec2<f32>) -> vec2<f32> {
    let half = DENSITY_MASK_EXTENT_M * 0.5;
    return clamp((p_xz + vec2<f32>(half, half)) / DENSITY_MASK_EXTENT_M,
                 vec2<f32>(0.0), vec2<f32>(1.0));
}

/// Project the view ray to the reference cloud altitude and sample
/// the top-down density mask there. For nearly-horizontal rays the
/// horizontal travel can be enormous and end up outside the mask
/// extent — `mask_uv_from_world` clamps, so distant horizon pixels
/// see the mask boundary value (a deliberate stationary-camera
/// limitation matching the ground pass). For downward rays we sample
/// at the camera's XZ.
fn sample_overcast_at_view(view_dir: vec3<f32>) -> f32 {
    let cam_world = frame.camera_position_world.xyz;
    var sample_xz = cam_world.xz;
    if (view_dir.y > 1.0e-3) {
        let dh = max(CLOUD_REFERENCE_ALT_M - cam_world.y, 0.0);
        let t = dh / view_dir.y;
        sample_xz = cam_world.xz + view_dir.xz * t;
    }
    let mask = textureSampleLevel(
        overcast_field, density_mask_sampler,
        mask_uv_from_world(sample_xz), 0.0,
    ).r;
    // Beer-Lambert mapping shared with the ground pass — keeps the
    // sky's overcast transition and the ground's overcast irradiance
    // moving together as coverage rises. The multiplier (10) was
    // calibrated so a moderate cumulus deck (slider ~0.5) lands
    // around 75% overcast rather than saturating to near-100% on the
    // way to it. The previous 40 was too steep: even slider 0.05
    // produced a near-fully-overcast sky.
    return 1.0 - exp(-10.0 * mask);
}

/// Overcast diffuse luminance. Mirrors the ground-pass term so the
/// horizon and the lit ground match under thick cloud — the sky
/// reads as the same flat-grey hemisphere the ground sees.
fn overcast_diffuse_luminance(sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_alt_factor = sqrt(clamp(sun_dir.y, 0.05, 1.0));
    let irradiance = frame.sun_illuminance.w * sun_alt_factor * 0.25;
    return vec3<f32>(irradiance);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct a world-space view direction.  We use the near plane
    // (reverse-Z NDC z = 1) and subtract the camera position, rather
    // than (near, far) — for an infinite-far perspective matrix, NDC
    // z = 0 maps to w = 0 in world-homogeneous space and yields NaN
    // after the perspective divide.
    let near_h = frame.inv_view_proj * vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let view_dir = normalize(near_p - frame.camera_position_world.xyz);

    let sun_dir = frame.sun_direction.xyz;

    // Camera radius in the atmosphere-local frame (planet-centred).
    let cam_atm = world_to_atmosphere_pos(frame.camera_position_world.xyz);
    let cam_r = max(length(cam_atm), world.planet_radius_m + 1.0);
    let uv = skyview_lut_uv(view_dir, sun_dir, cam_r);
    // Sky-view LUT is stored per-unit-illuminance to keep values inside
    // f16 range across all sun states; multiply by the current TOA
    // solar illuminance here.
    let clear_sky = textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb
                  * frame.sun_illuminance.rgb;

    // Phase 12.6b — mix the clear-sky LUT toward an overcast-grey
    // diffuse term where the cloud column is thick. We only do this
    // for above-horizon rays; below the horizon there is no cloud
    // deck above, and the geometric horizon should keep the LUT's
    // dark band intact.
    var luminance = clear_sky;
    if (view_dir.y > 0.0) {
        let cloud_blocking = sample_overcast_at_view(view_dir);
        luminance = mix(clear_sky, overcast_diffuse_luminance(sun_dir),
                        cloud_blocking);
    }

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
        let sun_visibility = sample_transmittance_lut(cam_atm, sun_dir);
        // The cloud march pass occludes the disk in image-space where
        // the ray actually passes through cloud, so we don't need to
        // gate the disk on `cloud_blocking` here — leaving the LUT
        // path correct keeps clear gaps lit by the disk as expected.
        luminance = luminance + l_disk * sun_visibility;
    }

    return vec4<f32>(max(luminance, vec3<f32>(0.0)), 1.0);
}
