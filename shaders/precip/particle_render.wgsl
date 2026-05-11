// Phase 8.1 / 8.4 — particle render shader.
//
// Each instance is a particle. Six vertices of a unit quad in particle-
// local space; the vertex shader orients the quad along the particle's
// velocity direction (for rain streaks) or as a screen-aligned splat
// (for snow). Length scales with velocity * exposure_time so faster
// particles produce longer streaks.

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    kind: u32,
};

struct Particles { items: array<Particle> };

struct PrecipUniforms {
    camera_position: vec4<f32>,
    wind_velocity: vec4<f32>,
    intensity_mm_per_h: f32,
    dt_seconds: f32,
    simulated_seconds: f32,
    kind: u32,
    particle_count: u32,
    spawn_radius_m: f32,
    spawn_top_m: f32,
    fall_speed_mps: f32,
    _pad_0: f32,
    _pad_1: f32,
    _pad_2: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<storage, read> particles: Particles;
@group(1) @binding(1) var<uniform> precip: PrecipUniforms;
@group(1) @binding(2) var top_down_density_mask: texture_2d<f32>;
@group(1) @binding(3) var density_sampler: sampler;

const MASK_EXTENT_M: f32 = 32000.0;
// Streaks scaled up from physical reality so individual rain drops are
// visible without thousands of pixels per particle. Real droplets are
// ~1 mm radius and motion-blur to a few cm; the per-particle radius is
// instead derived from the Marshall-Palmer mean drop diameter and
// scaled to a renderable size below.
const EXPOSURE_TIME_S: f32 = 1.0 / 30.0;
// Multiplier from Marshall-Palmer mean drop diameter (mm) to our
// rendered streak radius (m). Picked so 5 mm/h gives ~2 cm radius
// streaks (enough to be visible at a few metres without dominating).
const MP_RADIUS_GAIN: f32 = 0.05;
// Pool particle density (particles per m^3) the renderer assumes when
// converting MP volumetric density to per-particle alpha. Matches the
// 8000 particles / pi*50^2*30 m^3 cylinder used by the compute pass.
const POOL_DENSITY_PER_M3: f32 = 0.034;
const SNOW_RADIUS_M: f32 = 0.04;
const PI_R: f32 = 3.14159265358979;

/// Marshall-Palmer (1948): N(D) = N0 * exp(-Lambda * D).
/// N0 = 8000 m^-3 mm^-1, Lambda = 4.1 * I^-0.21 mm^-1.
/// Returns:
///   .x = total drops per m^3 (integrated over D from 0..inf)
///   .y = mean drop diameter (mm)
fn marshall_palmer(intensity_mm_per_h: f32) -> vec2<f32> {
    let i = max(intensity_mm_per_h, 1e-3);
    let lambda = 4.1 * pow(i, -0.21);
    let n0 = 8000.0; // m^-3 mm^-1
    // Integral of N0 * exp(-Lambda * D) dD from 0..inf = N0 / Lambda.
    let n_total = n0 / max(lambda, 1e-3);
    let d_mean = 1.0 / max(lambda, 1e-3);
    return vec2<f32>(n_total, d_mean);
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,        // local quad uv [-1, 1]
    @location(1) tint: vec3<f32>,      // colour multiplier (cloud/snow tinted)
    @location(2) alpha: f32,           // pre-tinted opacity
    @location(3) kind: f32,            // 0 = rain, 1 = snow (interpolant-friendly)
};

fn world_to_mask_uv(xz: vec2<f32>) -> vec2<f32> {
    return xz / MASK_EXTENT_M + vec2<f32>(0.5);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    var out: VsOut;
    let p = particles.items[iid];

    // Unit quad in local space: vid in [0..6).
    let quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0, 1.0),
    );
    let local = quad[vid];

    // Build a billboard frame oriented along the velocity for rain
    // streaks; for snow we use a screen-aligned splat.
    let view_pos = (frame.view * vec4<f32>(p.position, 1.0)).xyz;
    let cam_to_p = normalize(p.position - frame.camera_position_world.xyz);
    let v_world = p.velocity + precip.wind_velocity.xyz;
    let speed = length(v_world);

    var streak_axis: vec3<f32>;
    var width_axis: vec3<f32>;
    var radius: f32;
    var length_m: f32;
    let mp = marshall_palmer(precip.intensity_mm_per_h);
    if (p.kind == 0u) {
        // Rain: streak length = speed * exposure_time, with a small floor
        // so stationary droplets are still visible. Streak radius scales
        // with the MP mean drop diameter so heavy rain visibly thicker.
        radius = mp.y * MP_RADIUS_GAIN;
        length_m = max(speed * EXPOSURE_TIME_S, 0.01);
        // Streak axis = projection of world velocity onto the screen
        // plane (perpendicular to view direction at the particle).
        let v_dir = select(vec3<f32>(0.0, -1.0, 0.0),
                           v_world / max(speed, 1e-4),
                           speed > 1e-4);
        // Project velocity onto plane perpendicular to view direction.
        let v_perp = v_dir - cam_to_p * dot(v_dir, cam_to_p);
        let v_perp_len = length(v_perp);
        streak_axis = select(vec3<f32>(0.0, 1.0, 0.0),
                             v_perp / max(v_perp_len, 1e-4),
                             v_perp_len > 1e-4);
        width_axis = normalize(cross(streak_axis, cam_to_p));
    } else {
        // Snow: round screen-aligned billboard.
        radius = SNOW_RADIUS_M;
        length_m = SNOW_RADIUS_M;
        // Pick any vector not parallel to cam_to_p.
        let helper = select(vec3<f32>(0.0, 1.0, 0.0),
                            vec3<f32>(1.0, 0.0, 0.0),
                            abs(cam_to_p.y) > 0.9);
        width_axis = normalize(cross(helper, cam_to_p));
        streak_axis = normalize(cross(cam_to_p, width_axis));
    }

    let world_offset = width_axis * (local.x * radius)
                     + streak_axis * (local.y * length_m * 0.5);
    let world_pos = p.position + world_offset;
    out.clip_pos = frame.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = local;

    // Cloud occlusion: read the top-down density mask at the particle's
    // XZ position. mask=0 → no clouds above → no precipitation.
    let mask_uv = world_to_mask_uv(p.position.xz);
    let cloud_mask = textureSampleLevel(top_down_density_mask, density_sampler, mask_uv, 0.0).r;

    // Marshall-Palmer drop count per m^3 vs our pool density gives the
    // per-particle alpha: higher real-world density -> each particle
    // "represents" more drops -> brighter splat. Clamped to [0, 1] so
    // very heavy rain doesn't oversaturate the integrator.
    let drops_per_m3 = mp.x;
    let alpha_density = clamp(drops_per_m3 / max(POOL_DENSITY_PER_M3, 1e-3) / 65000.0, 0.0, 1.0);
    let alpha = cloud_mask * alpha_density;

    if (p.kind == 0u) {
        // Rain: cool grey-blue.
        out.tint = vec3<f32>(0.5, 0.55, 0.7);
        out.alpha = alpha;
    } else {
        // Snow: bright, white. Snow flakes are larger and brighter than
        // rain droplets at the same intensity; bias the alpha up.
        out.tint = vec3<f32>(0.95, 0.97, 1.0);
        out.alpha = clamp(alpha * 1.5, 0.0, 1.0);
    }
    out.kind = f32(p.kind);
    _ = view_pos;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var falloff: f32;
    if (in.kind < 0.5) {
        // Rain: smooth across the streak's width, full along its length.
        let d = abs(in.uv.x);
        falloff = clamp(1.0 - d * d, 0.0, 1.0);
    } else {
        // Snow: round splat.
        let d2 = dot(in.uv, in.uv);
        falloff = clamp(1.0 - d2, 0.0, 1.0);
    }
    let a = in.alpha * falloff;
    if (a <= 0.001) { discard; }
    // Pre-multiplied alpha output.
    return vec4<f32>(in.tint * a, a);
}
