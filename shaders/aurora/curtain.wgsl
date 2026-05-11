// Phase 12.5 — aurora curtain raymarch.
//
// Fullscreen Translucent pass. For each above-horizon pixel, march
// the view ray through an upper-atmosphere slab (80–300 km). At each
// step, evaluate a procedural curtain density that combines:
//
//   - a slow time-varying 2D noise driving the horizontal curtain
//     position (so the curtain "drifts" — the v3 punt for fast
//     moving rays still applies)
//   - a vertical envelope keeping emission concentrated in the
//     ~100–200 km altitude band
//   - the colour-biased emission scalar from the host
//
// Emission accumulates additively into the HDR target.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> world: WorldUniforms;

struct AuroraParams {
    emission: vec4<f32>,  // rgb = colour-biased peak emission, w = intensity gate
    config:   vec4<f32>,  // x = motion time (s), y = march steps, z = curtain extent (m), w unused
};
@group(2) @binding(0) var<uniform> aurora: AuroraParams;

const AURORA_BASE_M: f32 = 80000.0;
const AURORA_TOP_M:  f32 = 300000.0;

/// Hash a 2D coordinate to a pseudo-random scalar in `[-1, 1]`. Good
/// enough for a smooth gradient noise; not cryptographic.
fn hash2(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453;
    return fract(h) * 2.0 - 1.0;
}

/// 2D value noise with bicubic-smoothed interpolation.
fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Stack a few octaves for a richer curtain shape.
fn fbm2(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * vnoise2(p * freq);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return v;
}

/// Curtain density at a world-space point. Returns a scalar in
/// roughly `[0, 1]`. The curtain is laid out as a sheet running
/// E-W (constant Z) with horizontal undulation along X driven by
/// time-advected fbm; the vertical envelope concentrates emission
/// at ~110 km.
fn curtain_density(p: vec3<f32>, t: f32, extent: f32) -> f32 {
    let alt = p.y;
    if (alt < AURORA_BASE_M || alt > AURORA_TOP_M) {
        return 0.0;
    }
    // Vertical envelope — peak near 110 km, narrow below, broad
    // above. Models the typical green oxygen emission band.
    let alt_norm_low = (alt - AURORA_BASE_M) / 30000.0;
    let alt_norm_high = (AURORA_TOP_M - alt) / 200000.0;
    let envelope = clamp(alt_norm_low, 0.0, 1.0) * clamp(alt_norm_high, 0.0, 1.0);

    // Curtain XZ shape: a thin sheet that wanders in X with time.
    // The 2D fbm runs in (X/extent + t, Z/extent * 0.3) — Z runs
    // slow so curtains read as long ribbons rather than a spotty
    // grid.
    let sample_xz = vec2<f32>(p.x / extent + t, p.z / (extent * 3.0));
    let sheet = fbm2(sample_xz);

    // The sheet value drives a band: density peaks where sheet is
    // near zero and falls off either side. Squaring the inverse
    // distance gives a sharp curtain ridge.
    let band = exp(-sheet * sheet * 8.0);

    // Add a high-frequency vertical streak modulation so the
    // curtain has visible vertical structure inside the band.
    let streak = 0.6 + 0.4 * vnoise2(vec2<f32>(p.x / 800.0, alt / 800.0 + t * 0.3));

    return envelope * band * streak;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Early-out when the gate is closed — no point marching.
    if (aurora.emission.w < 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // World-space view direction (mirrors atmosphere/sky.wgsl).
    let near_h = frame.inv_view_proj * vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let view_dir = normalize(near_p - frame.camera_position_world.xyz);

    // Below-horizon rays don't hit the aurora layer.
    if (view_dir.y <= 0.001) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Compute t-range where the ray crosses the aurora slab. Camera
    // sits at low altitude (≪ 80 km) so the slab is always above —
    // entry at altitude AURORA_BASE_M, exit at AURORA_TOP_M.
    let cam_world = frame.camera_position_world.xyz;
    let t_enter = (AURORA_BASE_M - cam_world.y) / view_dir.y;
    let t_exit  = (AURORA_TOP_M  - cam_world.y) / view_dir.y;

    let n_steps = i32(aurora.config.y);
    let dt = (t_exit - t_enter) / max(f32(n_steps), 1.0);
    let extent = aurora.config.z;
    let t_motion = aurora.config.x;

    // Integrate density along the ray. Dividing by n_steps gives a
    // mean density independent of march resolution. Auroras are
    // optically thin so visible luminance is just emission ×
    // density (no 1/r falloff — line emission, not reflection).
    var density_accum = 0.0;
    var t = t_enter;
    for (var i = 0; i < n_steps; i = i + 1) {
        let p = cam_world + view_dir * t;
        density_accum = density_accum + curtain_density(p, t_motion, extent);
        t = t + dt;
    }
    let mean_density = density_accum / max(f32(n_steps), 1.0);
    let accum = aurora.emission.rgb * mean_density;

    // Clamp to non-negative so an over-aggressive emission scalar
    // never produces NaN-tail when the additive blend hits the HDR
    // target.
    return vec4<f32>(max(accum, vec3<f32>(0.0)), 1.0);
}
