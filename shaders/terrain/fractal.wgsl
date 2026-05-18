// Section 1.4 — fractal detail injection (Musgrave heterogeneous
// multifractal). Inline hash-based gradient noise; slope-masked.

struct FractalUniform {
    cell_size:               f32,
    amplitude_m:             f32,
    base_frequency:          f32,
    lacunarity:              f32,

    persistence:             f32,
    octaves:                 u32,
    ridged:                  f32,
    slope_mask_strength:     f32,

    slope_mask_threshold_tan: f32,
    _pad0:                    f32,
    _pad1:                    f32,
    _pad2:                    f32,
};

@group(0) @binding(0) var<uniform> u: FractalUniform;
@group(0) @binding(1) var terrain : texture_storage_2d<r32float, read_write>;

fn dims() -> vec2<i32> {
    return vec2<i32>(textureDimensions(terrain));
}
fn clamp_xy(p: vec2<i32>) -> vec2<i32> {
    let d = dims();
    return vec2<i32>(clamp(p.x, 0, d.x - 1), clamp(p.y, 0, d.y - 1));
}

// 32-bit integer hash → float in [0, 1].
fn hash21u(p: vec2<i32>) -> u32 {
    var x = u32(p.x) * 0x27d4eb2du + u32(p.y) * 0x165667b1u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    return x ^ (x >> 15u);
}
fn rand2(p: vec2<i32>) -> vec2<f32> {
    let h0 = hash21u(p);
    let h1 = hash21u(p + vec2<i32>(7919, 1));
    return vec2<f32>(
        f32(h0 & 0xffffu) / 65535.0,
        f32(h1 & 0xffffu) / 65535.0,
    );
}
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Standard 2D value noise. Range ~[-1, 1] approximately.
fn value_noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u_t = vec2<f32>(fade(f.x), fade(f.y));
    let h00 = rand2(i).x * 2.0 - 1.0;
    let h10 = rand2(i + vec2<i32>(1, 0)).x * 2.0 - 1.0;
    let h01 = rand2(i + vec2<i32>(0, 1)).x * 2.0 - 1.0;
    let h11 = rand2(i + vec2<i32>(1, 1)).x * 2.0 - 1.0;
    return mix(mix(h00, h10, u_t.x), mix(h01, h11, u_t.x), u_t.y);
}

// Heterogeneous multifractal (Musgrave). Each octave's contribution
// scales by the running sum, so high areas get sharper detail than
// low areas.
fn hetero_multifractal(pos: vec2<f32>, ridged: bool) -> f32 {
    var freq = u.base_frequency;
    var amp = 1.0;
    var n = value_noise(pos * freq);
    if (ridged) { n = 1.0 - abs(n); }
    var sum = n;
    for (var i: u32 = 1u; i < u.octaves; i = i + 1u) {
        freq = freq * u.lacunarity;
        amp = amp * u.persistence;
        var s = value_noise(pos * freq);
        if (ridged) { s = 1.0 - abs(s); }
        // Heterogeneous: scale the new octave by the running sum.
        sum = sum + s * amp * sum;
    }
    return sum;
}

@compute @workgroup_size(16, 16)
fn inject_fractal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let here = textureLoad(terrain, p).r;
    let h_l = textureLoad(terrain, clamp_xy(p + vec2<i32>(-1, 0))).r;
    let h_r = textureLoad(terrain, clamp_xy(p + vec2<i32>( 1, 0))).r;
    let h_t = textureLoad(terrain, clamp_xy(p + vec2<i32>( 0,-1))).r;
    let h_b = textureLoad(terrain, clamp_xy(p + vec2<i32>( 0, 1))).r;

    let dx = (h_r - h_l) * 0.5;
    let dy = (h_b - h_t) * 0.5;
    let slope_tan = sqrt(dx * dx + dy * dy) / max(u.cell_size, 1e-6);
    let mask = mix(
        1.0,
        smoothstep(u.slope_mask_threshold_tan * 0.5, u.slope_mask_threshold_tan, slope_tan),
        u.slope_mask_strength,
    );

    let pos = vec2<f32>(f32(p.x) * u.cell_size, f32(p.y) * u.cell_size);
    let n = hetero_multifractal(pos, u.ridged > 0.5);
    let delta = n * u.amplitude_m * mask;

    textureStore(terrain, p, vec4<f32>(here + delta, 0.0, 0.0, 0.0));
}
