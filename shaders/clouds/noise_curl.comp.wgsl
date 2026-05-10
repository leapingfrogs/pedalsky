// Phase 6.1 — curl noise bake.
//
// Output: 128² Rg8Unorm. RG = 2D curl of a Perlin scalar potential.
//
// Used to perturb the detail-noise lookup so cloud edges have organic
// turbulent eddies rather than crisp sphere-pack patterns.

@group(0) @binding(0) var output: texture_storage_2d<rg8unorm, write>;

const SIZE: u32 = 128u;

fn hash2(p: vec2<u32>) -> u32 {
    var x = p.x * 0x27d4eb2du + p.y * 0x165667b1u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    return x ^ (x >> 15u);
}

fn rand2(p: vec2<u32>) -> vec2<f32> {
    let h0 = hash2(p);
    let h1 = hash2(p + vec2<u32>(1u, 0u));
    return vec2<f32>(
        f32(h0 & 0xffffu) / 65535.0 * 2.0 - 1.0,
        f32(h1 & 0xffffu) / 65535.0 * 2.0 - 1.0,
    );
}

fn perlin_grad(c: vec2<i32>, freq: u32) -> vec2<f32> {
    let wrapped = vec2<u32>(
        u32((c.x % i32(freq) + i32(freq)) % i32(freq)),
        u32((c.y % i32(freq) + i32(freq)) % i32(freq)),
    );
    return normalize(rand2(wrapped));
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin2(p: vec2<f32>, freq: u32) -> f32 {
    let scaled = p * f32(freq);
    let c = vec2<i32>(floor(scaled));
    let f = scaled - floor(scaled);
    let u = vec2<f32>(fade(f.x), fade(f.y));
    let g00 = perlin_grad(c + vec2<i32>(0, 0), freq);
    let g10 = perlin_grad(c + vec2<i32>(1, 0), freq);
    let g01 = perlin_grad(c + vec2<i32>(0, 1), freq);
    let g11 = perlin_grad(c + vec2<i32>(1, 1), freq);
    let d00 = dot(g00, f - vec2<f32>(0.0, 0.0));
    let d10 = dot(g10, f - vec2<f32>(1.0, 0.0));
    let d01 = dot(g01, f - vec2<f32>(0.0, 1.0));
    let d11 = dot(g11, f - vec2<f32>(1.0, 1.0));
    let x0 = mix(d00, d10, u.x);
    let x1 = mix(d01, d11, u.x);
    return mix(x0, x1, u.y);
}

fn potential(p: vec2<f32>) -> f32 {
    return perlin2(p, 4u) + 0.5 * perlin2(p, 8u);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE || gid.y >= SIZE) { return; }
    let p = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / f32(SIZE);
    let eps = 1.0 / f32(SIZE);
    let dpdx = (potential(p + vec2<f32>(eps, 0.0)) - potential(p - vec2<f32>(eps, 0.0))) / (2.0 * eps);
    let dpdy = (potential(p + vec2<f32>(0.0, eps)) - potential(p - vec2<f32>(0.0, eps))) / (2.0 * eps);
    // 2D curl of scalar potential ψ: (∂ψ/∂y, −∂ψ/∂x). Squash to [0, 1].
    let curl = vec2<f32>(dpdy, -dpdx);
    let scaled = clamp(curl * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(scaled, 0.0, 1.0));
}
