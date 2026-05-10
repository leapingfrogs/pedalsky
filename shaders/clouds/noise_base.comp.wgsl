// Phase 6.1 — base shape noise volume bake.
//
// Output: 128³ Rgba8Unorm.
//   R = Perlin–Worley (Schneider 2015 §"Modeling Clouds"): Perlin fbm
//       remapped against a Worley cell to give organic-looking large-scale
//       coverage.
//   G = Worley FBM at frequency 2  (low frequency)
//   B = Worley FBM at frequency 8  (mid frequency)
//   A = Worley FBM at frequency 14 (high frequency)
//
// Tiling: every band uses tileable cell hashing so the volume tiles
// cleanly when sampled at scales > base_scale_m.
//
// The bake is deterministic for fixed (seed, dims) so the cache hash is
// stable across runs.

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;

const SIZE: u32 = 128u;

fn remap(v: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    return new_min + (v - old_min) * (new_max - new_min) / max(old_max - old_min, 1e-5);
}

fn hash3(p: vec3<u32>) -> u32 {
    var x = p.x * 0x27d4eb2du + p.y * 0x165667b1u + p.z * 0x9e3779b9u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    return x ^ (x >> 15u);
}

fn rand3(p: vec3<u32>) -> vec3<f32> {
    let h0 = hash3(p);
    let h1 = hash3(p + vec3<u32>(1u, 0u, 0u));
    let h2 = hash3(p + vec3<u32>(0u, 1u, 0u));
    return vec3<f32>(
        f32(h0 & 0xffffu) / 65535.0,
        f32(h1 & 0xffffu) / 65535.0,
        f32(h2 & 0xffffu) / 65535.0,
    );
}

/// Tileable Worley noise. `freq` is the number of cells per unit length;
/// `p` is in [0, 1)^3 (we wrap with `freq` so the result tiles).
fn worley_tile(p: vec3<f32>, freq: u32) -> f32 {
    let scaled = p * f32(freq);
    let cell = vec3<i32>(floor(scaled));
    let frac = scaled - floor(scaled);
    var min_d = 1.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighbour = cell + vec3<i32>(dx, dy, dz);
                let wrapped = vec3<u32>(
                    u32((neighbour.x % i32(freq) + i32(freq)) % i32(freq)),
                    u32((neighbour.y % i32(freq) + i32(freq)) % i32(freq)),
                    u32((neighbour.z % i32(freq) + i32(freq)) % i32(freq)),
                );
                let jitter = rand3(wrapped);
                let cell_centre = vec3<f32>(f32(dx), f32(dy), f32(dz)) + jitter;
                let d = length(cell_centre - frac);
                min_d = min(min_d, d);
            }
        }
    }
    // Worley returns the inverted distance so that "near a feature" is bright.
    return clamp(1.0 - min_d, 0.0, 1.0);
}

fn worley_fbm(p: vec3<f32>, freq: u32) -> f32 {
    let f1 = worley_tile(p, freq);
    let f2 = worley_tile(p, freq * 2u);
    let f3 = worley_tile(p, freq * 4u);
    return f1 * 0.625 + f2 * 0.25 + f3 * 0.125;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_grad_tile(c: vec3<i32>, freq: u32) -> vec3<f32> {
    let wrapped = vec3<u32>(
        u32((c.x % i32(freq) + i32(freq)) % i32(freq)),
        u32((c.y % i32(freq) + i32(freq)) % i32(freq)),
        u32((c.z % i32(freq) + i32(freq)) % i32(freq)),
    );
    return normalize(rand3(wrapped) * 2.0 - vec3<f32>(1.0));
}

fn perlin_tile(p: vec3<f32>, freq: u32) -> f32 {
    let scaled = p * f32(freq);
    let c = vec3<i32>(floor(scaled));
    let f = scaled - floor(scaled);
    let u = vec3<f32>(fade(f.x), fade(f.y), fade(f.z));

    var dots: array<f32, 8>;
    for (var iz = 0; iz < 2; iz = iz + 1) {
        for (var iy = 0; iy < 2; iy = iy + 1) {
            for (var ix = 0; ix < 2; ix = ix + 1) {
                let g = perlin_grad_tile(c + vec3<i32>(ix, iy, iz), freq);
                let d = f - vec3<f32>(f32(ix), f32(iy), f32(iz));
                dots[u32(ix) + u32(iy) * 2u + u32(iz) * 4u] = dot(g, d);
            }
        }
    }
    let x00 = mix(dots[0], dots[1], u.x);
    let x10 = mix(dots[2], dots[3], u.x);
    let x01 = mix(dots[4], dots[5], u.x);
    let x11 = mix(dots[6], dots[7], u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z) * 0.5 + 0.5;
}

fn perlin_fbm(p: vec3<f32>, base_freq: u32) -> f32 {
    let p1 = perlin_tile(p, base_freq);
    let p2 = perlin_tile(p, base_freq * 2u);
    let p3 = perlin_tile(p, base_freq * 4u);
    return p1 * 0.5 + p2 * 0.25 + p3 * 0.125;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE || gid.y >= SIZE || gid.z >= SIZE) { return; }
    let p = (vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) + vec3<f32>(0.5)) / f32(SIZE);

    let perlin = perlin_fbm(p, 4u);
    let worley_low = worley_fbm(p, 4u);
    // Perlin–Worley: remap perlin against worley to produce billowy shapes.
    let pw = clamp(remap(perlin, worley_low - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0);

    let r = pw;
    let g = worley_fbm(p, 2u);
    let b = worley_fbm(p, 8u);
    let a = worley_fbm(p, 14u);

    textureStore(output, vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
                 vec4<f32>(r, g, b, a));
}
