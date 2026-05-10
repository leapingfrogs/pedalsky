// Phase 6.1 — detail noise volume bake.
//
// Output: 32³ Rgba8Unorm.
//   R = Worley FBM at frequency 2
//   G = Worley FBM at frequency 8
//   B = Worley FBM at frequency 16
//   A = spare (0)

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;

const SIZE: u32 = 32u;

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
                let jitter = rand3(wrapped + vec3<u32>(73u, 19u, 41u));
                let cell_centre = vec3<f32>(f32(dx), f32(dy), f32(dz)) + jitter;
                let d = length(cell_centre - frac);
                min_d = min(min_d, d);
            }
        }
    }
    return clamp(1.0 - min_d, 0.0, 1.0);
}

fn worley_fbm(p: vec3<f32>, freq: u32) -> f32 {
    let f1 = worley_tile(p, freq);
    let f2 = worley_tile(p, freq * 2u);
    let f3 = worley_tile(p, freq * 4u);
    return f1 * 0.625 + f2 * 0.25 + f3 * 0.125;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE || gid.y >= SIZE || gid.z >= SIZE) { return; }
    let p = (vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) + vec3<f32>(0.5)) / f32(SIZE);

    let r = worley_fbm(p, 2u);
    let g = worley_fbm(p, 8u);
    let b = worley_fbm(p, 16u);
    textureStore(output, vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
                 vec4<f32>(r, g, b, 0.0));
}
