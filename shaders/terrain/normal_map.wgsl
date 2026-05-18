// Section 1.5 — normal map generation. Central differences on a
// heightmap, packed into rgba8unorm. The map is consumed by the
// ground fragment shader for lighting at full source resolution.

struct NormalMapUniform {
    cell_size: f32,
    _pad0:     f32,
    _pad1:     f32,
    _pad2:     f32,
};

@group(0) @binding(0) var<uniform> u: NormalMapUniform;
@group(0) @binding(1) var terrain : texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var nmap    : texture_storage_2d<rgba8unorm, write>;

fn dims() -> vec2<i32> {
    return vec2<i32>(textureDimensions(terrain));
}
fn clamp_xy(p: vec2<i32>) -> vec2<i32> {
    let d = dims();
    return vec2<i32>(clamp(p.x, 0, d.x - 1), clamp(p.y, 0, d.y - 1));
}

@compute @workgroup_size(16, 16)
fn compute_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let h_l = textureLoad(terrain, clamp_xy(p + vec2<i32>(-1, 0))).r;
    let h_r = textureLoad(terrain, clamp_xy(p + vec2<i32>( 1, 0))).r;
    let h_t = textureLoad(terrain, clamp_xy(p + vec2<i32>( 0,-1))).r;
    let h_b = textureLoad(terrain, clamp_xy(p + vec2<i32>( 0, 1))).r;

    let dx = (h_r - h_l) / (2.0 * u.cell_size);
    let dz = (h_b - h_t) / (2.0 * u.cell_size);
    let n = normalize(vec3<f32>(-dx, 1.0, -dz));
    // Pack [-1,1] -> [0,1].
    let packed = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    textureStore(nmap, p, packed);
}
