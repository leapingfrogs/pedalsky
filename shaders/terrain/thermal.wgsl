// Section 1.3 — thermal erosion (Olsen 2004 / Št'ava et al. 2008).
//
// Two compute passes:
//
//   compute_outflow : write a per-cell "material to send out" texture
//                     by scanning the Moore neighbourhood. Each cell
//                     only writes its own outflow; no races.
//   apply_outflow   : for each cell, sum the inflows from neighbours
//                     and apply the net change to its terrain height.
//
// The intermediate texture stores 8 channels packed into two rgba —
// but `rgba32float` storage textures already give us 4 channels per
// pixel and 8-direction movement only needs the magnitudes (the
// direction is implicit in the neighbour offset). We use a single
// rgba32float "outflow" texture where (r=L, g=R, b=T, a=B). Diagonals
// are subsumed by adding their share into the orthogonal axes —
// approximation; for full 8-direction split we'd need a second texture.

struct ThermalUniform {
    cell_size:     f32,
    talus_tan:     f32,
    erosion_rate:  f32,
    _pad:          f32,
};

@group(0) @binding(0) var<uniform> u: ThermalUniform;
@group(0) @binding(1) var terrain  : texture_storage_2d<r32float,    read_write>;
// 8 channels: rgba = (NW, N, NE, W), rg of binding 3 = (E, SW), ba = (S, SE).
@group(0) @binding(2) var outflow_a : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var outflow_b : texture_storage_2d<rgba32float, read_write>;

fn dims() -> vec2<i32> {
    return vec2<i32>(textureDimensions(terrain));
}
fn clamp_xy(p: vec2<i32>) -> vec2<i32> {
    let d = dims();
    return vec2<i32>(clamp(p.x, 0, d.x - 1), clamp(p.y, 0, d.y - 1));
}
fn load_h(p: vec2<i32>) -> f32 {
    return textureLoad(terrain, clamp_xy(p)).r;
}

// 8 neighbour offsets. Distance is `cell_size` for orthogonal,
// `cell_size * sqrt(2)` for diagonal.
//   0=NW  1=N  2=NE  3=W
//   4=E   5=SW 6=S   7=SE
const OFFSETS: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(-1,-1), vec2<i32>( 0,-1), vec2<i32>( 1,-1), vec2<i32>(-1, 0),
    vec2<i32>( 1, 0), vec2<i32>(-1, 1), vec2<i32>( 0, 1), vec2<i32>( 1, 1),
);

const SQRT2: f32 = 1.41421356;

fn neighbour_dist(i: u32) -> f32 {
    // diagonals: 0, 2, 5, 7
    let diag = (i == 0u) || (i == 2u) || (i == 5u) || (i == 7u);
    return select(u.cell_size, u.cell_size * SQRT2, diag);
}

@compute @workgroup_size(16, 16)
fn compute_outflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let here = load_h(p);
    var excess = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var excess_total: f32 = 0.0;
    var max_excess: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let n = p + OFFSETS[i];
        let dh = here - load_h(n);
        let dist = neighbour_dist(i);
        let e = dh - u.talus_tan * dist;
        if (e > 0.0) {
            excess[i] = e;
            excess_total = excess_total + e;
            max_excess = max(max_excess, e);
        }
    }
    let move_amt = min(0.5 * max_excess, u.erosion_rate * excess_total);
    // Pack outflows into the two rgba textures:
    //   outflow_a: (NW, N, NE, W) = (0, 1, 2, 3)
    //   outflow_b: (E, SW, S, SE) = (4, 5, 6, 7)
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    if (excess_total > 1e-6) {
        let scale = move_amt / excess_total;
        a = vec4<f32>(
            excess[0] * scale,
            excess[1] * scale,
            excess[2] * scale,
            excess[3] * scale,
        );
        b = vec4<f32>(
            excess[4] * scale,
            excess[5] * scale,
            excess[6] * scale,
            excess[7] * scale,
        );
    }
    textureStore(outflow_a, p, a);
    textureStore(outflow_b, p, b);
}

@compute @workgroup_size(16, 16)
fn apply_outflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let my_a = textureLoad(outflow_a, p);
    let my_b = textureLoad(outflow_b, p);
    let outflow = my_a.x + my_a.y + my_a.z + my_a.w
                + my_b.x + my_b.y + my_b.z + my_b.w;

    // Inflow from each neighbour is the neighbour's outflow channel
    // pointing back toward us. If neighbour at offset O sent material
    // in direction D == O, that's our inflow:
    //
    //   neighbour at (-1,-1) (NW) sent SE (idx 7) toward us.
    //   neighbour at ( 0,-1) (N)  sent S  (idx 6) toward us.
    //   neighbour at ( 1,-1) (NE) sent SW (idx 5) toward us.
    //   neighbour at (-1, 0) (W)  sent E  (idx 4) toward us.
    //   neighbour at ( 1, 0) (E)  sent W  (idx 3) toward us.
    //   neighbour at (-1, 1) (SW) sent NE (idx 2) toward us.
    //   neighbour at ( 0, 1) (S)  sent N  (idx 1) toward us.
    //   neighbour at ( 1, 1) (SE) sent NW (idx 0) toward us.

    let nw = textureLoad(outflow_b, clamp_xy(p + vec2<i32>(-1,-1))).w; // idx 7
    let n  = textureLoad(outflow_b, clamp_xy(p + vec2<i32>( 0,-1))).z; // idx 6
    let ne = textureLoad(outflow_b, clamp_xy(p + vec2<i32>( 1,-1))).y; // idx 5
    let w_ = textureLoad(outflow_b, clamp_xy(p + vec2<i32>(-1, 0))).x; // idx 4
    let e_ = textureLoad(outflow_a, clamp_xy(p + vec2<i32>( 1, 0))).w; // idx 3
    let sw = textureLoad(outflow_a, clamp_xy(p + vec2<i32>(-1, 1))).z; // idx 2
    let s_ = textureLoad(outflow_a, clamp_xy(p + vec2<i32>( 0, 1))).y; // idx 1
    let se = textureLoad(outflow_a, clamp_xy(p + vec2<i32>( 1, 1))).x; // idx 0

    let inflow = nw + n + ne + w_ + e_ + sw + s_ + se;
    let here = textureLoad(terrain, p).r;
    textureStore(terrain, p, vec4<f32>(here + inflow - outflow, 0.0, 0.0, 0.0));
}
