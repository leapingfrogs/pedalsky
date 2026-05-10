// Phase 5.2.1 — Transmittance LUT bake.
//
// Output: 256 × 64 Rgba16Float storage texture. Each texel stores
// `vec3<f32>` transmittance from the parametrised (height, view zenith)
// to the top of the atmosphere; alpha is unused (stored as 1.0).
//
// 40 trapezoidal steps per plan §5.2.1.
//
// Bindings (group 1 = WorldUniforms; group 2 = local):
//   group 2 binding 0 — output storage texture (Rgba16Float).

@group(1) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const N_STEPS: u32 = 40u;
const SIZE: vec2<u32> = vec2<u32>(256u, 64u);

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= SIZE.x || gid.y >= SIZE.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(SIZE);
    var p: vec3<f32>;
    var dir: vec3<f32>;
    transmittance_lut_uv_to_pos_dir(uv, &p, &dir);

    let optical_depth = integrate_optical_depth(p, dir, N_STEPS);
    let transmittance = exp(-optical_depth);

    textureStore(output, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(transmittance, 1.0));
}
