// Phase 4 §4.3 — shared math helpers.
//
// `compute_view_ray` requires `frame: FrameUniforms` to be in scope —
// shaders that include this file must first include `uniforms.wgsl` (or
// declare the binding themselves) and bind `frame` at group 0 binding 0.

fn remap(v: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    return new_min + (v - old_min) * (new_max - new_min) / max(old_max - old_min, 1e-5);
}

fn max3(v: vec3<f32>) -> f32 { return max(v.x, max(v.y, v.z)); }

struct ViewRay {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

/// Build a world-space ray from a screen-space fragment coordinate, honouring
/// the reverse-Z infinite-far perspective convention. depth=1 is the near
/// plane, depth=0 is at infinity.
fn compute_view_ray(frag_xy: vec2<f32>) -> ViewRay {
    let viewport = frame.viewport_size.xy;
    let ndc = vec2<f32>(
        (frag_xy.x / viewport.x) * 2.0 - 1.0,
        1.0 - (frag_xy.y / viewport.y) * 2.0,
    );
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_h  = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p  = far_h.xyz  / far_h.w;
    return ViewRay(near_p, normalize(far_p - near_p));
}

/// Returns true and the two intersection distances along `rd` if the ray
/// intersects the sphere; the smaller is `t.x`, the larger is `t.y`.
fn ray_sphere_intersect(ro: vec3<f32>, rd: vec3<f32>,
                        center: vec3<f32>, radius: f32,
                        t: ptr<function, vec2<f32>>) -> bool {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return false; }
    let sq = sqrt(disc);
    *t = vec2<f32>(-b - sq, -b + sq);
    return true;
}
