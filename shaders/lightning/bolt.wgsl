// Phase 12.3 — billboarded bolt segment.
//
// One quad per BoltSegment instance. The vertex shader builds a
// view-aligned billboard around the segment's a→b axis, expanding
// outward by `thickness/2` along the camera-facing perpendicular.
// The fragment shader writes a soft-edged emissive HDR colour,
// additively blended into the HDR target.

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct InstanceIn {
    @location(0) a_thickness: vec4<f32>,  // (a.xyz, thickness)
    @location(1) b_emission:  vec4<f32>,  // (b.xyz, emission)
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,           // -1..1 across thickness, 0..1 along axis
    @location(1) emission: f32,
}

const QUAD_UV: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, 0.0),
    vec2<f32>( 1.0, 0.0),
    vec2<f32>( 1.0, 1.0),
    vec2<f32>(-1.0, 0.0),
    vec2<f32>( 1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: InstanceIn) -> VsOut {
    let uv = QUAD_UV[vid];

    let a = inst.a_thickness.xyz;
    let b = inst.b_emission.xyz;
    let thickness = inst.a_thickness.w;

    // Point on the centreline at parameter uv.y.
    let centre = mix(a, b, uv.y);

    // View-aligned billboard: expand perpendicular to both the
    // segment axis and the view ray from the camera to centre.
    let axis = normalize(b - a);
    let view = normalize(frame.camera_position_world.xyz - centre);
    var perp = normalize(cross(axis, view));
    // Degenerate case: axis ≈ view → cross is zero. Fall back to
    // the world up vector's cross with axis.
    if (length(perp) < 1.0e-3) {
        perp = normalize(cross(axis, vec3<f32>(0.0, 1.0, 0.0)));
    }

    let world = centre + perp * (thickness * 0.5 * uv.x);

    var out: VsOut;
    out.pos = frame.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv;
    out.emission = inst.b_emission.w;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Soft cylindrical core: emissivity peaks along uv.x = 0 and
    // tapers to 0 at uv.x = ±1.
    let radial = 1.0 - in.uv.x * in.uv.x;
    // Slight taper along the segment ends so successive segments
    // visibly link rather than abrupt-meet.
    let along = 1.0 - 4.0 * (in.uv.y - 0.5) * (in.uv.y - 0.5);
    let core = clamp(radial, 0.0, 1.0) * clamp(along, 0.0, 1.0);
    // Tiny additive halo from the radial squared — gives the
    // characteristic blue-white glow around the channel.
    let halo = pow(clamp(radial, 0.0, 1.0), 4.0) * 0.3;

    // Slightly blue-white tint, matched to the cloud-illumination
    // colour bias in lib.rs::aggregate_cloud_illumination.
    let tint = vec3<f32>(0.90, 0.95, 1.0);
    let colour = tint * in.emission * (core + halo);
    return vec4<f32>(colour, 1.0);
}
