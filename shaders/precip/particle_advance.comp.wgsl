// Phase 8.1 / 8.4 — particle advance compute shader.
//
// Maintains a fixed-length pool of rain/snow particles around the camera.
// Each frame:
//   1. Integrate position by velocity * dt and add wind drift.
//   2. Re-seed particles that exit a cylinder of radius PRECIP_RADIUS_M
//      around the camera or fall below the ground (y < 0).
//   3. Increment age.
//
// Storage layout (matches the Rust Particle struct in ps-precip):
//   position : vec3<f32>   12 B
//   age      : f32          4 B
//   velocity : vec3<f32>   12 B
//   kind     : u32          4 B   (0 = rain, 1 = snow)
// Total: 32 B per particle.

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    kind: u32,
};

struct Particles {
    items: array<Particle>,
};

struct PrecipUniforms {
    camera_position: vec4<f32>,    // xyz = world pos
    wind_velocity: vec4<f32>,      // xyz = m/s, w = turbulence
    intensity_mm_per_h: f32,
    dt_seconds: f32,
    simulated_seconds: f32,
    kind: u32,                      // 0 = rain, 1 = snow
    particle_count: u32,
    spawn_radius_m: f32,            // emitter cylinder radius
    spawn_top_m: f32,               // top of spawn cylinder above camera
    fall_speed_mps: f32,            // terminal velocity
    user_seed: u32,                 // plan §Cross-Cutting/Determinism
    _pad_0: f32,
    _pad_1: f32,
    _pad_2: f32,
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: Particles;
@group(0) @binding(1) var<uniform> precip: PrecipUniforms;
@group(0) @binding(2) var wind_field: texture_3d<f32>;
@group(0) @binding(3) var wind_sampler: sampler;
@group(0) @binding(4) var<storage, read_write> draw_args: DrawIndirectArgs;

const WIND_EXTENT_M: f32 = 32000.0;
const WIND_TOP_M: f32 = 12000.0;

/// Trilinear lookup of the synthesised 3D wind field at world position p.
/// Returns the wind velocity in world space (m/s).
///
/// Channel mapping (matches `ps_synthesis::wind_field`):
///   R = u (east, +X)    G = v_horizontal (south, +Z)
///   B = w (vertical, +Y)   A = turbulence (unused here)
fn wind_at(p: vec3<f32>) -> vec3<f32> {
    let half = WIND_EXTENT_M * 0.5;
    let u = clamp((p.x + half) / WIND_EXTENT_M, 0.0, 1.0);
    let v = clamp(p.y / WIND_TOP_M, 0.0, 1.0);
    let w = clamp((p.z + half) / WIND_EXTENT_M, 0.0, 1.0);
    let s = textureSampleLevel(wind_field, wind_sampler, vec3<f32>(u, v, w), 0.0);
    return vec3<f32>(s.r, s.b, s.g);
}

// PCG-ish hash for deterministic respawn jitter.
fn hash3(seed: u32) -> vec3<f32> {
    var x = seed * 0x27d4eb2du + 0x9e3779b9u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    let r0 = f32(x & 0xffffffu) / 16777215.0;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    let r1 = f32(x & 0xffffffu) / 16777215.0;
    x = (x ^ (x >> 15u)) * 0x165667b1u;
    let r2 = f32(x & 0xffffffu) / 16777215.0;
    return vec3<f32>(r0, r1, r2);
}

fn respawn(particle_id: u32, frame_seed: u32) -> Particle {
    let r = hash3(particle_id * 1664525u + frame_seed);
    let theta = r.x * 6.2831853;
    let radius = sqrt(r.y) * precip.spawn_radius_m;
    let dx = cos(theta) * radius;
    let dz = sin(theta) * radius;
    let dy = r.z * precip.spawn_top_m;
    let pos = precip.camera_position.xyz + vec3<f32>(dx, dy, dz);
    let v = vec3<f32>(0.0, -precip.fall_speed_mps, 0.0);
    var p: Particle;
    p.position = pos;
    p.age = 0.0;
    p.velocity = v;
    p.kind = precip.kind;
    return p;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= precip.particle_count) { return; }

    var p = particles.items[i];

    // Cold start / re-seed: zero positions get a fresh slot. Track via
    // a sentinel age (negative ages flag fresh particles).
    if (p.age < 0.0) {
        let frame_seed = u32(precip.simulated_seconds * 60.0) ^ 0xa5a5u
                       ^ precip.user_seed;
        p = respawn(i, frame_seed);
        particles.items[i] = p;
        atomicAdd(&draw_args.instance_count, 1u);
        return;
    }

    // Advance. Snow has a much higher wind coupling than rain (drag-to-
    // mass ratio ~1000×) — flakes drift with the air almost 1:1, drops
    // mostly fall straight. Per plan §8.4 "stronger wind influence".
    var wind_gain: f32 = 0.3;
    if (p.kind == 1u) {
        wind_gain = 1.5;
    }
    // Sample the 3D wind field at the particle's own position so wind
    // varies along the particle's path (Ekman veer aloft, thermal
    // updrafts under cumulus, etc.) — plan §8.1 wind_at(p).
    let wind = wind_at(p.position);
    let v_total = p.velocity + wind * wind_gain;
    p.position = p.position + v_total * precip.dt_seconds;
    p.age = p.age + precip.dt_seconds;

    // Re-seed when leaving the cylinder or hitting the ground.
    let dxz = p.position.xz - precip.camera_position.xz;
    let r2 = dot(dxz, dxz);
    let max_r2 = precip.spawn_radius_m * precip.spawn_radius_m;
    let too_high = (p.position.y - precip.camera_position.y) > precip.spawn_top_m;
    let below_ground = p.position.y < 0.0;
    if (r2 > max_r2 || below_ground || too_high) {
        let frame_seed = u32(precip.simulated_seconds * 60.0) ^ (i * 31u)
                       ^ precip.user_seed;
        p = respawn(i, frame_seed);
    }

    particles.items[i] = p;
    // Atomic-increment the indirect-draw instance count. Plan §8.1: the
    // render pass uses draw_indirect to read the live count from this
    // counter. The counter is reset to 0 by the host each frame. Every
    // particle that survives `respawn`/`advance` is "live" and gets
    // counted.
    atomicAdd(&draw_args.instance_count, 1u);
}
