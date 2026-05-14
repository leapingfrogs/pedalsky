// Phase 8.2 — far rain screen-space streaks.
//
// Three layered fullscreen quads at virtual depths 50, 200, 1000 m.
// Each layer renders procedural streaks scrolled by (wind - camera_velocity)
// projected into screen space. Per-layer parallax + noise frequency.
//
// Single shader, three draw calls; layer index passed in via a uniform
// to avoid an extra bind group per layer.

struct PrecipUniforms {
    camera_position: vec4<f32>,
    wind_velocity: vec4<f32>,
    intensity_mm_per_h: f32,
    dt_seconds: f32,
    simulated_seconds: f32,
    kind: u32,
    particle_count: u32,
    spawn_radius_m: f32,
    spawn_top_m: f32,
    fall_speed_mps: f32,
    _pad_0: f32,
    _pad_1: f32,
    _pad_2: f32,
};

struct LayerUniform {
    depth_m: f32,
    streak_density: f32,
    streak_length_px: f32,
    intensity_scale: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> precip: PrecipUniforms;
@group(1) @binding(1) var overcast_field: texture_2d<f32>;
@group(1) @binding(2) var density_sampler: sampler;
@group(1) @binding(3) var<uniform> layer: LayerUniform;

const MASK_EXTENT_M: f32 = 32000.0;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    // Render at the layer's virtual depth (reverse-Z: smaller depth_m
    // means closer-to-camera; near plane = 1, far plane = 0).
    // Use proj to convert depth_m to clip space.
    let view_z = -layer.depth_m;
    let clip = frame.proj * vec4<f32>(0.0, 0.0, view_z, 1.0);
    let ndc_z = clip.z / clip.w;
    out.clip_pos = vec4<f32>(p, ndc_z, 1.0);
    out.ndc = p;
    return out;
}

fn hash21(p: vec2<i32>) -> u32 {
    var x = u32(p.x) * 0x27d4eb2du + u32(p.y) * 0x165667b1u;
    x = (x ^ (x >> 15u)) * 0x2c1b3c6du;
    x = (x ^ (x >> 12u)) * 0x297a2d39u;
    return x ^ (x >> 15u);
}

fn rand1(p: vec2<i32>) -> f32 {
    return f32(hash21(p) & 0xffffu) / 65535.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Estimate the world-space XZ position this fragment maps to at the
    // layer's depth so we can sample the cloud occlusion mask. NDC.xy
    // back-projected at view_z = -depth_m.
    let view_z = -layer.depth_m;
    let inv = frame.inv_view_proj;
    let clip = vec4<f32>(in.ndc.x, in.ndc.y, in.clip_pos.z, 1.0);
    let world_h = inv * clip;
    let world_p = world_h.xyz / world_h.w;
    let mask_uv = world_p.xz / MASK_EXTENT_M + vec2<f32>(0.5);
    let cloud_mask = textureSampleLevel(overcast_field, density_sampler, mask_uv, 0.0).r;
    // Phase 19.B — smoothstep gate so stratus / thin cloud columns
    // (mask ≈ 0.02) still register as "cloud overhead" rather than
    // being effectively zero. See particle_render.wgsl for rationale.
    let cloud_gate = smoothstep(0.005, 0.05, cloud_mask);
    let intensity = clamp(precip.intensity_mm_per_h / 50.0, 0.0, 1.0);
    if (cloud_gate * intensity * layer.intensity_scale < 0.001) {
        discard;
    }

    // Procedural streak field in screen space, scrolled by
    // (wind − camera_velocity) projected into screen space at the
    // layer's depth (plan §8.2). Move-through-the-rain at speed v thus
    // shifts streaks toward the back of the camera at v / depth ndc/sec.
    let proj_y = frame.proj[1][1];
    let rel_wind = precip.wind_velocity.xyz - frame.camera_velocity_world.xyz;
    let scroll = vec2<f32>(
        rel_wind.x * proj_y / layer.depth_m * precip.simulated_seconds,
        // Vertical scroll: own gravity (fall_speed) plus relative
        // vertical wind. Rain falls down → scroll downward in screen.
        (precip.fall_speed_mps - rel_wind.y) * proj_y / layer.depth_m
            * precip.simulated_seconds,
    );

    let viewport = frame.viewport_size.xy;
    let pixel = (in.ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * viewport;

    // Streaks: hash-based noise grid with vertical persistence.
    let cell_size = max(viewport.y / max(layer.streak_density, 1.0), 1.0);
    let scroll_px = scroll * viewport;
    let p_scrolled = pixel + scroll_px;
    let cell = vec2<i32>(floor(p_scrolled / cell_size));
    let frac = (p_scrolled / cell_size) - floor(p_scrolled / cell_size);
    let r = rand1(cell);
    // Streak shape: thin vertical line at horizontal x = jitter offset.
    let line_x = r;
    let dx = abs(frac.x - line_x);
    let line_alpha = clamp(1.0 - dx * 30.0, 0.0, 1.0);
    // Vertical falloff so streaks have a discrete length not a continuous
    // line.
    let cell_seed = hash21(cell);
    let length_phase = f32(cell_seed & 0xffu) / 255.0;
    let len_y = clamp(layer.streak_length_px / cell_size, 0.05, 1.0);
    let y_dist = abs(frac.y - length_phase);
    let len_alpha = clamp(1.0 - y_dist / len_y, 0.0, 1.0);
    let streak = line_alpha * len_alpha;

    let alpha = cloud_gate * intensity * layer.intensity_scale * streak * 0.5;
    if (alpha <= 0.001) { discard; }
    let tint = vec3<f32>(0.55, 0.6, 0.7);
    return vec4<f32>(tint * alpha, alpha);
}
