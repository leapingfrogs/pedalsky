// Section 1.2 — hydraulic erosion compute passes (Mei, Decaudin, Hu 2007).
//
// Four entry points, all 16×16 workgroups over the heightmap dims:
//
//   add_water_and_flux        — pass 1
//   update_water_and_velocity — pass 2
//   erosion_deposition        — pass 3
//   advect_sediment           — pass 4
//
// Bindings (group 0):
//
//   binding 0  uniforms (HydraulicUniformGpu)
//   binding 1  terrain_in    : texture_storage_2d<r32float, read_write>
//   binding 2  water_in      : texture_storage_2d<r32float, read_write>
//   binding 3  sediment_in   : texture_storage_2d<r32float, read_write>
//   binding 4  flux_in       : texture_storage_2d<rgba32float, read_write>
//   binding 5  velocity      : texture_storage_2d<rg32float, read_write>
//   binding 6  water_out     : texture_storage_2d<r32float, read_write>
//   binding 7  flux_out      : texture_storage_2d<rgba32float, read_write>
//   binding 8  sediment_out  : texture_storage_2d<r32float, read_write>
//
// We don't ping-pong terrain: the erosion pass writes back into
// `terrain_in` directly because each cell only modifies its own
// elevation. Water and flux ping-pong because pass 1's flux output
// becomes pass 2's input.

struct HydraulicUniform {
    cell_size:                  f32,
    dt:                         f32,
    rainfall_rate:              f32,
    evaporation_rate:           f32,

    pipe_cross_section:         f32,
    pipe_length:                f32,
    gravity:                    f32,
    sediment_capacity_constant: f32,

    dissolution_rate:           f32,
    deposition_rate:            f32,
    min_slope:                  f32,
    shallow_water_threshold:    f32,
};

@group(0) @binding(0) var<uniform> u: HydraulicUniform;

@group(0) @binding(1) var terrain      : texture_storage_2d<r32float,    read_write>;
@group(0) @binding(2) var water_in     : texture_storage_2d<r32float,    read_write>;
@group(0) @binding(3) var sediment_in  : texture_storage_2d<r32float,    read_write>;
@group(0) @binding(4) var flux_in      : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var velocity     : texture_storage_2d<rg32float,   read_write>;
@group(0) @binding(6) var water_out    : texture_storage_2d<r32float,    read_write>;
@group(0) @binding(7) var flux_out     : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(8) var sediment_out : texture_storage_2d<r32float,    read_write>;

fn dims() -> vec2<i32> {
    return vec2<i32>(textureDimensions(terrain));
}

fn clamp_xy(p: vec2<i32>) -> vec2<i32> {
    let d = dims();
    return vec2<i32>(
        clamp(p.x, 0, d.x - 1),
        clamp(p.y, 0, d.y - 1),
    );
}

fn load_terrain(p: vec2<i32>) -> f32 {
    return textureLoad(terrain, clamp_xy(p)).r;
}
fn load_water(p: vec2<i32>) -> f32 {
    return textureLoad(water_in, clamp_xy(p)).r;
}
fn load_flux(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(flux_in, clamp_xy(p));
}

// Pass 1 — add rainfall, then compute new outflow flux to each
// neighbour (L, R, T, B packed into rgba). Scaling step (Mei §3.2.1)
// caps total outflow so the cell can't lose more water than it
// has.
@compute @workgroup_size(16, 16)
fn add_water_and_flux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    var water = load_water(p) + u.rainfall_rate * u.dt;
    let here = load_terrain(p) + water;

    let h_l = load_terrain(p + vec2<i32>(-1, 0)) + load_water(p + vec2<i32>(-1, 0));
    let h_r = load_terrain(p + vec2<i32>( 1, 0)) + load_water(p + vec2<i32>( 1, 0));
    let h_t = load_terrain(p + vec2<i32>( 0,-1)) + load_water(p + vec2<i32>( 0,-1));
    let h_b = load_terrain(p + vec2<i32>( 0, 1)) + load_water(p + vec2<i32>( 0, 1));

    let coeff = u.dt * u.pipe_cross_section * u.gravity / max(u.pipe_length, 1e-4);
    let f_in = load_flux(p);
    var f = vec4<f32>(
        max(0.0, f_in.x + coeff * (here - h_l)),
        max(0.0, f_in.y + coeff * (here - h_r)),
        max(0.0, f_in.z + coeff * (here - h_t)),
        max(0.0, f_in.w + coeff * (here - h_b)),
    );

    // Scale step.
    let total = f.x + f.y + f.z + f.w;
    let cell_area = u.cell_size * u.cell_size;
    let k = select(1.0, min(1.0, water * cell_area / (total * u.dt + 1e-9)), total > 0.0);
    f = f * k;

    textureStore(flux_out, p, f);
    // Also stash the rainfall-augmented water back so pass 2 starts
    // from a consistent state. We write it through water_in here —
    // pass 2 will read flux_out (this pass's output) and water_in (the
    // rainfall-augmented value).
    textureStore(water_in, p, vec4<f32>(water, 0.0, 0.0, 0.0));
}

// Pass 2 — update water level from net flux, derive velocity.
@compute @workgroup_size(16, 16)
fn update_water_and_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    // Read the flux we just wrote in pass 1 (now in flux_out).
    let f_here = textureLoad(flux_out, p);
    let f_left  = textureLoad(flux_out, clamp_xy(p + vec2<i32>(-1, 0)));
    let f_right = textureLoad(flux_out, clamp_xy(p + vec2<i32>( 1, 0)));
    let f_top   = textureLoad(flux_out, clamp_xy(p + vec2<i32>( 0,-1)));
    let f_bot   = textureLoad(flux_out, clamp_xy(p + vec2<i32>( 0, 1)));

    // Inflow from neighbours into this cell:
    //   from left  -> our left neighbour's R flux
    //   from right -> our right neighbour's L flux
    //   from top   -> our top neighbour's B flux
    //   from bot   -> our bot neighbour's T flux
    let inflow  = f_left.y + f_right.x + f_top.w + f_bot.z;
    let outflow = f_here.x + f_here.y + f_here.z + f_here.w;
    let cell_area = u.cell_size * u.cell_size;
    let dw = u.dt * (inflow - outflow) / max(cell_area, 1e-6);

    let water_before = load_water(p);
    let water_after_in = water_before + dw;
    // Evaporation.
    let water_after = max(0.0, water_after_in * (1.0 - u.evaporation_rate * u.dt));

    textureStore(water_out, p, vec4<f32>(water_after, 0.0, 0.0, 0.0));

    // Velocity — average of the two neighbour-pair flux components,
    // divided by mean water depth on each axis.
    let mean_d_x = max(0.5 * (water_before + water_after_in), u.shallow_water_threshold);
    let mean_d_y = mean_d_x;
    let net_x = 0.5 * ((f_left.y - f_here.x) + (f_here.y - f_right.x));
    let net_y = 0.5 * ((f_top.w  - f_here.z) + (f_here.w - f_bot.z));
    let u_v = net_x / (mean_d_x * u.cell_size + 1e-6);
    let v_v = net_y / (mean_d_y * u.cell_size + 1e-6);
    textureStore(velocity, p, vec4<f32>(u_v, v_v, 0.0, 0.0));
}

// Pass 3 — erosion / deposition based on sediment capacity.
@compute @workgroup_size(16, 16)
fn erosion_deposition(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let here = load_terrain(p);
    let h_l = load_terrain(p + vec2<i32>(-1, 0));
    let h_r = load_terrain(p + vec2<i32>( 1, 0));
    let h_t = load_terrain(p + vec2<i32>( 0,-1));
    let h_b = load_terrain(p + vec2<i32>( 0, 1));

    // Local slope as sin(angle) from neighbour height differences.
    let dx = (h_r - h_l) * 0.5;
    let dy = (h_b - h_t) * 0.5;
    let denom = sqrt(dx * dx + dy * dy + u.cell_size * u.cell_size);
    let sin_tilt = sqrt(dx * dx + dy * dy) / max(denom, 1e-6);
    let tilt = max(sin_tilt, u.min_slope);

    let vel = textureLoad(velocity, p).xy;
    let speed = length(vel);

    let water = textureLoad(water_out, p).r;
    let depth_scale = mix(0.1, 1.0, smoothstep(0.0, u.shallow_water_threshold, water));

    let capacity = u.sediment_capacity_constant * tilt * speed * depth_scale;
    let sus = textureLoad(sediment_in, p).r;

    var new_terrain = here;
    var new_sed = sus;
    if (capacity > sus) {
        let amt = u.dissolution_rate * (capacity - sus) * u.dt;
        new_terrain = here - amt;
        new_sed = sus + amt;
    } else {
        let amt = u.deposition_rate * (sus - capacity) * u.dt;
        new_terrain = here + amt;
        new_sed = sus - amt;
    }

    textureStore(terrain, p, vec4<f32>(new_terrain, 0.0, 0.0, 0.0));
    // Stash the intermediate suspended sediment in sediment_out so
    // pass 4 can advect it.
    textureStore(sediment_out, p, vec4<f32>(new_sed, 0.0, 0.0, 0.0));
}

// Pass 4 — semi-Lagrangian advection of suspended sediment.
@compute @workgroup_size(16, 16)
fn advect_sediment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let d = dims();
    if (p.x >= d.x || p.y >= d.y) { return; }

    let vel = textureLoad(velocity, p).xy;
    let prev = vec2<f32>(f32(p.x), f32(p.y)) - vel * u.dt / u.cell_size;
    // Clamp the trace inside the domain.
    let cx = clamp(prev.x, 0.0, f32(d.x - 1));
    let cy = clamp(prev.y, 0.0, f32(d.y - 1));
    // Bilinear sample.
    let x0 = i32(floor(cx));
    let y0 = i32(floor(cy));
    let x1 = min(x0 + 1, d.x - 1);
    let y1 = min(y0 + 1, d.y - 1);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let s00 = textureLoad(sediment_out, vec2<i32>(x0, y0)).r;
    let s10 = textureLoad(sediment_out, vec2<i32>(x1, y0)).r;
    let s01 = textureLoad(sediment_out, vec2<i32>(x0, y1)).r;
    let s11 = textureLoad(sediment_out, vec2<i32>(x1, y1)).r;
    let top = mix(s00, s10, fx);
    let bot = mix(s01, s11, fx);
    let advected = mix(top, bot, fy);

    // Write to sediment_in for the next iteration (which becomes
    // pass 3's input).
    textureStore(sediment_in, p, vec4<f32>(advected, 0.0, 0.0, 0.0));
}
