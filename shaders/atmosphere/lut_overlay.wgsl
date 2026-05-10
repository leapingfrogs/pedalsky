// Phase 5 debug: LUT visualiser overlay.
//
// Runs AFTER the tone-map, writing into the swapchain. Reads the four
// atmosphere LUTs (group 3) and tiles them into a 2x2 grid in the
// top-left corner of the framebuffer:
//
//   ┌─────────────────┬─────────────────┐
//   │ transmittance   │ multi-scatter   │
//   │ (256x64)        │ (32x32)         │
//   ├─────────────────┼─────────────────┤
//   │ sky-view        │ AP middle slice │
//   │ (192x108)       │ (32x32 @ z=16)  │
//   └─────────────────┴─────────────────┘
//
// Each tile is a fixed pixel size (configurable via push constant in the
// future; baked here). Pixels outside the grid fall through to the
// existing swapchain content (LoadOp::Load).

struct OverlayUniforms {
    // (tile_w, tile_h, gap, _pad) — pixels.
    tile_layout: vec4<f32>,
    // Per-LUT visual scale factors. .r=transmittance, .g=multi-scatter,
    // .b=sky-view, .a=AP. Channel-wise multipliers applied before display.
    scales: vec4<f32>,
};

@group(0) @binding(0) var<uniform> overlay: OverlayUniforms;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(3) var skyview_lut: texture_2d<f32>;
@group(0) @binding(4) var ap_lut: texture_3d<f32>;
@group(0) @binding(5) var lut_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) frag: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    let p = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.frag = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tile_w = overlay.tile_layout.x;
    let tile_h = overlay.tile_layout.y;
    let gap = overlay.tile_layout.z;

    // We don't have a viewport size here directly — but the fullscreen
    // triangle gives in.frag in [0, 1]². To pin tiles to pixel sizes,
    // recover via in.pos.xy (clip space → pixel). pos.xy is already in
    // window-pixel coords (wgpu provides that on @builtin(position)).
    let px = in.pos.xy;

    // Tile positions (top-left origin, like wgpu).
    let pad = 8.0;
    let r0_y0 = pad;
    let r1_y0 = pad + tile_h + gap;
    let c0_x0 = pad;
    let c1_x0 = pad + tile_w + gap;

    // Helper: sample-or-skip. Returns vec4(rgb, in_tile) where in_tile=1
    // if pixel falls inside the tile (we then take it), 0 otherwise.
    var out_rgb = vec4<f32>(0.0, 0.0, 0.0, 0.0); // .a=1 means "we wrote here"

    // Top-left: transmittance (R/G/B). Scale factor in overlay.scales.r.
    if (px.x >= c0_x0 && px.x < c0_x0 + tile_w &&
        px.y >= r0_y0 && px.y < r0_y0 + tile_h) {
        let uv = vec2<f32>(
            (px.x - c0_x0) / tile_w,
            (px.y - r0_y0) / tile_h,
        );
        let v = textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
        out_rgb = vec4<f32>(clamp(v * overlay.scales.r, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    // Top-right: multi-scatter. Scale in overlay.scales.g.
    if (px.x >= c1_x0 && px.x < c1_x0 + tile_w &&
        px.y >= r0_y0 && px.y < r0_y0 + tile_h) {
        let uv = vec2<f32>(
            (px.x - c1_x0) / tile_w,
            (px.y - r0_y0) / tile_h,
        );
        let v = textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
        out_rgb = vec4<f32>(clamp(v * overlay.scales.g, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    // Bottom-left: sky-view. Scale in overlay.scales.b.
    if (px.x >= c0_x0 && px.x < c0_x0 + tile_w &&
        px.y >= r1_y0 && px.y < r1_y0 + tile_h) {
        let uv = vec2<f32>(
            (px.x - c0_x0) / tile_w,
            (px.y - r1_y0) / tile_h,
        );
        let v = textureSampleLevel(skyview_lut, lut_sampler, uv, 0.0).rgb;
        out_rgb = vec4<f32>(clamp(v * overlay.scales.b, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    // Bottom-right: AP middle slice (z=0.5 in the 3D LUT).
    if (px.x >= c1_x0 && px.x < c1_x0 + tile_w &&
        px.y >= r1_y0 && px.y < r1_y0 + tile_h) {
        let uvw = vec3<f32>(
            (px.x - c1_x0) / tile_w,
            (px.y - r1_y0) / tile_h,
            0.5,
        );
        let v = textureSampleLevel(ap_lut, lut_sampler, uvw, 0.0);
        // Visualise inscatter (rgb). Transmittance (.a) shown as a small
        // strip at the bottom of the tile.
        let in_strip = (px.y - r1_y0) / tile_h > 0.9;
        if (in_strip) {
            out_rgb = vec4<f32>(v.a, v.a, v.a, 1.0);
        } else {
            out_rgb = vec4<f32>(clamp(v.rgb * overlay.scales.a, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        }
    }

    // If we didn't write a tile, discard so the swapchain content stays.
    if (out_rgb.a < 0.5) {
        discard;
    }
    return vec4<f32>(out_rgb.rgb, 1.0);
}
