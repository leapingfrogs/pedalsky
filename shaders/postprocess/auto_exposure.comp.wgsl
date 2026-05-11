// Phase 9.2 — debug auto-exposure.
//
// Single-workgroup reduction of the HDR target's log2(luminance). Each
// thread strides across the image accumulating into workgroup-shared
// memory; thread 0 sums the 256 partial sums and writes a single f32
// to the output buffer. The host reads it back (one-frame lag) and
// derives EV100 to centre mid-grey.
//
// Inefficient (single workgroup) but correct, and only runs when
// [debug] auto_exposure = true.

struct AeOutput {
    log_lum_sum: f32,
    pixel_count: f32,
};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: AeOutput;

const WG_X: u32 = 16u;
const WG_Y: u32 = 16u;
const WG_TOTAL: u32 = WG_X * WG_Y;
// Stride between samples per thread; effectively a thinning factor.
// 4 means we sample 1 of every 16 pixels (4 in each axis), which is
// far more than enough for an exposure estimate.
const STRIDE: u32 = 4u;

var<workgroup> partials: array<f32, WG_TOTAL>;
var<workgroup> counts: array<f32, WG_TOTAL>;

@compute @workgroup_size(WG_X, WG_Y, 1)
fn cs_main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) _wid: vec3<u32>,
) {
    let dims = textureDimensions(hdr_tex);
    let w = dims.x;
    let h = dims.y;

    var sum: f32 = 0.0;
    var n: f32 = 0.0;
    let tx = lid % WG_X;
    let ty = lid / WG_X;
    // Each thread covers a regular sub-grid of the image.
    var y = ty * STRIDE;
    loop {
        if (y >= h) { break; }
        var x = tx * STRIDE;
        loop {
            if (x >= w) { break; }
            let rgb = textureLoad(hdr_tex, vec2<i32>(i32(x), i32(y)), 0).rgb;
            let lum = max(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-6);
            sum = sum + log2(lum);
            n = n + 1.0;
            x = x + WG_X * STRIDE;
        }
        y = y + WG_Y * STRIDE;
    }
    partials[lid] = sum;
    counts[lid] = n;
    workgroupBarrier();

    if (lid == 0u) {
        var total_sum: f32 = 0.0;
        var total_n: f32 = 0.0;
        for (var i = 0u; i < WG_TOTAL; i = i + 1u) {
            total_sum = total_sum + partials[i];
            total_n = total_n + counts[i];
        }
        output.log_lum_sum = total_sum;
        output.pixel_count = total_n;
    }
}
