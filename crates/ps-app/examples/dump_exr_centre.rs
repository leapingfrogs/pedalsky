//! Diagnostic — dump centre column of an EXR as RGB triples.
//!
//!     cargo run --example dump_exr_centre -p ps-app -- <path.exr>

use exr::prelude::*;

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).expect("usage: dump_exr_centre <path>");
    let image = read_first_rgba_layer_from_file(
        &path,
        |resolution, _channels| {
            let w = resolution.width();
            let h = resolution.height();
            vec![vec![(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32); w]; h]
        },
        |buf, position, (r, g, b, a): (f32, f32, f32, f32)| {
            buf[position.y()][position.x()] = (r, g, b, a);
        },
    )
    .expect("read EXR");

    let buf = &image.layer_data.channel_data.pixels;
    let h = buf.len();
    let w = buf[0].len();
    let cx = w / 2;
    println!("# {path}");
    println!("# resolution: {w}x{h}");

    // A small `--scan-y N` mode: dump R/G/B and R:B ratio across the
    // whole row at y=N. Useful for finding chromatic shifts across
    // cloud edges (Phase 12.2 RGB transmittance verification).
    let scan_y: Option<usize> = std::env::args()
        .position(|a| a == "--scan-y")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok());

    if let Some(y) = scan_y {
        println!("# scan row y={y}: x → R G B (R/B)");
        for x in (0..w).step_by(40) {
            let p = buf[y][x];
            let ratio = if p.2 > 1e-3 { p.0 / p.2 } else { 0.0 };
            println!(
                "x={x:4}  R={:.1}  G={:.1}  B={:.1}  R/B={ratio:.3}",
                p.0, p.1, p.2
            );
        }
    } else {
        println!("# centre column (x={cx}), y → R G B");
        for y in (0..h).step_by(40) {
            let p = buf[y][cx];
            println!("y={y:4}  R={:.3}  G={:.3}  B={:.3}", p.0, p.1, p.2);
        }
    }
    Ok(())
}
