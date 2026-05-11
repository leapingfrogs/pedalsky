//! One-shot generator for `tests/scenes/presets/mountain_wave_lozenges_128x128.bin`.
//!
//! Run with:
//!   cargo run --example generate_mountain_wave -p ps-app -- <out_path>
//!
//! Produces a 128x128 row-major little-endian f32 grid with lozenge-
//! shaped coverage patches (altocumulus lenticularis pattern). Each
//! lozenge is an axis-aligned ellipse modulated by a 2D Gaussian.

fn main() -> std::io::Result<()> {
    let size: usize = 128;
    let extent_m: f32 = 32_000.0;
    let half = extent_m * 0.5;
    let mut data = vec![0.0_f32; size * size];

    // Six lozenges arranged in two parallel "wave trains" running NE-SW
    // (the typical orographic wave pattern downwind of Scottish hills).
    let lozenges: [(f32, f32, f32, f32, f32); 6] = [
        // (centre_x_m, centre_z_m, semi_axis_x, semi_axis_z, peak_coverage)
        (-9_000.0, -6_000.0, 4_500.0, 1_400.0, 0.85),
        (-1_500.0, -3_000.0, 4_500.0, 1_400.0, 0.95),
        (6_000.0, 0.0, 4_500.0, 1_400.0, 0.85),
        (-6_000.0, 4_500.0, 4_000.0, 1_300.0, 0.75),
        (1_500.0, 7_500.0, 4_000.0, 1_300.0, 0.85),
        (9_000.0, 10_500.0, 4_000.0, 1_300.0, 0.65),
    ];

    for y in 0..size {
        for x in 0..size {
            let world_x = (x as f32 + 0.5) / size as f32 * extent_m - half;
            let world_z = (y as f32 + 0.5) / size as f32 * extent_m - half;
            let mut max_cov = 0.0_f32;
            for (cx, cz, ax, az, peak) in &lozenges {
                let dx = (world_x - cx) / ax;
                let dz = (world_z - cz) / az;
                let r2 = dx * dx + dz * dz;
                let cov = peak * (-2.0 * r2).exp();
                if cov > max_cov {
                    max_cov = cov;
                }
            }
            data[y * size + x] = max_cov.clamp(0.0, 1.0);
        }
    }

    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/scenes/presets/mountain_wave_lozenges_128x128.bin".to_string());
    std::fs::write(&path, &bytes)?;
    println!("wrote {} ({} bytes)", path, bytes.len());
    Ok(())
}
