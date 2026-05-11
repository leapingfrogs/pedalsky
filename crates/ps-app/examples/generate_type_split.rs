//! Generate a 128×128 split coverage + type grid for the
//! cumulus_meets_stratus diagnostic scene (Phase 12.1).
//!
//!     cargo run --example generate_type_split -p ps-app
//!
//! Writes two files alongside `tests/scenes/presets/`:
//!   coverage_split_128x128.bin  — f32 row-major, all 1.0 (full
//!                                  cloud cover everywhere)
//!   type_split_128x128.bin      — u8 row-major, left half = 0
//!                                  (Cumulus), right half = 1
//!                                  (Stratus)
//!
//! The cloud march samples both grids per-pixel and renders the two
//! halves with their respective NDF profiles, demonstrating
//! per-pixel cloud type blending.

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let size: usize = 128;
    let scenes_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/scenes/presets"));
    std::fs::create_dir_all(&scenes_dir)?;

    // Coverage: uniform 1.0 (cloud everywhere; the type grid is
    // what differentiates the two halves).
    let mut coverage = Vec::with_capacity(size * size * 4);
    for _ in 0..(size * size) {
        coverage.extend_from_slice(&1.0f32.to_le_bytes());
    }
    let cov_path = scenes_dir.join("coverage_split_128x128.bin");
    std::fs::write(&cov_path, &coverage)?;
    println!("wrote {} ({} bytes)", cov_path.display(), coverage.len());

    // Type: left half 0 (Cumulus), right half 1 (Stratus).
    let mut types = Vec::with_capacity(size * size);
    for _y in 0..size {
        for x in 0..size {
            let t: u8 = if x < size / 2 { 0 } else { 1 };
            types.push(t);
        }
    }
    let type_path = scenes_dir.join("type_split_128x128.bin");
    std::fs::write(&type_path, &types)?;
    println!("wrote {} ({} bytes)", type_path.display(), types.len());

    Ok(())
}
