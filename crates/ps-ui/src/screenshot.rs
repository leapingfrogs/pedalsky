//! Phase 10.4 screenshot encoders.
//!
//! - `write_png`: tonemapped PNG. Caller provides the `Rgba8Unorm`
//!   pixels (already linear→sRGB encoded by the tonemap pass).
//! - `write_exr`: HDR EXR. Caller provides `Rgba16Float` pixels
//!   (linear, post-AP, pre-tonemap).

use anyhow::{Context, Result};
use std::path::Path;

/// Write `Rgba8Unorm` pixels (top-left origin, tightly packed) to a PNG.
pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| anyhow::anyhow!("PNG buffer length mismatch"))?;
    img.save(path).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Write `Rgba16Float` pixels (top-left origin, tightly packed; 4 × f16
/// per pixel) to a 32-bit float EXR. Channels are written in RGBA order
/// with linear values.
pub fn write_exr(path: &Path, width: u32, height: u32, rgba_f16: &[half::f16]) -> Result<()> {
    if rgba_f16.len() < (width * height * 4) as usize {
        anyhow::bail!(
            "EXR buffer too small: have {} elements, need {}",
            rgba_f16.len(),
            width * height * 4
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Convert f16 → f32 for EXR encoding (the exr crate's high-level
    // writer prefers f32 channels; we deliberately store 32-bit so any
    // downstream tool reads with no precision loss).
    let pixel_count = (width * height) as usize;
    let mut r = Vec::with_capacity(pixel_count);
    let mut g = Vec::with_capacity(pixel_count);
    let mut b = Vec::with_capacity(pixel_count);
    let mut a = Vec::with_capacity(pixel_count);
    for chunk in rgba_f16.chunks_exact(4) {
        r.push(chunk[0].to_f32());
        g.push(chunk[1].to_f32());
        b.push(chunk[2].to_f32());
        a.push(chunk[3].to_f32());
    }
    use exr::prelude::*;
    let image = Image::from_channels(
        (width as usize, height as usize),
        SpecificChannels::rgba(|pos: Vec2<usize>| {
            let i = pos.y() * width as usize + pos.x();
            (r[i], g[i], b[i], a[i])
        }),
    );
    image
        .write()
        .to_file(path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}
