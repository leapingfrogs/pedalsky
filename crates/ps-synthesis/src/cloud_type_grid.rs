//! Phase 12.1 — per-pixel cloud-type grid (companion to the
//! coverage grid).
//!
//! When a scene's `[clouds.coverage_grid]` block sets `type_path`, the
//! file is a row-major u8 array of cloud-type indices (0..7 matching
//! [`ps_core::CloudType`]). The values are uploaded into a
//! `R8Uint` texture; the cloud march samples it via `textureLoad`
//! with integer coordinates (nearest-by-construction — interpolating
//! type indices is meaningless).
//!
//! When no `type_path` is set, or the file fails to load, the
//! texture is uploaded with a sentinel value of [`SENTINEL`] so the
//! cloud march falls back to the per-layer `cloud_type`.

use ps_core::Scene;
use tracing::warn;

/// Texture dimensions (square; matches the weather map).
pub const SIZE: u32 = 128;

/// Sentinel meaning "use the layer's cloud_type instead of a
/// per-pixel override". Picked outside the valid 0..=7 range; the
/// cloud march compares against this constant. Real cloud type
/// indices stay in 0..7 (eight enum variants in
/// [`ps_core::CloudType`]).
pub const SENTINEL: u8 = 255;

/// CPU-side cloud-type grid: row-major u8 of size [`SIZE`] × [`SIZE`].
#[derive(Debug, Clone)]
pub struct CloudTypeGrid {
    /// Tightly packed row-major u8 type indices.
    pub pixels: Vec<u8>,
    /// Width = height in pixels.
    pub size: u32,
}

impl CloudTypeGrid {
    /// Synthesise the grid for a scene. Returns a sentinel-filled
    /// grid when the scene has no `coverage_grid.type_path` or when
    /// the file fails to load (loader logs at `warn`).
    pub fn synthesise(scene: &Scene) -> Self {
        let pixels = (|| -> Option<Vec<u8>> {
            let grid = scene.clouds.coverage_grid.as_ref()?;
            let path = grid.type_path.as_ref()?;
            let [src_w, src_h] = grid.size;
            let expected_bytes = (src_w as usize) * (src_h as usize);
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        target: "ps_synthesis::cloud_type_grid",
                        path = %path.display(),
                        error = %e,
                        "cloud-type grid read failed; using sentinel"
                    );
                    return None;
                }
            };
            if bytes.len() != expected_bytes {
                warn!(
                    target: "ps_synthesis::cloud_type_grid",
                    path = %path.display(),
                    got = bytes.len(),
                    want = expected_bytes,
                    "cloud-type grid size mismatch; using sentinel"
                );
                return None;
            }
            // Resample the source grid onto our SIZE × SIZE target via
            // nearest-neighbour. Values that aren't valid type
            // indices are mapped to SENTINEL so the cloud march
            // doesn't index into an out-of-range NDF case (which
            // would return 0 and silently produce no cloud).
            let mut out = vec![SENTINEL; (SIZE * SIZE) as usize];
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let sx = (x as f32 / SIZE as f32 * src_w as f32) as u32;
                    let sy = (y as f32 / SIZE as f32 * src_h as f32) as u32;
                    let sx = sx.min(src_w - 1);
                    let sy = sy.min(src_h - 1);
                    let v = bytes[(sy * src_w + sx) as usize];
                    out[(y * SIZE + x) as usize] = if v <= 7 { v } else { SENTINEL };
                }
            }
            Some(out)
        })()
        .unwrap_or_else(|| vec![SENTINEL; (SIZE * SIZE) as usize]);

        Self { pixels, size: SIZE }
    }

    /// Allocate + upload the cloud-type grid as an `R8Uint` 2D
    /// texture. Returns the texture and its default view.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud-type-grid"),
            size: wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.size),
                rows_per_image: Some(self.size),
            },
            wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ps_core::{Aurora, Clouds, Lightning, Precipitation, Surface};

    fn empty_scene() -> Scene {
        Scene {
            schema_version: 1,
            surface: Surface::default(),
            clouds: Clouds::default(),
            precipitation: Precipitation::default(),
            lightning: Lightning::default(),
            aurora: Aurora::default(),
            water: None,
        }
    }

    #[test]
    fn no_grid_yields_sentinel_filled() {
        let grid = CloudTypeGrid::synthesise(&empty_scene());
        assert_eq!(grid.size, SIZE);
        assert_eq!(grid.pixels.len(), (SIZE * SIZE) as usize);
        assert!(grid.pixels.iter().all(|&v| v == SENTINEL));
    }
}
