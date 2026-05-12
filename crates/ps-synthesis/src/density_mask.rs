//! Phase 3 §3.2.5 — top-down density mask.
//!
//! For each ground sample on the same 32 km × 32 km grid as the weather
//! map (§3.2.3), integrate the cloud-density column above using the same
//! NDF the renderer uses (Phase 6.4 / [`crate::ndf`]). Step the column at
//! 100 m vertical intervals, multiply by `coverage * density_scale`, and
//! collapse to a [0, 1] R8 value. Used by:
//! - Phase 8 precipitation to gate rain streaks under low cloud
//! - Phase 12.6 ground shader for cloud-modulated overcast diffuse
//!
//! Phase 12.1 / followup #76 — when a `coverage_grid` is loaded the
//! mask becomes spatially varying: each output pixel multiplies the
//! layer's mean NDF and density_scale by the *gridded* coverage at
//! that XZ rather than the layer's flat scalar coverage. Without a
//! grid the mask stays uniform across the 32 km extent (the Phase 3
//! v1 behaviour).

use ps_core::CloudLayerGpu;

use crate::coverage_grid::LoadedCoverageGrid;
use crate::ndf;

/// Map dimensions (matches the weather map).
pub const SIZE: u32 = 128;
/// Spatial extent in metres (matches the weather map).
pub const EXTENT_M: f32 = 32_000.0;
/// Vertical integration step in metres.
const STEP_M: f32 = 100.0;

/// CPU-side top-down density mask, ready for upload.
#[derive(Debug, Clone)]
pub struct TopDownMask {
    /// Tightly packed `u8` row-major pixels.
    pub pixels: Vec<u8>,
    /// Width = height in pixels.
    pub size: u32,
    /// Spatial extent in metres.
    pub extent_m: f32,
}

impl TopDownMask {
    /// Synthesise from the cloud-layer envelopes. When `grid` is
    /// `Some(_)`, the per-pixel coverage is sampled from it; the
    /// resulting mask varies spatially. When `None`, the mask is
    /// uniform (the Phase 3 v1 behaviour) using each layer's flat
    /// `coverage` field.
    pub fn synthesise(
        layers: &[CloudLayerGpu],
        grid: Option<&LoadedCoverageGrid>,
    ) -> Self {
        let pixels = match grid {
            // Spatially-varying: integrate per pixel using the
            // gridded coverage at that XZ.
            Some(g) => {
                let mut out = vec![0u8; (SIZE * SIZE) as usize];
                let half = EXTENT_M * 0.5;
                for y in 0..SIZE {
                    for x in 0..SIZE {
                        // Pixel centre → world XZ on the same convention
                        // weather_map uses.
                        let fx = (x as f32 + 0.5) / SIZE as f32;
                        let fy = (y as f32 + 0.5) / SIZE as f32;
                        let gx = fx * EXTENT_M - half;
                        let gy = fy * EXTENT_M - half;
                        let cov = g.sample_world(gx, gy);
                        let col = integrate_column_with_coverage(layers, cov);
                        out[(y * SIZE + x) as usize] =
                            (col * 255.0).clamp(0.0, 255.0) as u8;
                    }
                }
                out
            }
            // Uniform v1 behaviour: integrate once with each layer's
            // own coverage and broadcast.
            None => {
                let column_density = integrate_column(layers);
                let value = (column_density * 255.0).clamp(0.0, 255.0) as u8;
                vec![value; (SIZE * SIZE) as usize]
            }
        };

        Self {
            pixels,
            size: SIZE,
            extent_m: EXTENT_M,
        }
    }

    /// Allocate + upload an R8Unorm 2D texture.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("top-down-density-mask"),
            size: wgpu::Extent3d {
                width: self.size,
                height: self.size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
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

/// Integrate vertical density across all layers using each layer's
/// own scalar coverage. Returns a normalised total in [0, 1]. Each
/// layer contributes `coverage · density_scale · ∫ NDF(h) dh`.
fn integrate_column(layers: &[CloudLayerGpu]) -> f32 {
    let mut total = 0.0_f32;
    for layer in layers {
        total += per_layer_density(layer, layer.coverage);
    }
    (total / 10.0).clamp(0.0, 1.0)
}

/// Like [`integrate_column`] but every layer uses the same external
/// `coverage` value (the gridded sample at this XZ). Layers retain
/// their per-layer `density_scale` and NDF profile.
fn integrate_column_with_coverage(layers: &[CloudLayerGpu], coverage: f32) -> f32 {
    let mut total = 0.0_f32;
    for layer in layers {
        total += per_layer_density(layer, coverage);
    }
    (total / 10.0).clamp(0.0, 1.0)
}

/// Single-layer column density contribution. Mean NDF × thickness ×
/// coverage × density_scale.
fn per_layer_density(layer: &CloudLayerGpu, coverage: f32) -> f32 {
    let thickness = (layer.top_m - layer.base_m).max(1.0);
    let n_steps = (thickness / STEP_M).ceil().max(1.0) as u32;
    let dh = 1.0 / n_steps as f32;
    let mut sum = 0.0_f32;
    for k in 0..n_steps {
        let h_norm = (k as f32 + 0.5) * dh;
        sum += ndf::ndf(h_norm, layer.cloud_type);
    }
    let mean_ndf = sum / n_steps as f32;
    mean_ndf * (thickness / 1000.0) * coverage * layer.density_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cumulus_layer(cov: f32) -> CloudLayerGpu {
        CloudLayerGpu {
            base_m: 1500.0,
            top_m: 2300.0,
            coverage: cov,
            density_scale: 1.0,
            cloud_type: 0, // Cumulus
            shape_bias: 0.0,
            detail_bias: 0.0,
            anvil_bias: 0.0,
            droplet_diameter_um: 20.0,
            _pad_after_droplets_0: 0.0,
            _pad_after_droplets_1: 0.0,
            _pad_after_droplets_2: 0.0,
        }
    }

    #[test]
    fn empty_layers_produce_zero_mask() {
        let mask = TopDownMask::synthesise(&[], None);
        assert!(mask.pixels.iter().all(|&v| v == 0));
    }

    #[test]
    fn cumulus_produces_nonzero_mask() {
        let mask = TopDownMask::synthesise(&[cumulus_layer(0.5)], None);
        assert!(mask.pixels[0] > 0, "mask should be > 0 under cumulus");
    }

    #[test]
    fn higher_coverage_gives_higher_mask() {
        let low = TopDownMask::synthesise(&[cumulus_layer(0.2)], None);
        let high = TopDownMask::synthesise(&[cumulus_layer(0.9)], None);
        assert!(high.pixels[0] > low.pixels[0]);
    }

    #[test]
    fn gridded_coverage_produces_spatially_varying_mask() {
        // 4×4 grid: full coverage on the left half, zero on the right.
        let data: Vec<f32> = (0..16)
            .map(|i| if (i % 4) < 2 { 1.0 } else { 0.0 })
            .collect();
        let grid = LoadedCoverageGrid {
            data,
            src_w: 4,
            src_h: 4,
            extent_m: EXTENT_M,
        };
        let mask = TopDownMask::synthesise(&[cumulus_layer(0.5)], Some(&grid));
        // Sample left side (world x ≈ -8000) vs right side (x ≈ +8000)
        // — they should differ.
        let left_mask = mask.pixels[(SIZE / 2 * SIZE + SIZE / 4) as usize];
        let right_mask =
            mask.pixels[(SIZE / 2 * SIZE + 3 * SIZE / 4) as usize];
        assert!(
            left_mask > right_mask,
            "left (full coverage) mask should exceed right (zero coverage) — \
             got left={left_mask}, right={right_mask}"
        );
        assert!(right_mask < 8, "right side should be near-zero (got {right_mask})");
    }
}
