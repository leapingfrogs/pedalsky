//! Phase 3 §3.2.5 — top-down density mask.
//!
//! For each ground sample on the same 32 km × 32 km grid as the weather
//! map (§3.2.3), integrate the cloud-density column above using the same
//! NDF the renderer uses (Phase 6.4 / [`crate::ndf`]). Step the column at
//! 100 m vertical intervals, multiply by `coverage * density_scale`, and
//! collapse to a [0, 1] R8 value. Used by Phase 8 precipitation to gate
//! rain streaks under low cloud.
//!
//! Currently the layer-coverage value is uniform across the grid (per
//! plan §3.2.3 R-channel). Once gridded coverage lands the mask becomes
//! spatially varying.

use ps_core::CloudLayerGpu;

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
    /// Synthesise from the cloud-layer envelopes.
    pub fn synthesise(layers: &[CloudLayerGpu]) -> Self {
        // For v1 the mask is spatially uniform per-layer. We sample the
        // tallest layer's column-integrated density at the world origin
        // and broadcast across the grid. Once gridded coverage lands the
        // inner integral becomes per-pixel.
        let column_density = integrate_column(layers);
        let value = (column_density * 255.0).clamp(0.0, 255.0) as u8;

        let pixels = vec![value; (SIZE * SIZE) as usize];

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

/// Integrate vertical density across all layers and return a normalised
/// total in [0, 1]. Each layer contributes
/// `coverage · density_scale · ∫ NDF(h) dh`.
fn integrate_column(layers: &[CloudLayerGpu]) -> f32 {
    let mut total = 0.0_f32;
    for layer in layers {
        let thickness = (layer.top_m - layer.base_m).max(1.0);
        let n_steps = (thickness / STEP_M).ceil().max(1.0) as u32;
        let dh = 1.0 / n_steps as f32;
        let mut sum = 0.0_f32;
        for k in 0..n_steps {
            let h_norm = (k as f32 + 0.5) * dh;
            sum += ndf::ndf(h_norm, layer.cloud_type);
        }
        // Average NDF over the layer × layer thickness in km × coverage × density.
        let mean_ndf = sum / n_steps as f32;
        let layer_density = mean_ndf * (thickness / 1000.0) * layer.coverage * layer.density_scale;
        total += layer_density;
    }
    // Saturate around "10 km of full-density cloud column"; this keeps the
    // [0, 1] range meaningful without specialising per cloud type.
    (total / 10.0).clamp(0.0, 1.0)
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
        }
    }

    #[test]
    fn empty_layers_produce_zero_mask() {
        let mask = TopDownMask::synthesise(&[]);
        assert!(mask.pixels.iter().all(|&v| v == 0));
    }

    #[test]
    fn cumulus_produces_nonzero_mask() {
        let mask = TopDownMask::synthesise(&[cumulus_layer(0.5)]);
        assert!(mask.pixels[0] > 0, "mask should be > 0 under cumulus");
    }

    #[test]
    fn higher_coverage_gives_higher_mask() {
        let low = TopDownMask::synthesise(&[cumulus_layer(0.2)]);
        let high = TopDownMask::synthesise(&[cumulus_layer(0.9)]);
        assert!(high.pixels[0] > low.pixels[0]);
    }
}
