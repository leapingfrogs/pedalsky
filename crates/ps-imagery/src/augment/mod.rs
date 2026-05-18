//! Imagery-level augment seam.
//!
//! Extension point for colour correction, sharpening, time-of-day
//! relight, cloud-mask compositing. v1 ships only [`PassthroughAugment`].

mod passthrough;
pub use passthrough::PassthroughAugment;

use crate::{tile::RgbTile, ImageryError};

/// RGB transform. Consumes one tile, returns one tile.
pub trait ImageryAugment {
    /// Run the transform.
    fn augment(&self, tile: RgbTile) -> Result<RgbTile, ImageryError>;

    /// Short label for tracing output.
    fn name(&self) -> &'static str;
}
