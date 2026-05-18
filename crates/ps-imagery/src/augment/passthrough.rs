//! Identity augment stage. v1 default.

use super::ImageryAugment;
use crate::{tile::RgbTile, ImageryError};

/// No-op augment. Returns the input tile unchanged.
pub struct PassthroughAugment;

impl ImageryAugment for PassthroughAugment {
    fn augment(&self, tile: RgbTile) -> Result<RgbTile, ImageryError> {
        Ok(tile)
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
