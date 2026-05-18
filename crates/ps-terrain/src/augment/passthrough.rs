//! Identity augment stage. v1 default.

use super::HeightmapAugment;
use crate::{tile::HeightmapTile, TerrainError};

/// No-op augment. Returns the input tile unchanged.
///
/// Exists so the pipeline's three-stage shape (fetch → augment →
/// mesh → simplify) is in place from v1. Real upsamplers
/// (fBm, super-resolution, erosion) drop in here later by implementing
/// [`HeightmapAugment`] and being swapped into the
/// `HeightmapPipeline::augment` field.
pub struct PassthroughAugment;

impl HeightmapAugment for PassthroughAugment {
    fn augment(&self, tile: HeightmapTile) -> Result<HeightmapTile, TerrainError> {
        Ok(tile)
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
