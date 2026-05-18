//! Identity augment stage.

use super::HeightmapAugment;
use crate::progress::TerrainProgressSink;
use crate::{tile::HeightmapTile, TerrainError};

/// No-op augment. Returns the input tile unchanged.
pub struct PassthroughAugment;

impl HeightmapAugment for PassthroughAugment {
    fn augment_with_progress(
        &self,
        tile: HeightmapTile,
        _progress: &dyn TerrainProgressSink,
    ) -> Result<HeightmapTile, TerrainError> {
        Ok(tile)
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
