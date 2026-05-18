//! Heightmap-level augment seam.
//!
//! The extension point for procedural detail synthesis. The
//! [`PassthroughAugment`] keeps the v1 contract; [`ErosionAugment`]
//! implements the full Section 1 pipeline from
//! `docs/pedalback_terrain_pipeline_spec.md` (upsample → hydraulic
//! erosion → thermal erosion → fractal detail → normal map).

mod erosion;
mod passthrough;
mod upsample;

pub use erosion::{ErosionAugment, ErosionParams};
pub use passthrough::PassthroughAugment;

use crate::progress::{NullProgressSink, TerrainProgressSink};
use crate::{tile::HeightmapTile, TerrainError};

/// Heightmap transform. Consumes one tile, returns one tile.
///
/// **Invariant for impls:** if `width` or `height` change, `heights_m`
/// must be resized to match (`len == width * height`). The pipeline
/// validates this between stages.
pub trait HeightmapAugment {
    /// Run the transform with no progress reporting. The default impl
    /// forwards to [`Self::augment_with_progress`] using a
    /// `NullProgressSink`; concrete impls override the progress-aware
    /// variant.
    fn augment(&self, tile: HeightmapTile) -> Result<HeightmapTile, TerrainError> {
        self.augment_with_progress(tile, &NullProgressSink)
    }

    /// Run the transform, streaming stage progress to the sink. Override
    /// this; `augment` will forward automatically.
    fn augment_with_progress(
        &self,
        tile: HeightmapTile,
        progress: &dyn TerrainProgressSink,
    ) -> Result<HeightmapTile, TerrainError>;

    /// Short label used in tracing output (e.g. `"passthrough"`,
    /// `"erosion"`).
    fn name(&self) -> &'static str;
}
