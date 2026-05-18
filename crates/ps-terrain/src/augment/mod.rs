//! Heightmap-level augment seam.
//!
//! This is the extension point for procedural detail synthesis (fBm
//! upsampling, hydrology-aware erosion, learned super-resolution,
//! multi-source fusion). v1 ships only [`PassthroughAugment`]; future
//! impls plug in here without touching fetch/decode/mesh.

mod passthrough;
pub use passthrough::PassthroughAugment;

use crate::{tile::HeightmapTile, TerrainError};

/// Heightmap transform. Consumes one tile, returns one tile.
///
/// **Invariant for impls:** if `width` or `height` change, `heights_m`
/// must be resized to match (`len == width * height`). The pipeline
/// validates this between stages.
pub trait HeightmapAugment {
    /// Run the transform. May allocate; designed to run on a worker
    /// thread.
    fn augment(&self, tile: HeightmapTile) -> Result<HeightmapTile, TerrainError>;

    /// Short label used in tracing output (e.g. `"passthrough"`,
    /// `"fbm-4x"`).
    fn name(&self) -> &'static str;
}
