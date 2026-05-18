//! Heightmap source seam — fetches + decodes a tile for a request.
//!
//! v1 ships [`copernicus_glo30::CopernicusGlo30`]; future impls
//! (EEA-10 for Europe, OpenTopography for non-AWS access, bundled
//! offline tiles) plug in here by implementing [`HeightmapSource`].

pub mod copernicus_glo30;

use crate::{tile::{HeightmapTile, TileRequest}, TerrainError};

/// Source of heightmap tiles.
///
/// Implementations are expected to be self-caching (the GLO-30 impl
/// uses [`crate::cache::BinaryCache`]); the pipeline does not cache
/// between source and augment.
pub trait HeightmapSource {
    /// Fetch the tile covering `req.lat / req.lon`. The returned tile
    /// may be larger than the requested `radius_m` — cropping happens
    /// in the pipeline after augment.
    fn fetch(&self, req: &TileRequest) -> Result<HeightmapTile, TerrainError>;
}
