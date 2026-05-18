//! Phase 16 terrain ingestion for PedalSky.
//!
//! Downloads Copernicus DEM tiles for a configured lat/lon, runs them
//! through a two-seam pipeline (heightmap-level augment, then mesh-level
//! simplify), and produces a `MeshData` value the ground subsystem
//! uploads to the GPU.
//!
//! Pipeline:
//!
//! ```text
//! Source(fetch+decode)
//!   -> Augment(*)     -- HeightmapTile -> HeightmapTile
//!   -> Mesh(grid)     -- one vertex per heightmap pixel
//!   -> Simplify(*)    -- MeshData -> MeshData (decimation)
//!   -> caller uploads to GPU
//! ```
//!
//! `(*)` are extension seams. v1 ships identity impls
//! (`PassthroughAugment`, `PassthroughSimplify`) so the dense GLO-30
//! grid flows through untouched. Follow-up work plugs in fBm /
//! super-resolution at `augment` and quadric-error decimation at
//! `simplify` without touching fetch/decode.

#![deny(missing_docs)]

pub mod augment;
pub mod cache;
pub mod mesh;
pub mod simplify;
pub mod source;
pub mod tile;

pub use augment::{HeightmapAugment, PassthroughAugment};
pub use cache::BinaryCache;
pub use mesh::{MeshData, build_grid_mesh};
pub use simplify::{MeshSimplify, PassthroughSimplify, SimplifyTarget};
pub use source::{HeightmapSource, copernicus_glo30::CopernicusGlo30};
pub use tile::{GeoExtent, HeightmapTile, TileRequest};

use std::path::PathBuf;

/// Errors raised by the terrain pipeline.
#[derive(Debug, thiserror::Error)]
pub enum TerrainError {
    /// HTTP / network / IO failure during fetch.
    #[error("terrain source fetch failed: {0}")]
    Fetch(#[source] anyhow::Error),
    /// TIFF / COG decode failure.
    #[error("terrain decode failed: {0}")]
    Decode(#[source] anyhow::Error),
    /// Augment stage returned an invalid tile (e.g. heights len didn't
    /// match width*height).
    #[error("augment stage produced invalid tile: {0}")]
    AugmentInvalid(String),
    /// Simplify stage returned an invalid mesh.
    #[error("simplify stage produced invalid mesh: {0}")]
    SimplifyInvalid(String),
    /// Mesh build failure (e.g. tile too small).
    #[error("mesh build failed: {0}")]
    Mesh(String),
}

/// Top-level pipeline composed of the three replaceable stages.
///
/// Construct via [`HeightmapPipeline::default_copernicus`] for the
/// stock GLO-30 + passthrough setup, or build by hand for tests /
/// extension impls.
pub struct HeightmapPipeline {
    /// Heightmap source — currently always GLO-30 in production builds.
    pub source: Box<dyn HeightmapSource + Send + Sync>,
    /// Heightmap-level augment seam. v1 default: identity.
    pub augment: Box<dyn HeightmapAugment + Send + Sync>,
    /// Mesh-level simplification seam. v1 default: identity.
    pub simplify: Box<dyn MeshSimplify + Send + Sync>,
}

impl HeightmapPipeline {
    /// Build the default v1 pipeline: Copernicus GLO-30 source,
    /// passthrough augment, passthrough simplify. Cache lives under
    /// `cache_root` (typically `./cache/terrain`).
    pub fn default_copernicus(cache_root: PathBuf) -> Self {
        Self {
            source: Box::new(CopernicusGlo30::new(cache_root)),
            augment: Box::new(PassthroughAugment),
            simplify: Box::new(PassthroughSimplify),
        }
    }

    /// Run the pipeline end-to-end. Returns a `MeshData` whose vertex
    /// count equals `width * height` of the (augmented) heightmap tile
    /// minus any observer-radius crop. Triangle count is
    /// `(W-1) * (H-1) * 2`.
    ///
    /// The caller is responsible for the GPU upload and any
    /// origin-shift the renderer needs (see ps-ground).
    pub fn run(&self, req: &TileRequest) -> Result<MeshData, TerrainError> {
        let raw = self.source.fetch(req)?;
        tracing::info!(
            target: "ps_terrain",
            source = raw.source,
            w = raw.width, h = raw.height,
            gsd_m = raw.gsd_m_centre,
            "terrain: fetched raw tile"
        );

        let augmented = self.augment.augment(raw)?;
        validate_tile(&augmented)
            .map_err(TerrainError::AugmentInvalid)?;
        tracing::info!(
            target: "ps_terrain",
            stage = self.augment.name(),
            w = augmented.width, h = augmented.height,
            "terrain: augment stage complete"
        );

        // Crop to the requested observer radius before building the
        // mesh. The crop is on the heightmap so the one-vertex-per-pixel
        // invariant in build_grid_mesh is unaffected.
        let cropped = tile::crop_to_radius(augmented, req);

        let mesh = build_grid_mesh(&cropped, req);
        tracing::info!(
            target: "ps_terrain",
            verts = mesh.positions.len(),
            tris = mesh.indices.len() / 3,
            "terrain: built dense grid mesh"
        );

        let simplified = self.simplify.simplify(
            mesh,
            req.simplify_target.unwrap_or(SimplifyTarget::Ratio(1.0)),
        )?;
        tracing::info!(
            target: "ps_terrain",
            stage = self.simplify.name(),
            verts = simplified.positions.len(),
            tris = simplified.indices.len() / 3,
            "terrain: simplify stage complete"
        );

        Ok(simplified)
    }
}

fn validate_tile(tile: &HeightmapTile) -> Result<(), String> {
    let expected = (tile.width as usize) * (tile.height as usize);
    if tile.heights_m.len() != expected {
        return Err(format!(
            "heights_m len {} != width*height {} (w={}, h={})",
            tile.heights_m.len(),
            expected,
            tile.width,
            tile.height
        ));
    }
    if tile.width < 2 || tile.height < 2 {
        return Err(format!("tile too small: {}x{}", tile.width, tile.height));
    }
    Ok(())
}
