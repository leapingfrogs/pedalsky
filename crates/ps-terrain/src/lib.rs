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
pub mod progress;
pub mod simplify;
pub mod source;
pub mod tile;

pub use augment::{ErosionAugment, ErosionParams, HeightmapAugment, PassthroughAugment};
pub use cache::BinaryCache;
pub use mesh::{build_grid_mesh, MeshData};
pub use progress::{NullProgressSink, TerrainProgressSink, TerrainStage};
pub use simplify::{
    DecimationParams, DelatinSimplify, MeshSimplify, PassthroughSimplify, SimplifyTarget,
};
pub use source::{copernicus_glo30::CopernicusGlo30, HeightmapSource};
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

    /// Build the pipeline used by the interactive PedalBack spec
    /// implementation: Copernicus source, GPU-driven `ErosionAugment`
    /// (Section 1 of `docs/pedalback_terrain_pipeline_spec.md`), and
    /// `DelatinSimplify` (Section 2). Caller supplies a wgpu device +
    /// queue so the augment can run compute passes off the render
    /// thread.
    pub fn pedalback(
        cache_root: PathBuf,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        erosion: ErosionParams,
        decimation: DecimationParams,
    ) -> Self {
        Self {
            source: Box::new(CopernicusGlo30::new(cache_root)),
            augment: Box::new(ErosionAugment::new(device, queue, erosion)),
            simplify: Box::new(DelatinSimplify::new(decimation)),
        }
    }

    /// Run the pipeline end-to-end with a null progress sink. See
    /// [`Self::run_with_progress`] for the variant that reports stage
    /// progress to a UI.
    pub fn run(&self, req: &TileRequest) -> Result<MeshData, TerrainError> {
        self.run_with_progress(req, &NullProgressSink)
    }

    /// Run the pipeline end-to-end and report stage progress to the
    /// caller's sink. Returns a `MeshData` ready for GPU upload.
    pub fn run_with_progress(
        &self,
        req: &TileRequest,
        progress: &dyn TerrainProgressSink,
    ) -> Result<MeshData, TerrainError> {
        progress.stage(TerrainStage::FetchingDem, 0, 1);
        let raw = self.source.fetch(req)?;
        progress.stage(TerrainStage::FetchingDem, 1, 1);
        tracing::info!(
            target: "ps_terrain",
            source = raw.source,
            w = raw.width, h = raw.height,
            gsd_m = raw.gsd_m_centre,
            "terrain: fetched raw tile"
        );

        let augmented = self.augment.augment_with_progress(raw, progress)?;
        validate_tile(&augmented).map_err(TerrainError::AugmentInvalid)?;
        tracing::info!(
            target: "ps_terrain",
            stage = self.augment.name(),
            w = augmented.width, h = augmented.height,
            "terrain: augment stage complete"
        );

        // Crop to the requested observer radius before building the
        // mesh. The crop is on the heightmap so the one-vertex-per-pixel
        // invariant in build_grid_mesh is unaffected.
        progress.stage(TerrainStage::Cropping, 0, 1);
        let cropped = tile::crop_to_radius(augmented, req);
        progress.stage(TerrainStage::Cropping, 1, 1);

        progress.stage(TerrainStage::BuildingMesh, 0, 1);
        let mesh = build_grid_mesh(&cropped, req);
        progress.stage(TerrainStage::BuildingMesh, 1, 1);
        tracing::info!(
            target: "ps_terrain",
            verts = mesh.positions.len(),
            tris = mesh.indices.len() / 3,
            "terrain: built dense grid mesh"
        );

        progress.stage(TerrainStage::Decimating, 0, 1);
        let simplified = self.simplify.simplify(
            mesh,
            req.simplify_target.unwrap_or(SimplifyTarget::Ratio(1.0)),
        )?;
        progress.stage(TerrainStage::Decimating, 1, 1);
        tracing::info!(
            target: "ps_terrain",
            stage = self.simplify.name(),
            verts = simplified.positions.len(),
            tris = simplified.indices.len() / 3,
            "terrain: simplify stage complete"
        );

        progress.stage(TerrainStage::Done, 1, 1);
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
