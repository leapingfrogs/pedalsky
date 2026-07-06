//! Phase 16 satellite imagery ingestion for PedalSky.
//!
//! Downloads RGB satellite imagery for a configured lat/lon and
//! makes it available to the ground subsystem as a texture overlay
//! that the WGSL shader can sample in place of the procedural Voronoi
//! albedo.
//!
//! Pipeline shape mirrors `ps-terrain`:
//!
//! ```text
//! Source(fetch+stitch)
//!   -> Augment(*)     -- RgbTile -> RgbTile (colour correction, etc.)
//!   -> caller uploads to GPU
//! ```
//!
//! `(*)` is an extension seam. v1 ships only [`PassthroughAugment`].
//!
//! v1 source: EOX s2cloudless (Sentinel-2 cloud-free yearly composite),
//! served over WMTS at <https://tiles.maps.eox.at> with no auth
//! required. CC BY 4.0 — attribution string is exposed via
//! [`ATTRIBUTION`].

#![deny(missing_docs)]

pub mod augment;
pub mod cache;
pub mod source;
pub mod tile;

pub use augment::{ImageryAugment, PassthroughAugment};
pub use cache::BinaryCache;
pub use source::{eox_s2cloudless::EoxS2Cloudless, ImagerySource};
pub use tile::{GeoExtent, ImageryRequest, ImageryResolution, RgbTile};

/// Progress sink for long-running tile fetches. Implementations
/// should be cheap and lock-free if possible — the source calls
/// `update` after every tile (or every cache hit).
///
/// `done` and `total` are absolute tile counts. `total` is published
/// once at the start (it doesn't change mid-fetch); `done` climbs
/// monotonically from 0 to `total`.
///
/// `Send + Sync` because the source runs on a worker thread but the
/// UI typically owns the sink and reads it from the main thread.
pub trait ImageryProgressSink: Send + Sync {
    /// Called whenever the source has finished one tile (downloaded
    /// or cache-hit). `total` is constant within one fetch.
    fn update(&self, done: u32, total: u32);
}

/// No-op sink for callers that don't care about progress.
pub struct NullProgressSink;

impl ImageryProgressSink for NullProgressSink {
    fn update(&self, _done: u32, _total: u32) {}
}

use std::path::PathBuf;

/// Attribution string required by the EOX s2cloudless CC BY 4.0
/// licence. Display this anywhere the satellite imagery is visible
/// (UI corner, screenshot watermark, etc.).
pub const ATTRIBUTION: &str = "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH \
     (Contains modified Copernicus Sentinel data 2024)";

/// Errors raised by the imagery pipeline.
#[derive(Debug, thiserror::Error)]
pub enum ImageryError {
    /// HTTP / network / IO failure during tile fetch.
    #[error("imagery source fetch failed: {0}")]
    Fetch(#[source] anyhow::Error),
    /// JPEG / PNG decode failure.
    #[error("imagery decode failed: {0}")]
    Decode(#[source] anyhow::Error),
    /// Augment stage produced an invalid tile.
    #[error("augment stage produced invalid tile: {0}")]
    AugmentInvalid(String),
}

/// Top-level pipeline composed of the two replaceable stages.
pub struct ImageryPipeline {
    /// Imagery source — currently always EOX s2cloudless in production.
    pub source: Box<dyn ImagerySource + Send + Sync>,
    /// Optional colour-correction / sharpening / relight seam.
    pub augment: Box<dyn ImageryAugment + Send + Sync>,
}

impl ImageryPipeline {
    /// Default v1 pipeline: EOX s2cloudless + passthrough augment.
    pub fn default_eox(cache_root: PathBuf) -> Self {
        Self {
            source: Box::new(EoxS2Cloudless::new(cache_root)),
            augment: Box::new(PassthroughAugment),
        }
    }

    /// Run the pipeline end-to-end. Returns an `RgbTile` covering at
    /// least the requested geographic extent. The caller uploads it
    /// to a GPU texture (see `ps-ground`).
    ///
    /// Convenience wrapper for [`Self::run_with_progress`] using a
    /// [`NullProgressSink`].
    pub fn run(&self, req: &ImageryRequest) -> Result<RgbTile, ImageryError> {
        self.run_with_progress(req, &NullProgressSink)
    }

    /// Run with a progress sink. The source calls `progress.update`
    /// after each tile (downloaded or cache-hit).
    pub fn run_with_progress(
        &self,
        req: &ImageryRequest,
        progress: &dyn ImageryProgressSink,
    ) -> Result<RgbTile, ImageryError> {
        let raw = self.source.fetch_with_progress(req, progress)?;
        tracing::info!(
            target: "ps_imagery",
            source = raw.source,
            w = raw.width, h = raw.height,
            "imagery: fetched raw tile"
        );

        let augmented = self.augment.augment(raw)?;
        validate_tile(&augmented).map_err(ImageryError::AugmentInvalid)?;
        tracing::info!(
            target: "ps_imagery",
            stage = self.augment.name(),
            w = augmented.width, h = augmented.height,
            "imagery: augment stage complete"
        );

        Ok(augmented)
    }
}

fn validate_tile(tile: &RgbTile) -> Result<(), String> {
    let expected = (tile.width as usize) * (tile.height as usize) * 4;
    if tile.pixels_rgba.len() != expected {
        return Err(format!(
            "pixels_rgba len {} != width*height*4 {} (w={}, h={})",
            tile.pixels_rgba.len(),
            expected,
            tile.width,
            tile.height
        ));
    }
    if tile.width < 1 || tile.height < 1 {
        return Err(format!("tile too small: {}x{}", tile.width, tile.height));
    }
    Ok(())
}
