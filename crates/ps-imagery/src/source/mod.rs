//! Imagery source seam.

pub mod eox_s2cloudless;

use crate::{
    tile::{ImageryRequest, RgbTile},
    ImageryError, ImageryProgressSink, NullProgressSink,
};

/// Source of RGB satellite imagery tiles.
pub trait ImagerySource {
    /// Fetch + stitch tiles covering the requested extent.
    ///
    /// The default impl forwards to [`Self::fetch_with_progress`] with
    /// a [`NullProgressSink`]; impls only need to override one method
    /// (the progress-aware one is the primary).
    fn fetch(&self, req: &ImageryRequest) -> Result<RgbTile, ImageryError> {
        self.fetch_with_progress(req, &NullProgressSink)
    }

    /// Fetch with progress reporting. Override this; the no-progress
    /// `fetch` will forward automatically.
    fn fetch_with_progress(
        &self,
        req: &ImageryRequest,
        progress: &dyn ImageryProgressSink,
    ) -> Result<RgbTile, ImageryError>;
}
