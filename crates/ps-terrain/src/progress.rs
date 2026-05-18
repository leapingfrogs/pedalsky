//! Progress reporting for the heightmap pipeline.
//!
//! Mirrors `ps_imagery::ImageryProgressSink`. The host implements the
//! trait against a `UiHandle` so the worker thread can stream stage +
//! per-iteration progress into the panel's progress bar.

/// Coarse-grained pipeline stage. The UI uses this both to label the
/// progress bar ("Hydraulic erosion (45/200)") and to drive a tiny
/// state machine that decides when to switch the label.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TerrainStage {
    /// Source fetch (Copernicus tile download or cache hit).
    FetchingDem,
    /// Stage 1.1 — bicubic upsample to the working resolution.
    Upsampling,
    /// Stage 1.2 — hydraulic erosion iterations.
    HydraulicErosion,
    /// Stage 1.3 — thermal erosion (relaxes overhangs).
    ThermalErosion,
    /// Stage 1.4 — high-frequency fractal detail injection.
    FractalDetail,
    /// Stage 1.5 — normal map generation.
    NormalMap,
    /// Observer-radius crop on the augmented heightmap.
    Cropping,
    /// CPU mesh build (one vertex per heightmap pixel).
    BuildingMesh,
    /// Section 2 — delatin / passthrough simplification.
    Decimating,
    /// Pipeline finished; UI should clear the bar.
    Done,
}

impl TerrainStage {
    /// Short human-readable label for the UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::FetchingDem => "Fetching DEM",
            Self::Upsampling => "Upsampling",
            Self::HydraulicErosion => "Hydraulic erosion",
            Self::ThermalErosion => "Thermal erosion",
            Self::FractalDetail => "Fractal detail",
            Self::NormalMap => "Normal map",
            Self::Cropping => "Cropping",
            Self::BuildingMesh => "Building mesh",
            Self::Decimating => "Decimating",
            Self::Done => "Done",
        }
    }
}

/// Progress sink for long-running terrain pipeline runs.
///
/// Implementations should be cheap and lock-light; `stage` is called
/// frequently (e.g. once per hydraulic iteration). The trait is
/// `Send + Sync` because the pipeline runs on a worker thread but the
/// UI typically owns the sink and reads it from the main thread.
pub trait TerrainProgressSink: Send + Sync {
    /// Called whenever the pipeline advances. `done` and `total` are
    /// in stage-local units (e.g. iteration counts for hydraulic
    /// erosion, `1/1` for one-shot CPU stages).
    fn stage(&self, stage: TerrainStage, done: u32, total: u32);
}

/// No-op sink for callers that don't care about progress.
pub struct NullProgressSink;

impl TerrainProgressSink for NullProgressSink {
    fn stage(&self, _stage: TerrainStage, _done: u32, _total: u32) {}
}
