//! Section 1 of the PedalBack terrain spec — interactive heightmap
//! enhancement: bicubic upsample, hydraulic erosion (Mei 2007),
//! thermal erosion (Olsen 2004), fractal detail (Musgrave), and
//! normal map generation. All but the upsample run on the GPU as
//! wgpu compute passes.

mod fractal;
mod hydraulic;
mod normal_map;
mod params;
mod runtime;
mod thermal;

pub use params::ErosionParams;

use std::sync::Arc;

use super::{upsample, HeightmapAugment};
use crate::progress::{TerrainProgressSink, TerrainStage};
use crate::tile::HeightmapTile;
use crate::TerrainError;

/// GPU-driven heightmap enhancement implementing Section 1 of
/// `docs/pedalback_terrain_pipeline_spec.md`.
///
/// Holds an `Arc<wgpu::Device>` + `Arc<wgpu::Queue>` so it can run on
/// the imagery-style worker thread without taking ownership of the
/// renderer's device. The compute pipelines are built lazily and
/// reused across runs (cheap params changes don't rebuild them).
pub struct ErosionAugment {
    runtime: Arc<runtime::ErosionRuntime>,
    params: ErosionParams,
}

impl ErosionAugment {
    /// Construct. The first call to [`augment`] builds the compute
    /// pipelines and caches them on the runtime.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        params: ErosionParams,
    ) -> Self {
        Self {
            runtime: Arc::new(runtime::ErosionRuntime::new(device, queue)),
            params,
        }
    }

    /// Replace the parameter set without rebuilding GPU resources.
    /// Cheap; intended for UI slider re-runs.
    pub fn set_params(&mut self, params: ErosionParams) {
        self.params = params;
    }

    /// Current parameter set.
    pub fn params(&self) -> &ErosionParams {
        &self.params
    }
}

impl HeightmapAugment for ErosionAugment {
    fn name(&self) -> &'static str {
        "erosion"
    }

    fn augment_with_progress(
        &self,
        tile: HeightmapTile,
        progress: &dyn TerrainProgressSink,
    ) -> Result<HeightmapTile, TerrainError> {
        let params = &self.params;

        // Stage 1.1 — bicubic upsample.
        progress.stage(TerrainStage::Upsampling, 0, 1);
        let upsampled = if params.target_resolution_m > 0.0
            && params.target_resolution_m < tile.gsd_m_centre
        {
            let (tw, th) = upsample::working_resolution(&tile, params.target_resolution_m);
            // Cap the working grid so we don't accidentally allocate
            // a 32k×32k texture on a 30m source if someone slides
            // target_resolution_m to 0.1 — wgpu texture limits would
            // reject it anyway. Cap at 2048 to keep within the
            // baseline texture-size guarantee.
            let tw = tw.min(params.max_working_dim);
            let th = th.min(params.max_working_dim);
            upsample::bicubic_upsample(tile, tw, th)
        } else {
            tile
        };
        progress.stage(TerrainStage::Upsampling, 1, 1);

        // Stages 1.2/1.3/1.4/1.5 — run on the GPU. The runtime
        // owns texture allocation + pipeline caches; we hand over the
        // params and let it execute the interleaved hydraulic/thermal
        // loop + fractal injection.
        let enhanced = runtime::run(&self.runtime, upsampled, params, progress)?;

        Ok(enhanced)
    }
}
