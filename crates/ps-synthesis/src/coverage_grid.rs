//! Phase 12.1 — loaded coverage-grid data, shared between
//! [`crate::weather_map`] (which writes the R channel) and
//! [`crate::density_mask`] (which integrates the column at each
//! pixel using the gridded coverage rather than a flat per-layer
//! scalar).
//!
//! The disk format is row-major little-endian f32, one value per
//! source-grid pixel, in [0, 1]. The grid spans
//! `[-extent_m/2, +extent_m/2]` on each XZ axis, centred on the
//! world origin.

use ps_core::Scene;
use tracing::warn;

/// In-memory copy of a scene's coverage grid plus its spatial
/// metadata. Sample with [`Self::sample`] which does bilinear
/// interpolation.
#[derive(Debug, Clone)]
pub struct LoadedCoverageGrid {
    /// Row-major f32 values in [0, 1] — one per source-grid pixel.
    pub data: Vec<f32>,
    /// Source grid width in pixels.
    pub src_w: u32,
    /// Source grid height in pixels.
    pub src_h: u32,
    /// Spatial extent in metres. The grid covers
    /// `[-extent/2, +extent/2]` on each axis.
    pub extent_m: f32,
}

impl LoadedCoverageGrid {
    /// Bilinear-sample at a source-grid UV in [0, 1]^2. Out-of-range
    /// UVs clamp to the edge.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let fx = u.clamp(0.0, 1.0) * (self.src_w as f32 - 1.0);
        let fy = v.clamp(0.0, 1.0) * (self.src_h as f32 - 1.0);
        let x0 = fx.floor() as u32;
        let y0 = fy.floor() as u32;
        let x1 = (x0 + 1).min(self.src_w - 1);
        let y1 = (y0 + 1).min(self.src_h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let idx = |x: u32, y: u32| (y * self.src_w + x) as usize;
        let v00 = self.data[idx(x0, y0)];
        let v10 = self.data[idx(x1, y0)];
        let v01 = self.data[idx(x0, y1)];
        let v11 = self.data[idx(x1, y1)];
        let bottom = v00 + (v10 - v00) * tx;
        let top = v01 + (v11 - v01) * tx;
        bottom + (top - bottom) * ty
    }

    /// Sample at a world-space (x, z) position. Maps via the grid's
    /// own extent then clamps.
    pub fn sample_world(&self, x: f32, z: f32) -> f32 {
        let half = self.extent_m * 0.5;
        let u = (x + half) / self.extent_m;
        let v = (z + half) / self.extent_m;
        self.sample(u, v).clamp(0.0, 1.0)
    }
}

/// Try to load the coverage grid for a scene. Returns `Ok(None)` when
/// the scene has no `coverage_grid` block (the common case);
/// `Some(_)` on successful load. Errors are logged at `warn` level
/// and `None` is returned so callers fall back to per-layer scalars.
pub fn load(scene: &Scene) -> Option<LoadedCoverageGrid> {
    let grid = scene.clouds.coverage_grid.as_ref()?;
    let [src_w, src_h] = grid.size;
    let expected_bytes = (src_w as usize) * (src_h as usize) * 4;
    let bytes = match std::fs::read(&grid.data_path) {
        Ok(b) => b,
        Err(e) => {
            warn!(
                target: "ps_synthesis::coverage_grid",
                path = %grid.data_path.display(),
                error = %e,
                "coverage grid read failed; falling back to scalar coverage",
            );
            return None;
        }
    };
    if bytes.len() != expected_bytes {
        warn!(
            target: "ps_synthesis::coverage_grid",
            path = %grid.data_path.display(),
            got = bytes.len(),
            want = expected_bytes,
            "coverage grid size mismatch; falling back to scalar coverage",
        );
        return None;
    }
    let mut data = Vec::with_capacity((src_w as usize) * (src_h as usize));
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(LoadedCoverageGrid {
        data,
        src_w,
        src_h,
        extent_m: grid.extent_m,
    })
}
