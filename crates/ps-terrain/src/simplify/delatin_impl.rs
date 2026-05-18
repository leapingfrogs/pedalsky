//! Section 2 — Garland-Heckbert greedy mesh decimation via the
//! `delatin` crate.
//!
//! delatin operates on a flat row-major heightmap with a single
//! `max_error` threshold; it returns a list of grid-index vertices +
//! triangle indices. We re-pack that into our `MeshData` format,
//! preserving the original mesh's world-space mapping by sampling
//! positions/normals from the input dense mesh.

use crate::mesh::{MeshData, MeshVertex};
use crate::simplify::{params::DecimationParams, MeshSimplify, SimplifyTarget};
use crate::TerrainError;

/// Greedy Garland-Heckbert decimation (delatin).
///
/// Single-LOD in v1 — uses `params.lod_max_errors_m[0]` unless the
/// caller overrides via `SimplifyTarget::MaxError`. Other LOD values
/// are carried for future use (see `docs/pedalback_terrain_pipeline_spec.md`
/// Section 2 — multi-LOD with distance-based selection).
pub struct DelatinSimplify {
    params: DecimationParams,
}

impl DelatinSimplify {
    /// Construct.
    pub fn new(params: DecimationParams) -> Self {
        Self { params }
    }

    /// Replace the parameter set. UI re-runs use this.
    pub fn set_params(&mut self, params: DecimationParams) {
        self.params = params;
    }

    /// Current parameter set.
    pub fn params(&self) -> &DecimationParams {
        &self.params
    }
}

impl MeshSimplify for DelatinSimplify {
    fn name(&self) -> &'static str {
        "delatin"
    }

    fn simplify(
        &self,
        mesh: MeshData,
        target: SimplifyTarget,
    ) -> Result<MeshData, TerrainError> {
        // Resolve effective max_error: explicit target overrides the
        // configured LOD 0; ratio targets fall back to the LOD-0
        // setting because delatin only understands error thresholds.
        let max_error_m = match target {
            SimplifyTarget::MaxError(e) => e.max(1e-4),
            SimplifyTarget::Ratio(r) if r >= 0.999 => {
                // Identity request — short-circuit.
                return Ok(mesh);
            }
            _ => self.params.lod_max_errors_m[0].max(1e-4),
        };

        // delatin needs a row-major f32 heightmap reconstructed from
        // our MeshData. The mesh builder produces a regular WxH grid
        // (one vertex per heightmap pixel) in row-major order with
        // Y = height. We can detect the grid dimensions by walking
        // the index buffer: the first quad's NW/NE indices are
        // 0 and 1, so the row width is the first index distance
        // greater than 1. Easier: re-derive from positions —
        // unique x values determine width.
        let (width, height) = grid_dims_from_positions(&mesh)?;
        let total_verts = (width as usize) * (height as usize);
        if mesh.positions.len() != total_verts {
            return Err(TerrainError::SimplifyInvalid(format!(
                "expected dense {}x{} grid ({} verts), got {}",
                width, height, total_verts, mesh.positions.len()
            )));
        }

        // delatin uses f64 heights internally.
        let mut heights = Vec::with_capacity(total_verts);
        for v in &mesh.positions {
            heights.push(v.position[1] as f64);
        }

        let (sel_points, sel_tris) = run_delatin(
            &heights,
            width as usize,
            height as usize,
            max_error_m,
            self.params.max_triangles_per_lod,
        )
        .map_err(|e| TerrainError::SimplifyInvalid(format!("delatin: {e}")))?;

        // Re-pack: each delatin point is a (col, row) into the dense
        // grid, so we look up the original MeshVertex (position +
        // normal) at that grid index and emit it in delatin's order.
        let mut positions: Vec<MeshVertex> = Vec::with_capacity(sel_points.len());
        for (col, row) in &sel_points {
            let idx = row * (width as usize) + col;
            positions.push(mesh.positions[idx]);
        }
        let mut indices: Vec<u32> = Vec::with_capacity(sel_tris.len() * 3);
        for (a, b, c) in &sel_tris {
            indices.push(*a as u32);
            indices.push(*b as u32);
            indices.push(*c as u32);
        }

        tracing::info!(
            target: "ps_terrain::simplify",
            max_error_m,
            in_verts = mesh.positions.len(),
            in_tris = mesh.indices.len() / 3,
            out_verts = positions.len(),
            out_tris = indices.len() / 3,
            "delatin decimation complete"
        );

        Ok(MeshData {
            positions,
            indices,
            reference_height_m: mesh.reference_height_m,
        })
    }
}

/// Recover the dense-grid dimensions from a MeshData built by
/// `build_grid_mesh`. The function relies on the regular-grid order
/// produced by that builder; if a future builder changes the layout
/// this needs to be reworked (probably better: pass dims through
/// MeshData).
fn grid_dims_from_positions(mesh: &MeshData) -> Result<(u32, u32), TerrainError> {
    if mesh.positions.is_empty() {
        return Err(TerrainError::SimplifyInvalid("empty mesh".into()));
    }
    // Row-major builder: x increases along the row, then z jumps.
    // Find the first index where the x value drops (start of row 1).
    let first_x = mesh.positions[0].position[0];
    let row_start_z = mesh.positions[0].position[2];
    let mut width = 0u32;
    for (i, v) in mesh.positions.iter().enumerate() {
        if i > 0 && v.position[2] != row_start_z {
            width = i as u32;
            break;
        }
        let _ = first_x;
    }
    if width == 0 {
        // Single-row mesh — fall back to total len as width.
        width = mesh.positions.len() as u32;
    }
    let height = mesh.positions.len() as u32 / width;
    if width * height != mesh.positions.len() as u32 {
        return Err(TerrainError::SimplifyInvalid(format!(
            "mesh isn't a regular grid (verts={}, derived w={}, h={})",
            mesh.positions.len(),
            width,
            height,
        )));
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::build_grid_mesh;
    use crate::tile::{GeoExtent, HeightmapTile, TileRequest};
    use std::path::PathBuf;

    fn pyramid_tile(side: u32) -> HeightmapTile {
        // Build a 4-sided pyramid heightmap so the delatin decimator
        // has features to preserve at the ridges.
        let mut heights = Vec::with_capacity((side * side) as usize);
        let mid = (side - 1) as f32 / 2.0;
        for y in 0..side {
            for x in 0..side {
                let dx = (x as f32 - mid).abs();
                let dy = (y as f32 - mid).abs();
                heights.push(mid - dx.max(dy));
            }
        }
        HeightmapTile {
            heights_m: heights,
            width: side,
            height: side,
            extent_deg: GeoExtent { west: 0.0, east: 1.0, south: 0.0, north: 1.0 },
            source: "test",
            gsd_m_centre: 1.0,
        }
    }

    fn req(lat: f64, lon: f64) -> TileRequest {
        TileRequest {
            lat,
            lon,
            radius_m: 1_000_000.0,
            cache_dir: PathBuf::from("."),
            simplify_target: None,
        }
    }

    #[test]
    fn delatin_reduces_triangle_count_below_dense_grid() {
        let tile = pyramid_tile(33);
        let mesh = build_grid_mesh(&tile, &req(0.5, 0.5));
        let dense_tris = mesh.indices.len() / 3;

        let simp = DelatinSimplify::new(DecimationParams {
            lod_max_errors_m: [0.5, 1.0, 2.0, 5.0],
            ..Default::default()
        });
        let out = simp
            .simplify(mesh, SimplifyTarget::MaxError(0.5))
            .expect("delatin");
        let out_tris = out.indices.len() / 3;
        assert!(
            out_tris < dense_tris,
            "delatin should produce fewer triangles than the dense grid: dense={dense_tris} out={out_tris}"
        );
        assert!(out_tris > 0);
    }

    #[test]
    fn tighter_max_error_yields_more_triangles() {
        let tile = pyramid_tile(33);
        let mesh = build_grid_mesh(&tile, &req(0.5, 0.5));
        let simp = DelatinSimplify::new(DecimationParams::default());
        let coarse = simp
            .simplify(mesh.clone_for_test(), SimplifyTarget::MaxError(2.0))
            .expect("delatin coarse");
        let fine = simp
            .simplify(mesh, SimplifyTarget::MaxError(0.05))
            .expect("delatin fine");
        assert!(
            fine.indices.len() >= coarse.indices.len(),
            "tighter error should retain at least as many triangles (fine={}, coarse={})",
            fine.indices.len() / 3,
            coarse.indices.len() / 3,
        );
    }
}

/// Wrap `delatin::triangulate`. delatin doesn't expose a hard
/// triangle cap, so we honour the spec's "memory safety net" by
/// re-running with progressively higher `max_error` until under the
/// cap.
fn run_delatin(
    heights: &[f64],
    width: usize,
    height: usize,
    max_error: f32,
    max_triangles: Option<u32>,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize, usize)>), String> {
    let mut err = max_error as f64;
    let mut last = delatin::triangulate(heights, (width, height), delatin::Error(err))
        .map_err(|e| format!("{e:?}"))?;
    if let Some(cap) = max_triangles {
        while last.1.len() > cap as usize && err < 1_000.0 {
            err *= 1.5;
            last = delatin::triangulate(heights, (width, height), delatin::Error(err))
                .map_err(|e| format!("{e:?}"))?;
        }
    }
    Ok(last)
}
