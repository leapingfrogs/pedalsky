//! CPU mesh builder: heightmap -> regular grid mesh (positions + normals
//! + indices).
//!
//! No wgpu dependencies — the GPU upload lives in `ps-ground`. This file
//! is deliberately pure compute so tests, exporters, and the future QEM
//! simplifier can all operate on the same in-memory form.

use crate::tile::{HeightmapTile, TileRequest};

/// Vertex packed for GPU upload. Matches `crates/ps-ground/src/lib.rs`
/// `Vertex` layout (`position: vec3<f32>` + `normal: vec3<f32>`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// World-space position (metres). Y is up.
    pub position: [f32; 3],
    /// World-space outward normal (unit length).
    pub normal: [f32; 3],
}

/// CPU-side mesh ready for GPU upload.
pub struct MeshData {
    /// Vertices in row-major heightmap order before simplification. A
    /// `MeshSimplify` impl may reorder these; the GPU upload does not
    /// care.
    pub positions: Vec<MeshVertex>,
    /// Triangle list indices into `positions`.
    pub indices: Vec<u32>,
    /// Reference height (metres) under the observer. The mesh is
    /// translated so this height maps to local Y=0 in world space —
    /// preserves AGL semantics for other subsystems that assume
    /// ground = Y=0.
    pub reference_height_m: f32,
}

impl MeshData {
    /// Empty mesh (no vertices, no indices). Used as the v1 startup
    /// state in `ps-ground` before the first terrain fetch completes.
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            reference_height_m: 0.0,
        }
    }

    /// Test-only deep clone. MeshData isn't `Clone` to discourage
    /// accidental copies in the hot path; tests that need to fork a
    /// dense mesh into multiple simplify calls go through this.
    #[doc(hidden)]
    pub fn clone_for_test(&self) -> Self {
        Self {
            positions: self.positions.clone(),
            indices: self.indices.clone(),
            reference_height_m: self.reference_height_m,
        }
    }
}

/// Build a one-vertex-per-pixel grid mesh from a heightmap.
///
/// Hard invariants asserted in tests:
/// - `positions.len() == tile.width * tile.height`
/// - `indices.len() == (tile.width - 1) * (tile.height - 1) * 6`
///
/// The mesh is centred so XZ=(0,0) lies under the requested observer
/// `(req.lat, req.lon)` and Y=0 is the height at that point — preserves
/// the existing flat-plane convention used by other subsystems.
pub fn build_grid_mesh(tile: &HeightmapTile, req: &TileRequest) -> MeshData {
    let w = tile.width as usize;
    let h = tile.height as usize;
    let gsd = tile.gsd_m_centre;

    // Pixel coordinates of the observer inside the (possibly cropped)
    // tile so we can re-centre.
    let lat_span = tile.extent_deg.north - tile.extent_deg.south;
    let lon_span = tile.extent_deg.east - tile.extent_deg.west;
    let row_f = if lat_span.abs() > 1e-9 {
        ((tile.extent_deg.north - req.lat) / lat_span) * (h as f64 - 1.0)
    } else {
        0.0
    };
    let col_f = if lon_span.abs() > 1e-9 {
        ((req.lon - tile.extent_deg.west) / lon_span) * (w as f64 - 1.0)
    } else {
        0.0
    };
    let obs_row = row_f.clamp(0.0, h as f64 - 1.0);
    let obs_col = col_f.clamp(0.0, w as f64 - 1.0);

    // Reference height = bilinearly-sampled height at the observer pixel.
    let reference_height_m = sample_bilinear(&tile.heights_m, w, h, obs_col, obs_row);

    // Compute positions first; normals come from finite differences on
    // the already-built positions.
    let mut positions: Vec<MeshVertex> = Vec::with_capacity(w * h);
    // Convention (matches existing ps-ground): X = +east, Z = +south.
    // Pixel (0,0) is northwest, so dx = +east increases with col, dz =
    // +south increases with row.
    for r in 0..h {
        let dz = (r as f64 - obs_row) as f32 * gsd;
        for c in 0..w {
            let dx = (c as f64 - obs_col) as f32 * gsd;
            let y = tile.heights_m[r * w + c] - reference_height_m;
            positions.push(MeshVertex {
                position: [dx, y, dz],
                normal: [0.0, 1.0, 0.0],
            });
        }
    }

    // Normals via central differences on Y. Edge cells use forward /
    // backward differences. `dy/dx` is in dimensionless (rise/run) form;
    // the cross product of the two tangent vectors gives the normal.
    for r in 0..h {
        for c in 0..w {
            let y_left = positions[r * w + c.saturating_sub(1)].position[1];
            let y_right = positions[r * w + (c + 1).min(w - 1)].position[1];
            let y_up = positions[r.saturating_sub(1) * w + c].position[1];
            let y_down = positions[(r + 1).min(h - 1) * w + c].position[1];

            // Step in metres for the finite difference: 2*gsd for centre
            // cells, gsd for edges.
            let dx_step = if c == 0 || c == w - 1 { gsd } else { 2.0 * gsd };
            let dz_step = if r == 0 || r == h - 1 { gsd } else { 2.0 * gsd };

            let nx = -(y_right - y_left) / dx_step;
            let nz = -(y_down - y_up) / dz_step;
            // Tangent_x = (dx_step, dy/dx*dx_step, 0); Tangent_z = (0, dy/dz*dz_step, dz_step).
            // Their cross product is (-dy/dx, 1, -dy/dz) up to scale.
            let len = (nx * nx + 1.0 + nz * nz).sqrt();
            positions[r * w + c].normal = [nx / len, 1.0 / len, nz / len];
        }
    }

    // Indices: two triangles per quad, CCW when viewed from above (+Y).
    //
    //   nw --- ne
    //   |  \   |
    //   |   \  |
    //   sw --- se
    //
    // Tri 1: nw, sw, se. Tri 2: nw, se, ne.
    let mut indices: Vec<u32> = Vec::with_capacity((w - 1) * (h - 1) * 6);
    for r in 0..(h - 1) {
        for c in 0..(w - 1) {
            let nw = (r * w + c) as u32;
            let ne = (r * w + c + 1) as u32;
            let sw = ((r + 1) * w + c) as u32;
            let se = ((r + 1) * w + c + 1) as u32;
            indices.extend_from_slice(&[nw, sw, se, nw, se, ne]);
        }
    }

    MeshData {
        positions,
        indices,
        reference_height_m,
    }
}

fn sample_bilinear(heights: &[f32], w: usize, h: usize, col: f64, row: f64) -> f32 {
    let c0 = col.floor().clamp(0.0, w as f64 - 1.0) as usize;
    let r0 = row.floor().clamp(0.0, h as f64 - 1.0) as usize;
    let c1 = (c0 + 1).min(w - 1);
    let r1 = (r0 + 1).min(h - 1);
    let tx = (col - c0 as f64) as f32;
    let ty = (row - r0 as f64) as f32;
    let a = heights[r0 * w + c0];
    let b = heights[r0 * w + c1];
    let c = heights[r1 * w + c0];
    let d = heights[r1 * w + c1];
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{GeoExtent, HeightmapTile, TileRequest};
    use std::path::PathBuf;

    fn tiny_tile() -> HeightmapTile {
        // 4x3 heightmap (12 pixels) covering a 1°x1° tile.
        let heights = vec![
            0.0, 1.0, 2.0, 3.0,
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 5.0,
        ];
        HeightmapTile {
            heights_m: heights,
            width: 4,
            height: 3,
            extent_deg: GeoExtent { west: -4.0, east: -3.0, south: 56.0, north: 57.0 },
            source: "test",
            gsd_m_centre: 30.0,
        }
    }

    fn req_at(lat: f64, lon: f64) -> TileRequest {
        TileRequest {
            lat,
            lon,
            radius_m: 1_000_000.0,
            cache_dir: PathBuf::from("."),
            simplify_target: None,
        }
    }

    #[test]
    fn one_vertex_per_pixel() {
        let tile = tiny_tile();
        let mesh = build_grid_mesh(&tile, &req_at(56.5, -3.5));
        assert_eq!(mesh.positions.len(), (tile.width * tile.height) as usize);
    }

    #[test]
    fn triangle_count_matches_grid() {
        let tile = tiny_tile();
        let mesh = build_grid_mesh(&tile, &req_at(56.5, -3.5));
        let expected = ((tile.width - 1) * (tile.height - 1) * 6) as usize;
        assert_eq!(mesh.indices.len(), expected);
    }

    #[test]
    fn normals_are_unit_length() {
        let tile = tiny_tile();
        let mesh = build_grid_mesh(&tile, &req_at(56.5, -3.5));
        for v in &mesh.positions {
            let [x, y, z] = v.normal;
            let len = (x * x + y * y + z * z).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal not unit: {len}");
        }
    }
}
