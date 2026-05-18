//! Section 2 parameters — see `docs/pedalback_terrain_pipeline_spec.md`.

/// All Section 2 (decimation) parameters. Fields named to match the
/// spec; the UI exposes the primary entries prominently and the rest
/// behind an Advanced collapse.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DecimationParams {
    /// Maximum allowed vertical error per LOD, in metres. v1 of the
    /// pipeline only emits LOD 0 (`lod_max_errors_m[0]`); higher-LOD
    /// values are stored for the future multi-LOD work referenced in
    /// Section 2 of the spec.
    pub lod_max_errors_m: [f32; 4],
    /// Distances at which LOD switches occur. Same future-LOD note as
    /// above.
    pub lod_distances_m: [f32; 3],
    /// Vertical depth of the skirt added at each tile boundary. v1
    /// skirts are out of scope; field carried so the parameter is
    /// already in the public API.
    pub skirt_depth_m: f32,
    /// Optional hard upper bound on triangle count per LOD. `None`
    /// = follow `lod_max_errors_m` alone.
    pub max_triangles_per_lod: Option<u32>,
}

impl Default for DecimationParams {
    fn default() -> Self {
        Self {
            lod_max_errors_m: [0.05, 0.25, 1.0, 5.0],
            lod_distances_m: [50.0, 200.0, 1000.0],
            skirt_depth_m: 10.0,
            max_triangles_per_lod: None,
        }
    }
}
