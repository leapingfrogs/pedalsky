//! Mesh-level simplification seam.
//!
//! Decimates a dense regular-grid mesh into a sparser mesh that
//! preserves silhouette detail. Concentrates triangles on ridges,
//! cliffs and channels (high curvature), collapses them on plains
//! (low curvature). The default v1 impl is identity.
//!
//! Pays off most once the [`crate::HeightmapAugment`] stage is doing
//! super-resolution or erosion: a 4× super-resolution turns an 8 M-tri
//! mesh into 128 M-tri, at which point QEM decimation becomes
//! non-optional.

mod passthrough;
pub use passthrough::PassthroughSimplify;

use crate::{mesh::MeshData, TerrainError};

/// What the simplify stage should aim for.
#[derive(Debug, Copy, Clone)]
pub enum SimplifyTarget {
    /// Keep at most this many triangles.
    MaxTriangles(u32),
    /// Aim for this fraction of the input triangle count (1.0 = identity).
    Ratio(f32),
    /// Decimate up to this maximum quadric error (in metres²).
    MaxError(f32),
}

/// Mesh transform. Consumes one mesh + a target, returns one mesh.
pub trait MeshSimplify {
    /// Run the simplification.
    fn simplify(
        &self,
        mesh: MeshData,
        target: SimplifyTarget,
    ) -> Result<MeshData, TerrainError>;

    /// Short label used in tracing output.
    fn name(&self) -> &'static str;
}
