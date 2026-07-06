//! Mesh-level simplification seam.

mod delatin_impl;
mod params;
mod passthrough;

pub use delatin_impl::DelatinSimplify;
pub use params::DecimationParams;
pub use passthrough::PassthroughSimplify;

use crate::{mesh::MeshData, TerrainError};

/// What the simplify stage should aim for.
#[derive(Debug, Copy, Clone)]
pub enum SimplifyTarget {
    /// Keep at most this many triangles.
    MaxTriangles(u32),
    /// Aim for this fraction of the input triangle count (1.0 = identity).
    Ratio(f32),
    /// Decimate up to this maximum vertical error (metres). Used by
    /// [`DelatinSimplify`].
    MaxError(f32),
}

/// Mesh transform. Consumes one mesh + a target, returns one mesh.
pub trait MeshSimplify {
    /// Run the simplification.
    fn simplify(&self, mesh: MeshData, target: SimplifyTarget) -> Result<MeshData, TerrainError>;

    /// Short label used in tracing output.
    fn name(&self) -> &'static str;
}
