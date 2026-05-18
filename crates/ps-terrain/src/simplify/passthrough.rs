//! Identity simplify stage. v1 default.

use super::{MeshSimplify, SimplifyTarget};
use crate::{mesh::MeshData, TerrainError};

/// No-op simplifier. Returns the input mesh unchanged regardless of
/// target. Drop-in replacement for a future QEM-based impl.
pub struct PassthroughSimplify;

impl MeshSimplify for PassthroughSimplify {
    fn simplify(
        &self,
        mesh: MeshData,
        _target: SimplifyTarget,
    ) -> Result<MeshData, TerrainError> {
        Ok(mesh)
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
