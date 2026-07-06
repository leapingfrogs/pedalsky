//! Identity simplify stage.

use super::{MeshSimplify, SimplifyTarget};
use crate::{mesh::MeshData, TerrainError};

/// No-op simplifier. Returns the input mesh unchanged regardless of
/// target. Useful as a fallback (e.g. when the user disables
/// decimation in the UI).
pub struct PassthroughSimplify;

impl MeshSimplify for PassthroughSimplify {
    fn simplify(&self, mesh: MeshData, _target: SimplifyTarget) -> Result<MeshData, TerrainError> {
        Ok(mesh)
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }
}
