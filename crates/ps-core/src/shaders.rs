//! Shared WGSL fragments, a composer, and a hot-reload-aware loader.
//!
//! WGSL has no preprocessor. To share the canonical `FrameUniforms` /
//! `WorldUniforms` declarations across subsystem shaders without
//! duplicating them, ps-core embeds the canonical source files at
//! compile time and exposes them as `&'static str`. Subsystems string-
//! prepend the relevant fragment(s) to their own shader module before
//! handing the result to `wgpu`.
//!
//! # Hot-reload
//!
//! [`load_shader`] returns the shader source for a relative path under
//! `shaders/`. When the workspace root has been registered via
//! [`set_workspace_root`] (the live `ps-app` binary does this at GPU
//! init time when `[debug] shader_hot_reload` is true) and the
//! corresponding file exists on disk, the *file* contents are returned;
//! otherwise the baked fallback is returned. This means a developer
//! can edit a WGSL file and trigger a pipeline rebuild without
//! recompiling, while production / test / headless paths continue to
//! use the inlined source.
//!
//! Callers pair `load_shader("clouds/cloud_march.wgsl",
//! include_str!("../../../shaders/clouds/cloud_march.wgsl"))` so the
//! baked fallback is correct even if the workspace root is wrong or
//! the file has been deleted.

/// Canonical WGSL declarations for `FrameUniforms` (group 0) and
/// `WorldUniforms` (group 1). Mirrors the Rust types in
/// `ps_core::frame_uniforms` and `ps_core::weather`.
pub const COMMON_UNIFORMS_WGSL: &str = include_str!("../../../shaders/common/uniforms.wgsl");

/// Shared math helpers (`remap`, `max3`, `ViewRay`, `compute_view_ray`,
/// `ray_sphere_intersect`). Requires `frame: FrameUniforms` to be in
/// scope (i.e. include [`COMMON_UNIFORMS_WGSL`] and bind group 0 first).
pub const COMMON_MATH_WGSL: &str = include_str!("../../../shaders/common/math.wgsl");

/// Atmosphere helpers (density profiles, phase functions, ray-planet
/// intersection, transmittance LUT mapping, optical-depth integrator).
/// Requires `world: WorldUniforms` to be in scope.
pub const COMMON_ATMOSPHERE_WGSL: &str = include_str!("../../../shaders/common/atmosphere.wgsl");

/// LUT sampling helpers (`sample_transmittance_lut`). Requires
/// `transmittance_lut` + `lut_sampler` bindings to be in scope.
pub const COMMON_ATMOSPHERE_LUT_SAMPLING_WGSL: &str =
    include_str!("../../../shaders/common/atmosphere_lut_sampling.wgsl");

/// Concatenate `parts` with newline separators. Convenience for the
/// "string-prepend then compile" pattern.
pub fn compose(parts: &[&str]) -> String {
    let mut out = String::with_capacity(parts.iter().map(|s| s.len() + 1).sum());
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(p);
    }
    out
}

// ---------------------------------------------------------------------------
// Shader hot-reload (plan §Cross-Cutting/Shader hot-reload)
// ---------------------------------------------------------------------------

use std::path::PathBuf;
use std::sync::OnceLock;

static WORKSPACE_SHADERS_ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Register the workspace `shaders/` directory so [`load_shader`] reads
/// runtime overrides from there. Pass `None` to keep the loader
/// purely-baked. Idempotent: only the first call wins.
pub fn set_workspace_root(workspace: Option<PathBuf>) {
    let _ = WORKSPACE_SHADERS_ROOT.set(workspace.map(|w| w.join("shaders")));
}


/// Resolve a shader source. `rel_path` is the path under `shaders/`
/// (e.g. `"clouds/cloud_march.wgsl"`); `baked_fallback` is the
/// `include_str!`-inlined source the subsystem ships with.
///
/// When [`set_workspace_root`] has been called with `Some(_)` and the
/// file exists, the on-disk contents are returned. On any read error
/// the baked fallback is used and the error is logged at `warn` level.
pub fn load_shader(rel_path: &str, baked_fallback: &'static str) -> String {
    if let Some(Some(root)) = WORKSPACE_SHADERS_ROOT.get() {
        let path = root.join(rel_path);
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                tracing::debug!(target: "ps_core::shaders",
                    path = %path.display(), "loaded shader from disk");
                return s;
            }
            Err(e) => {
                tracing::warn!(target: "ps_core::shaders",
                    path = %path.display(), error = %e,
                    "shader read failed; falling back to baked source");
            }
        }
    }
    baked_fallback.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_uniforms_declares_frame_uniforms() {
        assert!(COMMON_UNIFORMS_WGSL.contains("struct FrameUniforms"));
        assert!(COMMON_UNIFORMS_WGSL.contains("struct WorldUniforms"));
    }

    #[test]
    fn compose_joins_with_newlines() {
        let s = compose(&["a", "b", "c"]);
        assert_eq!(s, "a\nb\nc");
    }

    #[test]
    fn load_shader_falls_back_to_baked_when_workspace_unset() {
        // No call to set_workspace_root — loader returns the baked
        // fallback verbatim.
        let baked = "// baked sentinel\n";
        let s = load_shader("does/not/exist.wgsl", baked);
        assert_eq!(s, baked);
    }
}
