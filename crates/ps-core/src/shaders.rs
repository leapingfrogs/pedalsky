//! Shared WGSL fragments and a tiny composer.
//!
//! WGSL has no preprocessor. To share the canonical `FrameUniforms` /
//! `WorldUniforms` declarations across subsystem shaders without
//! duplicating them, ps-core embeds the canonical source files at
//! compile time and exposes them as `&'static str`. Subsystems string-
//! prepend the relevant fragment(s) to their own shader module before
//! handing the result to `wgpu`.

/// Canonical WGSL declarations for `FrameUniforms` (group 0) and
/// `WorldUniforms` (group 1). Mirrors the Rust types in
/// `ps_core::frame_uniforms` and `ps_core::weather`.
pub const COMMON_UNIFORMS_WGSL: &str = include_str!("../../../shaders/common/uniforms.wgsl");

/// Shared math helpers (`remap`, `max3`, `ViewRay`, `compute_view_ray`,
/// `ray_sphere_intersect`). Requires `frame: FrameUniforms` to be in
/// scope (i.e. include [`COMMON_UNIFORMS_WGSL`] and bind group 0 first).
pub const COMMON_MATH_WGSL: &str = include_str!("../../../shaders/common/math.wgsl");

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
}
