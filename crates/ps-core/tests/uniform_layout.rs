//! Phase 4 §4.1 std140 cross-check.
//!
//! Parses `shaders/common/uniforms.wgsl` with `naga` and verifies the
//! WGSL `FrameUniforms` and `WorldUniforms` struct sizes match the Rust
//! `repr(C)` `FrameUniformsGpu` and `WorldUniformsGpu` sizes. Catches
//! alignment drift between CPU and GPU.

use naga::front::wgsl::Frontend;
use ps_core::{FrameUniformsGpu, WorldUniformsGpu};

fn parse_module(src: &str) -> naga::Module {
    Frontend::new()
        .parse(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed: {e:#?}"))
}

fn struct_size(module: &naga::Module, name: &str) -> Option<u32> {
    for (_, ty) in module.types.iter() {
        if let Some(ref n) = ty.name {
            if n == name {
                if let naga::TypeInner::Struct { span, .. } = &ty.inner {
                    return Some(*span);
                }
            }
        }
    }
    None
}

#[test]
fn frame_uniforms_size_matches_wgsl() {
    let src = ps_core::shaders::COMMON_UNIFORMS_WGSL;
    let module = parse_module(src);
    let wgsl_size =
        struct_size(&module, "FrameUniforms").expect("WGSL FrameUniforms struct must be declared");
    let rust_size = std::mem::size_of::<FrameUniformsGpu>() as u32;
    assert_eq!(
        wgsl_size, rust_size,
        "WGSL FrameUniforms = {wgsl_size} B, Rust FrameUniformsGpu = {rust_size} B"
    );
}

#[test]
fn world_uniforms_size_matches_wgsl() {
    let src = ps_core::shaders::COMMON_UNIFORMS_WGSL;
    let module = parse_module(src);
    let wgsl_size =
        struct_size(&module, "WorldUniforms").expect("WGSL WorldUniforms struct must be declared");
    let rust_size = std::mem::size_of::<WorldUniformsGpu>() as u32;
    assert_eq!(
        wgsl_size, rust_size,
        "WGSL WorldUniforms = {wgsl_size} B, Rust WorldUniformsGpu = {rust_size} B"
    );
}
