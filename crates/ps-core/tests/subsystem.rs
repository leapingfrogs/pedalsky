//! Phase 1 Group A tests for the `RenderSubsystem` trait, `PassStage`
//! ordering, and the `PrepareContext` / `RenderContext` shapes.

use ps_core::PassStage;

#[test]
fn pass_stage_orders_correctly() {
    use PassStage::*;
    // From the plan §1.3: Compute < SkyBackdrop < Opaque < Translucent <
    // PostProcess < ToneMap < Overlay.
    let order = [
        Compute,
        SkyBackdrop,
        Opaque,
        Translucent,
        PostProcess,
        ToneMap,
        Overlay,
    ];
    for window in order.windows(2) {
        assert!(
            window[0] < window[1],
            "{:?} must be < {:?}",
            window[0],
            window[1]
        );
    }
    // sort_by_key relies on Ord; spot-check:
    let mut shuffled = vec![Overlay, Compute, Opaque, ToneMap, SkyBackdrop];
    shuffled.sort();
    assert_eq!(
        shuffled,
        vec![Compute, SkyBackdrop, Opaque, ToneMap, Overlay]
    );
}

#[test]
fn pass_stages_are_copy_clone_debug_eq_hash() {
    // Compile-time witness that the derive set on PassStage matches the spec.
    fn assert_traits<
        T: Copy + Clone + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + std::hash::Hash,
    >() {
    }
    assert_traits::<PassStage>();
}

/// Smoke test that `PrepareContext` and `RenderContext` have the field types
/// the plan specifies. Compiles on its own — never executed.
#[allow(dead_code, clippy::too_many_arguments)]
fn _ctx_smoke(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    framebuffer: &ps_core::HdrFramebuffer,
    bg: &wgpu::BindGroup,
    luts_bg: Option<&wgpu::BindGroup>,
    weather: &ps_core::WeatherState,
    world: &ps_core::WorldState,
    frame_uniforms: &ps_core::FrameUniforms,
    atmosphere_luts: Option<&ps_core::AtmosphereLuts>,
) {
    let _prep = ps_core::PrepareContext {
        device,
        queue,
        world,
        weather,
        frame_uniforms,
        atmosphere_luts,
        dt_seconds: 0.0,
    };
    let _rc = ps_core::RenderContext {
        device,
        queue,
        framebuffer,
        frame_bind_group: bg,
        world_bind_group: bg,
        luts_bind_group: luts_bg,
        frame_uniforms,
        weather,
    };
}
