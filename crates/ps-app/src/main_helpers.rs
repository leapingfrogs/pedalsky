//! Tiny helpers shared between `main.rs` (the windowed binary) and
//! `test_harness.rs` (the headless harness used by integration tests).

use ps_core::HdrFramebufferImpl;

/// One-pass colour + depth clear. Always clears depth (reverse-Z = 0.0);
/// only clears colour when no SkyBackdrop subsystem is going to do it.
pub fn encode_frame_clear(
    encoder: &mut wgpu::CommandEncoder,
    hdr: &HdrFramebufferImpl,
    clear_color_too: bool,
    clear_color: [f32; 4],
) {
    let color_load = if clear_color_too {
        wgpu::LoadOp::Clear(wgpu::Color {
            r: clear_color[0] as f64,
            g: clear_color[1] as f64,
            b: clear_color[2] as f64,
            a: clear_color[3] as f64,
        })
    } else {
        wgpu::LoadOp::Load
    };
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("frame-clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &hdr.color_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: color_load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &hdr.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}
