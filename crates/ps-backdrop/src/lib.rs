//! Phase 1 demo subsystem: clears the HDR target to a config-driven
//! solid colour. Registers one `PassStage::SkyBackdrop` pass.
//!
//! Exists purely to exercise the render-graph executor and the hot-reload
//! path. Phase 5's atmosphere subsystem will replace this with the real
//! sky/atmosphere pass at the same stage.

#![deny(missing_docs)]

use std::sync::{Arc, Mutex};

use ps_core::{
    Config, GpuContext, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory,
};

/// Stable subsystem name (matches `[render.subsystems].backdrop`).
pub const NAME: &str = "backdrop";

/// Phase 1 demo subsystem: clears the HDR target to a configurable RGB
/// colour at `PassStage::SkyBackdrop`.
pub struct BackdropSubsystem {
    enabled: bool,
    /// Shared with the pass closure so `reconfigure()` can update the
    /// colour without rebuilding the closure (or the subsystem).
    colour: Arc<Mutex<[f32; 4]>>,
}

impl BackdropSubsystem {
    /// Construct from the current config.
    pub fn new(config: &Config) -> Self {
        let [r, g, b] = config.render.backdrop.color;
        Self {
            enabled: true,
            colour: Arc::new(Mutex::new([r, g, b, 1.0])),
        }
    }
}

impl RenderSubsystem for BackdropSubsystem {
    fn name(&self) -> &'static str {
        "backdrop"
    }

    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {}

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let colour = self.colour.clone();
        vec![RegisteredPass {
            name: "backdrop-clear",
            stage: PassStage::SkyBackdrop,
            run: Box::new(move |encoder, ctx| {
                let [r, g, b, a] = *colour.lock().expect("backdrop colour lock poisoned");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("backdrop-clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ctx.framebuffer.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: r as f64,
                                g: g as f64,
                                b: b as f64,
                                a: a as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let _ = &mut pass;
            }),
        }]
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        let [r, g, b] = config.render.backdrop.color;
        *self.colour.lock().expect("backdrop colour lock poisoned") = [r, g, b, 1.0];
        Ok(())
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`.
pub struct BackdropFactory;

impl SubsystemFactory for BackdropFactory {
    fn name(&self) -> &'static str {
        "backdrop"
    }
    fn build(
        &self,
        config: &Config,
        _gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(BackdropSubsystem::new(config)))
    }
}
