//! Phase 12.5 — aurora curtain raymarch.
//!
//! Fullscreen Translucent pass that raymarches each above-horizon
//! view ray through an upper-atmosphere slab (≈80–300 km), sampling a
//! procedural 3D noise field laid out as vertical curtains. Emission
//! accumulates into the HDR target with additive blending.
//!
//! There is no precomputed 3D texture: the curtain field is evaluated
//! analytically in the shader from a small uniform buffer (intensity,
//! colour bias, time, latitude gate, march steps). This keeps the
//! crate self-contained — no synthesis upload, no extra compute pass.
//!
//! Geographic gating is applied CPU-side on the latitude from
//! `WorldState.latitude_deg` and combined with the scene's `kp_index`
//! and the optional `intensity_override` to produce the final
//! intensity scalar uploaded each frame.

#![deny(missing_docs)]

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use ps_core::{
    frame_bind_group_layout, world_bind_group_layout, Config, GpuContext, HdrFramebuffer,
    PassStage, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};
use tracing::debug;

const SHADER_REL: &str = "aurora/curtain.wgsl";
const SHADER_BAKED: &str = include_str!("../../../shaders/aurora/curtain.wgsl");

/// Stable subsystem name (matches `[render.subsystems].aurora`).
pub const NAME: &str = "aurora";

/// Per-frame uniform block uploaded to the aurora shader (group 2).
/// 32 bytes; padded to a vec4 boundary because std140.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct AuroraParamsGpu {
    /// `rgb` = colour-biased emission scalar in cd/m²·sr proxy units;
    /// `w` = intensity gate scalar (0..1) — the shader can early-out
    /// to zero when this is small.
    emission: [f32; 4],
    /// `x` = motion time (seconds); `y` = march steps as f32; `z` =
    /// curtain horizontal extent in metres; `w` = unused.
    config: [f32; 4],
}

/// Snapshot of `[render.aurora]` taken at construction / reconfigure.
#[derive(Clone, Copy, Debug)]
struct TuningSnapshot {
    march_steps: u32,
    peak_emission: f32,
    motion_hz: f32,
    min_lat: f32,
    peak_lat: f32,
    fade_lat: f32,
}

impl TuningSnapshot {
    fn from_config(config: &Config) -> Self {
        let a = &config.render.aurora;
        Self {
            march_steps: a.march_steps,
            peak_emission: a.peak_emission,
            motion_hz: a.motion_hz,
            min_lat: a.min_latitude_abs_deg,
            peak_lat: a.peak_latitude_abs_deg,
            fade_lat: a.fade_latitude_abs_deg,
        }
    }
}

/// Phase 12.5 aurora subsystem.
pub struct AuroraSubsystem {
    enabled: bool,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tuning: Arc<Mutex<TuningSnapshot>>,
    /// Cached most-recent intensity (debug log throttling).
    last_logged_intensity: Mutex<f32>,
}

impl AuroraSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
            &live_src,
        ]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora/curtain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-params-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<AuroraParamsGpu>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-pl"),
            bind_group_layouts: &[Some(&frame_layout), Some(&world_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("aurora-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HdrFramebuffer::COLOR_FORMAT,
                    // Additive HDR blend — emission accumulates onto
                    // whatever the sky/cloud passes already wrote.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            // Auroras live above the cloud layer and don't need to
            // depth-test against the scene; they read sky pixels and
            // additively brighten them.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora-params"),
            size: std::mem::size_of::<AuroraParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora-bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        let tuning = Arc::new(Mutex::new(TuningSnapshot::from_config(config)));
        debug!(target: "ps_aurora", "subsystem ready");
        Self {
            enabled: true,
            pipeline,
            layout,
            params_buf,
            bind_group,
            tuning,
            last_logged_intensity: Mutex::new(-1.0),
        }
    }
}

impl RenderSubsystem for AuroraSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let tuning = *self.tuning.lock().expect("aurora tuning lock");
        let lat_abs = ctx.world.latitude_deg.abs() as f32;
        let lat_gate = latitude_gate(
            lat_abs,
            tuning.min_lat,
            tuning.peak_lat,
            tuning.fade_lat,
        );

        let kp = ctx.weather.scene_aurora_kp.max(0.0);
        let intensity = if ctx.weather.scene_aurora_intensity_override >= 0.0 {
            ctx.weather.scene_aurora_intensity_override.clamp(0.0, 1.0)
        } else {
            // Map kp 0..9 to a 0..1 curve. Real activity is rare so a
            // mild knee (kp=4 → ~0.5) feels right; full saturation
            // by kp=8.
            (kp / 8.0).clamp(0.0, 1.0)
        };
        let combined = lat_gate * intensity;
        let bias = ctx.weather.scene_aurora_colour_bias;
        let emission = Vec3::new(bias[0], bias[1], bias[2]) * tuning.peak_emission * combined;

        let t_motion = ctx.frame_uniforms.simulated_seconds * tuning.motion_hz;
        // Curtain horizontal extent (m). 30 km matches the
        // top-down weather extent's order of magnitude — auroras seen
        // from the ground span tens-of-degrees so a single 30 km
        // curtain at 100 km altitude subtends ~17° — enough that the
        // effect dominates the visible field at high latitude.
        let curtain_extent = 30_000.0;

        let params = AuroraParamsGpu {
            emission: [emission.x, emission.y, emission.z, combined],
            config: [
                t_motion,
                tuning.march_steps as f32,
                curtain_extent,
                0.0,
            ],
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // One-time / change-throttled debug log so the gate behaviour
        // is visible in logs without spamming every frame.
        let mut last = self
            .last_logged_intensity
            .lock()
            .expect("aurora log lock");
        if (combined - *last).abs() > 0.05 {
            debug!(
                target: "ps_aurora",
                lat_abs_deg = lat_abs,
                lat_gate,
                kp,
                intensity,
                combined,
                "aurora intensity refreshed",
            );
            *last = combined;
        }
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let pipeline = self.pipeline.clone();
        let bind_group = self.bind_group.clone();
        vec![RegisteredPass {
            name: "aurora-curtain",
            stage: PassStage::Translucent,
            run: Box::new(move |encoder, ctx| {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("aurora-curtain"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ctx.framebuffer.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(
                        wgpu::RenderPassDepthStencilAttachment {
                            view: &ctx.framebuffer.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        },
                    ),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                pass.set_bind_group(1, ctx.world_bind_group, &[]);
                pass.set_bind_group(2, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }),
        }]
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        *self.tuning.lock().expect("aurora tuning lock") =
            TuningSnapshot::from_config(config);
        // Suppress the dead_code warning on `layout`; we hold it so
        // future hot-reload paths can rebuild bind groups without
        // re-creating the layout.
        let _ = &self.layout;
        Ok(())
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Latitude → intensity curve. Below `min` returns 0; ramps to 1 at
/// `peak`; decays linearly toward 0.5 by `fade`. Inputs and bounds
/// are absolute degrees.
pub fn latitude_gate(lat_abs_deg: f32, min: f32, peak: f32, fade: f32) -> f32 {
    if lat_abs_deg < min {
        return 0.0;
    }
    if lat_abs_deg <= peak {
        let t = (lat_abs_deg - min) / (peak - min).max(1e-3);
        return t.clamp(0.0, 1.0);
    }
    if lat_abs_deg <= fade {
        let t = (lat_abs_deg - peak) / (fade - peak).max(1e-3);
        // Decay from 1.0 at peak to 0.5 at fade (polar cap is dimmer
        // but still has some activity).
        return (1.0 - 0.5 * t).clamp(0.0, 1.0);
    }
    // Above fade: continue decaying linearly to 0 by lat 90°, but
    // never below zero.
    let t = (lat_abs_deg - fade) / (90.0 - fade).max(1e-3);
    (0.5 - 0.5 * t).clamp(0.0, 1.0)
}

/// Factory wired by `AppBuilder`.
pub struct AuroraFactory;

impl SubsystemFactory for AuroraFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.aurora
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(AuroraSubsystem::new(config, gpu)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latitude_gate_zero_below_min() {
        assert_eq!(latitude_gate(40.0, 50.0, 65.0, 80.0), 0.0);
        assert_eq!(latitude_gate(0.0, 50.0, 65.0, 80.0), 0.0);
    }

    #[test]
    fn latitude_gate_peaks_at_peak() {
        let v = latitude_gate(65.0, 50.0, 65.0, 80.0);
        assert!((v - 1.0).abs() < 1e-3, "expected 1.0 at peak, got {v}");
    }

    #[test]
    fn latitude_gate_decays_above_fade() {
        let at_fade = latitude_gate(80.0, 50.0, 65.0, 80.0);
        let at_85 = latitude_gate(85.0, 50.0, 65.0, 80.0);
        let at_90 = latitude_gate(90.0, 50.0, 65.0, 80.0);
        assert!((at_fade - 0.5).abs() < 1e-3);
        assert!(at_85 < at_fade);
        assert!(at_90 <= at_85);
        assert!(at_90 >= 0.0);
    }

    #[test]
    fn latitude_gate_ramp_between_min_and_peak() {
        let mid = latitude_gate(57.5, 50.0, 65.0, 80.0);
        assert!((mid - 0.5).abs() < 1e-2, "expected ~0.5 at midpoint, got {mid}");
    }
}
