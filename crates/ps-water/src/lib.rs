//! Phase 13.5 — water surface (ponds, lakes, sea).
//!
//! A flat rectangular plane at the configured altitude, rendered with
//! GGX/Smith specular, a Fresnel-weighted sky reflection sampled from
//! the atmosphere sky-view LUT at the reflected view direction, and a
//! wind-advected procedural normal map. Refraction is out of scope —
//! the v1 plan has no DEM, so there's nothing under the water to
//! refract toward; the body Lambertian term proxies the absorbed light
//! path with a fixed colour.
//!
//! The subsystem activates only when the loaded scene contains a
//! `[water]` block (`scene.water = Some(_)`). Otherwise it builds
//! cleanly but its pass closure no-ops.
//!
//! Render-graph slot: `PassStage::Opaque`. Registered *after* the
//! ground subsystem in `AppBuilder`, so within-stage registration
//! order puts the water pass after the ground. Both write to the
//! depth buffer at depth-compare `Greater` (reverse-Z), so the
//! depth-test correctly handles overlap between the two planes.

#![deny(missing_docs)]

use bytemuck::{bytes_of, Pod, Zeroable};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout, Config,
    GpuContext, HdrFramebuffer, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory,
};

const SHADER_BAKED: &str = include_str!("../../../shaders/water/water.wgsl");
const SHADER_REL: &str = "water/water.wgsl";

/// Stable subsystem name (matches `[render.subsystems].water`).
pub const NAME: &str = "water";

/// Per-side subdivisions of the water grid mesh. 16×16 = 512
/// triangles total; per-pixel normals from the procedural noise
/// carry the high-frequency detail, so the geometry just needs
/// enough subdivision to make finite-frustum-clip not chunk the
/// horizon.
const GRID_RES: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct WaterParamsGpu {
    /// `(xmin, xmax, zmin, zmax)`.
    bounds: [f32; 4],
    /// `(altitude_m, roughness_min, roughness_max, simulated_seconds)`.
    config: [f32; 4],
    /// `(wind_dir_deg, wind_speed_mps, _, _)`.
    wind: [f32; 4],
}

/// Phase 13.5 water subsystem.
pub struct WaterSubsystem {
    enabled: bool,
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    /// Active flag: `true` only when the live `WeatherState.scene_water`
    /// is `Some`. The pass closure reads this via the shared cell.
    active: std::sync::Arc<std::sync::Mutex<bool>>,
}

impl WaterSubsystem {
    /// Construct.
    pub fn new(_config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &live_src,
        ]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water/water.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-params-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<WaterParamsGpu>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let lut_layout = atmosphere_lut_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&params_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HdrFramebuffer::COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Reverse-Z + Greater means a strictly larger depth
                // wins. Water at the same altitude as the ground would
                // tie and be rejected; bias the water slightly toward
                // the camera so coplanar water+ground renders water.
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Build a unit grid: positions are (s, t) in [0,1] and the
        // vertex shader maps them into the configured bounds.
        let (vertices, indices) = build_grid_mesh(GRID_RES);
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-vb"),
            size: std::mem::size_of_val(vertices.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        vertex_buf.unmap();

        let index_count = indices.len() as u32;
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-ib"),
            size: std::mem::size_of_val(indices.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        index_buf.unmap();

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-params-ub"),
            size: std::mem::size_of::<WaterParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-params-bg"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        Self {
            enabled: true,
            pipeline,
            vertex_buf,
            index_buf,
            index_count,
            params_buf,
            params_bg,
            active: std::sync::Arc::new(std::sync::Mutex::new(false)),
        }
    }
}

impl RenderSubsystem for WaterSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let Some(water) = ctx.weather.scene_water.as_ref() else {
            *self.active.lock().expect("water active lock") = false;
            return;
        };
        let surface = &ctx.weather.surface;
        let params = WaterParamsGpu {
            bounds: [water.xmin, water.xmax, water.zmin, water.zmax],
            config: [
                water.altitude_m,
                water.roughness_min,
                water.roughness_max,
                ctx.frame_uniforms.simulated_seconds,
            ],
            wind: [surface.wind_dir_deg, surface.wind_speed_mps, 0.0, 0.0],
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytes_of(&params));
        *self.active.lock().expect("water active lock") = true;
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let pipeline = self.pipeline.clone();
        let vertex_buf = self.vertex_buf.clone();
        let index_buf = self.index_buf.clone();
        let index_count = self.index_count;
        let params_bg = self.params_bg.clone();
        let active = self.active.clone();
        vec![RegisteredPass {
            name: "water",
            stage: PassStage::Opaque,
            run: Box::new(move |encoder, ctx| {
                if !*active.lock().expect("water active lock") {
                    return;
                }
                let Some(luts_bg) = ctx.luts_bind_group else {
                    return;
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("water-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &ctx.framebuffer.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &ctx.framebuffer.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                pass.set_bind_group(1, ctx.world_bind_group, &[]);
                pass.set_bind_group(2, &params_bg, &[]);
                pass.set_bind_group(3, luts_bg, &[]);
                pass.set_vertex_buffer(0, vertex_buf.slice(..));
                pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..index_count, 0, 0..1);
            }),
        }]
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Build a unit-extent grid mesh in `[0,1]²` with `n × n` subdivisions.
/// Returns `(vertices, indices)` where each vertex is `[s, t]` and the
/// index list is triangle list, 6 indices per quad.
fn build_grid_mesh(n: u32) -> (Vec<[f32; 2]>, Vec<u32>) {
    let mut vertices: Vec<[f32; 2]> = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
    for j in 0..=n {
        for i in 0..=n {
            vertices.push([
                i as f32 / n as f32,
                j as f32 / n as f32,
            ]);
        }
    }
    let stride = n + 1;
    let mut indices: Vec<u32> = Vec::with_capacity((n * n * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let i0 = j * stride + i;
            let i1 = j * stride + i + 1;
            let i2 = (j + 1) * stride + i;
            let i3 = (j + 1) * stride + i + 1;
            indices.push(i0);
            indices.push(i1);
            indices.push(i3);
            indices.push(i0);
            indices.push(i3);
            indices.push(i2);
        }
    }
    (vertices, indices)
}

/// Factory wired by `AppBuilder`.
pub struct WaterFactory;

impl SubsystemFactory for WaterFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.water
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(WaterSubsystem::new(config, gpu)))
    }
}
