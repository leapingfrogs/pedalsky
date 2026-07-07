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
//! Geometry comes from one of two sources:
//!
//! - **Internal grid** (default): when the loaded scene contains a
//!   `[water]` block (`scene.water = Some(_)`), a `GRID_RES`-subdivided
//!   rectangle is built on the CPU in world coordinates from the
//!   scene's bounds + altitude and uploaded on change.
//! - **External mesh** (host injection): a host embedding this
//!   subsystem can supply arbitrary world-space geometry (lake / river
//!   polygons draped on its own terrain) via
//!   [`WaterSubsystem::set_external_mesh`]. While an external mesh is
//!   set it takes precedence over the scene rectangle and the
//!   subsystem renders whenever weather state is available — no
//!   `[water]` block is required (hosts have no scene). Roughness
//!   bounds fall back to the `[water]` defaults, overridable via
//!   [`WaterSubsystem::set_roughness_range`].
//!
//! With neither source present the subsystem builds cleanly but its
//! pass closure no-ops.
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
    GpuContext, HdrFramebuffer, PassDescriptor, PassId, PassStage, PrepareContext, RenderContext,
    RenderSubsystem, SubsystemFactory,
};

const PASS_WATER: PassId = 0;

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

/// Fallback GGX roughness bounds for the external-mesh path when the
/// scene supplies no `[water]` block. Match `ps_core::Water::default()`.
const DEFAULT_ROUGHNESS_MIN: f32 = 0.02;
const DEFAULT_ROUGHNESS_MAX: f32 = 0.10;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct WaterParamsGpu {
    /// `(xmin, xmax, zmin, zmax)`. **Vestigial** — vertices now carry
    /// world positions directly, so the shader no longer remaps into
    /// these bounds. Kept (and still populated from the scene rect
    /// when one exists) purely for uniform-layout stability.
    bounds: [f32; 4],
    /// `(altitude_m, roughness_min, roughness_max, simulated_seconds)`.
    /// `altitude_m` is vestigial like `bounds` (altitude is baked into
    /// the vertex Y); roughness min/max drive the wind-roughness lerp
    /// in the fragment shader. `simulated_seconds` duplicates the frame
    /// uniform and is unused by the shader.
    config: [f32; 4],
    /// `(wind_dir_deg, wind_speed_mps, _, _)`.
    wind: [f32; 4],
}

/// Host-injected world-space geometry (see
/// [`WaterSubsystem::set_external_mesh`]).
struct ExternalMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

/// Phase 13.5 water subsystem.
pub struct WaterSubsystem {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    /// Host-injected geometry; takes precedence over the internal
    /// scene-rect grid while `Some`.
    external: Option<ExternalMesh>,
    /// Roughness `(min, max)` used by the external-mesh path when the
    /// scene supplies no `[water]` block.
    external_roughness: (f32, f32),
    /// The `(xmin, xmax, zmin, zmax, altitude_m)` rect the internal
    /// grid vertex buffer currently holds; `None` until first upload.
    internal_rect: Option<[f32; 5]>,
    /// Active flag: `true` when there is geometry to draw — the live
    /// `WeatherState.scene_water` is `Some`, or an external mesh is
    /// set. Latched by `prepare`, read by `dispatch_pass`.
    active: bool,
}

impl WaterSubsystem {
    /// Construct.
    pub fn new(_config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed =
            ps_core::shaders::compose(&[ps_core::shaders::COMMON_UNIFORMS_WGSL, &live_src]);
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
                        std::mem::size_of::<WaterParamsGpu>() as u64
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
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
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

        // Internal grid: fixed topology, world-space vertices written
        // by `prepare` when the scene rect first appears (or changes).
        // The vertex buffer is allocated up front at the grid's fixed
        // size and left zeroed — nothing draws until `prepare` has
        // latched a scene rect and uploaded real positions.
        let vertex_count = (GRID_RES + 1) * (GRID_RES + 1);
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-vb"),
            size: u64::from(vertex_count) * std::mem::size_of::<[f32; 3]>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices = grid_indices(GRID_RES);
        let index_count = indices.len() as u32;
        let index_buf = create_init_buffer(
            device,
            "water-ib",
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            bytemuck::cast_slice(&indices),
        );

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
            pipeline,
            vertex_buf,
            index_buf,
            index_count,
            params_buf,
            params_bg,
            external: None,
            external_roughness: (DEFAULT_ROUGHNESS_MIN, DEFAULT_ROUGHNESS_MAX),
            internal_rect: None,
            active: false,
        }
    }

    /// Inject host-provided world-space water geometry, replacing the
    /// internal scene-rect grid until [`Self::clear_external_mesh`].
    ///
    /// `vertices` are world positions `[x, y, z]` (typically OSM lake /
    /// river polygons triangulated by the host, XZ footprint with a
    /// per-vertex surface height in Y). `indices` is a `u32` triangle
    /// list into `vertices`. Wave animation is a per-pixel normal
    /// perturbation — the mesh needs only enough tessellation for
    /// frustum clipping, not per-wave detail.
    ///
    /// While an external mesh is set the subsystem is active whenever
    /// weather state is available; no `scene.water` block is required.
    /// Wind direction/speed come from `WeatherState.surface` and the
    /// wave clock from the frame uniforms, exactly as for the internal
    /// path. Roughness min/max come from `scene.water` when present,
    /// else from [`Self::set_roughness_range`] (default 0.02 / 0.10).
    ///
    /// Passing an empty vertex or index slice is equivalent to
    /// [`Self::clear_external_mesh`].
    pub fn set_external_mesh(
        &mut self,
        device: &wgpu::Device,
        vertices: &[[f32; 3]],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.external = None;
            return;
        }
        let vertex_buf = create_init_buffer(
            device,
            "water-external-vb",
            wgpu::BufferUsages::VERTEX,
            bytemuck::cast_slice(vertices),
        );
        let index_buf = create_init_buffer(
            device,
            "water-external-ib",
            wgpu::BufferUsages::INDEX,
            bytemuck::cast_slice(indices),
        );
        self.external = Some(ExternalMesh {
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        });
    }

    /// Drop any host-injected geometry and return to the internal
    /// `scene.water` rectangle path.
    pub fn clear_external_mesh(&mut self) {
        self.external = None;
    }

    /// Override the GGX roughness `(min, max)` used by the
    /// external-mesh path when the scene supplies no `[water]` block.
    /// The fragment shader lerps between the two by surface wind
    /// speed. Defaults to `(0.02, 0.10)` — the `[water]` defaults.
    pub fn set_roughness_range(&mut self, min: f32, max: f32) {
        self.external_roughness = (min, max);
    }
}

/// Create a buffer initialised with `bytes` (`bytes.len()` must be a
/// multiple of `wgpu::COPY_BUFFER_ALIGNMENT`, which holds for `[f32; 3]`
/// vertices and `u32` indices).
fn create_init_buffer(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
    buf.unmap();
    buf
}

impl RenderSubsystem for WaterSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let scene_water = ctx.weather.scene_water.as_ref();

        // Internal path: needs a scene rect; (re)upload world-space
        // grid vertices when it first appears or changes. The external
        // path carries its own geometry and skips all of this.
        if self.external.is_none() {
            let Some(water) = scene_water else {
                self.active = false;
                return;
            };
            let rect = [
                water.xmin,
                water.xmax,
                water.zmin,
                water.zmax,
                water.altitude_m,
            ];
            if self.internal_rect != Some(rect) {
                let vertices = grid_world_vertices(
                    GRID_RES,
                    [water.xmin, water.xmax, water.zmin, water.zmax],
                    water.altitude_m,
                );
                ctx.queue
                    .write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices));
                self.internal_rect = Some(rect);
            }
        }

        // Roughness bounds come from the scene when it has a `[water]`
        // block; the external path falls back to the configured
        // defaults so `prepare` stays total without a scene.
        let (roughness_min, roughness_max) = scene_water
            .map(|w| (w.roughness_min, w.roughness_max))
            .unwrap_or(self.external_roughness);
        let surface = &ctx.weather.surface;
        let params = WaterParamsGpu {
            // bounds + altitude are vestigial (see WaterParamsGpu);
            // populated from the scene rect when one exists.
            bounds: scene_water
                .map(|w| [w.xmin, w.xmax, w.zmin, w.zmax])
                .unwrap_or_default(),
            config: [
                scene_water.map(|w| w.altitude_m).unwrap_or_default(),
                roughness_min,
                roughness_max,
                ctx.frame_uniforms.simulated_seconds,
            ],
            wind: [surface.wind_dir_deg, surface.wind_speed_mps, 0.0, 0.0],
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytes_of(&params));
        self.active = true;
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![PassDescriptor {
            name: "water",
            stage: PassStage::Opaque,
            id: PASS_WATER,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        if !self.active {
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
        pass.set_bind_group(1, ctx.world_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bg, &[]);
        pass.set_bind_group(3, luts_bg, &[]);
        let (vertex_buf, index_buf, index_count) = match self.external.as_ref() {
            Some(mesh) => (&mesh.vertex_buf, &mesh.index_buf, mesh.index_count),
            None => (&self.vertex_buf, &self.index_buf, self.index_count),
        };
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..index_count, 0, 0..1);
    }
}

/// World-space vertices for an `n × n`-subdivided rectangle covering
/// `bounds = (xmin, xmax, zmin, zmax)` at `altitude_m`.
///
/// The remap from grid parameter `(s, t) ∈ [0,1]²` to world XZ
/// reproduces the WGSL `mix()` the vertex shader historically applied
/// on the GPU — `e1 * (1 - e3) + e2 * e3` — bit-for-bit in f32, so
/// moving it to the CPU feeds the shader identical world positions.
fn grid_world_vertices(n: u32, bounds: [f32; 4], altitude_m: f32) -> Vec<[f32; 3]> {
    let [xmin, xmax, zmin, zmax] = bounds;
    let mix = |e1: f32, e2: f32, e3: f32| e1 * (1.0 - e3) + e2 * e3;
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
    for j in 0..=n {
        for i in 0..=n {
            let s = i as f32 / n as f32;
            let t = j as f32 / n as f32;
            vertices.push([mix(xmin, xmax, s), altitude_m, mix(zmin, zmax, t)]);
        }
    }
    vertices
}

/// Triangle-list indices for an `n × n`-subdivided grid laid out
/// row-major as produced by [`grid_world_vertices`]; 6 indices per quad.
fn grid_indices(n: u32) -> Vec<u32> {
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
    indices
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
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(WaterSubsystem::new(config, gpu)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU world-coordinate generation must reproduce the remap
    /// the vertex shader used to perform on the GPU: for grid
    /// parameter `(s, t) = (i/n, j/n)`, `x = mix(xmin, xmax, s)` and
    /// `z = mix(zmin, zmax, t)` with WGSL `mix(e1, e2, e3) =
    /// e1 * (1 - e3) + e2 * e3`, at the configured altitude.
    #[test]
    fn grid_world_vertices_match_shader_remap() {
        let n = GRID_RES;
        let bounds = [-25.0_f32, 25.0, -15.0, 15.0];
        let altitude = 0.1_f32;
        let vertices = grid_world_vertices(n, bounds, altitude);
        assert_eq!(vertices.len(), ((n + 1) * (n + 1)) as usize);
        for j in 0..=n {
            for i in 0..=n {
                let s = i as f32 / n as f32;
                let t = j as f32 / n as f32;
                let expected_x = bounds[0] * (1.0 - s) + bounds[1] * s;
                let expected_z = bounds[2] * (1.0 - t) + bounds[3] * t;
                let v = vertices[(j * (n + 1) + i) as usize];
                assert_eq!(v[0].to_bits(), expected_x.to_bits(), "x at ({i},{j})");
                assert_eq!(v[1].to_bits(), altitude.to_bits(), "y at ({i},{j})");
                assert_eq!(v[2].to_bits(), expected_z.to_bits(), "z at ({i},{j})");
            }
        }
        // Corners land exactly on the bounds (s, t ∈ {0, 1}).
        let last = vertices.len() - 1;
        assert_eq!(vertices[0][0].to_bits(), bounds[0].to_bits());
        assert_eq!(vertices[0][2].to_bits(), bounds[2].to_bits());
        assert_eq!(vertices[last][0].to_bits(), bounds[1].to_bits());
        assert_eq!(vertices[last][2].to_bits(), bounds[3].to_bits());
    }

    /// Grid topology: 6 indices per quad, all in range, matching the
    /// row-major vertex layout.
    #[test]
    fn grid_indices_cover_all_quads_in_range() {
        let n = GRID_RES;
        let indices = grid_indices(n);
        assert_eq!(indices.len(), (n * n * 6) as usize);
        let vertex_count = (n + 1) * (n + 1);
        assert!(indices.iter().all(|&i| i < vertex_count));
        // First quad spells out the historical winding.
        assert_eq!(&indices[..6], &[0, 1, n + 2, 0, n + 2, n + 1]);
    }

    /// External-mesh lifecycle: setting host geometry installs it,
    /// empty input is equivalent to clearing, and
    /// `clear_external_mesh` returns to the internal path. Skips
    /// silently on machines without a GPU adapter.
    #[test]
    fn external_mesh_set_and_clear() {
        let gpu = match ps_core::gpu::init_headless() {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("skipping external_mesh_set_and_clear — no GPU adapter: {e}");
                return;
            }
        };
        let config = Config::default();
        let mut sub = WaterSubsystem::new(&config, &gpu);
        assert!(sub.external.is_none());

        let vertices = [[0.0_f32, 0.1, 0.0], [10.0, 0.1, 0.0], [0.0, 0.1, 10.0]];
        let indices = [0_u32, 1, 2];
        sub.set_external_mesh(&gpu.device, &vertices, &indices);
        assert_eq!(sub.external.as_ref().map(|m| m.index_count), Some(3));

        // Empty vertex or index input clears rather than installing a
        // degenerate mesh.
        sub.set_external_mesh(&gpu.device, &[], &indices);
        assert!(sub.external.is_none());
        sub.set_external_mesh(&gpu.device, &vertices, &[]);
        assert!(sub.external.is_none());

        sub.set_external_mesh(&gpu.device, &vertices, &indices);
        sub.clear_external_mesh();
        assert!(sub.external.is_none());
    }
}
