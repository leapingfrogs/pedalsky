//! Phase 13.6 — 3D wind sock.
//!
//! A small cone of geometry anchored at a fixed camera-relative
//! offset (5 m forward, 1.5 m up) so the sock follows the user but
//! reads as a real object in the scene. The cone's axis points
//! along the downwind direction (where the wind is blowing *to*)
//! and droops toward vertical as the wind speed falls.
//!
//! Render-graph slot: `PassStage::Opaque` (depth-write on, depth-test
//! `Greater` against the reverse-Z depth buffer). The shader applies
//! aerial perspective from the atmosphere LUTs so the sock fades
//! cleanly when viewed from far away (e.g. a fly-camera looking back
//! at the windsock).
//!
//! Bind groups mirror the ground subsystem so wiring stays uniform
//! across opaque-stage geometry:
//!   - 0 — `FrameUniforms`
//!   - 1 — `WindsockParams` (model matrix, banded albedo)
//!   - 2 — atmosphere LUTs (shared `atmosphere_lut_bind_group_layout`)
//!
//! Disabled cleanly when atmosphere is off: the pass closure no-ops
//! when `ctx.luts_bind_group` is `None` (same convention as ground).

#![deny(missing_docs)]

use bytemuck::{bytes_of, Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, Config, GpuContext, HdrFramebuffer,
    PassStage, PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};

const SHADER_BAKED: &str = include_str!("../../../shaders/windsock/windsock.wgsl");
const SHADER_REL: &str = "windsock/windsock.wgsl";

/// Stable subsystem name (matches `[render.subsystems].windsock`).
pub const NAME: &str = "windsock";

/// Cone tessellation: number of segments around the base ring.
const RING_SEGMENTS: u32 = 24;

/// Geometric size of the sock. Picked so the hub is roughly the size of
/// a real airfield windsock seen at conversational distance: a 1.2 m
/// long bag with a 0.4 m base mouth.
const SOCK_LENGTH_M: f32 = 1.2;
const SOCK_BASE_RADIUS_M: f32 = 0.4;

/// Camera-relative anchor in metres (forward, up).
const ANCHOR_FORWARD_M: f32 = 5.0;
const ANCHOR_UP_M: f32 = 1.5;

/// Wind speed (m/s) at which the sock holds fully horizontal. Below this
/// the droop angle scales linearly toward `MAX_DROOP_DEG` at zero wind.
const SOCK_HORIZONTAL_AT_MPS: f32 = 8.0;
const MAX_DROOP_DEG: f32 = 70.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    axial_t: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct WindsockParamsGpu {
    model: [[f32; 4]; 4],
    albedo: [f32; 4],
    stripe: [f32; 4],
}

/// Phase 13.6 windsock subsystem.
pub struct WindsockSubsystem {
    enabled: bool,
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    params_buf: wgpu::Buffer,
    /// Group-1 bind group built once (only holds the uniform; the LUT
    /// bind group on group 2 comes from the shared frame context).
    params_bg: wgpu::BindGroup,
}

impl WindsockSubsystem {
    /// Construct pipeline + cone mesh.
    pub fn new(_config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            &live_src,
        ]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("windsock/windsock.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = frame_bind_group_layout(device);
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("windsock-params-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<WindsockParamsGpu>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let lut_layout = atmosphere_lut_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("windsock-pl"),
            bind_group_layouts: &[Some(&frame_layout), Some(&params_layout), Some(&lut_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("windsock-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, // position_local
                        1 => Float32x3, // normal_local
                        2 => Float32,   // axial_t
                    ],
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
                // Cone is open at the base; both faces are visible.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (vertices, indices) = build_cone_mesh();
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("windsock-vb"),
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
            label: Some("windsock-ib"),
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
            label: Some("windsock-params-ub"),
            size: std::mem::size_of::<WindsockParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("windsock-params-bg"),
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
        }
    }
}

impl RenderSubsystem for WindsockSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let params = build_params(ctx);
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytes_of(&params));
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let pipeline = self.pipeline.clone();
        let vertex_buf = self.vertex_buf.clone();
        let index_buf = self.index_buf.clone();
        let index_count = self.index_count;
        let params_bg = self.params_bg.clone();
        vec![RegisteredPass {
            name: "windsock",
            stage: PassStage::Opaque,
            run: Box::new(move |encoder, ctx| {
                let Some(luts_bg) = ctx.luts_bind_group else {
                    // No LUTs available — skip (shader requires the AP
                    // 3D texture). Matches the ground subsystem.
                    return;
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("windsock-pass"),
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
                pass.set_bind_group(1, &params_bg, &[]);
                pass.set_bind_group(2, luts_bg, &[]);
                pass.set_vertex_buffer(0, vertex_buf.slice(..));
                pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint16);
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

/// Build the per-frame `WindsockParamsGpu`: position the cone in front
/// of the camera, orient it downwind, droop based on wind speed.
fn build_params(ctx: &PrepareContext<'_>) -> WindsockParamsGpu {
    let surface = &ctx.weather.surface;
    let wind_dir_deg = surface.wind_dir_deg;
    let wind_speed = surface.wind_speed_mps.max(0.0);

    // Downwind direction (where wind blows TO) in world space:
    // PedalSky convention is +X east, +Y up, +Z south, with azimuth
    // measured clockwise from north (= -Z). Meteorological wind
    // direction is the direction the wind comes FROM; downwind is
    // therefore that + 180°.
    let downwind_deg = wind_dir_deg + 180.0;
    let theta = downwind_deg.to_radians();
    let (sz, cz) = theta.sin_cos();
    let downwind_horiz = Vec3::new(sz, 0.0, -cz).normalize_or_zero();

    // Droop: linearly interpolate from full-horizontal at
    // `SOCK_HORIZONTAL_AT_MPS` to `MAX_DROOP_DEG` at zero wind.
    let droop_t = 1.0 - (wind_speed / SOCK_HORIZONTAL_AT_MPS).clamp(0.0, 1.0);
    let droop_rad = (droop_t * MAX_DROOP_DEG).to_radians();
    // Compose the sock-local axis (+Z_local maps to downwind) as a
    // rotation that tilts the horizontal downwind direction downward
    // by `droop_rad`.
    let cone_axis = (downwind_horiz * droop_rad.cos() + Vec3::NEG_Y * droop_rad.sin())
        .normalize_or_zero();

    // Build an orthonormal basis with `cone_axis` as the Z column.
    // Pick an "up" reference that isn't parallel to the axis (the
    // axis is never world-up: even fully drooped it points -Y,
    // and the world-up reference is +Y).
    let world_up = Vec3::Y;
    let x_axis = world_up.cross(cone_axis).normalize_or_zero();
    let y_axis = cone_axis.cross(x_axis).normalize_or_zero();
    let z_axis = cone_axis;

    // Anchor: 5 m along the camera horizontal forward, 1.5 m up. We
    // ignore the camera's pitch component so the sock floats at a
    // constant height regardless of where the user is looking.
    let cam_pos = ctx.frame_uniforms.camera_position_world.truncate();
    let cam_forward = camera_horizontal_forward(ctx.frame_uniforms);
    let anchor = cam_pos + cam_forward * ANCHOR_FORWARD_M + Vec3::Y * ANCHOR_UP_M;

    // Model matrix: scale local cone by (radius, radius, length), then
    // rotate by the basis, then translate to the anchor. Columns are
    // `s * axis_in_world` so the cone mesh's local +Z (apex direction)
    // ends up along `cone_axis` after multiplication.
    let sx = SOCK_BASE_RADIUS_M;
    let sy = SOCK_BASE_RADIUS_M;
    let sz_scale = SOCK_LENGTH_M;
    let model = Mat4::from_cols(
        Vec4::new(x_axis.x * sx, x_axis.y * sx, x_axis.z * sx, 0.0),
        Vec4::new(y_axis.x * sy, y_axis.y * sy, y_axis.z * sy, 0.0),
        Vec4::new(z_axis.x * sz_scale, z_axis.y * sz_scale, z_axis.z * sz_scale, 0.0),
        Vec4::new(anchor.x, anchor.y, anchor.z, 1.0),
    );

    WindsockParamsGpu {
        model: model.to_cols_array_2d(),
        // Standard aviation-orange primary band, alpha = stripe cutoff.
        albedo: [0.95, 0.45, 0.05, 0.30],
        // White secondary band (the base 30 % of the cone).
        stripe: [0.95, 0.95, 0.95, 0.0],
    }
}

/// Derive the camera's *horizontal* forward direction from the
/// inverse-view-projection. We need the camera's look direction
/// projected to the world horizontal plane so the windsock anchors
/// at a constant elevation no matter the camera pitch.
fn camera_horizontal_forward(uniforms: &ps_core::FrameUniforms) -> Vec3 {
    // The view matrix's third row of its rotation part is the camera
    // -Z axis in world space (right-handed). In `look_to_rh`, the
    // matrix is built so that `view * vec3_in_world = vec3_in_camera`;
    // therefore the world-space forward is the *negated* third column
    // of the inverse view, which is `inv_view.col(2).truncate() * -1`.
    // Easier: pull the basis directly from `view`.
    let v = uniforms.view;
    // The camera-space -Z direction in world coords is given by
    // -(view⁻¹) * vec4(0,0,1,0). For an orthonormal view matrix this
    // is simply the third row of the rotation block, negated. The
    // third row of `view` is the camera-space Z axis expressed in
    // world; negate to get forward (which looks down -Z in camera
    // space).
    let forward_world = Vec3::new(-v.row(2).x, -v.row(2).y, -v.row(2).z);
    // Project onto the horizontal plane.
    let horiz = Vec3::new(forward_world.x, 0.0, forward_world.z);
    horiz.normalize_or_zero()
}

/// Build a closed cone in local space: apex at +Z=1, base ring at Z=0
/// with radius 1, plus a base cap so the sock isn't see-through from
/// behind. Returns `(vertices, indices)`.
fn build_cone_mesh() -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let n = RING_SEGMENTS;

    // --- Side ring (used by side triangles). Duplicated below as a
    // cap ring so the base cap can carry its own straight-down normal.
    let side_ring_first: u16 = vertices.len() as u16;
    let cone_slant = (1.0_f32).hypot(1.0); // base_radius=1, length=1 → slant=√2.
    let cos_slant = 1.0 / cone_slant;
    let sin_slant = 1.0 / cone_slant;
    for i in 0..n {
        let phi = (i as f32) / (n as f32) * std::f32::consts::TAU;
        let (s, c) = phi.sin_cos();
        // Side normal: in cylindrical coords the side surface has a
        // radial component `cos_slant` and an axial (+Z) component
        // `sin_slant` (apex narrower than base).
        let nx = c * cos_slant;
        let ny = s * cos_slant;
        let nz = sin_slant;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        vertices.push(Vertex {
            position: [c, s, 0.0],
            normal: [nx / len, ny / len, nz / len],
            axial_t: 0.0,
        });
    }
    // Apex vertices — one per side triangle so each gets the matching
    // side normal interpolated cleanly to the apex.
    let apex_first = vertices.len() as u16;
    for i in 0..n {
        let phi = ((i as f32) + 0.5) / (n as f32) * std::f32::consts::TAU;
        let (s, c) = phi.sin_cos();
        let nx = c * cos_slant;
        let ny = s * cos_slant;
        let nz = sin_slant;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        vertices.push(Vertex {
            position: [0.0, 0.0, 1.0],
            normal: [nx / len, ny / len, nz / len],
            axial_t: 1.0,
        });
    }
    for i in 0..n {
        let a = side_ring_first + i as u16;
        let b = side_ring_first + ((i + 1) % n) as u16;
        let p = apex_first + i as u16;
        indices.push(a);
        indices.push(b);
        indices.push(p);
    }

    // --- Cap ring (used by the closed base).
    let cap_ring_first: u16 = vertices.len() as u16;
    for i in 0..n {
        let phi = (i as f32) / (n as f32) * std::f32::consts::TAU;
        let (s, c) = phi.sin_cos();
        vertices.push(Vertex {
            position: [c, s, 0.0],
            normal: [0.0, 0.0, -1.0],
            axial_t: 0.0,
        });
    }
    let cap_centre = vertices.len() as u16;
    vertices.push(Vertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        axial_t: 0.0,
    });
    for i in 0..n {
        let a = cap_ring_first + i as u16;
        let b = cap_ring_first + ((i + 1) % n) as u16;
        indices.push(cap_centre);
        indices.push(b);
        indices.push(a);
    }

    let _ = (side_ring_first, cap_ring_first);
    debug_assert!(vertices.len() <= u16::MAX as usize);
    (vertices, indices)
}

/// Factory wired by `AppBuilder`.
pub struct WindsockFactory;

impl SubsystemFactory for WindsockFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.windsock
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(WindsockSubsystem::new(config, gpu)))
    }
}
