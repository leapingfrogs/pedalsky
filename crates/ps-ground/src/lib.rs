//! Phase 7 ground subsystem: PBR ground with wet surface and snow.
//!
//! Replaces the Phase 0 procedural checker. The fragment shader
//! implements:
//! - GGX/Smith specular + Lambertian diffuse over a 5 m Voronoi-tiled
//!   3-entry palette (plan §7.1).
//! - Lagarde 2013 wet surface chain (plan §7.2): darkened albedo,
//!   reduced roughness, optional thin water layer for puddles above
//!   `surface.puddle_start`.
//! - Snow layer gated by `temp_c < 0.5 && snow_depth_m > 0` (plan §7.3).
//! - Aerial perspective applied in-shader from the AP LUT (plan §7.4).
//!
//! Bind groups:
//! - 0 — `FrameUniforms` (engine-wide).
//! - 1 — `WorldUniforms` (engine-wide).
//! - 2 — ground-owned `SurfaceParams` uniform (this crate).
//! - 3 — atmosphere LUTs (Phase 5).
//!
//! `SurfaceParams` is uploaded each frame in `prepare()` from
//! `PrepareContext::weather.surface`.

#![deny(missing_docs)]

use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout,
    BindGroupCache, Config, GpuContext, HdrFramebuffer, PassDescriptor, PassId, PassStage,
    PrepareContext, RenderContext, RenderSubsystem, SubsystemFactory, SurfaceParams,
};
use ps_imagery::RgbTile;
use ps_terrain::MeshData;

/// Mailbox shared between the terrain-fetch worker thread and the
/// `GroundSubsystem`. The worker writes a completed `MeshData` here;
/// `GroundSubsystem::prepare` drains it on the next frame and uploads
/// to the GPU.
///
/// `Arc<Mutex<Option<MeshData>>>` is intentionally minimal — terrain
/// fetches are infrequent (a few seconds apart at most) so the lock is
/// uncontended. A more elaborate channel would be overkill.
#[derive(Clone, Default)]
pub struct PendingMeshInbox(Arc<Mutex<Option<MeshData>>>);

impl PendingMeshInbox {
    /// New empty inbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Post a completed mesh from a worker thread. Overwrites any
    /// previous unconsumed mesh (the latest fetch wins).
    pub fn post(&self, mesh: MeshData) {
        *self.0.lock().expect("PendingMeshInbox lock") = Some(mesh);
    }

    /// Take any pending mesh. Called once per frame by the
    /// `GroundSubsystem`.
    pub fn take(&self) -> Option<MeshData> {
        self.0.lock().expect("PendingMeshInbox lock").take()
    }
}

/// Mailbox shared between the imagery-fetch worker thread and the
/// `GroundSubsystem`. Same shape as [`PendingMeshInbox`] but carries a
/// stitched [`RgbTile`].
#[derive(Clone, Default)]
pub struct PendingImageryInbox(Arc<Mutex<Option<RgbTile>>>);

impl PendingImageryInbox {
    /// New empty inbox.
    pub fn new() -> Self {
        Self::default()
    }

    /// Post a completed RGB tile from a worker thread.
    pub fn post(&self, tile: RgbTile) {
        *self.0.lock().expect("PendingImageryInbox lock") = Some(tile);
    }

    /// Take any pending tile. Called once per frame by the
    /// `GroundSubsystem`.
    pub fn take(&self) -> Option<RgbTile> {
        self.0.lock().expect("PendingImageryInbox lock").take()
    }
}

/// Runtime toggle for the satellite overlay. Wired through the
/// `GroundOverlayController` so the UI can flip it without rebuilding
/// the subsystem.
#[derive(Clone, Default)]
pub struct GroundOverlayController(Arc<Mutex<GroundOverlayState>>);

#[derive(Default, Copy, Clone)]
struct GroundOverlayState {
    satellite_enabled: bool,
}

impl GroundOverlayController {
    /// New, satellite-disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle the satellite overlay on/off.
    pub fn set_satellite_enabled(&self, enabled: bool) {
        self.0
            .lock()
            .expect("GroundOverlayController lock")
            .satellite_enabled = enabled;
    }

    /// Current satellite-overlay state.
    pub fn satellite_enabled(&self) -> bool {
        self.0
            .lock()
            .expect("GroundOverlayController lock")
            .satellite_enabled
    }
}

/// GPU-side uniform that tells the shader where the satellite raster
/// covers (in world XZ metres relative to the observer) and whether to
/// sample it at all. Matches the WGSL `GroundOverlayParams` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct GroundOverlayParamsGpu {
    /// World-space XZ centre of the imagery raster (the observer
    /// position at fetch time, in metres).
    centre_xz: [f32; 2],
    /// Half-extent of the raster in metres, EW and NS. The shader
    /// computes UV = (p.xz - centre) / (2 * half_extent) + 0.5.
    half_extent_xz: [f32; 2],
    /// Non-zero when the satellite texture should be sampled.
    /// Stored as `f32` so we don't pull in any padding rules.
    enabled: f32,
    /// Pad to 16 B alignment.
    _pad: [f32; 3],
}

const PASS_GROUND: PassId = 0;

/// Baked shader source — used as the fallback when no runtime
/// override is registered (the default for headless tests and
/// production builds without `[debug] shader_hot_reload`).
const SHADER_BAKED: &str = include_str!("../../../shaders/ground/pbr.wgsl");
/// Path of the shader file relative to `shaders/`. The hot-reload
/// loader uses this to find the live source on disk.
const SHADER_REL: &str = "ground/pbr.wgsl";
const QUAD_HALF_EXTENT_M: f32 = 100_000.0;

/// Per-vertex format. Position + outward normal. Matches
/// `ps_terrain::mesh::MeshVertex`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Stable subsystem name (matches `[render.subsystems].ground`).
pub const NAME: &str = "ground";

/// Build the bind-group layout for the ground subsystem's group 2:
///
///   binding 0: SurfaceParams uniform
///   binding 1: top-down cloud density mask (Phase 12.6 — overcast
///              diffuse modulation). The mask is sampled in the
///              ground shader at the surface point's XZ to determine
///              how much cloud is overhead.
///   binding 2: linear-clamp sampler for the cloud mask.
///   binding 3: Phase 16 — satellite imagery (RGBA8). Sampled in
///              place of the procedural Voronoi when
///              `overlay.enabled != 0`.
///   binding 4: linear-clamp sampler for the satellite texture.
///   binding 5: GroundOverlayParams uniform — UV mapping + enable
///              flag for the satellite overlay.
fn surface_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ground-surface-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<SurfaceParams>() as u64
                    ),
                },
                count: None,
            },
            // Phase 12.6 — top-down cloud density mask.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Phase 16 — satellite imagery texture.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // Phase 16 — GroundOverlayParams uniform.
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                        GroundOverlayParamsGpu,
                    >() as u64),
                },
                count: None,
            },
        ],
    })
}

/// Procedural ground plane. Constructed via [`GroundSubsystem`]; kept
/// public so tests / host code can build one directly.
///
/// The group-2 bind group is rebuilt each frame in `prepare()` rather
/// than cached, because it references the top-down density mask
/// texture view from the live WeatherState (which can be replaced
/// when synthesis re-runs for a hot-reload).
///
/// Phase 16 — geometry is now indexed: a vertex buffer + an index
/// buffer, sized by whatever mesh the host has uploaded. Initial state
/// is a flat 6-vertex quad equivalent so the first frame still
/// renders before any terrain fetch completes.
pub struct PbrGround {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    /// Allocated capacity (vertices) of `vertex_buf`. Reallocated when
    /// a larger mesh is uploaded.
    vertex_capacity: u32,
    index_buf: wgpu::Buffer,
    /// Allocated capacity (indices) of `index_buf`.
    index_capacity: u32,
    /// Number of valid indices currently in `index_buf`. The draw call
    /// reads `0..index_count`.
    index_count: u32,
    surface_buf: wgpu::Buffer,
    /// Layout used to rebuild the group-2 bind group each frame.
    surface_layout: wgpu::BindGroupLayout,
    /// Cached sampler for the density-mask texture binding.
    density_mask_sampler: wgpu::Sampler,
    /// Phase 16 — satellite imagery texture. 1×1 transparent on
    /// startup; replaced via [`Self::upload_satellite_image`].
    satellite_tex: wgpu::Texture,
    satellite_view: wgpu::TextureView,
    /// Allocated `(width, height)` of `satellite_tex`. Reallocated when
    /// a larger image is uploaded.
    satellite_size: (u32, u32),
    /// Linear-clamp sampler for the satellite texture.
    satellite_sampler: wgpu::Sampler,
    /// GPU-side overlay params (UV mapping + enable flag).
    overlay_buf: wgpu::Buffer,
    /// CPU mirror — written each frame in `prepare`.
    overlay_params: GroundOverlayParamsGpu,
    /// Monotonic counter bumped every time `upload_satellite_image`
    /// runs so the bind-group cache invalidates and the new texture
    /// view is picked up.
    satellite_revision: u64,
}

impl PbrGround {
    /// Build the pipeline, vertex buffer, and surface uniform buffer +
    /// bind group.
    pub fn new(device: &wgpu::Device) -> Self {
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_LUT_SAMPLING_WGSL,
            &live_src,
        ]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground/pbr.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });

        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let surface_layout = surface_bind_group_layout(device);
        let lut_layout = atmosphere_lut_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ground-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&surface_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ground-rp"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    // Phase 16 — position + normal (per-vertex).
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
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
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Phase 16 — startup geometry: the legacy 6-vertex flat quad,
        // expressed as an indexed draw using the *same* triangulation
        // and vertex order as the pre-Phase-16 unindexed version so
        // the goldens stay byte-identical until a terrain fetch
        // replaces this mesh.
        //
        // Original (pre-Phase-16) vertex stream was:
        //   tri 1: NW(-h,-h), NE(h,-h), SE(h,h)
        //   tri 2: NW(-h,-h), SE(h,h),  SW(-h,h)
        // Shared diagonal: NW-SE.
        //
        // We deduplicate NW + SE into a 4-vertex buffer and re-emit
        // the exact same triangle sequence via indices.
        let h = QUAD_HALF_EXTENT_M;
        let up = [0.0_f32, 1.0, 0.0];
        let vertices: [Vertex; 4] = [
            Vertex {
                position: [-h, 0.0, -h],
                normal: up,
            }, // 0 NW
            Vertex {
                position: [h, 0.0, -h],
                normal: up,
            }, // 1 NE
            Vertex {
                position: [h, 0.0, h],
                normal: up,
            }, // 2 SE
            Vertex {
                position: [-h, 0.0, h],
                normal: up,
            }, // 3 SW
        ];
        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-vb"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vertex_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&vertices));
        vertex_buf.unmap();

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-ib"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        index_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        index_buf.unmap();

        let vertex_capacity = vertices.len() as u32;
        let index_capacity = indices.len() as u32;
        let index_count = indices.len() as u32;

        let surface_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-surface-ub"),
            size: std::mem::size_of::<SurfaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Phase 12.6 — sampler for the top-down density mask. Linear
        // filtering smooths cloud-edge transitions across grid cells;
        // clamp-to-edge means surface points outside the 32 km mask
        // extent get the boundary value (a v1 stationary-camera
        // limitation).
        let density_mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ground-density-mask-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Phase 16 — placeholder satellite texture: 1×1 black pixel.
        // The shader only reads from it when `overlay.enabled != 0`,
        // so this never gets sampled until a real upload happens.
        let satellite_size = (1u32, 1u32);
        let satellite_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ground-satellite-tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let satellite_view = satellite_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let satellite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ground-satellite-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let overlay_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground-overlay-ub"),
            size: std::mem::size_of::<GroundOverlayParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buf,
            vertex_capacity,
            index_buf,
            index_capacity,
            index_count,
            surface_buf,
            surface_layout,
            density_mask_sampler,
            satellite_tex,
            satellite_view,
            satellite_size,
            satellite_sampler,
            overlay_buf,
            overlay_params: GroundOverlayParamsGpu::default(),
            satellite_revision: 0,
        }
    }

    /// Replace the current ground mesh.
    ///
    /// Reallocates `vertex_buf` and/or `index_buf` if `mesh` is larger
    /// than the current capacity; otherwise writes in place via
    /// `queue.write_buffer`. Either way the draw count updates to
    /// `mesh.indices.len()`.
    ///
    /// Called from the host's terrain-fetch completion handler in
    /// `ps-app::main`. Designed to be safe to call between frames; do
    /// not call mid-render.
    pub fn upload_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &MeshData) {
        let v_bytes: &[u8] = bytemuck::cast_slice(&mesh.positions);
        let i_bytes: &[u8] = bytemuck::cast_slice(&mesh.indices);

        let needed_verts = mesh.positions.len() as u32;
        if needed_verts > self.vertex_capacity {
            self.vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ground-vb"),
                size: v_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = needed_verts;
        }
        queue.write_buffer(&self.vertex_buf, 0, v_bytes);

        let needed_indices = mesh.indices.len() as u32;
        if needed_indices > self.index_capacity {
            self.index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ground-ib"),
                size: i_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = needed_indices;
        }
        queue.write_buffer(&self.index_buf, 0, i_bytes);

        self.index_count = needed_indices;
        tracing::info!(
            target: "ps_ground",
            verts = mesh.positions.len(),
            tris = mesh.indices.len() / 3,
            "ground: uploaded terrain mesh"
        );
    }

    /// Replace the satellite texture with a fresh [`RgbTile`]. Updates
    /// the overlay-params uniform so the shader knows the new
    /// geographic extent in world XZ.
    ///
    /// Reallocates the texture if the new image is larger than the
    /// previous one; otherwise writes in place.
    ///
    /// `centre_world_xz` is the world-space XZ position the satellite
    /// raster is anchored to — set this to the observer position at
    /// the time of fetch so the imagery lines up with the terrain
    /// mesh, which is also centred on the observer (see
    /// `ps_terrain::build_grid_mesh`).
    pub fn upload_satellite_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile: &RgbTile,
        centre_world_xz: [f32; 2],
    ) {
        // Recreate the texture if the size has changed.
        if (tile.width, tile.height) != self.satellite_size {
            self.satellite_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ground-satellite-tex"),
                size: wgpu::Extent3d {
                    width: tile.width,
                    height: tile.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.satellite_view = self
                .satellite_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.satellite_size = (tile.width, tile.height);
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.satellite_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tile.pixels_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile.width * 4),
                rows_per_image: Some(tile.height),
            },
            wgpu::Extent3d {
                width: tile.width,
                height: tile.height,
                depth_or_array_layers: 1,
            },
        );

        // Compute the world-space half-extent of the raster. The
        // ps-imagery tile carries its geographic extent; we convert
        // degrees to metres using the centre latitude so the rectangle
        // matches the terrain mesh which uses metric XZ.
        //
        // X = +east, Z = +south. EW span = (east-west) * deg_to_m_lon;
        // NS span = (north-south) * deg_to_m_lat.
        let lat_centre = (tile.extent_deg.north + tile.extent_deg.south) * 0.5;
        let m_per_deg_lat = 111_320.0_f64;
        let m_per_deg_lon = 111_320.0 * lat_centre.to_radians().cos().abs().max(0.05);
        let half_ew = ((tile.extent_deg.east - tile.extent_deg.west) * 0.5 * m_per_deg_lon) as f32;
        let half_ns =
            ((tile.extent_deg.north - tile.extent_deg.south) * 0.5 * m_per_deg_lat) as f32;

        self.overlay_params = GroundOverlayParamsGpu {
            centre_xz: centre_world_xz,
            half_extent_xz: [half_ew, half_ns],
            // `enabled` is driven separately each frame from the
            // controller; we leave it whatever the prepare loop wrote
            // last (the overlay-params buffer is rewritten on every
            // frame in prepare).
            enabled: self.overlay_params.enabled,
            _pad: [0.0; 3],
        };
        self.satellite_revision = self.satellite_revision.wrapping_add(1);

        tracing::info!(
            target: "ps_ground",
            w = tile.width, h = tile.height,
            half_ew_m = half_ew, half_ns_m = half_ns,
            "ground: uploaded satellite imagery"
        );
    }
}

/// `RenderSubsystem` wrapper around [`PbrGround`].
pub struct GroundSubsystem {
    inner: PbrGround,
    /// Most-recent SurfaceParams (uploaded each frame in `prepare()`).
    surface: SurfaceParams,
    /// `true` when `[render.subsystems].wet_surface` is on. When off the
    /// ground shader still runs but `prepare()` zeros out the wetness +
    /// snow inputs so the dry BRDF path is taken.
    wet_surface_enabled: bool,
    /// Live group-2 bind group published by `prepare()` for `dispatch_pass`.
    /// Built via a revision-keyed cache so the wgpu hub touch only
    /// happens when the underlying density mask view changes (synthesis
    /// rerun), not every frame.
    live_surface_bg: Option<Arc<wgpu::BindGroup>>,
    /// Revision-keyed cache for the surface bind group.
    surface_bg_cache: BindGroupCache<u64>,
    /// Phase 16 — inbox drained in `prepare()`. The host-side terrain
    /// fetch worker writes a completed mesh here.
    pending_mesh: PendingMeshInbox,
    /// Phase 16 — inbox for completed satellite imagery uploads.
    pending_imagery: PendingImageryInbox,
    /// Phase 16 — runtime toggle for the satellite overlay
    /// (UI-driven). Read once per frame in `prepare`.
    overlay: GroundOverlayController,
    /// Last observer XZ position at imagery upload time — drives the
    /// shader's UV mapping. Tracked separately because we don't have
    /// the world XZ inside `prepare` (only world lat/lon).
    /// The host is expected to pass `[0.0, 0.0]` when the camera is
    /// observer-centred (the existing convention); see
    /// `centre_world_xz` in `upload_satellite_image`.
    last_overlay_centre_xz: [f32; 2],
}

impl GroundSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        Self::with_inboxes(
            config,
            gpu,
            PendingMeshInbox::new(),
            PendingImageryInbox::new(),
            GroundOverlayController::new(),
        )
    }

    /// Construct with existing inboxes + controller so a host can hold
    /// the sender sides and drive them from worker threads / the UI.
    pub fn with_inboxes(
        config: &Config,
        gpu: &GpuContext,
        pending_mesh: PendingMeshInbox,
        pending_imagery: PendingImageryInbox,
        overlay: GroundOverlayController,
    ) -> Self {
        Self {
            inner: PbrGround::new(&gpu.device),
            surface: SurfaceParams::default(),
            wet_surface_enabled: config.render.subsystems.wet_surface,
            live_surface_bg: None,
            surface_bg_cache: BindGroupCache::new(),
            pending_mesh,
            pending_imagery,
            overlay,
            last_overlay_centre_xz: [0.0, 0.0],
        }
    }
}

impl RenderSubsystem for GroundSubsystem {
    fn name(&self) -> &'static str {
        "ground"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // Phase 16 — drain any pending terrain mesh posted by the host
        // worker thread and upload it to the GPU. Cheap when empty.
        if let Some(mesh) = self.pending_mesh.take() {
            self.inner.upload_mesh(ctx.device, ctx.queue, &mesh);
        }

        // Phase 16 — drain any pending satellite imagery. The host
        // posts to the inbox; we upload to the GPU and recompute UV
        // mapping here. The terrain mesh is centred on the observer,
        // and the host posts the centre XZ as the world origin
        // ([0,0]) because that's where the terrain mesh anchors too.
        if let Some(tile) = self.pending_imagery.take() {
            self.inner
                .upload_satellite_image(ctx.device, ctx.queue, &tile, [0.0, 0.0]);
            self.last_overlay_centre_xz = [0.0, 0.0];
        }

        let mut surface = ctx.weather.surface;
        // The wet_surface master toggle (Phase 7.2 / 7.3) gates only the
        // wetness + puddle features — snow is a distinct ground material
        // and stays driven by the scene's snow_depth_m + temperature_c.
        if !self.wet_surface_enabled {
            surface.ground_wetness = 0.0;
            surface.puddle_coverage = 0.0;
        }
        ctx.queue
            .write_buffer(&self.inner.surface_buf, 0, bytes_of(&surface));
        self.surface = surface;

        // Phase 16 — refresh the overlay-params uniform. The
        // enable-flag comes from the UI-driven controller; the
        // UV-mapping fields are whatever the last `upload_satellite_image`
        // computed.
        self.inner.overlay_params.enabled = if self.overlay.satellite_enabled() {
            1.0
        } else {
            0.0
        };
        ctx.queue.write_buffer(
            &self.inner.overlay_buf,
            0,
            bytes_of(&self.inner.overlay_params),
        );

        // Phase 12.6 + Phase 16 — publish the group-2 bind group for
        // `dispatch_pass`. Cache key combines `weather.revision`
        // (rebuilt on synthesis re-run; affects the density mask
        // texture view) with `satellite_revision` (rebuilt on every
        // imagery upload; affects the satellite texture view). Two
        // 32-bit halves keep the key cheap.
        let cache_key =
            (ctx.weather.revision << 32) | (self.inner.satellite_revision & 0xFFFF_FFFF);
        let inner = &self.inner;
        let bg = self.surface_bg_cache.get_or_build(cache_key, || {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ground-surface-bg"),
                layout: &inner.surface_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inner.surface_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &ctx.weather.textures.overcast_field_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&inner.density_mask_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&inner.satellite_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&inner.satellite_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: inner.overlay_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.live_surface_bg = Some(bg);
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![PassDescriptor {
            name: "ground-pbr",
            stage: PassStage::Opaque,
            id: PASS_GROUND,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        // Phase 12.6 — read the live group-2 bind group built in
        // `prepare()` against this frame's density mask.
        let Some(surface_bg) = self.live_surface_bg.as_ref() else {
            // prepare() hasn't run yet (first frame in some headless
            // test paths); skip cleanly.
            return;
        };
        let Some(luts_bg) = ctx.luts_bind_group else {
            // Atmosphere disabled — skip; the shader needs the LUTs.
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ground-pass"),
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
        pass.set_pipeline(&self.inner.pipeline);
        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
        pass.set_bind_group(1, ctx.world_bind_group, &[]);
        pass.set_bind_group(2, surface_bg.as_ref(), &[]);
        pass.set_bind_group(3, luts_bg, &[]);
        pass.set_vertex_buffer(0, self.inner.vertex_buf.slice(..));
        pass.set_index_buffer(self.inner.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.inner.index_count, 0, 0..1);
    }

    /// Phase 19.A — refresh runtime-tunable flags so the UI subsystem
    /// checkboxes actually take effect mid-session.
    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        self.wet_surface_enabled = config.render.subsystems.wet_surface;
        Ok(())
    }
}

/// Factory wired by `AppBuilder`.
///
/// Holds the `PendingMeshInbox`, `PendingImageryInbox`, and
/// `GroundOverlayController` so the host can drive all three from
/// worker threads and the UI without rebuilding the subsystem.
pub struct GroundFactory {
    pending_mesh: PendingMeshInbox,
    pending_imagery: PendingImageryInbox,
    overlay: GroundOverlayController,
}

impl Default for GroundFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl GroundFactory {
    /// Construct a factory with fresh inboxes + overlay controller.
    pub fn new() -> Self {
        Self {
            pending_mesh: PendingMeshInbox::new(),
            pending_imagery: PendingImageryInbox::new(),
            overlay: GroundOverlayController::new(),
        }
    }

    /// Clone the mesh inbox for the terrain-fetch worker thread.
    pub fn mesh_inbox(&self) -> PendingMeshInbox {
        self.pending_mesh.clone()
    }

    /// Clone the imagery inbox for the imagery-fetch worker thread.
    pub fn imagery_inbox(&self) -> PendingImageryInbox {
        self.pending_imagery.clone()
    }

    /// Clone the overlay controller for the UI thread.
    pub fn overlay_controller(&self) -> GroundOverlayController {
        self.overlay.clone()
    }
}

impl SubsystemFactory for GroundFactory {
    fn name(&self) -> &'static str {
        "ground"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        Ok(Box::new(GroundSubsystem::with_inboxes(
            config,
            gpu,
            self.pending_mesh.clone(),
            self.pending_imagery.clone(),
            self.overlay.clone(),
        )))
    }
}
