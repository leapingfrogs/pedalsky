//! Phase 5 atmosphere subsystem (Hillaire 2020).
//!
//! Owns the [`AtmosphereLuts`] textures (group 3) and the four compute
//! pipelines that bake them, plus the sky raymarch fragment pipeline at
//! `PassStage::SkyBackdrop`.
//!
//! Pass schedule (registered with `App::frame`):
//! 1. `Compute` — transmittance LUT bake (only when atmosphere params change).
//! 2. `Compute` — multi-scatter LUT bake (only when atmosphere params change).
//! 3. `Compute` — sky-view LUT bake (every frame).
//! 4. `Compute` — aerial-perspective LUT bake (every frame).
//! 5. `SkyBackdrop` — fullscreen sky raymarch.
//!
//! The `AtmosphereLuts` bind group (group 3) is published on the host
//! side via [`AtmosphereSubsystem::luts`] each frame so the ground shader
//! can sample the AP LUT.

#![deny(missing_docs)]

pub mod lut_overlay;
pub mod lut_viewer;

use std::sync::{Arc, Mutex};

use ps_core::{
    atmosphere_lut_bind_group_layout, atmosphere_static_only_bind_group,
    atmosphere_static_only_bind_group_layout, atmosphere_transmittance_only_bind_group,
    atmosphere_transmittance_only_bind_group_layout, frame_bind_group_layout,
    world_bind_group_layout, AtmosphereLuts, BindGroupCache, Config, GpuContext,
    HdrFramebuffer, PassStage, PrepareContext, RegisteredPass, RenderSubsystem,
    SubsystemFactory,
};
use tracing::{debug, info};

/// Stable subsystem name (matches `[render.subsystems].atmosphere`).
pub const NAME: &str = "atmosphere";

// Baked shader sources (compile-time `include_str!`) paired with their
// path under `shaders/` for the hot-reload loader (plan
// §Cross-Cutting/Shader hot-reload).
const TRANSMITTANCE_BAKED: &str =
    include_str!("../../../shaders/atmosphere/transmittance.comp.wgsl");
const TRANSMITTANCE_REL: &str = "atmosphere/transmittance.comp.wgsl";
const MULTISCATTER_BAKED: &str =
    include_str!("../../../shaders/atmosphere/multiscatter.comp.wgsl");
const MULTISCATTER_REL: &str = "atmosphere/multiscatter.comp.wgsl";
const SKYVIEW_BAKED: &str = include_str!("../../../shaders/atmosphere/skyview.comp.wgsl");
const SKYVIEW_REL: &str = "atmosphere/skyview.comp.wgsl";
const AP_BAKED: &str = include_str!("../../../shaders/atmosphere/aerialperspective.comp.wgsl");
const AP_REL: &str = "atmosphere/aerialperspective.comp.wgsl";
const SKY_FS_BAKED: &str = include_str!("../../../shaders/atmosphere/sky.wgsl");
const SKY_FS_REL: &str = "atmosphere/sky.wgsl";

/// Per-LUT storage bind group: just one storage texture at binding 0.
fn storage_2d_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atmosphere::lut-storage-2d"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    })
}

fn storage_3d_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atmosphere::lut-storage-3d"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        }],
    })
}

/// Bind group 2 for the sky pass: top-down cloud density mask + a
/// linear-clamp sampler. Layout matches the binding declarations in
/// `shaders/atmosphere/sky.wgsl` (Phase 12.6b / followup #74).
fn sky_density_mask_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("atmosphere::sky-density-mask-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: true,
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Phase 5 atmosphere subsystem.
pub struct AtmosphereSubsystem {
    enabled: bool,
    /// LUT textures + sampling bind group (group 3).
    luts: Arc<AtmosphereLuts>,

    transmittance_pipeline: wgpu::ComputePipeline,
    multiscatter_pipeline: wgpu::ComputePipeline,
    skyview_pipeline: wgpu::ComputePipeline,
    ap_pipeline: wgpu::ComputePipeline,
    sky_pipeline: wgpu::RenderPipeline,

    /// Layout used to rebuild the sky pass's group-2 bind group each
    /// frame against the live density-mask view.
    sky_density_mask_layout: wgpu::BindGroupLayout,
    /// Cached sampler shared by every per-frame group-2 bind group.
    density_mask_sampler: wgpu::Sampler,

    /// Storage-target bind groups (group 2 of each compute pipeline).
    transmittance_storage: wgpu::BindGroup,
    multiscatter_storage: wgpu::BindGroup,
    skyview_storage: wgpu::BindGroup,
    ap_storage: wgpu::BindGroup,

    /// Sampling bind group with only the transmittance LUT — used by
    /// the multi-scatter bake (which writes to multi-scatter so can't
    /// have multi-scatter in the sampling group concurrently).
    sample_trans_only: wgpu::BindGroup,
    /// Sampling bind group with transmittance + multi-scatter — used by
    /// the sky-view and AP bakes (each writes its own LUT, but reads
    /// these two).
    sample_static_only: wgpu::BindGroup,

    /// True until the static LUTs (transmittance, multi-scatter) have been
    /// baked. They depend only on `WorldUniforms`/`AtmosphereParams`, so
    /// re-bake only on `reconfigure`.
    static_dirty: Mutex<bool>,

    /// Plan §10 tuning toggle: when false the multi-scatter LUT is
    /// cleared to zero so the sky-view bake's MS contribution
    /// collapses. The LUT texture is allocated either way. Behind an
    /// Arc so the multi-scatter pass closure can read it through a
    /// shared reference.
    tuning_multi_scattering: Arc<Mutex<bool>>,
    /// Sky-pass density-mask bind group cache, keyed on the weather
    /// revision. The mask view only changes when `ps-synthesis`
    /// reruns; the previous code rebuilt this bind group every frame.
    sky_density_mask_cache: Arc<Mutex<BindGroupCache<u64>>>,
}

impl AtmosphereSubsystem {
    /// Construct.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let luts = Arc::new(AtmosphereLuts::new(gpu));

        // Common helpers prepended to every shader.
        // The transmittance bake doesn't sample any LUT; everything else does.
        let common = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
        ]);
        let common_with_luts = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_LUT_SAMPLING_WGSL,
        ]);

        // Layouts.
        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let lut_sample_layout = atmosphere_lut_bind_group_layout(device);
        let trans_only_layout = atmosphere_transmittance_only_bind_group_layout(device);
        let static_only_layout = atmosphere_static_only_bind_group_layout(device);
        let storage_2d = storage_2d_layout(device);
        let storage_3d = storage_3d_layout(device);

        // Compute pipelines. Load each WGSL via the hot-reload-aware
        // loader; each returns the disk file when shader hot-reload is
        // armed and the baked source otherwise.
        let transmittance_src =
            ps_core::shaders::load_shader(TRANSMITTANCE_REL, TRANSMITTANCE_BAKED);
        let multiscatter_src =
            ps_core::shaders::load_shader(MULTISCATTER_REL, MULTISCATTER_BAKED);
        let skyview_src = ps_core::shaders::load_shader(SKYVIEW_REL, SKYVIEW_BAKED);
        let ap_src = ps_core::shaders::load_shader(AP_REL, AP_BAKED);
        let sky_fs_src = ps_core::shaders::load_shader(SKY_FS_REL, SKY_FS_BAKED);

        let transmittance_pipeline = make_compute_pipeline_with_optional(
            device,
            "atmosphere::transmittance",
            &common,
            &transmittance_src,
            // bind group 1 (world), 2 (storage). 0 unbound.
            &[None, Some(&world_layout), Some(&storage_2d)],
        );
        let multiscatter_pipeline = make_compute_pipeline_with_optional(
            device,
            "atmosphere::multiscatter",
            &common_with_luts,
            &multiscatter_src,
            // bind group 1 (world), 2 (storage), 3 (transmittance only). 0 unbound.
            &[
                None,
                Some(&world_layout),
                Some(&storage_2d),
                Some(&trans_only_layout),
            ],
        );
        let skyview_pipeline = make_compute_pipeline(
            device,
            "atmosphere::skyview",
            &common_with_luts,
            &skyview_src,
            // bind groups 0 (frame), 1 (world), 2 (storage), 3 (T+MS).
            &[
                &frame_layout,
                &world_layout,
                &storage_2d,
                &static_only_layout,
            ],
        );
        let ap_pipeline = make_compute_pipeline(
            device,
            "atmosphere::aerialperspective",
            &common_with_luts,
            &ap_src,
            &[
                &frame_layout,
                &world_layout,
                &storage_3d,
                &static_only_layout,
            ],
        );

        // Sky raymarch render pipeline.
        let sky_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atmosphere::sky.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                ps_core::shaders::compose(&[&common_with_luts, &sky_fs_src]).into(),
            ),
        });
        let sky_density_mask_layout = sky_density_mask_layout(device);
        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atmosphere::sky-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&sky_density_mask_layout),
                Some(&lut_sample_layout),
            ],
            immediate_size: 0,
        });
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("atmosphere::sky-rp"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_module,
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
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: HdrFramebuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Reverse-Z: write only at the absolute far plane (depth=0).
                // Use Always so fragment shader's depth=0 always passes,
                // and writes 0 — which subsequent ground passes (CompareGreater)
                // win against.
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Phase 12.6b — sampler shared by every per-frame sky-pass
        // density-mask bind group. Linear so cloud-edge transitions
        // smooth across mask cells; clamp-to-edge so views beyond the
        // 32 km mask extent get the boundary value.
        let density_mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere::sky-density-mask-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Sampling bind groups for the bake passes.
        let sample_trans_only = atmosphere_transmittance_only_bind_group(
            device,
            &trans_only_layout,
            &luts.transmittance_view,
            &luts.sampler,
        );
        let sample_static_only = atmosphere_static_only_bind_group(
            device,
            &static_only_layout,
            &luts.transmittance_view,
            &luts.multiscatter_view,
            &luts.sampler,
        );

        // Storage bind groups (one per LUT).
        let transmittance_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere::transmittance-storage-bg"),
            layout: &storage_2d,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&luts.transmittance_view),
            }],
        });
        let multiscatter_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere::multiscatter-storage-bg"),
            layout: &storage_2d,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&luts.multiscatter_view),
            }],
        });
        let skyview_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere::skyview-storage-bg"),
            layout: &storage_2d,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&luts.skyview_view),
            }],
        });
        let ap_storage = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmosphere::ap-storage-bg"),
            layout: &storage_3d,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&luts.aerial_perspective_view),
            }],
        });

        Self {
            enabled: true,
            luts,
            transmittance_pipeline,
            multiscatter_pipeline,
            skyview_pipeline,
            ap_pipeline,
            sky_pipeline,
            sky_density_mask_layout,
            density_mask_sampler,
            transmittance_storage,
            multiscatter_storage,
            skyview_storage,
            ap_storage,
            sample_trans_only,
            sample_static_only,
            static_dirty: Mutex::new(true),
            tuning_multi_scattering: Arc::new(Mutex::new(
                config.render.atmosphere.multi_scattering,
            )),
            sky_density_mask_cache: Arc::new(Mutex::new(BindGroupCache::new())),
        }
    }

    /// Reference to the atmosphere LUTs bundle for the host to publish
    /// into `PrepareContext::atmosphere_luts` and `RenderContext::luts_bind_group`.
    pub fn luts(&self) -> &Arc<AtmosphereLuts> {
        &self.luts
    }
}

fn make_compute_pipeline_with_optional(
    device: &wgpu::Device,
    label: &'static str,
    common: &str,
    main: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(ps_core::shaders::compose(&[common, main]).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    common: &str,
    main: &str,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(ps_core::shaders::compose(&[common, main]).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bind_group_layouts
            .iter()
            .map(|l| Some(*l))
            .collect::<Vec<_>>(),
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

impl RenderSubsystem for AtmosphereSubsystem {
    fn name(&self) -> &'static str {
        "atmosphere"
    }

    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {
        // No CPU prepare work — uniform writes happen in ps-app.
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        // We register the four LUT bakes + sky pass. Only sky-view + AP
        // re-run every frame; transmittance + multi-scatter only on first
        // frame (and on `reconfigure`). Track this via the shared
        // `static_dirty` flag.
        let static_dirty = Arc::new(Mutex::new(true));
        let static_dirty_check = static_dirty.clone();
        let static_dirty_init = static_dirty.clone();
        // Snapshot `self.static_dirty` once, then forget the original —
        // closures own their own copy after this.
        {
            let mut guard = self.static_dirty.lock().expect("atmosphere dirty lock");
            *static_dirty.lock().unwrap() = *guard;
            *guard = false; // Compute pipelines below latch the dirty bit.
            let _ = static_dirty_init;
        }

        let trans_pipeline = self.transmittance_pipeline.clone();
        let trans_storage = self.transmittance_storage.clone();
        let ms_pipeline = self.multiscatter_pipeline.clone();
        let ms_storage = self.multiscatter_storage.clone();
        let ms_texture = self.luts.multiscatter.clone();
        let ms_toggle = self.tuning_multi_scattering.clone();
        let sv_pipeline = self.skyview_pipeline.clone();
        let sv_storage = self.skyview_storage.clone();
        let ap_pipeline = self.ap_pipeline.clone();
        let ap_storage = self.ap_storage.clone();
        let sample_trans_only = self.sample_trans_only.clone();
        let sample_static_only = self.sample_static_only.clone();
        let lut_bg = self.luts.bind_group.clone();
        let sky_pipeline = self.sky_pipeline.clone();

        vec![
            // 1. Transmittance — only when dirty.
            RegisteredPass {
                name: "atmosphere::transmittance",
                stage: PassStage::Compute,
                run: Box::new({
                    let dirty = static_dirty_check.clone();
                    move |encoder, ctx| {
                        if !*dirty.lock().unwrap() {
                            return;
                        }
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("atmosphere::transmittance-bake"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&trans_pipeline);
                        pass.set_bind_group(1, ctx.world_bind_group, &[]);
                        pass.set_bind_group(2, &trans_storage, &[]);
                        pass.dispatch_workgroups(32, 8, 1); // 256/8, 64/8
                    }
                }),
            },
            // 2. Multi-scatter — only when dirty. Latches the dirty bit
            //    off after running, so subsequent frames skip the bake.
            //    When the multi_scattering tuning toggle is off, clear
            //    the LUT to zero instead of dispatching the bake.
            RegisteredPass {
                name: "atmosphere::multiscatter",
                stage: PassStage::Compute,
                run: Box::new({
                    let dirty = static_dirty_check.clone();
                    let bg = sample_trans_only.clone();
                    let toggle = ms_toggle.clone();
                    let texture = ms_texture.clone();
                    move |encoder, ctx| {
                        let mut g = dirty.lock().unwrap();
                        if !*g {
                            return;
                        }
                        let on = *toggle.lock().expect("ms toggle lock");
                        if on {
                            let mut pass = encoder.begin_compute_pass(
                                &wgpu::ComputePassDescriptor {
                                    label: Some("atmosphere::multiscatter-bake"),
                                    timestamp_writes: None,
                                },
                            );
                            pass.set_pipeline(&ms_pipeline);
                            pass.set_bind_group(1, ctx.world_bind_group, &[]);
                            pass.set_bind_group(2, &ms_storage, &[]);
                            pass.set_bind_group(3, &bg, &[]);
                            pass.dispatch_workgroups(4, 4, 1); // 32/8, 32/8
                        } else {
                            // Multi-scatter disabled: zero the 32x32
                            // Rgba16Float LUT so the sky-view bake's
                            // MS contribution collapses to 0.
                            let zeros = vec![0u8; 32 * 32 * 8];
                            ctx.queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &texture,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                &zeros,
                                wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(32 * 8),
                                    rows_per_image: Some(32),
                                },
                                wgpu::Extent3d {
                                    width: 32,
                                    height: 32,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                        *g = false;
                    }
                }),
            },
            // 3. Sky-view — every frame.
            RegisteredPass {
                name: "atmosphere::skyview",
                stage: PassStage::Compute,
                run: Box::new({
                    let bg = sample_static_only.clone();
                    move |encoder, ctx| {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("atmosphere::skyview-bake"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&sv_pipeline);
                        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                        pass.set_bind_group(1, ctx.world_bind_group, &[]);
                        pass.set_bind_group(2, &sv_storage, &[]);
                        pass.set_bind_group(3, &bg, &[]);
                        pass.dispatch_workgroups(24, 14, 1); // 192/8, 108/8 round up
                    }
                }),
            },
            // 4. Aerial-perspective — every frame.
            RegisteredPass {
                name: "atmosphere::aerialperspective",
                stage: PassStage::Compute,
                run: Box::new({
                    let bg = sample_static_only.clone();
                    move |encoder, ctx| {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("atmosphere::ap-bake"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&ap_pipeline);
                        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                        pass.set_bind_group(1, ctx.world_bind_group, &[]);
                        pass.set_bind_group(2, &ap_storage, &[]);
                        pass.set_bind_group(3, &bg, &[]);
                        // 32/4, 32/4, 64/4 — Phase 13.1 doubled
                        // the Z extent to 64 slices.
                        pass.dispatch_workgroups(8, 8, 16);
                    }
                }),
            },
            // 5. Sky raymarch fragment pass at SkyBackdrop.
            RegisteredPass {
                name: "atmosphere::sky",
                stage: PassStage::SkyBackdrop,
                run: Box::new({
                    let lut_bg = lut_bg.clone();
                    let mask_layout = self.sky_density_mask_layout.clone();
                    let mask_sampler = self.density_mask_sampler.clone();
                    let mask_cache = self.sky_density_mask_cache.clone();
                    move |encoder, ctx| {
                        // The density-mask bind group only changes when
                        // synthesis reruns (bumping `weather.revision`).
                        // Cache by revision so the per-frame fast path
                        // is an `Arc::clone` rather than a wgpu hub
                        // touch.
                        let density_bg = mask_cache
                            .lock()
                            .expect("atmosphere: sky_density_mask_cache lock")
                            .get_or_build(ctx.weather.revision, || {
                                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some("atmosphere::sky-density-mask-bg"),
                                    layout: &mask_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: wgpu::BindingResource::TextureView(
                                                &ctx.weather.textures
                                                    .top_down_density_mask_view,
                                            ),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: wgpu::BindingResource::Sampler(
                                                &mask_sampler,
                                            ),
                                        },
                                    ],
                                })
                            });

                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("atmosphere::sky"),
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
                        pass.set_pipeline(&sky_pipeline);
                        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                        pass.set_bind_group(1, ctx.world_bind_group, &[]);
                        pass.set_bind_group(2, density_bg.as_ref(), &[]);
                        pass.set_bind_group(3, &lut_bg, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }),
            },
        ]
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        // Mark static LUTs dirty so they re-bake on the next frame
        // (caller invoked us because some atmosphere parameter changed).
        *self.static_dirty.lock().expect("dirty lock") = true;
        // Pull the multi-scatter tuning toggle. The skyview pass
        // unconditionally samples the MS LUT; when this is off we
        // overwrite the LUT with zeros after the regular bake (see
        // `register_passes`).
        *self
            .tuning_multi_scattering
            .lock()
            .expect("ms tuning lock") = config.render.atmosphere.multi_scattering;
        debug!("atmosphere: reconfigure → static LUTs marked dirty");
        Ok(())
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Factory wired by `AppBuilder`. Constructs the live subsystem and
/// publishes a clone of its `Arc<AtmosphereLuts>` via the channel passed
/// at registration; ps-app reads the channel to plumb the LUTs into
/// `PrepareContext` / `RenderContext`.
pub struct AtmosphereFactory {
    /// Shared cell where the constructed subsystem deposits its LUTs handle.
    pub luts_publish: Arc<Mutex<Option<Arc<AtmosphereLuts>>>>,
}

impl AtmosphereFactory {
    /// Build an `AtmosphereFactory` paired with a publish cell the host
    /// reads after `AppBuilder::build` to pick up the LUTs handle.
    pub fn new() -> (Self, Arc<Mutex<Option<Arc<AtmosphereLuts>>>>) {
        let cell = Arc::new(Mutex::new(None));
        (
            Self {
                luts_publish: cell.clone(),
            },
            cell,
        )
    }
}

impl SubsystemFactory for AtmosphereFactory {
    fn name(&self) -> &'static str {
        "atmosphere"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        let subsys = AtmosphereSubsystem::new(config, gpu);
        // Publish the LUTs handle so ps-app can plumb it into
        // PrepareContext + RenderContext + ground's bind group 3.
        *self
            .luts_publish
            .lock()
            .map_err(|e| anyhow::anyhow!("luts_publish lock poisoned: {e}"))? =
            Some(subsys.luts().clone());
        info!(target: "ps_atmosphere", "atmosphere LUTs allocated and published");
        Ok(Box::new(subsys))
    }
}
