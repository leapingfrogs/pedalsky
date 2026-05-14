//! PedalSky volumetric cloud subsystem (Phase 6: Schneider/Hillaire).
//!
//! Implements Schneider Nubis (2015/2017/2022) base/detail/curl noise
//! volumes plus the Hillaire 2016 multi-octave multiple-scattering
//! approximation. Wired as a `Translucent`-stage `RenderSubsystem`
//! that composites premultiplied cloud luminance over the HDR target.
//!
//! Pass schedule:
//! - `Translucent` — cloud raymarch into a dedicated cloud RT.
//! - `Translucent` — composite cloud RT over HDR with premultiplied
//!   `One, OneMinusSrcAlpha` blend.

#![deny(missing_docs)]

pub mod noise;
pub mod params;
pub mod pipeline;

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use ps_core::{
    AtmosphereLuts, BindGroupCache, CloudLayerGpu, Config, GpuContext, HdrFramebuffer,
    PassDescriptor, PassId, PassStage, PrepareContext, RenderContext, RenderSubsystem,
    SubsystemFactory,
};
use tracing::warn;

/// Subsystem-local pass IDs. Their numeric values are passed back into
/// [`CloudsSubsystem::dispatch_pass`] so the match branches on them.
const PASS_MARCH: PassId = 0;
const PASS_TAA: PassId = 1;
const PASS_COMPOSITE: PassId = 2;

/// Scale factor applied to the framebuffer size when
/// `clouds.half_res_render` is enabled. 0.5 = 1/4 the area.
const HALF_RES_SCALE: f32 = 0.5;

/// TAA exponential-blend weight on the **current** sample once the
/// history is valid. 1/8 ≈ 87.5% history, 12.5% current — gives
/// roughly 8-frame accumulation. Matches Frostbite / Decima defaults.
const TAA_BLEND_WEIGHT: f32 = 1.0 / 8.0;

/// Cache-key tuple for the cloud-march `data_bg` cache: weather
/// revision, framebuffer width, framebuffer height, half-res flag.
/// Captures every input that can change which bind-group object
/// the march should be using.
type DataBgKey = (u64, u32, u32, bool);

/// CPU mirror of the WGSL `CloudCompositeParams` uniform. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct CloudCompositeParamsGpu {
    /// `(1/w, 1/h, w, h)` of the cloud RT (texel space).
    cloud_rt_size: [f32; 4],
    /// 0 = bilinear passthrough, 1 = Catmull-Rom 9-tap.
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// CPU mirror of the WGSL `CloudTaaParams` uniform. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct CloudTaaParamsGpu {
    /// `.x` = blend weight on the current sample (0.0–1.0; 1.0 means
    /// "ignore history"). `.y` = history_valid (1.0 if reprojection
    /// is meaningful, 0.0 if not). `.zw` reserved.
    config: [f32; 4],
}

pub use noise::CloudNoise;
pub use params::{CloudParamsGpu, MAX_CLOUD_LAYERS};
pub use pipeline::CloudPipelines;

/// Stable subsystem name (matches `[render.subsystems].clouds`).
pub const NAME: &str = "clouds";

/// Phase 6 volumetric cloud subsystem.
///
/// Per audit §2.4, this used to hold ~9 `Arc<Mutex<…>>` / `Arc<Atomic*>`
/// cells purely so the per-pass `Box<dyn Fn>` closures could share
/// mutable state. With `dispatch_pass(&mut self)` those collapse to
/// plain fields — the per-frame mutex traffic across the cloud passes
/// drops to zero.
pub struct CloudsSubsystem {
    noise: CloudNoise,
    params: CloudParamsGpu,
    cpu_layers: [CloudLayerGpu; MAX_CLOUD_LAYERS as usize],

    /// Cloud RT bundle (recreated on resize).
    rt: CloudRt,

    /// Cached pipelines (depend only on shader source).
    pipelines: CloudPipelines,

    /// LUTs handle (populated by the factory before the subsystem is
    /// installed into the App). `None` if atmosphere is disabled — in
    /// which case the cloud-march pass skips rendering.
    luts: Option<Arc<AtmosphereLuts>>,

    /// Per-frame uniform buffer for CloudParamsGpu.
    params_buffer: wgpu::Buffer,
    /// Per-frame storage buffer for the layer array.
    layers_buffer: wgpu::Buffer,
    /// Per-frame uniform for `CloudCompositeParams` (cloud RT size +
    /// upsample mode flag). Rebuilt when `half_res_render` toggles or
    /// when the framebuffer resizes.
    composite_params_buffer: wgpu::Buffer,
    /// Per-frame uniform for the TAA pass — blend weight + history
    /// validity flag.
    taa_params_buffer: wgpu::Buffer,
    /// Live half-res toggle, mirrored from `clouds.half_res_render` in
    /// `reconfigure`. Drives both the cloud RT allocation size and the
    /// composite shader's upsample mode.
    half_res_render: bool,
    /// Live TAA toggle. Pulled from `clouds.temporal_taa`; auto-gated
    /// off in the march pass when `freeze_time` is set (so paused
    /// screenshots reflect a single frame rather than a temporal
    /// blend).
    temporal_taa: bool,
    /// Tracks the previous frame's effective `taa_on` so the march
    /// pass can detect off→on transitions and discard the now-stale
    /// history.
    prev_taa_on: bool,
    /// Per-frame TAA dispatch state — set by the march pass for the
    /// subsequent TAA + composite passes to read. `None` means "TAA
    /// off this frame"; `Some(slot)` is the history-slot index that
    /// the TAA pass will write into and the composite pass will read
    /// from.
    taa_dispatch: Option<u32>,
    /// Cache for the cloud-march group-2 bind group. Keyed on
    /// `(weather.revision, fb_w, fb_h, half_res)` because the bind
    /// group's entries depend on: the weather-state texture views
    /// (revision), the framebuffer's depth view (resize → size
    /// change), and the layout choice (half_res selects between
    /// `noise.layout` and `noise.layout_halfres`). With this cache
    /// the 12-13 entry bind group only gets rebuilt on synthesis,
    /// window resize, or the half-res toggle — not every frame.
    data_bg_cache: BindGroupCache<DataBgKey>,
    /// Plan §6.9 freeze-time toggle. The cloud march itself doesn't
    /// read `simulated_seconds` (the base/detail/curl noise volumes are
    /// purely spatial — plan principle #9 / Phase 6 design), so this
    /// flag only gates TAA on/off.
    freeze_time: bool,
    /// Audit §L2 — the cloud-layer Vec only changes when synthesis
    /// reruns (`weather.revision` bump). Tracking the revision lets
    /// `prepare()` skip the per-frame `cpu_layers` zero-fill +
    /// element copy + 256 B `queue.write_buffer` of the layers
    /// storage buffer.
    layers_uploaded_at_revision: Option<u64>,
}

/// Cloud render target bundle. Two `Rgba16Float` attachments —
/// luminance (premultiplied along the ray, AP-applied) and RGB
/// transmittance through the cloud column. The composite pass reads
/// both and uses dual-source blending to apply
/// `dst = luminance + dst * transmittance` per channel into the HDR
/// target (plan §6.6 RGB transmittance / Phase 12.2).
pub struct CloudRt {
    /// Premultiplied luminance attachment (Rgba16Float; matches HDR).
    /// Cloud march writes to this each frame; with TAA off the
    /// composite reads it directly, with TAA on the TAA pass reads
    /// it as the "current frame" input.
    luminance: wgpu::Texture,
    /// View on `luminance`.
    luminance_view: wgpu::TextureView,
    /// RGB transmittance attachment (Rgba16Float; .a unused).
    transmittance: wgpu::Texture,
    /// View on `transmittance`.
    transmittance_view: wgpu::TextureView,
    /// Bind group used by the full-res composite pass when TAA is
    /// **off** — reads the scratch (luminance + transmittance) +
    /// shared sampler.
    composite_bg: wgpu::BindGroup,
    /// Bind group used by the half-res composite pass when TAA is
    /// **off** — same texture sources as `composite_bg` plus a
    /// `CloudCompositeParams` uniform binding.
    composite_halfres_bg: wgpu::BindGroup,
    /// TAA ping-pong history pair + bind groups. `None` when TAA
    /// has never been enabled; allocated lazily on the first frame
    /// the toggle is on. The fields are kept inside `Option` so a
    /// user who never touches TAA pays zero VRAM cost (an extra
    /// 4 RTs at cloud-RT size).
    taa: Option<TaaState>,
    /// Pixel size — used to detect framebuffer resize.
    size: (u32, u32),
    /// Cached sampler so the bind-group rebuild on resize is cheap.
    sampler: wgpu::Sampler,
}

/// Ping-pong history textures + per-slot bind groups for the TAA pass.
/// Each slot holds the resolved (luminance, transmittance) pair from
/// one frame, written by the TAA shader and consumed by the next
/// frame's TAA shader (and by the current frame's composite, which
/// reads the just-written slot when TAA is on).
struct TaaState {
    history: [TaaHistoryPair; 2],
    /// Bind group selected by `write_slot`: reads scratch + reads
    /// `history[1 - write_slot]`. `taa_input_bg[i]` corresponds to
    /// the pass that WRITES to `history[i]`.
    taa_input_bg: [wgpu::BindGroup; 2],
    /// Composite bind groups that read from each history slot
    /// (full-res variant — 3 bindings).
    composite_bg: [wgpu::BindGroup; 2],
    /// Composite bind groups that read from each history slot
    /// (half-res variant — 4 bindings; the 4th is the
    /// `CloudCompositeParams` uniform shared with the non-TAA path).
    composite_halfres_bg: [wgpu::BindGroup; 2],
    /// Index into `history[]` that the upcoming frame writes to. Flips
    /// every TAA frame so the read/write sides ping-pong.
    write_slot: u32,
    /// `true` once at least one valid resolved frame has been written.
    /// Reset to `false` on size change or when TAA is freshly enabled.
    /// The shader treats `false` as "no history" and outputs the
    /// current sample verbatim.
    history_valid: bool,
}

struct TaaHistoryPair {
    #[allow(dead_code)]
    luminance: wgpu::Texture,
    luminance_view: wgpu::TextureView,
    #[allow(dead_code)]
    transmittance: wgpu::Texture,
    transmittance_view: wgpu::TextureView,
}

impl CloudRt {
    fn new(
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        composite_halfres_layout: &wgpu::BindGroupLayout,
        composite_params: &wgpu::Buffer,
        (w, h): (u32, u32),
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clouds-rt-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self::build(
            device,
            composite_layout,
            composite_halfres_layout,
            composite_params,
            &sampler,
            (w, h),
        )
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        composite_halfres_layout: &wgpu::BindGroupLayout,
        composite_params: &wgpu::Buffer,
        size: (u32, u32),
    ) {
        let new = Self::build(
            device,
            composite_layout,
            composite_halfres_layout,
            composite_params,
            &self.sampler,
            size,
        );
        self.luminance = new.luminance;
        self.luminance_view = new.luminance_view;
        self.transmittance = new.transmittance;
        self.transmittance_view = new.transmittance_view;
        self.composite_bg = new.composite_bg;
        self.composite_halfres_bg = new.composite_halfres_bg;
        self.size = size;
        // Any cached TAA history is at the old size and now references
        // dangling views. Drop it; the next TAA-on frame reallocates
        // and sets `history_valid = false` so the shader bootstraps.
        self.taa = None;
    }

    fn build(
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        composite_halfres_layout: &wgpu::BindGroupLayout,
        composite_params: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        (w, h): (u32, u32),
    ) -> Self {
        let make_attachment = |label: &'static str| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HdrFramebuffer::COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (luminance, luminance_view) = make_attachment("clouds-rt-luminance");
        let (transmittance, transmittance_view) =
            make_attachment("clouds-rt-transmittance");
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-composite-bg"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&luminance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let composite_halfres_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-composite-halfres-bg"),
            layout: composite_halfres_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&luminance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: composite_params.as_entire_binding(),
                },
            ],
        });
        Self {
            luminance,
            luminance_view,
            transmittance,
            transmittance_view,
            composite_bg,
            composite_halfres_bg,
            taa: None,
            size: (w.max(1), h.max(1)),
            sampler: sampler.clone(),
        }
    }

    /// Allocate (or reallocate) the TAA history pair + bind groups.
    /// Called by the cloud-march pass closure the first frame TAA is
    /// active, and again whenever the cloud RT resizes or TAA is
    /// freshly re-enabled. Marks `history_valid = false` so the
    /// shader bootstraps from the current sample on this frame.
    fn enable_taa(
        &mut self,
        device: &wgpu::Device,
        taa_layout: &wgpu::BindGroupLayout,
        composite_layout: &wgpu::BindGroupLayout,
        composite_halfres_layout: &wgpu::BindGroupLayout,
        composite_params: &wgpu::Buffer,
        taa_params: &wgpu::Buffer,
    ) {
        let (w, h) = self.size;
        let make_pair = |label_lum: &'static str, label_trans: &'static str| {
            let make = |label: &'static str| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w.max(1),
                        height: h.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HdrFramebuffer::COLOR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (tex, view)
            };
            let (lum, lum_view) = make(label_lum);
            let (trans, trans_view) = make(label_trans);
            TaaHistoryPair {
                luminance: lum,
                luminance_view: lum_view,
                transmittance: trans,
                transmittance_view: trans_view,
            }
        };
        let history = [
            make_pair("clouds-taa-history-0-lum", "clouds-taa-history-0-trans"),
            make_pair("clouds-taa-history-1-lum", "clouds-taa-history-1-trans"),
        ];

        // Build the TAA input bind groups. `taa_input_bg[i]` is bound
        // when the pass writes to `history[i]`; it therefore reads
        // from `history[1 - i]`.
        let make_taa_input_bg = |read_slot: usize| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clouds-taa-input-bg"),
                layout: taa_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.luminance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &history[read_slot].luminance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &history[read_slot].transmittance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: taa_params.as_entire_binding(),
                    },
                ],
            })
        };
        // taa_input_bg[0] writes history[0] → reads history[1].
        // taa_input_bg[1] writes history[1] → reads history[0].
        let taa_input_bg = [make_taa_input_bg(1), make_taa_input_bg(0)];

        // Composite bind groups that read each history slot.
        let make_composite_bg = |slot: usize| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clouds-composite-taa-bg"),
                layout: composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &history[slot].luminance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &history[slot].transmittance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            })
        };
        let make_composite_halfres_bg = |slot: usize| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clouds-composite-taa-halfres-bg"),
                layout: composite_halfres_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &history[slot].luminance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &history[slot].transmittance_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: composite_params.as_entire_binding(),
                    },
                ],
            })
        };
        let composite_bg = [make_composite_bg(0), make_composite_bg(1)];
        let composite_halfres_bg = [
            make_composite_halfres_bg(0),
            make_composite_halfres_bg(1),
        ];

        self.taa = Some(TaaState {
            history,
            taa_input_bg,
            composite_bg,
            composite_halfres_bg,
            write_slot: 0,
            history_valid: false,
        });
    }
}

impl CloudsSubsystem {
    /// Construct the subsystem.
    pub fn new(config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let noise = CloudNoise::bake(gpu);
        let pipelines = CloudPipelines::new(
            device,
            &noise.layout,
            &noise.layout_halfres,
        );

        // Seed CloudParamsGpu from config so the very first frame
        // honours the user's initial cloud_steps / detail_strength etc.
        let mut params = CloudParamsGpu::default();
        let c = &config.render.clouds;
        params.cloud_steps = c.cloud_steps;
        params.light_steps = c.light_steps;
        params.multi_scatter_octaves = c.multi_scatter_octaves;
        params.detail_strength = c.detail_strength;
        params.powder_strength = c.powder_strength;
        params.droplet_diameter_bias = c.droplet_diameter_bias;
        params.cone_light_sampling = u32::from(c.cone_light_sampling);
        params.forward_bias = c.forward_bias;
        // Phase 13.9 — temporal jitter is gated by both the live flag
        // and `freeze_time` (paused screenshots must not shimmer).
        params.temporal_jitter = u32::from(c.temporal_jitter && !c.freeze_time);
        // Phase 14.C — wind drift follows the same freeze_time pattern
        // so paused screenshots don't continue to advect noise after
        // the user clicks pause.
        params.wind_drift_strength = if c.freeze_time { 0.0 } else { c.wind_drift_strength };
        // Phase 18 — diurnal modulation is sun-direction driven, not
        // simulated_seconds driven, so freeze_time doesn't need to
        // gate it (the world clock pausing already freezes the sun).
        // Users wanting a documentation screenshot at full character
        // regardless of time-of-day can set `diurnal_strength = 0`
        // explicitly via the slider.
        params.diurnal_strength = c.diurnal_strength;
        // Phase 14.H — skew with height is a spatial effect (not
        // time-driven), so it stays on while paused. Users who want
        // a "no lean" screenshot can set wind_skew_strength = 0
        // explicitly.
        params.wind_skew_strength = c.wind_skew_strength;
        let mut cpu_layers = [CloudLayerGpu::zeroed(); MAX_CLOUD_LAYERS as usize];
        // Default to a single demonstration cumulus layer so the subsystem
        // produces visible output before WeatherState is wired in.
        cpu_layers[0] = CloudLayerGpu {
            base_m: 1500.0,
            top_m: 3500.0,
            coverage: 0.4,
            density_scale: 1.0,
            cloud_type: 0,
            shape_bias: 0.0,
            detail_bias: 0.0,
            anvil_bias: 0.0,
            // Cumulus water-droplet effective diameter (~20 µm).
            // Drives the Approximate Mie phase function in the cloud
            // march. Matches
            // `ps_core::default_droplet_diameter_um(CloudType::Cumulus)`
            // — duplicated rather than imported to keep the crate
            // boundary clean.
            droplet_diameter_um: 20.0,
            _pad_after_droplets_0: 0.0,
            _pad_after_droplets_1: 0.0,
            _pad_after_droplets_2: 0.0,
        };

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-params-ub"),
            size: std::mem::size_of::<CloudParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layers_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-layers-sb"),
            size: (std::mem::size_of::<CloudLayerGpu>() * MAX_CLOUD_LAYERS as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let composite_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-composite-params-ub"),
            size: std::mem::size_of::<CloudCompositeParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let taa_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-taa-params-ub"),
            size: std::mem::size_of::<CloudTaaParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rt = CloudRt::new(
            device,
            &pipelines.composite_layout,
            &pipelines.composite_halfres_layout,
            &composite_params_buffer,
            (1, 1),
        );

        Self {
            noise,
            params,
            cpu_layers,
            rt,
            pipelines,
            luts: None,
            params_buffer,
            layers_buffer,
            composite_params_buffer,
            taa_params_buffer,
            half_res_render: c.half_res_render,
            temporal_taa: c.temporal_taa,
            prev_taa_on: false,
            taa_dispatch: None,
            data_bg_cache: BindGroupCache::new(),
            freeze_time: c.freeze_time,
            layers_uploaded_at_revision: None,
        }
    }

    /// Reference to the baked noise textures (for diagnostic readback).
    pub fn noise(&self) -> &CloudNoise {
        &self.noise
    }

    /// Plumb the atmosphere LUTs reference (called by the factory once
    /// AtmosphereSubsystem has published its bundle, before the
    /// subsystem is installed into the App).
    pub fn set_atmosphere_luts(&mut self, luts: Arc<AtmosphereLuts>) {
        self.luts = Some(luts);
    }

    /// Cloud raymarch into the cloud RT.
    fn dispatch_march(&mut self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext<'_>) {
        let Some(luts) = self.luts.as_ref() else {
            // Atmosphere disabled. The TAA + composite passes still
            // run; clear `taa_dispatch` so the TAA pass skips and the
            // composite reads the (possibly stale) scratch as before.
            self.taa_dispatch = None;
            return;
        };
        let fb_size = (ctx.framebuffer.size.0, ctx.framebuffer.size.1);
        // Cloud RT dimensions = framebuffer × scale. Half-res reduces
        // both axes by `HALF_RES_SCALE` (= 0.5 ⇒ 1/4 the area, 1/3-1/4
        // the cloud-march cost). `.max(1)` so a tiny window doesn't
        // produce a zero-sized RT.
        let half_res = self.half_res_render;
        let rt_size = if half_res {
            (
                ((fb_size.0 as f32 * HALF_RES_SCALE) as u32).max(1),
                ((fb_size.1 as f32 * HALF_RES_SCALE) as u32).max(1),
            )
        } else {
            fb_size
        };
        if self.rt.size != rt_size {
            self.rt.rebuild(
                ctx.device,
                &self.pipelines.composite_layout,
                &self.pipelines.composite_halfres_layout,
                &self.composite_params_buffer,
                rt_size,
            );
        }

        // TAA bookkeeping. Active when the user toggle is on and
        // `freeze_time` is off. Allocate the history textures lazily
        // on first activation, and flip the ping-pong slot at the
        // start of every TAA frame. The chosen slot is latched in
        // `self.taa_dispatch` for the subsequent TAA + composite
        // passes to read.
        let taa_on = self.temporal_taa && !self.freeze_time;
        let just_enabled = taa_on && !self.prev_taa_on;
        self.prev_taa_on = taa_on;
        let taa_dispatch_slot: Option<u32> = if taa_on {
            if self.rt.taa.is_none() {
                self.rt.enable_taa(
                    ctx.device,
                    &self.pipelines.taa_layout,
                    &self.pipelines.composite_layout,
                    &self.pipelines.composite_halfres_layout,
                    &self.composite_params_buffer,
                    &self.taa_params_buffer,
                );
            }
            let taa_state = self.rt.taa.as_mut().expect("just-allocated TAA state");
            // Off→on transition: discard any leftover history (the
            // camera may have moved arbitrarily far while TAA was
            // off, so reprojection of the old data would produce
            // ghosting). Bootstrap from the current sample this
            // frame.
            if just_enabled {
                taa_state.history_valid = false;
            }
            // Flip ping-pong on every TAA frame so the pass's write
            // target is opposite to the history read target. On the
            // very first frame after activation, both slots are
            // stale and `history_valid = false`; the shader bootstraps
            // from the current sample.
            let write_slot = if taa_state.history_valid {
                taa_state.write_slot ^ 1
            } else {
                0
            };
            taa_state.write_slot = write_slot;
            let blend_weight = if taa_state.history_valid {
                TAA_BLEND_WEIGHT
            } else {
                1.0
            };
            let history_valid_flag = if taa_state.history_valid { 1.0 } else { 0.0 };
            let taa_params = CloudTaaParamsGpu {
                config: [blend_weight, history_valid_flag, 0.0, 0.0],
            };
            ctx.queue.write_buffer(
                &self.taa_params_buffer,
                0,
                bytemuck::bytes_of(&taa_params),
            );
            // After this frame's TAA pass writes the new resolved
            // slot, next frame's history is unconditionally valid.
            taa_state.history_valid = true;
            Some(write_slot)
        } else {
            None
        };
        self.taa_dispatch = taa_dispatch_slot;

        // Upload the cloud-RT-size uniform. The buffer is bound twice
        // — at binding 12 of the half-res march pipeline's data bind
        // group (where the patched shader reads `cloud_rt_uniform.xy`
        // for NDC, AP-UV, and depth-scale math) and at binding 3 of
        // the half-res composite bind group (where the Catmull-Rom
        // kernel reads `.zw` for the texel-size weights). The
        // full-res pipelines don't read this buffer; binding it for
        // them is a no-op.
        let rt_size_f32 = [rt_size.0 as f32, rt_size.1 as f32];
        let composite_params = CloudCompositeParamsGpu {
            cloud_rt_size: [
                rt_size_f32[0],
                rt_size_f32[1],
                1.0 / rt_size_f32[0],
                1.0 / rt_size_f32[1],
            ],
            mode: u32::from(half_res),
            ..Default::default()
        };
        ctx.queue.write_buffer(
            &self.composite_params_buffer,
            0,
            bytemuck::bytes_of(&composite_params),
        );
        // Build the group-2 bind group via the cache. Keyed on
        // (weather.revision, fb_w, fb_h, half_res) so the 12–13 entry
        // bind group only gets rebuilt on synthesis, window resize,
        // or the half-res toggle — not every frame. The full-res and
        // half-res pipelines use different bind-group layouts, so
        // the cache key includes `half_res` to fire whenever the
        // layout choice flips.
        let cache_key = (
            ctx.weather.revision,
            ctx.framebuffer.size.0,
            ctx.framebuffer.size.1,
            half_res,
        );
        let noise = &self.noise;
        let params_buffer = &self.params_buffer;
        let layers_buffer = &self.layers_buffer;
        let composite_params_buffer = &self.composite_params_buffer;
        let data_bg = self.data_bg_cache.get_or_build(cache_key, || {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&noise.base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise.detail_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&noise.curl_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&noise.blue_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&noise.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&noise.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(
                        &ctx.weather.textures.weather_map_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&ctx.framebuffer.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(
                        &ctx.weather.textures.cloud_type_grid_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(
                        &ctx.weather.textures.wind_field_view,
                    ),
                },
            ];
            let data_layout = if half_res {
                entries.push(wgpu::BindGroupEntry {
                    binding: 12,
                    resource: composite_params_buffer.as_entire_binding(),
                });
                &noise.layout_halfres
            } else {
                &noise.layout
            };
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clouds-data-bg"),
                layout: data_layout,
                entries: &entries,
            })
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clouds::march"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.rt.luminance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.rt.transmittance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let march_pipeline = if half_res {
            &self.pipelines.march_halfres
        } else {
            &self.pipelines.march
        };
        pass.set_pipeline(march_pipeline);
        pass.set_bind_group(0, ctx.frame_bind_group, &[]);
        pass.set_bind_group(1, ctx.world_bind_group, &[]);
        pass.set_bind_group(2, data_bg.as_ref(), &[]);
        pass.set_bind_group(3, &luts.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// TAA blend pass — runs only when TAA is active this frame
    /// (toggle on + freeze_time off). Reads the current scratch +
    /// previous history slot, writes the new resolved slot. Skipped
    /// via early return when `taa_dispatch` is None.
    fn dispatch_taa(&mut self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext<'_>) {
        let Some(write_slot) = self.taa_dispatch else {
            return;
        };
        let Some(taa_state) = self.rt.taa.as_ref() else {
            return;
        };
        let target = &taa_state.history[write_slot as usize];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clouds::taa"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.luminance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.transmittance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipelines.taa);
        pass.set_bind_group(0, &taa_state.taa_input_bg[write_slot as usize], &[]);
        pass.set_bind_group(1, ctx.frame_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Composite the cloud RT (or TAA-resolved history) over the HDR
    /// target.
    fn dispatch_composite(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        let half_res = self.half_res_render;
        let taa_slot = self.taa_dispatch;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clouds::composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.framebuffer.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Pick the right (pipeline, bind group) pair from the
        // cartesian of (full-res / half-res) × (TAA off / TAA on).
        // The TAA-on bind groups point at the just-written history
        // slot; the TAA-off bind groups point at the scratch.
        if half_res {
            pass.set_pipeline(&self.pipelines.composite_halfres);
            if let (Some(slot), Some(taa_state)) = (taa_slot, self.rt.taa.as_ref()) {
                pass.set_bind_group(0, &taa_state.composite_halfres_bg[slot as usize], &[]);
            } else {
                pass.set_bind_group(0, &self.rt.composite_halfres_bg, &[]);
            }
        } else {
            pass.set_pipeline(&self.pipelines.composite);
            if let (Some(slot), Some(taa_state)) = (taa_slot, self.rt.taa.as_ref()) {
                pass.set_bind_group(0, &taa_state.composite_bg[slot as usize], &[]);
            } else {
                pass.set_bind_group(0, &self.rt.composite_bg, &[]);
            }
        }
        pass.draw(0..3, 0..1);
    }
}

impl RenderSubsystem for CloudsSubsystem {
    fn name(&self) -> &'static str {
        "clouds"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // Audit §L2 — pull synthesised layers + write the layers
        // storage buffer only when synthesis has rerun (revision bump).
        // The Vec content is otherwise stable across frames.
        let revision = ctx.weather.revision;
        let layers_changed = self.layers_uploaded_at_revision != Some(revision);
        if layers_changed {
            if !ctx.weather.cloud_layers.is_empty() {
                self.cpu_layers = [CloudLayerGpu::zeroed(); MAX_CLOUD_LAYERS as usize];
                for (i, l) in ctx
                    .weather
                    .cloud_layers
                    .iter()
                    .take(MAX_CLOUD_LAYERS as usize)
                    .enumerate()
                {
                    self.cpu_layers[i] = *l;
                }
                self.params.cloud_layer_count = ctx
                    .weather
                    .cloud_layers
                    .len()
                    .min(MAX_CLOUD_LAYERS as usize) as u32;
            } else {
                self.params.cloud_layer_count = 1;
            }
            ctx.queue.write_buffer(
                &self.layers_buffer,
                0,
                bytemuck::cast_slice(&self.cpu_layers),
            );
            self.layers_uploaded_at_revision = Some(revision);
        }

        // Params can change every frame via the UI's reconfigure path,
        // so this upload stays unconditional. It's a single 16-byte
        // uniform; the cost vs the layers storage buffer (~256 B) is
        // marginal.
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        // The group-2 bind group is rebuilt inside the cloud march pass
        // closure because it depends on the HDR depth view (only
        // available via RenderContext).
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![
            PassDescriptor {
                name: "clouds::march",
                stage: PassStage::Translucent,
                id: PASS_MARCH,
            },
            PassDescriptor {
                name: "clouds::taa",
                stage: PassStage::Translucent,
                id: PASS_TAA,
            },
            PassDescriptor {
                name: "clouds::composite",
                stage: PassStage::Translucent,
                id: PASS_COMPOSITE,
            },
        ]
    }

    fn dispatch_pass(
        &mut self,
        id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        match id {
            PASS_MARCH => self.dispatch_march(encoder, ctx),
            PASS_TAA => self.dispatch_taa(encoder, ctx),
            PASS_COMPOSITE => self.dispatch_composite(encoder, ctx),
            _ => panic!("clouds: unknown pass id {id}"),
        }
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        // Phase 10.3: pull live UI edits into CloudParamsGpu so the
        // next frame's prepare uploads the new values.
        let c = &config.render.clouds;
        self.params.cloud_steps = c.cloud_steps;
        self.params.light_steps = c.light_steps;
        self.params.multi_scatter_octaves = c.multi_scatter_octaves;
        self.params.detail_strength = c.detail_strength;
        self.params.powder_strength = c.powder_strength;
        self.params.droplet_diameter_bias = c.droplet_diameter_bias;
        self.params.cone_light_sampling = u32::from(c.cone_light_sampling);
        self.params.forward_bias = c.forward_bias;
        self.params.temporal_jitter = u32::from(c.temporal_jitter && !c.freeze_time);
        // Phase 14.C — wind drift mirrors the temporal_jitter gating.
        self.params.wind_drift_strength = if c.freeze_time { 0.0 } else { c.wind_drift_strength };
        // Phase 14.H — skew is spatial; not gated by freeze_time.
        self.params.wind_skew_strength = c.wind_skew_strength;
        // Phase 18 — diurnal modulation; the world clock pausing
        // already freezes the sun, so no freeze_time gate needed.
        self.params.diurnal_strength = c.diurnal_strength;
        // freeze_time: latch a non-advancing simulated_seconds for the
        // cloud march. The shader reads frame.simulated_seconds; we
        // overwrite the cpu-side params if a future tunable drives it,
        // but the canonical pause path is `WorldClock::set_paused`.
        self.freeze_time = c.freeze_time;
        // Half-res toggle. The cloud-march dispatch reads this each
        // frame; on change, `rt.size != rt_size` triggers a rebuild
        // of the cloud RT at the new dimensions.
        self.half_res_render = c.half_res_render;
        // TAA toggle. Going off → on triggers history reallocation +
        // bootstrap on the next frame; going on → off keeps the
        // history textures around (they're a few MB) so a re-enable
        // doesn't churn the allocator, but the CloudRt.taa fields get
        // dropped naturally on the next resize.
        self.temporal_taa = c.temporal_taa;
        Ok(())
    }
}

/// Factory wired by `AppBuilder`.
pub struct CloudsFactory {
    /// Cell to receive the AtmosphereSubsystem's LUTs handle. The host
    /// fills this *before* invoking AppBuilder::build (via
    /// AtmosphereFactory::new()'s shared cell), so the cloud factory can
    /// pick it up and forward it to the constructed CloudsSubsystem.
    pub atmosphere_luts: Arc<Mutex<Option<Arc<AtmosphereLuts>>>>,
}

impl CloudsFactory {
    /// Construct an unwired factory (atmosphere LUTs cell empty).
    pub fn new() -> Self {
        Self {
            atmosphere_luts: Arc::new(Mutex::new(None)),
        }
    }
    /// Construct a factory that shares an atmosphere-LUTs cell with the
    /// AtmosphereFactory.
    pub fn with_atmosphere_luts(cell: Arc<Mutex<Option<Arc<AtmosphereLuts>>>>) -> Self {
        Self {
            atmosphere_luts: cell,
        }
    }
}

impl Default for CloudsFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl SubsystemFactory for CloudsFactory {
    fn name(&self) -> &'static str {
        "clouds"
    }
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        let mut subsys = CloudsSubsystem::new(config, gpu);
        if let Some(luts) = self
            .atmosphere_luts
            .lock()
            .map_err(|e| anyhow::anyhow!("clouds factory: luts lock: {e}"))?
            .clone()
        {
            subsys.set_atmosphere_luts(luts);
        } else {
            warn!(
                target: "ps_clouds",
                "atmosphere LUTs not available — cloud subsystem will skip rendering"
            );
        }
        Ok(Box::new(subsys))
    }
}
