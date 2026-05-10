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

use bytemuck::Zeroable;
use ps_core::{
    AtmosphereLuts, CloudLayerGpu, Config, GpuContext, HdrFramebuffer, PassStage,
    PrepareContext, RegisteredPass, RenderSubsystem, SubsystemFactory,
};
use tracing::warn;

pub use noise::CloudNoise;
pub use params::{CloudParamsGpu, MAX_CLOUD_LAYERS};
pub use pipeline::CloudPipelines;

/// Stable subsystem name (matches `[render.subsystems].clouds`).
pub const NAME: &str = "clouds";

/// Phase 6 volumetric cloud subsystem.
pub struct CloudsSubsystem {
    enabled: bool,
    noise: Arc<CloudNoise>,
    params: CloudParamsGpu,
    cpu_layers: [CloudLayerGpu; MAX_CLOUD_LAYERS as usize],

    /// Cloud RT bundle (recreated on resize). `Arc<Mutex<>>` so the
    /// pass closures can both observe size and trigger a rebuild when
    /// the framebuffer changes.
    rt: Arc<Mutex<CloudRt>>,

    /// Cached pipelines (depend only on shader source).
    pipelines: Arc<CloudPipelines>,

    /// LUTs handle (set once by the factory after AtmosphereSubsystem
    /// publishes its bundle). `None` if atmosphere is disabled — in
    /// which case we skip rendering.
    luts: Arc<Mutex<Option<Arc<AtmosphereLuts>>>>,

    /// Per-frame uniform buffer for CloudParamsGpu.
    params_buffer: Arc<wgpu::Buffer>,
    /// Per-frame storage buffer for the layer array.
    layers_buffer: Arc<wgpu::Buffer>,
    /// Group-2 bind group (noise textures + samplers + cloud uniforms).
    cloud_data_bg: Arc<wgpu::BindGroup>,
}

/// Cloud render target bundle. Owned by `Arc<Mutex<>>` so pass closures
/// can both inspect and replace it without taking ownership of `Self`.
pub struct CloudRt {
    /// Color attachment (Rgba16Float; same format as the HDR target).
    color: wgpu::Texture,
    /// View on `color`.
    color_view: wgpu::TextureView,
    /// Bind group used by the composite pass to sample `color`.
    composite_bg: wgpu::BindGroup,
    /// Pixel size — used to detect framebuffer resize.
    size: (u32, u32),
    /// Cached sampler so the bind-group rebuild on resize is cheap.
    sampler: wgpu::Sampler,
}

impl CloudRt {
    fn new(
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
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
        Self::build(device, composite_layout, &sampler, (w, h))
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        size: (u32, u32),
    ) {
        let new = Self::build(device, composite_layout, &self.sampler, size);
        self.color = new.color;
        self.color_view = new.color_view;
        self.composite_bg = new.composite_bg;
        self.size = size;
    }

    fn build(
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        (w, h): (u32, u32),
    ) -> Self {
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("clouds-rt"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HdrFramebuffer::COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-composite-bg"),
            layout: composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            color,
            color_view,
            composite_bg,
            size: (w.max(1), h.max(1)),
            sampler: sampler.clone(),
        }
    }
}

impl CloudsSubsystem {
    /// Construct the subsystem.
    pub fn new(_config: &Config, gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let noise = Arc::new(CloudNoise::bake(gpu));
        let pipelines = Arc::new(CloudPipelines::new(device, &noise.layout));

        let params = CloudParamsGpu::default();
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
        };

        let params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-params-ub"),
            size: std::mem::size_of::<CloudParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let layers_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clouds-layers-sb"),
            size: (std::mem::size_of::<CloudLayerGpu>() * MAX_CLOUD_LAYERS as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let cloud_data_bg = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clouds-data-bg"),
            layout: &noise.layout,
            entries: &[
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
            ],
        }));

        let rt = Arc::new(Mutex::new(CloudRt::new(
            device,
            &pipelines.composite_layout,
            (1, 1),
        )));

        Self {
            enabled: true,
            noise,
            params,
            cpu_layers,
            rt,
            pipelines,
            luts: Arc::new(Mutex::new(None)),
            params_buffer,
            layers_buffer,
            cloud_data_bg,
        }
    }

    /// Reference to the baked noise textures (for diagnostic readback).
    pub fn noise(&self) -> &CloudNoise {
        &self.noise
    }

    /// Plumb the atmosphere LUTs reference (called by the factory once
    /// AtmosphereSubsystem has published its bundle).
    pub fn set_atmosphere_luts(&self, luts: Arc<AtmosphereLuts>) {
        *self.luts.lock().expect("clouds: luts lock") = Some(luts);
    }
}

impl RenderSubsystem for CloudsSubsystem {
    fn name(&self) -> &'static str {
        "clouds"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        // Pull in the synthesised cloud layers if the weather pipeline
        // has produced any; otherwise keep the default cumulus layer.
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

        // Upload uniforms.
        ctx.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        ctx.queue
            .write_buffer(&self.layers_buffer, 0, bytemuck::cast_slice(&self.cpu_layers));
    }

    fn register_passes(&self) -> Vec<RegisteredPass> {
        let pipelines_march = self.pipelines.clone();
        let pipelines_composite = self.pipelines.clone();
        let luts_for_march = self.luts.clone();
        let cloud_data_bg = self.cloud_data_bg.clone();
        let rt_march = self.rt.clone();
        let rt_composite = self.rt.clone();

        vec![
            // 1. Cloud raymarch into the cloud RT.
            RegisteredPass {
                name: "clouds::march",
                stage: PassStage::Translucent,
                run: Box::new(move |encoder, ctx| {
                    let luts_guard = luts_for_march.lock().expect("clouds: luts lock");
                    let Some(luts) = luts_guard.as_ref() else {
                        return; // atmosphere disabled — skip
                    };
                    // Resize cloud RT if the framebuffer dimensions have
                    // changed since last frame.
                    let fb_size = (ctx.framebuffer.size.0, ctx.framebuffer.size.1);
                    let mut rt = rt_march.lock().expect("clouds: rt lock");
                    if rt.size != fb_size {
                        rt.rebuild(ctx.device, &pipelines_march.composite_layout, fb_size);
                    }
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clouds::march"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &rt.color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&pipelines_march.march);
                    pass.set_bind_group(0, ctx.frame_bind_group, &[]);
                    pass.set_bind_group(1, ctx.world_bind_group, &[]);
                    pass.set_bind_group(2, cloud_data_bg.as_ref(), &[]);
                    pass.set_bind_group(3, &luts.bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }),
            },
            // 2. Composite cloud RT over HDR target.
            RegisteredPass {
                name: "clouds::composite",
                stage: PassStage::Translucent,
                run: Box::new(move |encoder, ctx| {
                    let rt = rt_composite.lock().expect("clouds: rt lock");
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
                    pass.set_pipeline(&pipelines_composite.composite);
                    pass.set_bind_group(0, &rt.composite_bg, &[]);
                    pass.draw(0..3, 0..1);
                }),
            },
        ]
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
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
        let subsys = CloudsSubsystem::new(config, gpu);
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
