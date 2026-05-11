//! GPU device initialisation and the [`GpuContext`] handle that owns the
//! `wgpu` device, queue, and (when windowed) surface.
//!
//! The headless variant takes `compatible_surface: None` for tests and
//! offline rendering; the windowed variant binds to a winit surface.

use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors returned by [`GpuContext`] construction.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No `wgpu::Adapter` matched the requested options.
    #[error("no compatible GPU adapter found")]
    NoAdapter,
    /// `Adapter::request_device` failed.
    #[error("device request failed: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    /// Surface creation failed (windowed only).
    #[error("surface creation failed: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
}

/// A handle to the active `wgpu` instance, adapter, device, and queue.
///
/// Cheap to clone — internal state is `Arc`-shared. Subsystems receive a
/// `&GpuContext` and never need to construct their own.
#[derive(Clone)]
pub struct GpuContext {
    /// The shared `wgpu::Instance`.
    pub instance: Arc<wgpu::Instance>,
    /// The selected `wgpu::Adapter`.
    pub adapter: Arc<wgpu::Adapter>,
    /// The logical `wgpu::Device`.
    pub device: Arc<wgpu::Device>,
    /// The submission `wgpu::Queue`.
    pub queue: Arc<wgpu::Queue>,
}

/// GPU features the engine wants per plan §0.2.
///
/// `request_device` would error if we requested features the adapter doesn't
/// advertise, so we intersect with `adapter.features()` and `warn!` on each
/// missing one. Phase 0 / Phase 1 don't strictly need any of these — the
/// real consumers are Phase 5 (atmosphere LUTs) and Phase 6 (cloud noise
/// volumes). Keeping the request best-effort here means the test harness
/// runs on integrated GPUs without these features; the affected phases
/// will fail explicitly when they reach for the feature.
fn required_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let wanted = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::FLOAT32_FILTERABLE
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        // Phase 12.2 — RGB cloud transmittance via dual-source blending
        // (composite pass outputs src0 = luminance, src1 = transmittance,
        // blend factor (One, Src1Color) per channel).
        | wgpu::Features::DUAL_SOURCE_BLENDING;
    let granted = wanted & adapter_features;
    let missing = wanted - granted;
    if !missing.is_empty() {
        warn!(
            ?missing,
            "adapter does not advertise some plan §0.2 features; phases that depend on them will fail when they reach for the missing feature"
        );
    }
    granted
}

/// Build the `wgpu::InstanceFlags` to use.
///
/// `force_validation = true` flips on the wgpu validation layer in addition
/// to whatever the build-profile default would have been (matches
/// `[debug] gpu_validation = true` in the engine config). When `false` we
/// stick with `InstanceFlags::default()` — debug builds get validation,
/// release builds don't.
fn instance_flags(force_validation: bool) -> wgpu::InstanceFlags {
    let mut flags = wgpu::InstanceFlags::default();
    if force_validation {
        flags |= wgpu::InstanceFlags::VALIDATION;
    }
    flags
}

/// Per-feature limits we bump above the wgpu defaults. See plan §0.2.
fn required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
    let base = wgpu::Limits::default();
    wgpu::Limits {
        max_storage_buffer_binding_size: base
            .max_storage_buffer_binding_size
            .max(128 << 20)
            .min(adapter_limits.max_storage_buffer_binding_size),
        max_sampled_textures_per_shader_stage: base
            .max_sampled_textures_per_shader_stage
            .max(32)
            .min(adapter_limits.max_sampled_textures_per_shader_stage),
        max_color_attachments: base
            .max_color_attachments
            .max(8)
            .min(adapter_limits.max_color_attachments),
        ..base
    }
}

/// A windowed [`GpuContext`] paired with its surface and surface configuration.
///
/// `surface` borrows from `window` for the lifetime `'window`; in practice
/// `ps-app` keeps the window alive by storing it in an `Arc`, which makes
/// the surface effectively `'static`.
pub struct WindowedGpu<'window> {
    /// Shared GPU handles.
    pub gpu: GpuContext,
    /// The bound `wgpu::Surface` on which the swapchain renders.
    pub surface: wgpu::Surface<'window>,
    /// Current surface configuration (format, present mode, size).
    pub surface_config: wgpu::SurfaceConfiguration,
}

impl<'window> WindowedGpu<'window> {
    /// Reconfigure the surface after a resize.
    pub fn resize(&mut self, (w, h): (u32, u32)) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
    }

    /// Phase 10 — switch the surface's present mode at runtime. Best-
    /// effort: if the requested mode isn't supported by the surface
    /// the call falls back to FIFO (the spec-required mode).
    pub fn set_vsync(&mut self, vsync: bool) {
        let caps = self.surface.get_capabilities(&self.gpu.adapter);
        let want = if vsync {
            wgpu::PresentMode::AutoVsync
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        if self.surface_config.present_mode == want {
            return;
        }
        self.surface_config.present_mode = want;
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
    }
}

/// Construct a [`WindowedGpu`] bound to `window`.
///
/// `target` accepts anything `wgpu::Instance::create_surface` accepts —
/// commonly an `Arc<winit::window::Window>` so the surface outlives the
/// closure that drove its creation.
///
/// `vsync = true` requests `PresentMode::AutoVsync`; `false` requests
/// `Immediate` and falls back to `Fifo` if Immediate isn't supported.
///
/// `gpu_validation = true` forces wgpu's validation layer on regardless
/// of build profile (matches `[debug] gpu_validation` in the engine
/// config). `false` lets wgpu pick its build-profile default (on in
/// debug builds, off in release).
pub fn init_windowed<'window, T>(
    target: T,
    initial_size: (u32, u32),
    vsync: bool,
    gpu_validation: bool,
    gpu_trace_dir: Option<std::path::PathBuf>,
) -> Result<WindowedGpu<'window>, GpuError>
where
    T: Into<wgpu::SurfaceTarget<'window>>,
{
    pollster::block_on(init_windowed_async(
        target,
        initial_size,
        vsync,
        gpu_validation,
        gpu_trace_dir,
    ))
}

async fn init_windowed_async<'window, T>(
    target: T,
    (width, height): (u32, u32),
    vsync: bool,
    gpu_validation: bool,
    gpu_trace_dir: Option<std::path::PathBuf>,
) -> Result<WindowedGpu<'window>, GpuError>
where
    T: Into<wgpu::SurfaceTarget<'window>>,
{
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: instance_flags(gpu_validation),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let surface = instance.create_surface(target)?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| GpuError::NoAdapter)?;
    info!(name = %adapter.get_info().name, backend = ?adapter.get_info().backend,
          "selected GPU adapter (windowed)");

    let limits = required_limits(&adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("pedalsky-device"),
            required_features: required_features(adapter.features()),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: trace_from_dir(gpu_trace_dir.as_deref()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await?;

    // Pick the first sRGB-suffixed format the surface advertises; fall back
    // to Bgra8UnormSrgb only if it appears in the list.
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .or_else(|| {
            caps.formats
                .iter()
                .copied()
                .find(|f| matches!(f, wgpu::TextureFormat::Bgra8UnormSrgb))
        })
        .unwrap_or_else(|| {
            warn!("no sRGB surface format available — falling back to first listed");
            caps.formats[0]
        });

    let present_mode = if vsync {
        wgpu::PresentMode::AutoVsync
    } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        wgpu::PresentMode::Immediate
    } else {
        wgpu::PresentMode::Fifo
    };

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &surface_config);
    debug!(
        format = ?surface_config.format,
        present_mode = ?surface_config.present_mode,
        width, height,
        "configured surface"
    );

    Ok(WindowedGpu {
        gpu: GpuContext {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
        },
        surface,
        surface_config,
    })
}

/// Construct a headless [`GpuContext`] (no surface).
///
/// Used by integration tests and the `ps-app render` headless subcommand.
/// Validation defaults to wgpu's build-profile default (on in debug, off
/// in release); use [`init_headless_with_validation`] to force it on.
pub fn init_headless() -> Result<GpuContext, GpuError> {
    pollster::block_on(init_headless_async(false))
}

/// As [`init_headless`], but lets the caller force wgpu validation on
/// regardless of build profile (matches `[debug] gpu_validation`).
pub fn init_headless_with_validation(gpu_validation: bool) -> Result<GpuContext, GpuError> {
    pollster::block_on(init_headless_async(gpu_validation))
}

async fn init_headless_async(gpu_validation: bool) -> Result<GpuContext, GpuError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: instance_flags(gpu_validation),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| GpuError::NoAdapter)?;
    info!(name = %adapter.get_info().name, backend = ?adapter.get_info().backend,
          "selected GPU adapter (headless)");

    let limits = required_limits(&adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("pedalsky-device"),
            required_features: required_features(adapter.features()),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: trace_from_dir(None),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await?;
    debug!("wgpu device + queue ready (headless)");

    Ok(GpuContext {
        instance: Arc::new(instance),
        adapter: Arc::new(adapter),
        device: Arc::new(device),
        queue: Arc::new(queue),
    })
}

/// Build a [`wgpu::Trace`] from an optional output directory. When the
/// directory is `Some`, this requires the `wgpu` crate to be built with
/// the `trace` feature; the workspace pins it on. The directory must
/// exist before `request_device` is called or wgpu will silently emit
/// to a sub-directory tree it cannot create.
fn trace_from_dir(dir: Option<&std::path::Path>) -> wgpu::Trace {
    match dir {
        Some(path) => {
            if let Err(e) = std::fs::create_dir_all(path) {
                warn!(error = %e, path = %path.display(),
                      "failed to create gpu-trace directory; tracing disabled");
                return wgpu::Trace::Off;
            }
            info!(path = %path.display(), "wgpu trace enabled — writing to directory");
            wgpu::Trace::Directory(path.to_path_buf())
        }
        None => wgpu::Trace::Off,
    }
}
