//! GPU device initialisation and the [`GpuContext`] handle that owns the
//! `wgpu` device, queue, and (when windowed) surface.
//!
//! The headless variant takes `compatible_surface: None` for tests and
//! offline rendering; the windowed variant binds to a winit surface.

use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, info};

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

/// Construct a headless [`GpuContext`] (no surface).
///
/// Used by integration tests and the `ps-app render` headless subcommand.
/// Returns the context paired with `None` for the surface so the windowed
/// and headless code paths share a single API.
pub fn init_headless() -> Result<GpuContext, GpuError> {
    pollster::block_on(init_headless_async())
}

async fn init_headless_async() -> Result<GpuContext, GpuError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
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
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
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
