//! `AppBuilder`, `App`, and the `SubsystemFactory` trait that wires
//! configuration to instantiated subsystems via dependency injection.
//!
//! See plan §1.5.

use std::collections::HashMap;

use thiserror::Error;
use tracing::{info, warn};

use crate::config::Config;
use crate::contexts::{GpuContext, PrepareContext, RenderContext};
use crate::subsystem::{PassStage, RegisteredPass, RenderSubsystem};

/// Errors raised by [`AppBuilder::build`].
#[derive(Debug, Error)]
pub enum AppError {
    /// A subsystem factory returned an error during construction.
    #[error("subsystem '{name}' failed to construct: {source}")]
    FactoryFailed {
        /// Subsystem name (matches `[render.subsystems].<name>`).
        name: &'static str,
        /// Underlying error reported by the factory.
        #[source]
        source: anyhow::Error,
    },
    /// Multiple factories registered the same subsystem name.
    #[error("duplicate subsystem name: '{0}'")]
    DuplicateName(&'static str),
    /// `[render.subsystems].<name>` is `true` but no factory was registered.
    #[error(
        "subsystem '{0}' is enabled in config but no factory was registered for it; \
         did you forget to add it to AppBuilder::with_factory()?"
    )]
    NoFactory(String),
}

/// A factory that knows how to build a single subsystem.
///
/// One factory is registered per subsystem; whether it actually constructs
/// the subsystem depends on the matching `[render.subsystems].<name>` flag.
pub trait SubsystemFactory: Send + Sync {
    /// Stable subsystem name (matches the `[render.subsystems].<name>` flag).
    fn name(&self) -> &'static str;

    /// Whether this subsystem is currently enabled in `config`. The default
    /// implementation reads the matching `[render.subsystems].<name>` flag
    /// for the names known by Phase 1; factories may override.
    fn enabled(&self, config: &Config) -> bool {
        let f = &config.render.subsystems;
        match self.name() {
            "ground" => f.ground,
            "atmosphere" => f.atmosphere,
            "clouds" => f.clouds,
            "precipitation" => f.precipitation,
            "wet_surface" => f.wet_surface,
            "backdrop" => f.backdrop,
            "tint" => f.tint,
            other => {
                warn!(
                    name = other,
                    "factory queried unknown subsystem flag — defaulting to false"
                );
                false
            }
        }
    }

    /// Construct the subsystem.
    fn build(&self, config: &Config, gpu: &GpuContext) -> anyhow::Result<Box<dyn RenderSubsystem>>;
}

/// Builds an [`App`] from a `&Config`, a `&GpuContext`, and a list of
/// factories. See plan §1.5.
pub struct AppBuilder {
    factories: Vec<Box<dyn SubsystemFactory>>,
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AppBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    /// Register a factory. The order of registration determines the
    /// tie-break ordering when two factories produce passes at the same
    /// `PassStage` (registration order is preserved).
    pub fn with_factory(mut self, factory: Box<dyn SubsystemFactory>) -> Self {
        self.factories.push(factory);
        self
    }

    /// Drive every factory whose subsystem is enabled in `config`, build the
    /// subsystem, register its passes, and assemble an [`App`].
    pub fn build(self, config: &Config, gpu: &GpuContext) -> Result<App, AppError> {
        let mut seen_names: HashMap<&'static str, ()> = HashMap::new();
        for f in &self.factories {
            if seen_names.insert(f.name(), ()).is_some() {
                return Err(AppError::DuplicateName(f.name()));
            }
        }

        let mut subsystems: Vec<Box<dyn RenderSubsystem>> = Vec::new();
        for factory in &self.factories {
            if !factory.enabled(config) {
                info!(target: "ps_core::app", subsystem = factory.name(), "skipping (disabled in config)");
                continue;
            }
            info!(target: "ps_core::app", subsystem = factory.name(), "constructing");
            let s = factory
                .build(config, gpu)
                .map_err(|source| AppError::FactoryFailed {
                    name: factory.name(),
                    source,
                })?;
            subsystems.push(s);
        }

        // Flatten and stably sort the pass list by PassStage. Within a stage,
        // registration order is preserved (Rust's sort_by is stable).
        let mut passes: Vec<(usize, RegisteredPass)> = Vec::new();
        for (subsys_idx, sys) in subsystems.iter().enumerate() {
            for pass in sys.register_passes() {
                passes.push((subsys_idx, pass));
            }
        }
        passes.sort_by_key(|(_, p)| p.stage);
        let passes: Vec<RegisteredPass> = passes.into_iter().map(|(_, p)| p).collect();

        // For prepare() ordering, sort subsystem indices by the minimum
        // PassStage among each subsystem's registered passes. This ensures
        // that e.g. atmosphere (which has a Compute LUT pass) prepares
        // before clouds (which only has Translucent passes).
        let mut subsystem_min_stage: Vec<(usize, PassStage)> = subsystems
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let min = s
                    .register_passes()
                    .into_iter()
                    .map(|p| p.stage)
                    .min()
                    .unwrap_or(PassStage::Overlay);
                (i, min)
            })
            .collect();
        subsystem_min_stage.sort_by_key(|(_, stage)| *stage);
        let prepare_order: Vec<usize> = subsystem_min_stage.into_iter().map(|(i, _)| i).collect();

        // Phase 10 GPU-timestamp infrastructure. Allocate up to a small
        // ceiling (>= current pass count, with headroom for runtime
        // reconfigure adding passes). Skip silently when the GPU lacks
        // the required feature.
        let timings = build_timings(gpu, passes.len());

        Ok(App {
            subsystems,
            passes,
            prepare_order,
            factories: self.factories,
            timings,
        })
    }
}

const TIMING_CAP_PADDING: u32 = 8;

fn build_timings(gpu: &GpuContext, pass_count: usize) -> Option<TimingsState> {
    let features = gpu.device.features();
    if !features.contains(wgpu::Features::TIMESTAMP_QUERY)
        || !features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    {
        return None;
    }
    let capacity_passes = (pass_count as u32 + TIMING_CAP_PADDING).max(8);
    let capacity = capacity_passes * 2;
    let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ps-core::frame-timings"),
        ty: wgpu::QueryType::Timestamp,
        count: capacity,
    });
    let buf_size = (capacity as u64) * std::mem::size_of::<u64>() as u64;
    let resolve_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ps-core::frame-timings-resolve"),
        size: buf_size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ps-core::frame-timings-staging"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    Some(TimingsState {
        query_set,
        resolve_buf,
        staging_buf,
        last_durations_s: std::sync::Mutex::new(Vec::new()),
        period_ns: gpu.queue.get_timestamp_period(),
        capacity,
    })
}

/// The assembled application. Holds the subsystem list (used for
/// `prepare()` and UI panels), the flattened sorted pass list (used for the
/// per-frame render loop), and the factory list (used by hot-reload to
/// reconstruct subsystems whose `reconfigure()` returned an error).
pub struct App {
    subsystems: Vec<Box<dyn RenderSubsystem>>,
    passes: Vec<RegisteredPass>,
    prepare_order: Vec<usize>,
    factories: Vec<Box<dyn SubsystemFactory>>,
    /// Phase 10: GPU timestamp infrastructure for per-pass profiling.
    /// `None` when the device doesn't expose
    /// `Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`.
    timings: Option<TimingsState>,
}

struct TimingsState {
    query_set: wgpu::QuerySet,
    /// Tightly packed `u64` resolved timestamps. Capacity = 2 * passes.
    resolve_buf: wgpu::Buffer,
    /// MAP_READ staging buffer copied from `resolve_buf` each frame.
    staging_buf: wgpu::Buffer,
    /// Last drained frame's per-pass durations in seconds, paired with
    /// pass names. The host reads via `App::gpu_timings` and pushes
    /// into the UI.
    last_durations_s: std::sync::Mutex<Vec<(String, f32)>>,
    /// Nanoseconds-per-tick conversion (queue.get_timestamp_period() returns f32 ns).
    period_ns: f32,
    /// Number of slots reserved in the query set (= 2 × passes registered
    /// at App-build time). The pass list can change on `reconfigure`; the
    /// extra slots stay unused on shrink and we cap at construction time.
    capacity: u32,
}

impl App {
    /// Run one frame: call `prepare()` on each subsystem in `PassStage`
    /// order, then walk the flattened pass list in `PassStage` order
    /// invoking each `RegisteredPass::run`.
    ///
    /// This is the executor described in plan §4.4.
    pub fn frame(
        &mut self,
        prepare_ctx: &mut PrepareContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
        render_ctx: &RenderContext<'_>,
    ) {
        for &i in &self.prepare_order {
            self.subsystems[i].prepare(prepare_ctx);
        }

        // Phase 10: per-pass GPU timestamp writes.
        let timings_active = self
            .timings
            .as_ref()
            .map(|t| (self.passes.len() as u32 * 2) <= t.capacity)
            .unwrap_or(false);

        let mut pass_names: Vec<&'static str> = Vec::with_capacity(self.passes.len());

        for (idx, pass) in self.passes.iter().enumerate() {
            tracing::trace!(target: "ps_core::app", pass = pass.name, stage = ?pass.stage, "running pass");
            if timings_active {
                if let Some(t) = &self.timings {
                    encoder.write_timestamp(&t.query_set, (idx as u32) * 2);
                }
            }
            (pass.run)(encoder, render_ctx);
            if timings_active {
                if let Some(t) = &self.timings {
                    encoder.write_timestamp(&t.query_set, (idx as u32) * 2 + 1);
                }
            }
            pass_names.push(pass.name);
        }

        // Resolve the query set into the resolve buffer and stage for
        // the host's next-frame read-back.
        if timings_active {
            if let Some(t) = &self.timings {
                let n = self.passes.len() as u32 * 2;
                encoder.resolve_query_set(&t.query_set, 0..n, &t.resolve_buf, 0);
                encoder.copy_buffer_to_buffer(
                    &t.resolve_buf,
                    0,
                    &t.staging_buf,
                    0,
                    n as u64 * std::mem::size_of::<u64>() as u64,
                );
                // Stash the names for the upcoming read-back; the host
                // calls `drain_gpu_timings` after queue.submit.
                t.last_durations_s
                    .lock()
                    .expect("timings names lock")
                    .clear();
                for name in &pass_names {
                    t.last_durations_s
                        .lock()
                        .expect("timings names lock")
                        .push((name.to_string(), 0.0));
                }
            }
        }
    }

    /// Phase 10 — read back the previous frame's GPU timestamps. Call
    /// after `queue.submit(...)` (blocks on `device.poll`). Returns the
    /// per-pass durations in milliseconds, or an empty Vec when
    /// timestamps are unsupported / haven't run yet.
    pub fn drain_gpu_timings(&self, gpu: &GpuContext) -> Vec<(String, f32)> {
        let Some(t) = &self.timings else { return Vec::new() };
        let names = t
            .last_durations_s
            .lock()
            .expect("timings names lock")
            .clone();
        if names.is_empty() {
            return Vec::new();
        }
        let n_passes = names.len();
        let bytes_needed = (n_passes * 2 * std::mem::size_of::<u64>()) as u64;
        let slice = t.staging_buf.slice(..bytes_needed);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return Vec::new();
        }
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            return Vec::new();
        }
        let bytes = slice.get_mapped_range().to_vec();
        t.staging_buf.unmap();
        let mut out = Vec::with_capacity(n_passes);
        for i in 0..n_passes {
            let begin = u64::from_le_bytes(
                bytes[i * 16..i * 16 + 8]
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            let end = u64::from_le_bytes(
                bytes[i * 16 + 8..i * 16 + 16]
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            let ticks = end.saturating_sub(begin) as f32;
            let ms = ticks * t.period_ns / 1_000_000.0;
            out.push((names[i].0.clone(), ms));
        }
        out
    }

    /// Apply a new `Config` to every live subsystem.
    ///
    /// For each subsystem, calls `reconfigure(&config, gpu)`. If that returns
    /// an error, the subsystem is dropped and rebuilt via its factory; if no
    /// factory is registered, that's a fatal `AppError::NoFactory`.
    ///
    /// Also adds subsystems newly enabled in `config` and removes ones newly
    /// disabled. Used by the hot-reload loop.
    pub fn reconfigure(&mut self, config: &Config, gpu: &GpuContext) -> Result<(), AppError> {
        // 1. For each existing subsystem: reconfigure or recreate.
        let mut new_subsystems: Vec<Box<dyn RenderSubsystem>> = Vec::new();
        for s in self.subsystems.drain(..) {
            let name = s.name();
            // If the subsystem is no longer enabled, drop it.
            let factory = self.factories.iter().find(|f| f.name() == name);
            let still_enabled = factory.map(|f| f.enabled(config)).unwrap_or(false);
            if !still_enabled {
                info!(target: "ps_core::app", subsystem = name, "disabled by reconfigure → dropping");
                drop(s);
                continue;
            }
            let mut s = s;
            match s.reconfigure(config, gpu) {
                Ok(()) => {
                    new_subsystems.push(s);
                }
                Err(err) => {
                    warn!(target: "ps_core::app", subsystem = name, error = %err,
                          "reconfigure failed → dropping and rebuilding via factory");
                    drop(s);
                    let factory = factory.ok_or_else(|| AppError::NoFactory(name.to_string()))?;
                    let rebuilt = factory
                        .build(config, gpu)
                        .map_err(|source| AppError::FactoryFailed { name, source })?;
                    new_subsystems.push(rebuilt);
                }
            }
        }

        // 2. Add newly-enabled subsystems (factory enabled() is true but no
        //    matching live subsystem present).
        for factory in &self.factories {
            let already = new_subsystems.iter().any(|s| s.name() == factory.name());
            if already {
                continue;
            }
            if factory.enabled(config) {
                info!(target: "ps_core::app", subsystem = factory.name(),
                      "newly enabled by reconfigure → constructing");
                let s = factory
                    .build(config, gpu)
                    .map_err(|source| AppError::FactoryFailed {
                        name: factory.name(),
                        source,
                    })?;
                new_subsystems.push(s);
            }
        }

        self.subsystems = new_subsystems;

        // Re-flatten and sort the pass list.
        let mut passes: Vec<RegisteredPass> = self
            .subsystems
            .iter()
            .flat_map(|s| s.register_passes())
            .collect();
        passes.sort_by_key(|p| p.stage);
        self.passes = passes;

        // Rebuild prepare order.
        let mut subsystem_min_stage: Vec<(usize, PassStage)> = self
            .subsystems
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let min = s
                    .register_passes()
                    .into_iter()
                    .map(|p| p.stage)
                    .min()
                    .unwrap_or(PassStage::Overlay);
                (i, min)
            })
            .collect();
        subsystem_min_stage.sort_by_key(|(_, stage)| *stage);
        self.prepare_order = subsystem_min_stage.into_iter().map(|(i, _)| i).collect();
        Ok(())
    }

    /// Number of live subsystems (test-only convenience).
    pub fn subsystem_count(&self) -> usize {
        self.subsystems.len()
    }

    /// Names of live subsystems in registration order (test-only convenience).
    pub fn subsystem_names(&self) -> Vec<&'static str> {
        self.subsystems.iter().map(|s| s.name()).collect()
    }

    /// Number of registered passes after sorting (test-only convenience).
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Stages of every registered pass in execution order (test-only convenience).
    pub fn pass_stages(&self) -> Vec<PassStage> {
        self.passes.iter().map(|p| p.stage).collect()
    }

    /// Names of every registered pass in execution order (test-only convenience).
    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name).collect()
    }

    /// Names of subsystems in `prepare()` order. Test-only convenience: this
    /// is the order each subsystem's `prepare()` will be called in by
    /// [`App::frame`].
    pub fn prepare_order_names(&self) -> Vec<&'static str> {
        self.prepare_order
            .iter()
            .map(|&i| self.subsystems[i].name())
            .collect()
    }
}
