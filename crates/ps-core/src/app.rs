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

        Ok(App {
            subsystems,
            passes,
            prepare_order,
            factories: self.factories,
        })
    }
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
        for pass in &self.passes {
            tracing::trace!(target: "ps_core::app", pass = pass.name, stage = ?pass.stage, "running pass");
            (pass.run)(encoder, render_ctx);
        }
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
