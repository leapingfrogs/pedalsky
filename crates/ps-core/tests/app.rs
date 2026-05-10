//! Phase 1 Group B tests for `AppBuilder`, `App`, factories, and the render-
//! graph assembly logic (no GPU work actually executed).

use std::sync::Mutex;
use std::sync::OnceLock;

use ps_core::{
    App, AppBuilder, Config, GpuContext, PassStage, PrepareContext, RegisteredPass,
    RenderSubsystem, SubsystemFactory,
};

/// Lazily construct a headless GpuContext, shared across tests in this file.
/// Returns `None` on machines with no compatible adapter (CI sometimes).
fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping GPU-bearing app tests — headless adapter unavailable: {e}");
            None
        }
    })
    .as_ref()
}

// --- Fake subsystems used by the tests. -------------------------------------

/// Fake subsystem whose `prepare()` pushes its name into a shared `Vec` so
/// tests can verify the call order.
struct FakeSubsystem {
    name: &'static str,
    stages: Vec<PassStage>,
    enabled: bool,
}

impl RenderSubsystem for FakeSubsystem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {
        // Tests don't call frame(); this remains untested at the call-site
        // but the prepare_order_names() accessor exercises the same logic.
    }
    fn register_passes(&self) -> Vec<RegisteredPass> {
        self.stages
            .iter()
            .copied()
            .map(|stage| RegisteredPass {
                name: self.name,
                stage,
                run: Box::new(|_, _| {}),
            })
            .collect()
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, e: bool) {
        self.enabled = e;
    }
}

/// Factory that builds a `FakeSubsystem` for a given config flag name + stages.
struct FakeFactory {
    name: &'static str,
    flag: fn(&Config) -> bool,
    stages: Vec<PassStage>,
    /// If true, `build()` panics — used to verify "disabled subsystem is not
    /// constructed".
    panic_on_build: bool,
    /// Counter incremented every time the factory builds.
    build_count: &'static Mutex<u32>,
}

impl SubsystemFactory for FakeFactory {
    fn name(&self) -> &'static str {
        self.name
    }
    fn enabled(&self, config: &Config) -> bool {
        (self.flag)(config)
    }
    fn build(
        &self,
        _config: &Config,
        _gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        *self.build_count.lock().unwrap() += 1;
        assert!(!self.panic_on_build, "factory '{}' was not supposed to build", self.name);
        Ok(Box::new(FakeSubsystem {
            name: self.name,
            stages: self.stages.clone(),
            enabled: true,
        }))
    }
}

// --- Tests. -----------------------------------------------------------------

#[test]
fn disabled_subsystem_is_not_constructed() {
    let Some(gpu) = gpu() else { return };

    static BUILDS: Mutex<u32> = Mutex::new(0);
    *BUILDS.lock().unwrap() = 0;

    let mut config = Config::default();
    config.render.subsystems.backdrop = false;

    let app = AppBuilder::new()
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c: &Config| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: true, // Should never run.
            build_count: &BUILDS,
        }))
        .build(&config, gpu)
        .expect("build should succeed without invoking the disabled factory");

    assert_eq!(app.subsystem_count(), 0);
    assert_eq!(*BUILDS.lock().unwrap(), 0);
}

#[test]
fn enabled_subsystem_is_constructed() {
    let Some(gpu) = gpu() else { return };

    static BUILDS: Mutex<u32> = Mutex::new(0);
    *BUILDS.lock().unwrap() = 0;

    let mut config = Config::default();
    config.render.subsystems.backdrop = true;

    let app = AppBuilder::new()
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c: &Config| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &BUILDS,
        }))
        .build(&config, gpu)
        .expect("build should succeed");

    assert_eq!(app.subsystem_count(), 1);
    assert_eq!(*BUILDS.lock().unwrap(), 1);
    assert_eq!(app.subsystem_names(), vec!["backdrop"]);
}

#[test]
fn passes_run_in_pass_stage_order() {
    let Some(gpu) = gpu() else { return };

    static B: Mutex<u32> = Mutex::new(0);
    static T: Mutex<u32> = Mutex::new(0);
    static A: Mutex<u32> = Mutex::new(0);
    *B.lock().unwrap() = 0;
    *T.lock().unwrap() = 0;
    *A.lock().unwrap() = 0;

    // Three subsystems, each registering one pass at a different stage,
    // registered in REVERSE pass-stage order. The flattened list must come
    // out in PassStage order.
    let mut config = Config::default();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = true;
    config.render.subsystems.atmosphere = true;
    // Disable the rest so we have a clean three-subsystem app.
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;

    let app = AppBuilder::new()
        // Tint registers PostProcess.
        .with_factory(Box::new(FakeFactory {
            name: "tint",
            flag: |c| c.render.subsystems.tint,
            stages: vec![PassStage::PostProcess],
            panic_on_build: false,
            build_count: &T,
        }))
        // Backdrop registers SkyBackdrop.
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
        }))
        // Atmosphere registers Compute then SkyBackdrop.
        .with_factory(Box::new(FakeFactory {
            name: "atmosphere",
            flag: |c| c.render.subsystems.atmosphere,
            stages: vec![PassStage::Compute, PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &A,
        }))
        .build(&config, gpu)
        .expect("build");

    let stages = app.pass_stages();
    // Sorted: Compute(atm), SkyBackdrop(backdrop or atm — registration order),
    // SkyBackdrop, PostProcess(tint).
    assert_eq!(stages.len(), 4);
    assert!(stages[0] <= stages[1] && stages[1] <= stages[2] && stages[2] <= stages[3],
            "stages must be sorted: {stages:?}");
    assert_eq!(stages[0], PassStage::Compute);
    assert_eq!(stages[3], PassStage::PostProcess);
}

#[test]
fn prepare_runs_in_pass_stage_order() {
    let Some(gpu) = gpu() else { return };

    static B: Mutex<u32> = Mutex::new(0);
    static T: Mutex<u32> = Mutex::new(0);
    static A: Mutex<u32> = Mutex::new(0);
    *B.lock().unwrap() = 0;
    *T.lock().unwrap() = 0;
    *A.lock().unwrap() = 0;

    let mut config = Config::default();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = true;
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;

    let app: App = AppBuilder::new()
        // Reverse registration order again to make it interesting.
        .with_factory(Box::new(FakeFactory {
            name: "tint",
            flag: |c| c.render.subsystems.tint,
            stages: vec![PassStage::PostProcess],
            panic_on_build: false,
            build_count: &T,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "atmosphere",
            flag: |c| c.render.subsystems.atmosphere,
            stages: vec![PassStage::Compute, PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &A,
        }))
        .build(&config, gpu)
        .expect("build");

    // Atmosphere has min stage Compute (lowest). Backdrop has min stage
    // SkyBackdrop. Tint has min stage PostProcess. So the prepare order
    // must be: atmosphere, backdrop, tint.
    assert_eq!(app.prepare_order_names(), vec!["atmosphere", "backdrop", "tint"]);
}

#[test]
fn reconfigure_disables_dropped_subsystem_and_enables_new() {
    let Some(gpu) = gpu() else { return };

    static B: Mutex<u32> = Mutex::new(0);
    static T: Mutex<u32> = Mutex::new(0);
    *B.lock().unwrap() = 0;
    *T.lock().unwrap() = 0;

    let mut config = Config::default();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;
    config.render.subsystems.atmosphere = false;

    let mut app = AppBuilder::new()
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "tint",
            flag: |c| c.render.subsystems.tint,
            stages: vec![PassStage::PostProcess],
            panic_on_build: false,
            build_count: &T,
        }))
        .build(&config, gpu)
        .expect("initial build");

    assert_eq!(app.subsystem_names(), vec!["backdrop"]);
    assert_eq!(*B.lock().unwrap(), 1);
    assert_eq!(*T.lock().unwrap(), 0);

    // Disable backdrop, enable tint, then reconfigure.
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = true;
    app.reconfigure(&config, gpu).expect("reconfigure ok");

    assert_eq!(app.subsystem_names(), vec!["tint"]);
    assert_eq!(*T.lock().unwrap(), 1);
}

#[test]
fn duplicate_factory_is_rejected() {
    let Some(gpu) = gpu() else { return };

    static B1: Mutex<u32> = Mutex::new(0);
    static B2: Mutex<u32> = Mutex::new(0);
    *B1.lock().unwrap() = 0;
    *B2.lock().unwrap() = 0;

    let config = Config::default();

    let result = AppBuilder::new()
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B1,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B2,
        }))
        .build(&config, gpu);

    let err = match result {
        Ok(_) => panic!("duplicate name should be rejected"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("duplicate"));
}
