//! Phase 1 Group B tests for `AppBuilder`, `App`, factories, and the render-
//! graph assembly logic (no GPU work actually executed).

use std::sync::{Arc, Mutex, OnceLock};

use ps_core::{
    App, AppBuilder, Config, GpuContext, PassDescriptor, PassId, PassStage, PrepareContext,
    RenderContext, RenderSubsystem, SubsystemFactory,
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

type PrepareLog = Arc<Mutex<Vec<&'static str>>>;
type PassLog = Arc<Mutex<Vec<(&'static str, PassStage)>>>;

/// Fake subsystem whose `prepare()` pushes its name into a shared `Vec` and
/// whose `dispatch_pass` pushes `(name, stage)` into another, so tests can
/// verify the actual call order under `App::frame`.
struct FakeSubsystem {
    name: &'static str,
    stages: Vec<PassStage>,
    prepare_log: Option<PrepareLog>,
    pass_log: Option<PassLog>,
}

impl RenderSubsystem for FakeSubsystem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn prepare(&mut self, _ctx: &mut PrepareContext<'_>) {
        if let Some(log) = &self.prepare_log {
            log.lock().unwrap().push(self.name);
        }
    }
    fn register_passes(&self) -> Vec<PassDescriptor> {
        self.stages
            .iter()
            .copied()
            .enumerate()
            .map(|(i, stage)| PassDescriptor {
                name: self.name,
                stage,
                id: i as PassId,
            })
            .collect()
    }
    fn dispatch_pass(
        &mut self,
        id: PassId,
        _encoder: &mut wgpu::CommandEncoder,
        _ctx: &RenderContext<'_>,
    ) {
        if let Some(log) = &self.pass_log {
            let stage = self.stages[id as usize];
            log.lock().unwrap().push((self.name, stage));
        }
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
    /// Optional log shared with the `FakeSubsystem` it builds.
    prepare_log: Option<PrepareLog>,
    pass_log: Option<PassLog>,
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
        assert!(
            !self.panic_on_build,
            "factory '{}' was not supposed to build",
            self.name
        );
        Ok(Box::new(FakeSubsystem {
            name: self.name,
            stages: self.stages.clone(),
            prepare_log: self.prepare_log.clone(),
            pass_log: self.pass_log.clone(),
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
            prepare_log: None,
            pass_log: None,
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
            prepare_log: None,
            pass_log: None,
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
            prepare_log: None,
            pass_log: None,
        }))
        // Backdrop registers SkyBackdrop.
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
            prepare_log: None,
            pass_log: None,
        }))
        // Atmosphere registers Compute then SkyBackdrop.
        .with_factory(Box::new(FakeFactory {
            name: "atmosphere",
            flag: |c| c.render.subsystems.atmosphere,
            stages: vec![PassStage::Compute, PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &A,
            prepare_log: None,
            pass_log: None,
        }))
        .build(&config, gpu)
        .expect("build");

    let stages = app.pass_stages();
    // Sorted: Compute(atm), SkyBackdrop(backdrop or atm — registration order),
    // SkyBackdrop, PostProcess(tint).
    assert_eq!(stages.len(), 4);
    assert!(
        stages[0] <= stages[1] && stages[1] <= stages[2] && stages[2] <= stages[3],
        "stages must be sorted: {stages:?}"
    );
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
            prepare_log: None,
            pass_log: None,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
            prepare_log: None,
            pass_log: None,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "atmosphere",
            flag: |c| c.render.subsystems.atmosphere,
            stages: vec![PassStage::Compute, PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &A,
            prepare_log: None,
            pass_log: None,
        }))
        .build(&config, gpu)
        .expect("build");

    // Atmosphere has min stage Compute (lowest). Backdrop has min stage
    // SkyBackdrop. Tint has min stage PostProcess. So the prepare order
    // must be: atmosphere, backdrop, tint.
    assert_eq!(
        app.prepare_order_names(),
        vec!["atmosphere", "backdrop", "tint"]
    );
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
            prepare_log: None,
            pass_log: None,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "tint",
            flag: |c| c.render.subsystems.tint,
            stages: vec![PassStage::PostProcess],
            panic_on_build: false,
            build_count: &T,
            prepare_log: None,
            pass_log: None,
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
            prepare_log: None,
            pass_log: None,
        }))
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B2,
            prepare_log: None,
            pass_log: None,
        }))
        .build(&config, gpu);

    let err = match result {
        Ok(_) => panic!("duplicate name should be rejected"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("duplicate"));
}

#[test]
fn frame_actually_calls_prepare_and_passes_in_pass_stage_order() {
    // Behavioural test: drive `App::frame` end-to-end on a headless GPU and
    // assert that `prepare()` calls and pass-closure invocations happen in
    // the order PassStage prescribes.
    let Some(gpu) = gpu() else { return };

    static B: Mutex<u32> = Mutex::new(0);
    static A: Mutex<u32> = Mutex::new(0);
    static T: Mutex<u32> = Mutex::new(0);
    *B.lock().unwrap() = 0;
    *A.lock().unwrap() = 0;
    *T.lock().unwrap() = 0;

    let prepare_log: PrepareLog = Arc::new(Mutex::new(Vec::new()));
    let pass_log: PassLog = Arc::new(Mutex::new(Vec::new()));

    let mut config = Config::default();
    config.render.subsystems.backdrop = true;
    config.render.subsystems.tint = true;
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;

    let mut app = AppBuilder::new()
        // Register in REVERSE pass-stage order.
        .with_factory(Box::new(FakeFactory {
            name: "tint",
            flag: |c| c.render.subsystems.tint,
            stages: vec![PassStage::PostProcess],
            panic_on_build: false,
            build_count: &T,
            prepare_log: Some(prepare_log.clone()),
            pass_log: Some(pass_log.clone()),
        }))
        .with_factory(Box::new(FakeFactory {
            name: "backdrop",
            flag: |c| c.render.subsystems.backdrop,
            stages: vec![PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &B,
            prepare_log: Some(prepare_log.clone()),
            pass_log: Some(pass_log.clone()),
        }))
        .with_factory(Box::new(FakeFactory {
            name: "atmosphere",
            flag: |c| c.render.subsystems.atmosphere,
            stages: vec![PassStage::Compute, PassStage::SkyBackdrop],
            panic_on_build: false,
            build_count: &A,
            prepare_log: Some(prepare_log.clone()),
            pass_log: Some(pass_log.clone()),
        }))
        .build(&config, gpu)
        .expect("build");

    // Build a tiny RenderContext + PrepareContext to drive frame().
    let hdr = ps_core::HdrFramebufferImpl::new(gpu, (4, 4));
    let stub = build_stub_bind_group(&gpu.device);
    let world = ps_core::WorldState::default();
    let weather = ps_core::WeatherState::stub_for_tests(gpu);
    let frame_uniforms = ps_core::FrameUniforms::default();
    let mut prepare_ctx = PrepareContext {
        device: &gpu.device,
        queue: &gpu.queue,
        world: &world,
        weather: &weather,
        frame_uniforms: &frame_uniforms,
        atmosphere_luts: None,
        dt_seconds: 1.0 / 60.0,
    };
    let render_ctx = ps_core::RenderContext {
        device: &gpu.device,
        queue: &gpu.queue,
        framebuffer: &hdr,
        frame_bind_group: &stub,
        world_bind_group: &stub,
        luts_bind_group: None,
        frame_uniforms: &frame_uniforms,
        weather: &weather,
        tonemap_target: None,
        tonemap_target_format: wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test-frame"),
        });
    app.frame(&mut prepare_ctx, &mut encoder, &render_ctx);
    gpu.queue.submit([encoder.finish()]);

    // Atmosphere (min stage Compute) prepares first; backdrop next; tint last.
    assert_eq!(
        *prepare_log.lock().unwrap(),
        vec!["atmosphere", "backdrop", "tint"],
        "prepare() must run in min-PassStage order across subsystems"
    );

    // Pass closures: Compute(atmosphere) → SkyBackdrop(atmosphere) →
    // SkyBackdrop(backdrop) → PostProcess(tint). Within a stage, registration
    // order is preserved. Atmosphere registered AFTER backdrop, but
    // atmosphere's SkyBackdrop pass is registered as the SECOND of its two
    // passes — both subsystems contribute SkyBackdrop, and registration order
    // matters: backdrop was registered before atmosphere as a factory, so
    // backdrop's SkyBackdrop pass appears first within that stage.
    let log = pass_log.lock().unwrap().clone();
    let stages: Vec<PassStage> = log.iter().map(|(_, s)| *s).collect();
    assert!(
        stages.windows(2).all(|w| w[0] <= w[1]),
        "pass closures must run in PassStage order: {stages:?}"
    );
    assert_eq!(stages.first(), Some(&PassStage::Compute));
    assert_eq!(stages.last(), Some(&PassStage::PostProcess));
}

fn build_stub_bind_group(device: &wgpu::Device) -> wgpu::BindGroup {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stub-uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stub-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stub-bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    })
}
