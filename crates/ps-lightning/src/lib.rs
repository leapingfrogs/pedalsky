//! Phase 12.3 — lightning subsystem.
//!
//! Stochastic Poisson-process strike generator with per-strike
//! fractal-bolt geometry, billboarded additive quad rendering, and a
//! cloud-illumination uniform exposed to the cloud march via
//! `frame.lightning_illuminance` + `frame.lightning_origin_world`.
//!
//! Pass schedule:
//! - `Translucent` — bolt billboard pass, additive HDR blend, depth
//!   test off (per scope-doc v1 punt: bolts are always-on-top
//!   emissive geometry; cloud-internal flash carries the in-volume
//!   presence).
//!
//! Cloud illumination is plumbed through the per-frame uniform block
//! by writing into `PrepareContext::frame_uniforms` during this
//! subsystem's `prepare()` call. The cloud shader picks the values up
//! the next frame.

#![deny(missing_docs)]

pub mod bolt;
pub mod rng;
pub mod strike;

use std::sync::{Arc, Mutex};

use glam::{Vec3, Vec4};
use ps_core::{
    Config, GpuContext, PassDescriptor, PassId, PassStage, PrepareContext, RenderContext,
    RenderSubsystem, SubsystemFactory,
};

const PASS_BOLTS: PassId = 0;
use tracing::debug;

use crate::strike::{ActiveStrike, StrikeStore};

mod render;

/// Shared cell the host reads after `App::frame` has run subsystem
/// prepares — carries the current `(lightning_illuminance,
/// lightning_origin_world)` so ps-app can splice them into
/// `FrameUniforms` before uploading.
///
/// The pattern mirrors `AtmosphereFactory::luts_publish` —
/// `PrepareContext::frame_uniforms` is `&` so subsystems can't mutate
/// it directly; the host reads this cell, mutates the uniforms, and
/// then writes the buffer.
pub type LightningPublish = Arc<Mutex<LightningSnapshot>>;

/// Per-frame lightning state published from the subsystem to the host.
#[derive(Default, Clone, Copy, Debug)]
pub struct LightningSnapshot {
    /// `rgb` = aggregated cloud-illumination emission proxy;
    /// `w` = horizontal falloff radius (m).
    pub illuminance: Vec4,
    /// `xyz` = strongest currently-active strike origin (world);
    /// `w` unused.
    pub origin_world: Vec4,
}

/// Stable subsystem name (matches `[render.subsystems].lightning`).
pub const NAME: &str = "lightning";

/// Per-strike total lifetime (s). Two-pulse envelope spans this
/// window: fast attack → first peak → quick decay → quiet → second
/// peak → final decay.
const STRIKE_LIFETIME_S: f32 = 0.20;

/// Snapshot of the LightningTuning block, taken at construction and
/// reconfigure(), so the subsystem doesn't have to re-borrow Config
/// each frame.
#[derive(Clone, Copy, Debug)]
struct TuningSnapshot {
    seed: u64,
    peak_cloud_illuminance: f32,
    max_active_strikes: u32,
    bolt_peak_emission: f32,
    illumination_radius_m: f32,
}

impl TuningSnapshot {
    fn from_config(config: &Config) -> Self {
        let l = &config.render.lightning;
        Self {
            seed: l.seed,
            peak_cloud_illuminance: l.peak_cloud_illuminance,
            max_active_strikes: l.max_active_strikes,
            bolt_peak_emission: l.bolt_peak_emission,
            illumination_radius_m: l.illumination_radius_m,
        }
    }
}

/// Phase 12.3 lightning subsystem.
pub struct LightningSubsystem {
    tuning: TuningSnapshot,
    strikes: StrikeStore,
    /// Per-pass GPU resources (vertex buffers, pipeline). Built once
    /// at construction; the vertex buffer is re-uploaded per frame
    /// from active strikes' geometry.
    render: render::BoltRender,
    /// Published snapshot of the current cloud-illumination state —
    /// the host reads this after `prepare()` returns and splices it
    /// into the next `FrameUniforms` upload. Still behind `Arc<Mutex>`
    /// because external callers (ps-app) hold a clone for cross-frame
    /// reads outside the subsystem's `&mut self` access window.
    publish: LightningPublish,
}

impl LightningSubsystem {
    /// Construct. The returned `publish` cell stays linked to this
    /// subsystem; the host reads it each frame to fold the
    /// lightning illumination into `FrameUniforms` before upload.
    pub fn new(config: &Config, gpu: &GpuContext) -> (Self, LightningPublish) {
        let tuning = TuningSnapshot::from_config(config);
        let strikes = StrikeStore::new(tuning.seed, tuning.max_active_strikes);
        let render = render::BoltRender::new(&gpu.device, tuning.max_active_strikes);
        let publish: LightningPublish = Arc::new(Mutex::new(LightningSnapshot::default()));
        debug!(
            target: "ps_lightning",
            "subsystem ready: seed={} max_active={}",
            tuning.seed, tuning.max_active_strikes
        );
        let publish_clone = publish.clone();
        (
            Self {
                tuning,
                strikes,
                render,
                publish: publish_clone,
            },
            publish,
        )
    }
}

impl RenderSubsystem for LightningSubsystem {
    fn name(&self) -> &'static str {
        NAME
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) {
        let tuning = self.tuning;
        let store = &mut self.strikes;

        // Advance lifetimes, evict expired.
        store.advance(ctx.dt_seconds);

        // Spawn new strikes from the Poisson trigger model. The
        // scene exposes strikes_per_min_per_km2; we count visible
        // 32×32 km² of cloud area, integrate over `dt`, and Poisson-
        // sample.
        let scene_rate = ctx
            .weather
            .scene_strikes_per_min_per_km2
            .max(0.0);
        let visible_area_km2 = 32.0 * 32.0; // top-down mask extent.
        let dt_minutes = ctx.dt_seconds / 60.0;
        let expected = scene_rate * visible_area_km2 * dt_minutes;
        let n_new = store.poisson_sample(expected);
        for _ in 0..n_new {
            // Pick a horizontal origin under the cloud field. Today
            // we pick uniformly within the mask extent; a more
            // careful version would importance-sample the top-down
            // density mask. Cloud base ≈ 1500m matches the v1 cloud
            // reference altitude.
            let xz = store.uniform_xz_in_extent(32_000.0);
            let origin = Vec3::new(xz.x, 1500.0, xz.y);
            // Ground attach point: 1–5 km horizontal from origin.
            let attach_xz = store.uniform_attach_offset(1000.0, 5000.0) + xz;
            let attach = Vec3::new(attach_xz.x, 0.0, attach_xz.y);
            let bolt = bolt::generate_bolt(origin, attach, store.rng_mut());
            store.push(ActiveStrike {
                origin,
                attach,
                age_s: 0.0,
                bolt,
            });
        }

        // Aggregate cloud illumination across active strikes. Pick
        // the strongest as the "origin" for the localised falloff.
        let (illum_rgb, origin) = aggregate_cloud_illumination(
            store.active(),
            tuning.peak_cloud_illuminance,
        );
        // Publish the snapshot for the host to splice into
        // FrameUniforms before the GPU upload — see
        // `LightningPublish` doc.
        *self.publish.lock().expect("lightning publish lock") = LightningSnapshot {
            illuminance: Vec4::new(
                illum_rgb.x,
                illum_rgb.y,
                illum_rgb.z,
                tuning.illumination_radius_m,
            ),
            origin_world: Vec4::new(origin.x, origin.y, origin.z, 0.0),
        };

        // Push bolt geometry to the GPU vertex buffer for the render
        // pass.
        self.render
            .upload_active_strikes(ctx.queue, store.active(), tuning.bolt_peak_emission);
    }

    fn register_passes(&self) -> Vec<PassDescriptor> {
        vec![PassDescriptor {
            name: "lightning-bolts",
            stage: PassStage::Translucent,
            id: PASS_BOLTS,
        }]
    }

    fn dispatch_pass(
        &mut self,
        _id: PassId,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext<'_>,
    ) {
        self.render.draw(encoder, ctx);
    }

    fn reconfigure(&mut self, config: &Config, _gpu: &GpuContext) -> anyhow::Result<()> {
        let snapshot = TuningSnapshot::from_config(config);
        self.tuning = snapshot;
        // Re-seed the RNG when seed changes for deterministic
        // headless renders.
        self.strikes.reseed(snapshot.seed);
        Ok(())
    }
}

/// Sum the per-strike pulse intensities, weighted by a colour bias
/// (slightly blue-white). Returns the aggregate `(rgb, origin)` where
/// `origin` is the strongest currently-active strike (used for the
/// cloud shader's distance falloff).
fn aggregate_cloud_illumination(
    active: &[ActiveStrike],
    peak: f32,
) -> (Vec3, Vec3) {
    let mut total = 0.0_f32;
    let mut strongest_intensity = 0.0_f32;
    let mut strongest_origin = Vec3::ZERO;
    for s in active {
        let i = strike::pulse_envelope(s.age_s, STRIKE_LIFETIME_S);
        total += i;
        if i > strongest_intensity {
            strongest_intensity = i;
            strongest_origin = s.origin;
        }
    }
    if total <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    // Colour bias — lightning is slightly blue-white (CCT ~28000 K).
    let colour = Vec3::new(0.85, 0.92, 1.0);
    (colour * peak * total, strongest_origin)
}

/// Factory wired by `AppBuilder`. Mirrors `AtmosphereFactory`: when
/// the subsystem is built, the constructed snapshot publish cell is
/// deposited into `publish_dest` so the host can read it after each
/// `App::frame` and splice the values into `FrameUniforms` before
/// upload.
pub struct LightningFactory {
    /// Shared cell where the constructed subsystem deposits its
    /// snapshot publish handle.
    pub publish_dest: Arc<Mutex<Option<LightningPublish>>>,
}

impl LightningFactory {
    /// Build a `LightningFactory` paired with a publish-cell receiver
    /// the host reads after `AppBuilder::build` to obtain the
    /// snapshot handle.
    pub fn new() -> (Self, Arc<Mutex<Option<LightningPublish>>>) {
        let cell = Arc::new(Mutex::new(None));
        (
            Self {
                publish_dest: cell.clone(),
            },
            cell,
        )
    }
}

impl SubsystemFactory for LightningFactory {
    fn name(&self) -> &'static str {
        NAME
    }
    fn enabled(&self, config: &Config) -> bool {
        config.render.subsystems.lightning
    }
    fn build(
        &self,
        config: &Config,
        gpu: &GpuContext,
    ) -> anyhow::Result<Box<dyn RenderSubsystem>> {
        let (subsys, publish) = LightningSubsystem::new(config, gpu);
        *self
            .publish_dest
            .lock()
            .map_err(|e| anyhow::anyhow!("lightning publish_dest lock poisoned: {e}"))? =
            Some(publish);
        Ok(Box::new(subsys))
    }
}
