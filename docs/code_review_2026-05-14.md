# PedalSky — Consolidated Code Review

Date: 2026-05-14
Reviewers: four parallel review agents covering Rust idioms, architecture, CPU
performance, and GPU/shader performance. This document deduplicates their
findings, cross-links them where the same root cause surfaced in multiple
reports, and re-ranks the merged set into a single impact-ordered taxonomy.

---

## Executive summary

The codebase is **fundamentally healthy**. The plug-in subsystem model is
working as intended (one-line subsystem enables, clean stage-ordered
execution), the numerical care for planet-scale rendering is consistent, and
the safety-critical paths (HDR pipeline, dual-source blend, depth-aware
termination, atmospheric LUT bake invariants) are correct.

Three substantive themes emerge across the four reviews:

1. **The closure-based pass registration model has accumulated mutex traffic
   far beyond what it set out to do.** `CloudsSubsystem` now carries nine
   `Arc<Mutex<…>>`/`Arc<AtomicBool>` cells driving three captured closures.
   Most of these are flag mirrors that only the main thread reads, but the
   pattern is fragile and ~11 lock calls per frame is pure overhead. Three of
   the four agents flagged this with different framings; the architectural
   fix (dispatch via `&mut self` instead of captured closures) is the
   highest-leverage single change.

2. **Several per-frame paths do work that should be cached.** The two most
   visible CPU costs are `device.poll(WaitIndefinitely)` paired calls
   (timestamp + auto-exposure read-back) that can stall the CPU for hundreds
   of microseconds to milliseconds, and ~8–12 `device.create_bind_group()`
   calls across subsystems that fire even when their inputs are stable.

3. **The cloud march has clear, mechanical GPU wins.** Hoisting transcendental
   work out of the 4-octave multi-scatter loop, splitting `sample_density`
   into "primary" and "light-march" variants (skipping curl + detail fetches
   in the latter), and hoisting the weather-map sample out of the light-march
   together project to a **30–50% reduction** in the heaviest shader's
   per-pixel cost. This alone can recover several ms at 1080p on a typical
   broken-cumulus scene.

There are no critical bugs requiring immediate attention. The one
correctness item worth surfacing is silent dropping of `wgpu::map_async`
errors in the user-triggered screenshot path; everything else is
maintenance, performance, or architectural clarity.

### Top 10 actionable items, ranked by impact

| # | Item | Severity | Where | Section |
|---|---|---|---|---|
| 1 | Pipeline `device.poll(WaitIndefinitely)` read-backs (timestamps + auto-exposure) | High (CPU) | `ps-core/app.rs:339-345`, `ps-postprocess/subsystem.rs:165-179` | §3.1 |
| 2 | Hoist transcendentals out of cloud-march multi-scatter loop | High (GPU, ~1 ms) | `shaders/clouds/cloud_march.wgsl:1019-1027` | §3.2 |
| 3 | Split `sample_density` into primary + light-march variants | High (GPU, ~0.7 ms) | `shaders/clouds/cloud_march.wgsl:993-1027` | §3.3 |
| 4 | Replace closure pass-registration with `dispatch_pass(&mut self)` | High (arch + CPU) | `ps-core/subsystem.rs`, all subsystems | §3.4 |
| 5 | Hoist `weather_map` sample out of cloud light-march | High (GPU, ~0.4 ms) | `shaders/clouds/cloud_march.wgsl:697-700, 988-991` | §3.5 |
| 6 | Cache per-frame bind-group rebuilds (clouds/ground/precip/godrays) | Medium (CPU, ~80 µs) | five subsystems | §4.1 |
| 7 | Synthesis-owned "overcast field" instead of `top_down_density_mask` | Medium (arch) | atmosphere + ground + precip all read it | §4.2 |
| 8 | `String`-typed enums in config (`tone_mapper`, `reprojection`, etc.) | Medium (Rust) | `ps-core/config.rs:166, 322, 576` | §4.3 |
| 9 | Replace cloud subsystem `Arc<Mutex<bool>>` flag mirrors with `AtomicBool` | Medium (Rust + CPU) | `ps-clouds/lib.rs:79-127` | §4.4 |
| 10 | `map_async` errors silently dropped before `get_mapped_range()` | Medium (Rust, correctness) | `ps-app/main.rs:1554-1572, 1624-1638` | §4.5 |

The full list is enumerated below. Each finding is tagged with the agents
that surfaced it: **(R)** Rust idioms · **(A)** Architecture · **(C)** CPU
performance · **(G)** GPU/shader.

---

## 1 Critical

None observed. No undefined behaviour, no memory-safety hazards, no
mandatory-panic on hot paths that would crash the renderer. The macOS Metal
timestamp-poll deadlock is already gated (see `crates/ps-core/src/gpu.rs`).

---

## 2 High

### 2.1 `device.poll(WaitIndefinitely)` pair stalls the CPU every frame (C)

**Where:** `crates/ps-core/src/app.rs:339-345` (`drain_gpu_timings`),
`crates/ps-postprocess/src/subsystem.rs:165-179` (`drain_auto_exposure`).

Both run on the main thread, both block until the GPU finishes the previous
frame's read-back. Stacking two `WaitIndefinitely` polls per frame
**directly serialises the CPU with the GPU pipeline depth**. Estimated cost:
0.5–4 ms/frame depending on driver queue depth — invisible in `cargo
test`-style single-frame renders, significant in the live app.

**Fix shape:** pipeline both read-backs one frame. Submit the map this
frame, check completion with `Maintain::Poll` (non-blocking) next frame,
fall back to wait only if the map hasn't completed by the *following*
submit. For auto-exposure specifically, dropping the read-back to N-of-M
frames is also acceptable since EV100 isn't visually sensitive at 60 Hz.

### 2.2 Cloud-march multi-scatter loop recomputes transcendentals each octave (G)

**Where:** `shaders/clouds/cloud_march.wgsl:1019-1027` (the 4-octave loop)
and `cloud_phase` at line 248.

`cloud_phase(cos_theta, layer, phase_h, c)` is invoked once per octave. Inside
it, `g_hg, g_d, alpha, w_d, ice_fraction, chromatic_mie_modulation` are all
**loop-invariant** in `n` — only `g_hg * c` and `g_d * c` (and the halo
`width/c`) actually depend on the octave. The current implementation
re-derives the 5 `exp()` + 2 `pow(.., 1.5)` calls per octave.

At 4 default octaves × ~10 dense samples × ~25% screen coverage at 1080p,
hoisting the loop-invariant work projects **0.6–1.2 ms saved on Apple
Silicon**.

**Fix shape:** extract a `CloudPhaseConstants` struct populated once per
dense sample (`g_hg, g_d, alpha, w_d, ice_fraction, chromatic_mod, halo_22,
halo_46`), pass it into a leaner `cloud_phase_with(c)` that does only the
per-octave `c`-dependent multiplications. Also hoist `cos_theta`-only
derived terms (e.g. Draine's `1 + alpha * cos²θ` numerator) since
`cos_theta` is per-pixel constant.

### 2.3 `sample_density` runs full noise tap chain for every light-march step (G)

**Where:** `shaders/clouds/cloud_march.wgsl:993-1027` (in `march_to_light` and
`march_to_light_cone`).

Each call to `sample_density` issues 3 sampled texture fetches (`noise_base`,
`noise_curl`, `noise_detail`) plus a `textureLoad` of the cloud-type grid.
The light-march calls it 6 times per dense primary sample, the cone variant
6 times. Cone offsets aside, **`noise_curl` and `noise_detail` contribute
almost nothing at the light-march's spatial scale** — they're refining edges
of the cloud at metre-scale, and the light march spans hundreds of metres.

At ~10 dense samples × 6 light-march steps × 3 texture fetches per call,
that's ~180 wasted fetches per dense ray. **Estimated 0.5–1.0 ms saved.**

**Fix shape:** split `sample_density` into `sample_density_full` (primary
ray, current behaviour) and `sample_density_light` (skip detail + curl
taps; keep base shape, weather, NDF, coverage remap). Cone-tap and straight
light-march both call the lite variant.

### 2.4 Pass-registration via `Box<dyn Fn>` closures creates mutex proliferation (R + A + C)

**Where:** `crates/ps-clouds/src/lib.rs:79-127, 697-705`; same pattern in
`ps-precip/src/lib.rs:736-746`, `ps-godrays/src/lib.rs:364-375`,
`ps-bloom/src/lib.rs:85-100`.

`CloudsSubsystem` carries nine `Arc<…>`/`Arc<Mutex<…>>` cells that exist
*only* because `register_passes()` returns `Box<dyn Fn>` closures that need
to read shared state. The closures don't run in parallel — they execute
sequentially in `App::frame` on the main thread. The mutex traffic is pure
satisfying-the-borrow-checker overhead: ~11 `Mutex::lock()` calls across the
three cloud passes per frame for state that is single-writer.

Two costs:
- **Runtime:** ~1–2 µs/frame in lock-call overhead; more importantly, the
  pattern obscures invariants and any panic poisons the entire subsystem's
  cloud state.
- **Maintenance:** adding any per-frame state requires three `clone()`s
  (one per closure), a parallel `Arc` field, and matching teardown in
  `set_enabled`. Easy to get wrong (the `half_res_render_composite` parallel
  to `half_res_render` exists for exactly this reason).

**Fix shape:** add `fn dispatch_pass(&mut self, pass_id: PassId, encoder:
&mut wgpu::CommandEncoder, ctx: &RenderContext)` to the `RenderSubsystem`
trait. `register_passes` returns lightweight `PassDescriptor { name,
stage, pass_id }` values. The executor dispatches by `(subsystem_index,
pass_id)`. Pass-internal state becomes plain `&mut self` fields — the
~9 `Arc<Mutex<>>` cells in `CloudsSubsystem` collapse to ~4 plain `bool`s
and one `Option<TaaState>`. Roughly 80% reduction in mutex traffic across
all subsystems.

This is the single most leverage-y refactor in the report.

### 2.5 `weather_map` sample redundant inside cloud light-march (G)

**Where:** `shaders/clouds/cloud_march.wgsl:697-700` (in `march_to_light`),
identical pattern in `march_to_light_cone`.

The weather map varies on a 32 km tile sampled with linear-repeat. A 6-step
light march spans at most a few km — the weather value is essentially
identical across all 6 taps. Currently re-fetched per step.

**Estimated 0.3–0.6 ms** at 1080p typical scene.

**Fix shape:** the primary ray's `weather_sample` (already fetched at
`fs_main:933`) gets threaded into `march_to_light(_cone)` and reused for
every light-march step. Mechanical change, no behavioural impact.

### 2.6 `RenderSubsystem::enabled()` / `set_enabled()` are dead trait methods (R + A)

**Where:** `crates/ps-core/src/subsystem.rs:98-103`. Never called by
`App::frame` or `App::reconfigure`. Every subsystem implements them, stores
a private `enabled: bool` it never reads, and `set_enabled` is invoked only
from tests.

The runtime enable/disable is actually handled by `SubsystemFactory::enabled(&config)`
which the executor consults at `App::reconfigure` time to drop and rebuild
subsystems. The trait methods exist from an earlier design that didn't
survive.

**Fix shape:** delete both methods from the trait, fold any teardown
(currently only `CloudsSubsystem::set_enabled` drops the TAA state — see
2.4's fix subsumes this) into `Drop`. Modest line-count saving, large
conceptual clarification.

---

## 3 Medium

### 3.1 Per-frame `device.create_bind_group()` calls without revision tracking (C)

**Where:**
- `crates/ps-clouds/src/lib.rs:863` — cloud-march data bind group (12-13 entries, every frame).
- `crates/ps-godrays/src/lib.rs:463, 523` — radial + composite bind groups, both rebuilt every frame.
- `crates/ps-ground/src/lib.rs:292` — density-mask bind group every `prepare()`.
- `crates/ps-precip/src/lib.rs:714-732` — up to **5** bind groups every frame even when precip is off.
- `crates/ps-atmosphere/src/lib.rs:632` — sky-pass density-mask bind group every frame.

Each `create_bind_group` is ~5–30 µs depending on entry count. Aggregate
cost: ~80–150 µs/frame from rebuilds that are unnecessary (the underlying
texture views only change when synthesis re-runs).

**Fix shape:** introduce a `ps_core::WeatherBindGroupCache` helper keyed on
texture-view identity (compare by `Arc::as_ptr` or via a synthesis-side
revision counter `WeatherState::revision: u64`); subsystems call
`cache.get_or_rebuild(&PrepareContext)` and only pay the wgpu cost on a real
change. Same helper used at 4–5 sites.

### 3.2 Sky / ground / precip all read `top_down_density_mask` regardless of clouds-subsystem state (A)

**Where:** `shaders/atmosphere/sky.wgsl:25-26` (sky overcast modulation),
`crates/ps-ground/src/lib.rs:291` (ground overcast irradiance), and
`crates/ps-precip/src/lib.rs:698-699` (precip cloud-mask gate). All bind
the texture from `ctx.weather.textures.top_down_density_mask_view` regardless
of whether the clouds subsystem is active. This is the coupling that
surfaced during a recent user-facing diagnostic (disabling clouds left the
sky pale-grey).

**Fix shape:** treat the density mask as a synthesis deliverable (not a
"clouds extension"). Rename to `WeatherTextures::overcast_field` (the
intent), add a `WeatherState::cloud_render_active: bool` propagated from
the executor's view of `[render.subsystems].clouds`, and gate the
modulation in the sky/ground shaders on the flag. The cleaner contract:
synthesis describes the world; subsystems decide whether to honour cloud
presence.

### 3.3 `String`-typed enums in config (R)

**Where:** `crates/ps-core/src/config.rs:166` (`tone_mapper: String`), `:322`
(`reprojection: String`), `:576` (`Aurora.predominant_colour: String`).

Each is compared by `==` at runtime; a typo round-trips to the default with
only a `warn!`. The TOML surface would be unchanged with `#[serde(rename_all
= "snake_case")]` enums but the validation moves to parse-time and the
runtime cost drops.

**Fix shape:** convert to `enum` with serde tags, delete the
`TonemapMode::from_config(&str)` / `aurora_colour_bias(&str)` lookups.

### 3.4 Cloud subsystem `Arc<Mutex<bool>>` flags should be `AtomicBool` (R + C)

**Where:** `crates/ps-clouds/src/lib.rs:79-127` — `half_res_render`,
`temporal_taa`, `freeze_time_live` are all single-writer (main thread sets,
closure reads). Each `.lock().expect("…")` panics on poison; an `AtomicBool`
is cheaper and can't poison.

Subsumed by §2.4 if that lands. Still worth a small standalone fix
otherwise.

**Fix shape:** convert the three `bool` mirrors to `AtomicBool`; the
`Mutex<Option<u32>>` taa_dispatch can become `AtomicU32` with `u32::MAX` as
the "None" sentinel.

### 3.5 `wgpu::map_async` errors silently dropped before `get_mapped_range()` (R)

**Where:** `crates/ps-app/src/main.rs:1554-1572, 1624-1638`. The screenshot
read-back path does `rx.recv().ok().and_then(|r| r.ok());` then immediately
`buf.slice(..).get_mapped_range()`. If `map_async` failed, the
`get_mapped_range` panics — and these are user-triggered (Render-panel
screenshot button), not background tasks.

**Fix shape:** match on the `Result`, return an `anyhow::Error` with a
useful message if mapping failed.

### 3.6 `App::register_passes` called twice per build/reconfigure (R + A + C)

**Where:** `crates/ps-core/src/app.rs:144-156, 432-438, 441-454`. The
flattening pass calls each subsystem's `register_passes`; then the stage-min
derivation calls them all again. Each invocation rebuilds N
`Box<dyn Fn>` closures with all their cloned `Arc<>` captures — non-trivial
build cost, doubled.

**Fix shape:** flatten once into `Vec<RegisteredPass>`, derive the
per-subsystem `min(stage)` from the flattened list. Or expose `fn
min_stage(&self) -> PassStage` separately on the trait. Subsumed by §2.4 if
`PassDescriptor` replaces closures.

### 3.7 Cloud-march density gate doesn't short-circuit on zero coverage (G)

**Where:** `shaders/clouds/cloud_march.wgsl:946-998` — the
`if (density > 1e-3)` gate is correct, but `sample_density` runs in full
(3 texture fetches + Schneider remaps) on every primary step regardless of
coverage. When the layer's effective coverage is below the Schneider
visible threshold (~0.4) the result will always be zero.

Clear-sky pixels are typically 50%+ of the frame in non-overcast scenes.

**Fix shape:** add an early-exit at the top of `sample_density` when
`weather.r * layer.coverage < 0.01`. Saves 3 fetches × ~150 empty steps for
clear-sky pixels. **~0.2 ms saved.**

### 3.8 `sample_sky_ambient` called per-step but varies only with altitude (G)

**Where:** `shaders/clouds/cloud_march.wgsl:1052-1056`. The sky-view LUT
sample varies with the local-up direction and altitude, both of which
change negligibly across a single layer's march (~few degrees of local-up
over a layer-thickness span).

**Fix shape:** sample once at the layer midpoint, cache `ambient_layer`,
modulate by `ambient_height_gradient(h_norm)` per step. **~0.1–0.2 ms.**

### 3.9 `cloud_phase`'s octave-decay `exp(-sigma_t * od_to_sun * b)` (G)

**Where:** `shaders/clouds/cloud_march.wgsl:1019-1023`. The octave loop
recomputes `exp(-sigma_t * od_to_sun * b)` four times; the base
`beer = exp(-sigma_t * od_to_sun)` is already computed at line 998.

**Fix shape:** `beer_b = pow(beer, b)` ⇒ one `pow` per octave instead of
three `exp` (vec3). Or unroll the `b` schedule and use incremental
multiplications `beer^b_n = beer^(b_n − b_{n−1}) * prev`. **~0.1–0.2 ms.**

### 3.10 TAA neighbourhood clamp does 9 bilinear samples × 2 textures (G)

**Where:** `shaders/clouds/cloud_taa.wgsl:115-134`. 18 fetches per pixel for
the 3×3 AABB clamp.

Karis 2014's original TAA paper uses a 5-tap plus-shaped neighbourhood —
visibly indistinguishable on cloud content. Cuts to 5×2=10 fetches.

**Fix shape:** replace 3×3 with 5-tap "+" pattern (centre + N/S/E/W).
**~0.1–0.3 ms** in TAA-on mode.

### 3.11 Per-frame `cpu_layers` rewrite in cloud subsystem (C)

**Where:** `crates/ps-clouds/src/lib.rs:649-677`. Every frame zeros
`[CloudLayerGpu; 8]` then re-copies from `weather.cloud_layers`, then
`queue.write_buffer` for the entire array. Layer data changes only on scene
hot-reload / synthesis re-run.

**Fix shape:** dirty-bit on synthesis revision; only write when the
revision changes. Similar pattern for `params_buffer` (line 671) and
`surface_buf` in ground (line 283).

### 3.12 `WeatherState` carries scene pass-through fields (A)

**Where:** `crates/ps-core/src/weather.rs:281-294` — `scene_strikes_per_min_per_km2`,
`scene_aurora_kp`, `scene_aurora_intensity_override`, `scene_water`. These
are not synthesised; they're plumbed because `PrepareContext` doesn't
expose `&Scene`.

**Fix shape:** add `PrepareContext::scene: &Scene`, drop the `scene_*`
fields from `WeatherState`. Subsystems that need lightning/aurora/water
read from `ctx.scene` directly.

---

## 4 Low

### 4.1 Half-res WGSL string-patcher is brittle (A)

**Where:** `crates/ps-clouds/src/pipeline.rs:252-300`. Four `.replace()`
calls with `expect()` markers; one shader edit can silently shift a marker
and produce wrong code. The byte-identical-MSL constraint that motivates
this approach is real but addressable via WGSL `override` constants instead.

**Fix shape:** switch to a single WGSL source with `override`-keyed booleans;
add a golden-MSL test to detect codegen drift instead of maintaining two
source variants.

### 4.2 egui tessellation runs every frame regardless of input (C)

**Where:** `crates/ps-ui/src/lib.rs:243-269`. `ctx.tessellate(...)` produces
`ClippedPrimitive` data every frame. Likely 100–300 µs at 1080p.

**Fix shape:** gate on `full_output.needs_repaint` or cap to ~10 Hz with a
separate clock. Recovers ~few hundred µs/frame when panels are static.

### 4.3 Sub-trivial pattern noise (R)

- `crates/ps-app/src/main.rs:18, 242-243, …` — `std::sync::Arc` fully
  qualified in ~12 places despite a local `use Arc`. Quick global replace.
- `crates/ps-core/src/app.rs:106` — `HashMap<&'static str, ()>` used as a
  set; use `HashSet`.
- `crates/ps-core/src/app.rs:347, 357, 361` — `unwrap_or([0; 8])` on
  `try_into` silently swallows out-of-bounds reads in the timestamps path;
  bail with a `warn!` instead.
- `crates/ps-core/src/scene.rs:579-587` — `intensity_override: -1.0` as a
  sentinel; use `Option<f32>`.

### 4.4 Per-frame `tracing::trace!` per pass (C)

**Where:** `crates/ps-core/src/app.rs:274`. Filtered cheaply at the
`tracing-subscriber` level but the eager `pass.name` + `stage` formatting
still happens. Verify the trace-level call is compile-time-stripped at
release; otherwise gate.

### 4.5 LUT bake constants (G)

- `shaders/atmosphere/skyview.comp.wgsl:46` — `192×108` dispatched as
  `24,14,1`; ~28 idle threads per group due to bounds-check. Cosmetic.
- `shaders/atmosphere/aerialperspective.comp.wgsl:41` — `workgroup_size(4,4,4)`
  = 64 threads, fine on Metal threadgroup ≥ 32. Could explore `8,8,4` to
  amortise group launch overhead. Borderline.
- `shaders/atmosphere/multiscatter.comp.wgsl:34` — `phi = 2π·fract(i·golden)`
  per direction recomputes `sin/cos/sqrt` per output texel. Pre-bake the 64
  sphere directions into a constant array. MS LUT bakes only on
  `reconfigure`, so impact is one-frame stalls.

### 4.6 Two-stage factory publishes via `Arc<Mutex<Option<Arc<T>>>>` (A)

**Where:** `ps-atmosphere` publishes its LUTs via a shared cell, `ps-clouds`
consumes; `ps-lightning` publishes a similar cell consumed by `ps-app`. With
more producer/consumer pairs likely (water, aurora outputs), the manual
shared-cell pattern will multiply.

**Fix shape:** a typed `ResourceHub` in `ps-core` with
`publish<T>(handle)` / `consume<T>() -> Option<Arc<T>>`. Three live
producer/consumer pairs today justify a small helper.

### 4.7 Other minor items

- `crates/ps-core/src/config.rs:438-457` — `CloudsTuning::Default` duplicates
  each literal even though `default_*` helpers above each field return the
  same constant. Use the helpers in `Default` for a single source of truth.
- `crates/ps-core/src/atmosphere_luts.rs:43-73` — `AtmosphereLuts` exposes
  both texture and view publicly; subsystems only need the view. Make the
  textures `pub(crate)`.
- `crates/ps-core/src/atmosphere_luts.rs:80-100, 101-121` — `make_2d` /
  `make_3d` duplicate ~90% of code. Extract a `fn make_lut_texture(…)`.
- `crates/ps-app/src/main.rs:863-887, 958-973` — `thread::Builder.spawn(…).expect("spawn weather-fetch thread")`
  panics on spawn failure mid-frame; should `warn!` + clear in-flight flag.
- `crates/ps-clouds/src/lib.rs:697-699` — dead `let _ = prev_taa_on;` line
  from a deleted refactor.

---

## 5 Calibration — what the codebase gets right

Three reviewers flagged these explicitly. Including them is a calibration
signal: the critical items above are findings *despite* the codebase doing
many things well.

- **Numerical stability at planet scale.** `altitude_from_entry` and the
  algebraic identity `(2t·r₀·cos_view + t²) / (r₀ + |pᵢ|)` for the height
  delta avoids catastrophic cancellation across every atmosphere LUT bake
  and the cloud march. (`shaders/clouds/cloud_march.wgsl:354-358`,
  `shaders/atmosphere/transmittance.comp.wgsl:206`,
  `shaders/atmosphere/skyview.comp.wgsl`, etc.)

- **Cloud-march early-exit at two granularities.** The recent
  `max3(transmittance) < 0.01` checks at *both* the inner step loop and
  the outer layer loop handle multi-layer scenes correctly.
  (`shaders/clouds/cloud_march.wgsl:952, 1097`.)

- **`PrepareContext` / `RenderContext` split.** Borrowed-only, no `Arc`s
  leaked, lifetimes scoped per call. The discipline at
  `crates/ps-core/src/subsystem.rs:62-72` about not caching bind groups
  across frames is respected throughout.

- **PassStage ordering with stable sort + registration tiebreak.** Simple,
  produces correct interleaving (atmosphere LUTs before clouds, godrays
  before tonemap) without per-subsystem ordering knowledge.
  (`crates/ps-core/src/app.rs:131-138`.)

- **`Config::parse` with `deny_unknown_fields + default`.** Typos surface as
  parse errors; partial files compose. The 800-line schema is large but
  per-field comments justify intent.

- **Defensive engineering against Metal MSL emission drift.** Keeping the
  full-res cloud pipeline's bind-group layout *untouched* (the half-res
  uses a sibling layout) was the right call given how sensitive Naga's
  MSL generation is to binding indices. (`crates/ps-clouds/src/noise.rs:380-505`
  vs `:517-637`.)

- **Curl + detail erosion gated by `if (cloud > 0.0)`.** The two most
  expensive shader fetches after the base are correctly skipped in empty
  atmospheric volume. (`shaders/clouds/cloud_march.wgsl:640`.)

---

## 6 Where to start

Suggested sequencing if you want maximum return per unit of work:

**Sprint 1 — quick CPU wins (1–2 days)**
- §2.1 pipeline `device.poll` read-backs.
- §3.1 cache per-frame bind-group rebuilds with a `WeatherBindGroupCache`.
- §3.4 cloud-subsystem `Arc<Mutex<bool>>` → `AtomicBool` (cheap precursor
  to §2.4 even if you don't do the full refactor).

These three together should be visible in frame times immediately on a
broken-cumulus afternoon and unblock real profiling.

**Sprint 2 — cloud-march GPU wins (1 day shader-side + bench)**
- §2.2 hoist transcendentals from multi-scatter loop.
- §2.3 split `sample_density` into primary + light-march variants.
- §2.5 hoist `weather_map` from light-march.
- §3.7 early-exit on zero coverage.

Bench before and after; this set is mechanical and the projected savings
sum to 1.5–3 ms at 1080p on dense scenes.

**Sprint 3 — pass-registration refactor (2–3 days)**
- §2.4 `dispatch_pass(&mut self)` model.

This is the bigger architectural change; do it when you have headroom
because it touches every subsystem. The payoff is a much simpler closure-
free pass model and ~80% reduction in mutex traffic across the codebase.

**Sprint 4 — schema + correctness polish (1 day)**
- §3.3 `String`-typed enums → real enums.
- §3.5 `map_async` error handling.
- §3.2 synthesis-owned overcast field (small refactor with UX clarity).
- §2.6 delete `enabled()`/`set_enabled()` from the trait (and probably
  fold into the §2.4 refactor naturally).

The rest of the items are nice-to-have polish on a roughly quarterly
cadence — pick them up alongside whatever feature work is next in that area.
