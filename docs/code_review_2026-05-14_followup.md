# PedalSky perf review — post-Sprint-4 follow-up

Date: 2026-05-14
Supplements: `docs/code_review_2026-05-14.md`

Two-agent pass (one GPU, one CPU) run after Sprints 1-4 shipped. New findings
only — items already addressed by §2.1, §2.2, §2.3, §2.4, §2.5, §3.1, §3.4,
§3.7, §3.2, §3.3, §3.5, §2.6 are excluded.

---

## Top-level take

Projected steady-state recovery on dense cumulus @ 1080p, 144 Hz:

- **GPU**: 0.5–1.3 ms/frame achievable via three mechanical light-march
  hoists (S.G1–G3) plus a storage→uniform swap (S.G4).
- **CPU**: 150–350 µs/frame achievable via four BindGroupCache adoptions
  (H1) plus two upload-elision items (H2, H3) plus the mutex remnants
  Sprint 3 missed (M1, M2).

Roughly half of the GPU recovery and most of the CPU recovery is "missed
during Sprint 1/2/3" — same patterns, applied to subsystems the original
sprints didn't reach.

---

# GPU / shader findings

## High

### S.G1 — `march_to_light_cone` recomputes per-sample geometry already known to the caller

**Where:** `shaders/clouds/cloud_march.wgsl:845-848` and `:1070-1073`.

The light march re-derives `r0 = length(p - centre)` and `h0` per dense step,
but the caller already has `p_alt` as a parameter — so `r0 = world.planet_radius_m + p_alt`
and `h0 = p_alt` are pure substitutions. Two `length()` calls per dense sample
× ~10 dense steps × half-res 1080p = ~10 M `length()` per frame.

**Impact:** 0.1–0.25 ms @ 1080p half-res dense cumulus.

**Fix shape:** substitute the known-good values; only `cos_view` (depends on
`sun_dir`) needs to stay derived.

### S.G2 — `effective_cloud_type` re-loads the cloud-type grid for every light-march tap

**Where:** `shaders/clouds/cloud_march.wgsl:740` (inside `sample_density_light`).

The 6-tap cone spans at most a few km — well below the cloud-type grid's
32 km × 128-texel resolution (250 m / texel). Cone taps almost always land
in the same texel as the primary sample, so 6 `textureLoad(cloud_type_grid, …)`
per dense step is wasted.

**Impact:** 0.15–0.3 ms @ 1080p dense cumulus. ~10 dense × 6 light × ~25%
screen coverage = ~3 M extra texel loads / frame.

**Fix shape:** caller computes `cloud_type` once per dense sample and threads
it into `sample_density_light` as a `u32` param — mirrors the
`weather_sample` hoist from Sprint 2 §2.5.

### S.G3 — `layer_wind_mps` 3D-texture sample fires once per dense step instead of once per layer

**Where:** `shaders/clouds/cloud_march.wgsl:480-487` (`layer_wind_mps`),
called from `:467, 502, 785, 849` and indirectly via `layer_skew_xz` at `:629`.

`layer_wind_mps` runs one 3D `textureSampleLevel` per call. With Sprint 2's
light-march refactor it gets called:
- once per primary layer setup — correct,
- but **also from `layer_skew_xz` inside both `sample_density` and
  `sample_density_light`** — once per dense sample, once per light-march tap.

3D texture samples are the most expensive fetch type. The wind vector for a
single layer is identical across the entire pixel's march; we're paying
~10 redundant 3D-texture-sample per pixel for a per-layer constant.

**Impact:** 0.3–0.6 ms @ 1080p dense cumulus.

**Fix shape:** caller computes `wind_mps_xz` + the derived `wind_offset_view`
and `skew_thickness` once per layer; pass into
`sample_density`/`sample_density_light`/`march_to_light` as struct params.
The per-sample `layer_skew_xz` becomes a vector multiply.

## Medium

### S.G4 — `cloud_layers` is `var<storage, read>` but is a 256 B fixed-size constant

**Where:** `shaders/clouds/cloud_march.wgsl:32`; `crates/ps-clouds/src/noise.rs` BGL.

`CloudLayerArray = array<CloudLayerGpu, 8>` (256 B). On Metal/Apple Silicon
uniform buffers hit the constant cache (free across the warp) — storage
buffers hit general L1.

**Impact:** 0.05–0.15 ms; small but mechanical.

**Fix shape:** swap binding to `var<uniform>`, BGL entry from
`Storage { read_only: true }` to `Uniform`. ~10-line diff.

### S.G5 — Ground PBR evaluates `value_noise(p.xz * 0.15)` twice for the same coord

**Where:** `shaders/ground/pbr.wgsl:530` (puddle), `:568` (snow).

Snow/rain scenes evaluate the same 4-hash + bilerp noise call twice across
two separate `if` blocks. Shader CSE not guaranteed across blocks.

**Impact:** 0.05–0.1 ms in snow scenes only.

### S.G6 — Ground ripple kernel runs 18 inner iterations per pixel even at zero ripple density

**Where:** `shaders/ground/pbr.wgsl:268-320`.

Gated only on `wetness > 0.5 && precip > 0`. When that's true (whole wet
ground), the full 3×3×2 Halton loop runs per pixel even though most
spawn cells are inactive at any moment. `halton2`'s inner `loop` causes
branch divergence per pixel.

**Impact:** 0.1–0.2 ms over wet area in rainy scenes.

**Fix shape:** pre-bake a 64×64 ripple distribution texture sampled once
per pixel; rebuild on simulation tick. Eliminates the per-fragment Halton
sequence.

### S.G7 — Godrays radial pass: 64+ unmasked taps per pixel even with sun far off-screen

**Where:** `shaders/godrays/radial.wgsl:82-87`.

`bright_pass` divides by `max(threshold, 1.0)` inside the N-sample loop
(loop-invariant; compiler probably hoists). Bigger issue: no depth-aware
early-out when `sun_uv` is far off-screen.

**Impact:** 0.1–0.3 ms recoverable.

**Fix shape:** early-out for pixels where the accumulated delta from
`sun_uv` would carry the read off-screen for >50% of taps; precompute
`inv_threshold` outside the loop.

### S.G8 — `chromatic_mie_modulation` uses raw `layer.droplet_diameter_um`, not the Cb-mixed value

**Where:** `shaders/clouds/cloud_march.wgsl:1062`.

Per-layer hoist is correct for non-Cb layers. For Cumulonimbus, the
chromatic shift doesn't follow the anvil ice transition that
`cloud_phase_constants` does (line 254-256). Likely intentional
approximation, but flag for the visual team — same kind of layer-scope vs
sample-scope mismatch you'd catch while doing S.G3.

**Impact:** Correctness/consistency check, not perf.

### S.G9 — Aerial-perspective LUT bakes 16 steps per froxel regardless of slice altitude

**Where:** `shaders/atmosphere/aerialperspective.comp.wgsl:39, 80`.

Z spacing exponential 50 m → 100 km across 64 slices. First 20 slices
cover 50 m → 1 km — 16 steps at dt = 12.5 m wildly over-resolved for
Rayleigh extinction. Last few slices cover 50 km → 100 km — under-resolved.

**Impact:** ~30% bake-time reduction on reconfigure with strictly better
accuracy at altitude.

**Fix shape:** variable step count per slice
(`n_steps = clamp(8 + i32(z_norm * 16.0), 4, 24)`).

## Low

### S.G10 — `cone_kernel_offset` switch could be a `const array<vec3<f32>, 6>`

**Where:** `shaders/clouds/cloud_march.wgsl:816-825`.

WGSL `const array` is supported in wgpu 29 / naga; constant-buffer fetch on
Metal vs branch lookup.

### S.G11 — TAA clamp-neighbourhood: fuse the two textures' 3×3 loops

**Where:** `shaders/clouds/cloud_taa.wgsl:115-134, 174, 176`.

Combine with prior audit's §3.10 (plus-pattern reduces 9 → 5 taps): a fused
loop over both attachments saves redundant offset computation and improves
cache locality.

### S.G12 — Sky still samples `overcast_field` per fullscreen pixel when clouds statically disabled

**Where:** `shaders/atmosphere/sky.wgsl:120-140`.

Sprint 4 zeros the field, so the mix collapses to clear sky, but the
texture sample + exp still fires per pixel.

**Impact:** ~0.05 ms @ 1080p — only worth doing alongside another sky-pass
refactor.

### S.G13 — Half-res WGSL string-patcher (prior §4.1 unresolved)

**Where:** `crates/ps-clouds/src/pipeline.rs:295-299`.

Four `.replace()` calls with a marker-count assertion; doesn't catch
silent wrong-text replacement. Move to `override` constants with one
shader source.

---

# CPU / host findings

## High

### H1 — Four subsystems still rebuild bind groups every frame (Sprint 1 §3.1 follow-up)

**Where:**
- `crates/ps-godrays/src/lib.rs:460-477, 510-527` — radial + composite BGs.
- `crates/ps-bloom/src/lib.rs:536-553, 584-601, 631-648, 676-693` — bright,
  N-1 down, N-1 up, composite (~7-11 BGs depending on `PYRAMID_LEVELS`).
- `crates/ps-precip/src/lib.rs:702-720` — compute + render + every far-layer
  BG rebuilt every prepare, even at zero intensity.
- `crates/ps-tint/src/lib.rs:232-249` — `tint-bg` rebuilt every frame;
  `scratch.view` only changes on resize.

Sprint 1's `BindGroupCache` reached clouds/ground/atmosphere-sky but
missed these four.

**Impact:** ~80-200 µs/frame aggregate. Bloom alone is likely 40-70 µs of it.

**Fix shape:** same one-line pattern as ground — godrays/tint key on
`scratch.full_size`; bloom one cache per level keyed on
`(full_size, level_index)`; precip on `(weather.revision, kind, far_layer_count)`.

### H2 — `WorldUniformsGpu` re-uploaded every frame despite being scene-stable

**Where:** `crates/ps-core/src/bind_groups.rs:131`, called from
`crates/ps-app/src/main.rs:1139-1143`.

`AtmosphereParams` (planet radius, Rayleigh/Mie/ozone, atmosphere top) only
changes on hot-reload or weather synthesis. Each frame uploads the full
struct (256-512 B) blindly — ~50 KB/s of pointless PCIe + a `queue.write_buffer`
submission. The frame-uniforms half *does* change; the world half does not.

**Impact:** ~10-30 µs/frame at the queue level plus reduced staging churn.

**Fix shape:** split `FrameWorldBindings::write` into `write_frame` +
`write_world`. Call `write_world` only from `App::reconfigure` and
`resynthesise_weather`.

### H3 — `refresh_overcast_field_visibility` per-frame call should be event-driven

**Where:** `crates/ps-app/src/main.rs:1186-1187` →
`crates/ps-core/src/weather.rs:328`.

The function is a no-op when `cloud_render_active=true` (the common case)
but the comparison + branch happens every frame. On a clouds-toggle frame
it issues a 16 KB texture upload — currently fires unconditionally on every
frame after the toggle until clouds re-enable.

**Fix shape:** move the upload into `App::reconfigure` + `resynthesise_weather`;
drop the per-frame call.

## Medium

### M1 — `TonemapSubsystem` still wraps three nested `Mutex`es behind an `Arc<TonemapShared>`

**Where:** `crates/ps-postprocess/src/subsystem.rs:48-55, 100-117`;
`tonemap.rs:161, 200`.

Sprint 3 made `dispatch_pass` take `&mut self`, but the inner state survives
as `Arc<TonemapShared>` with `Mutex<TonemapState>`, `Mutex<Option<f32>>`,
`Mutex<wgpu::BindGroup>`. Each tonemap pass: 3 lock-acquires for ~24 B of state.

**Impact:** ~3-5 µs/frame plus one less poison failure mode.

**Fix shape:** `TonemapState` → atomics (TonemapMode is 4 variants → `AtomicU32`;
ev100 → `AtomicU32` bitcast); `bind_group` only changes on resize → plain
field with rebuild from `rebuild_bindings()`; collapse `Arc<TonemapShared>`
into `TonemapSubsystem` fields directly.

### M2 — `LightningPublish = Arc<Mutex<LightningSnapshot>>` for a 32-byte payload

**Where:** `crates/ps-lightning/src/lib.rs:49, 191`; consumer
`crates/ps-app/src/main.rs:1109-1113`.

Single-writer (lightning prepare), single-reader (main frame). Both on
main thread post-Sprint 3.

**Fix shape:** replace with an `Arc<Cell<LightningSnapshot>>` (needs `Send`),
or — better — explicit accessor `App::lightning_snapshot(&self)`. Same
pattern as `instance_count: Mutex<u32>` in `lightning/render.rs:137`.

### M3 — `App::drain_gpu_timings` allocates and clones strings every frame

**Where:** `crates/ps-core/src/app.rs:358-421`.

`pending[slot_idx].clone()` clones a `Vec<&'static str>`; loop creates
`String` from `&'static str` per pass; `last_durations_s.clone()` runs twice
(return + re-assign). 15-25 passes × `String::from` + 3 `Vec` clones / frame.

**Impact:** ~5-15 µs/frame plus allocator churn.

**Fix shape:** store `&'static str` end-to-end (pass names are static
literals); UI side allocates `String` once at panel-build time.

### M4 — `pass_names: Vec<&'static str>` allocated every frame

**Where:** `crates/ps-core/src/app.rs:298`.

Same allocation each frame; capacity known at `App::reconfigure` time.

**Fix shape:** hoist to a field, clear-and-extend each frame.

### M5 — `ui_handle.lock()` called ~15 times per frame in `main::frame`

**Where:** `crates/ps-app/src/main.rs` — eight read-only single-field reads
plus the bigger write blocks (1226, 1325).

**Impact:** ~3-8 µs/frame.

**Fix shape:** consolidate into one early `let ui_snapshot = …` that copies
out everything needed; one late lock for writes. Or `parking_lot::Mutex`.

### M6 — egui tessellation every frame regardless of input (§4.2 still applies)

**Where:** `crates/ps-ui/src/lib.rs:241-269` (unchanged).

`ctx.tessellate(full_output.shapes, …)` runs every frame; ~100-300 µs at
1080p when panels are static.

**Fix shape:** check `ctx.has_requested_repaint()` or throttle to 10-15 Hz;
cache previous paint_jobs.

## Low

### L1 — `pending_names: Mutex<[Vec<&'static str>; 2]>` is main-thread-only

`crates/ps-core/src/app.rs:224, 262, 340-342, 380-382`. Same pattern as M2.

### L2 — `cpu_layers` array rewrite each frame in clouds (§3.11 still applies)

`crates/ps-clouds/src/lib.rs:1007-1031`. Zeroes and re-copies
`[CloudLayerGpu; 8]` + writes 2 KB regardless of whether layers changed.

**Fix shape:** gate on `weather.revision`.

### L3 — Aurora / Water / Windsock / Bloom upload uniforms unconditionally

`ps-aurora/src/lib.rs:249-250`, `ps-water/src/lib.rs:235-236`,
`ps-windsock/src/lib.rs:231-232`, `ps-bloom/src/lib.rs:462-473`. Small
(~32-64 B) uniforms uploaded every frame even when contents identical.

**Impact:** Steady ~5-8 µs of redundant queue writes aggregate.

**Fix shape:** cache last-uploaded value; gate `queue.write_buffer` on diff.

### L4 — `register_passes()` called twice during build/reconfigure (§3.6)

`crates/ps-core/src/app.rs:157-189`. Not per-frame; negligible impact.

### L5 — `scene.surface.winds_aloft.clone()` every frame for the UI compass

`crates/ps-app/src/main.rs:1248`. Clone fires unconditionally; data only
changes on scene reload.

### L6 — Per-frame `tracing::trace!` per pass (§4.4 still applies)

`crates/ps-core/src/app.rs:301`. Verify with release-mode bench that
`tracing` level filter elides the formatting.

---

## Recommended sprint sequencing

**Sprint 5 — cloud-march light-path cleanups (1 day)**
GPU side, all mechanical, all mirror Sprint 2 patterns:
- S.G2 — cloud_type_grid hoist from light march.
- S.G3 — wind-field per-layer hoist (the biggest single win).
- S.G1 — light-march geometry substitution.

Projected: 0.5–1.0 ms @ 1080p dense cumulus.

**Sprint 6 — CPU bind-group + upload elision (0.5 day)**
Pure pattern application, low risk:
- H1 — BindGroupCache adoption in godrays/bloom/precip/tint.
- H2 — split world-uniform upload out of the per-frame path.
- H3 — overcast-field upload to reconfigure path.

Projected: 150–300 µs/frame on the steady state.

**Sprint 7 — Sprint 3 cleanup tail (0.5 day)**
The remnants Sprint 3 didn't reach:
- M1 — TonemapShared → atomics.
- M2 — LightningPublish → Cell or accessor.
- M3/M4 — drain_gpu_timings + pass_names allocations.
- L1, L2, L3 — small mutex/upload polish.

**Outstanding from prior audit, still fair game:**
§3.8 (sample_sky_ambient per-step), §3.9 (octave-decay `exp` hoist),
§3.10 (TAA 3×3 → plus — bundle with S.G11), §4.1 (string-patcher), §4.5
(LUT bake nits — bundle with S.G9).
