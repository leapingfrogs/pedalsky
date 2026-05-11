# Phase 13 — Scope Document

**Status:** draft. Lower-priority quality improvements that surfaced
during Phase 11 acceptance and the Phase 12 design pass. No
dependencies on Phase 12; some items become obsolete if Phase 12
extensions land first (noted per item).

**Theme:** rendering correctness polish + developer ergonomics. No
new renderable systems; everything here improves what already
exists.

---

## 13.1 — Aerial perspective LUT range extension

**Plan reference:** §5.2.4 documented "32 km linear is fine for
ground-level viewing. For high-altitude camera flights the far slice
does not cover space-to-ground viewing. Documented for v2; out of
scope for v1's stationary test harness."

### Current state

`shaders/atmosphere/aerialperspective.comp.wgsl` bakes a 32×32×32
froxel volume covering 32 km along the camera frustum (quadratic Z
spacing). Above ~5 km altitude or for views where mountain ranges
sit beyond 32 km, the AP haze flattens and the distance fade looks
wrong.

### Scope

1. Bump the LUT depth dimension to 64 slices.
2. Switch to an exponential spacing: `t_target = AP_NEAR_M *
   exp(z_norm * ln(AP_FAR_M / AP_NEAR_M))` covering ~50 m to ~100
   km.
3. Update the shader's `AP_FAR_M` constant in the consuming sites
   (cloud_march, ground/pbr).
4. Document that the new range trades near-camera detail for far
   reach — user-visible only on high-altitude or panoramic views.

### Estimate

~2 hours.

### Risks

- Existing goldens will shift slightly because the AP haze on the
  ground at horizon distances changes. Re-bless after.

---

## 13.2 — Cloud march temporal reprojection

**Plan reference:** §6.8 — `[render.clouds].reprojection = "off"`
is the only supported value in v1; "checker_2x2" and "quad_4x4" are
reserved.

### Current state

Full-resolution cloud march at every frame. ~12 ms on a 4070-class
GPU at 1080p (per the Phase 11 GPU timings test). The bulk of frame
budget.

### Scope

1. **Checker_2x2 mode.** Render only every other pixel (2×2 checker
   pattern jittered by frame index). Reproject the previous frame's
   result into the un-rendered pixels via velocity vectors derived
   from camera motion.
2. **Quad_4x4 mode.** Render 1/16 of pixels per frame, more
   aggressive reprojection.
3. **Disocclusion handling.** When the camera moves fast or
   geometry changes, reprojected pixels may be invalid. Detect via
   velocity-vector length + depth discontinuity; fall back to a
   filtered neighbourhood sample for those pixels.
4. **History buffer.** Add a persistent cloud-luminance + cloud-
   alpha texture pair that survives across frames; resize on
   framebuffer resize; clear on subsystem reconfigure.

### Estimate

~12–18 hours. Substantial because reprojection is hard to get
right without artifacts (ghosting, bleeding, history corruption on
scene cuts).

### Risks

- Reprojection introduces the *exact* shimmer-when-paused that the
  v1 plan went out of its way to avoid via spatial-only blue
  noise. Need to disable temporal accumulation when
  `[render.clouds].freeze_time = true`.
- Visible ghosting around the analytic sun disk if the reprojection
  pulls in stale pixels along the sun's trail. Mask the sun region
  out of reprojection.

---

## 13.3 — Bloom

**Plan reference:** plan §0.6 mentions "ACES Filmic and a
passthrough/clamp option for debugging" but bloom is not explicit.
The crate `ps-postprocess` exists and would be the natural home.

### Current state

No bloom. The analytic sun disk renders at ~10⁹ cd/m² (correct
photometric magnitude) which after ACES tonemap saturates to white,
but doesn't bleed into surrounding pixels.

### Scope

1. **Bright-pass.** Threshold the HDR target at an EV-relative
   brightness; downsample to half-res scratch.
2. **5-tap Gaussian blur pyramid** at quarter and eighth res with
   physically-based tap weights.
3. **Composite back additive.** Tunable intensity slider 0..1.
4. **UI exposure.** `[render.bloom]` block with `enabled` (default
   true), `threshold_ev100`, `intensity`.

### Estimate

~5–7 hours. Standard technique.

### Risks

- The sun disk is tiny in screen space; bloom has to reach beyond
  the disk's pixel footprint to look natural. Tune the pyramid
  depth.

---

## 13.4 — Surface material library

**Plan reference:** §7.1 specifies "single Voronoi-tiled albedo
with a 3-entry palette". Real-world ground varies enormously
(grass / bare soil / tarmac / sand / water).

### Current state

`shaders/ground/pbr.wgsl` voronoi_palette returns one of three
hard-coded RGB triples. The cell size is 5 m. Result: every scene
has the same generic grey-tan-cool ground.

### Scope

1. **`SurfaceMaterial` enum** in `ps_core::scene`: `Grass | BareSoil
   | Tarmac | Sand | WaterEdge`. Default `Grass`.
2. **Per-material palettes.** Each material has its own 3-entry
   colour palette with appropriate roughness + F0 defaults. Tarmac
   is darker and rougher; sand is lighter and slightly more
   specular; water-edge is a thin blue stripe along the visible
   horizon.
3. **Per-scene override** in TOML: `[surface] material = "Tarmac"`.
4. **UI exposure.** Material picker in the Wet Surface panel.

### Estimate

~3–4 hours.

### Risks

- None significant; this is a classic "extend an enum, add a
  switch in the shader" pattern.

---

## 13.5 — Water (ponds, lakes, sea)

**Plan reference:** plan §"Out of Scope" excludes "real-time
volumetric water" but a flat water plane that *renders* as water
(wave-perturbed normals, Fresnel sky reflection) is in scope.

### Current state

`Wetness::puddle_coverage` produces small Lagarde-2013 puddles on
the ground. There is no extended water body; sea/lake/large
puddle types are unrepresented.

### Scope

1. **`[scene.water]` block** with bounds + altitude + roughness
   range. Optional; absent → behave as today.
2. **Water surface pass.** A textured plane at the water altitude
   with:
   - GGX/Smith specular (high roughness 0.02–0.10).
   - Fresnel-weighted sky reflection (sample sky-view LUT at the
     reflected ray; reuse the cloud sky-ambient infrastructure).
   - Animated normal map driven by a 2D noise advected with the
     wind field at 10 m AGL.
3. **No refraction in v2.** Plan punts on terrain depth; without a
   ground-under-water there's nothing to refract toward. Document.

### Estimate

~6–8 hours.

### Risks

- The 200×200 km ground plane already covers everything; a water
  plane needs to either replace it within bounds or composite
  cleanly along its edges. Use a stencil or a per-pixel alpha mask
  derived from the water bounds.

---

## 13.6 — Wind sock / weather instrument overlay

**Plan reference:** none — this is a developer-ergonomics nice-to-
have that appeared during the Phase 11 acceptance pass.

### Current state

Wind direction + speed are scene parameters. The user has no
in-world cue showing where the wind is coming from. Particle drift
is visible in the rain but only when there's precipitation.

### Scope

1. **3D windsock geometry** at a fixed offset from the camera (say
   5 m forward, 1.5 m up). Cone shape with horizontal axis pointing
   downwind, drooping toward vertical at low wind speed.
2. **Compass rose** drawn in the egui overlay: a small N/E/S/W
   indicator showing the camera's current yaw against geographic
   north.
3. **Sun/moon icon** in the compass at the sun/moon's azimuth +
   altitude.

### Estimate

~3–5 hours.

### Risks

- Visible 3D geometry needs the existing render-graph stages to
  cooperate (depth-tested, applied AP). Pick PassStage::Opaque +
  add to existing depth target.

---

## 13.7 — Headless render: animation sequences

**Plan reference:** plan §11.2 documents `--scene <toml> --time
<ISO8601> --output <path>` for single-frame headless renders.
Animation is implied (you can iterate the binary externally) but
not built-in.

### Current state

`ps-app render` renders one frame per invocation. Producing a
24-frame animation requires 24 invocations + paying GPU init cost
each time.

### Scope

1. **`--time-range <start> <end> --fps <n>`** CLI variant that
   produces a sequence of PNGs (and optionally a single EXR
   sequence) at a configurable framerate.
2. **Single GPU init** for the whole sequence; tear down only at
   the end.
3. **Output naming**: `<output>.0000.png`, `.0001.png`, …
4. **Optional video encoding**: out of scope for v2 — punt to a
   separate tool that consumes the PNG sequence.

### Estimate

~4–6 hours.

### Risks

- Re-synthesising the WeatherState on every frame would defeat the
  purpose; add a "synthesise once" mode unless the scene's
  `simulated_seconds`-driven systems (cloud noise advection, precip)
  are part of the animation.

---

## 13.8 — Subsystem-level enable/disable from UI without restart

**Plan reference:** plan §"Acceptance" for Phase 1 already says
"Toggling `[render.subsystems].atmosphere = false` while the app is
running removes the subsystem cleanly with no GPU validation
errors." This works for the global checkboxes but the path is
fragile around bind-group rebuilds.

### Current state

Toggling a subsystem in the UI's Subsystems panel writes
`live_config.render.subsystems.{name}` and triggers
`App::reconfigure`. The existing reconfigure path handles
add/remove/replace per the comment in `crates/ps-core/src/app.rs`.
Has not been stressed under all combinations.

### Scope

1. **Stress-test all 2^N combinations** of subsystem on/off via a
   property-style integration test. ~256 combinations (8
   subsystems); each runs one frame and asserts no GPU validation
   errors.
2. **Bind-group rebuild correctness.** When a subsystem is toggled
   off then back on, its bind-group references to atmosphere LUTs
   must point at the still-live LUT instance. Today this works
   because LUTs survive `reconfigure` via Arc; document the
   invariant or add a test.
3. **UI: greyed-out dependencies.** When atmosphere is off, clouds
   and ground both lose the LUT bindings — UI should grey out
   those toggles + show a tooltip explaining the dependency.

### Estimate

~4–6 hours.

### Risks

- The 256-combination test will be slow (each combo creates a fresh
  GPU context). Sample down to e.g. 32 representative combinations
  if the runtime exceeds 2 minutes.

---

## 13.9 — Per-frame stable randomness for blue-noise jitter

**Plan reference:** plan §6.1 + §Cross-Cutting/Determinism state
"Blue noise is **spatial only** so the cloud march does not
shimmer when paused."

### Current state

`shaders/clouds/cloud_march.wgsl` reads blue noise via
`textureLoad(blue_noise, jitter_xy & vec2<i32>(63), 0).r` — purely
spatial, no temporal index. Correct.

### Scope (this is a v2 want)

1. Optional **temporal jitter** for users who run with TAA
   downstream (post-process). Plan §"Anti-aliasing" notes "TAA off
   in v1; if added later, blue-noise jitter changes from spatial-
   only to temporally varying — but this is a v2 concern."
2. New `[render.clouds].temporal_jitter: bool` (default false). When
   true, the blue-noise lookup XOR's `jitter_xy` with a frame-index-
   derived offset producing a 16-frame rotation.
3. **Auto-disable when frozen.** When `freeze_time = true`,
   force temporal jitter off so paused screenshots are stable.

### Estimate

~2 hours. Trivial shader change + UI checkbox.

### Risks

- Without TAA downstream, temporal jitter looks like obvious frame-
  by-frame noise. The default-off + clear UI label handles this.

---

## 13.10 — Probe pixel: optical depth at sun

**Plan reference:** plan §10.2 Debug panel — "Probe-point readouts
for transmittance/optical-depth at a selected screen pixel."

### Current state

Debug panel shows transmittance at the probe pixel (working — wired
through `ps_app::probe`). Optical depth from the surface to the sun
along the actual sun-direction ray is **not** displayed — only the
transmittance, which is `exp(-OD_total)`.

### Scope

1. Compute `OD = -ln(transmittance)` per channel and display in the
   Debug panel.
2. Add separate optical-depth contributions for Rayleigh, Mie, and
   ozone (re-run the LUT bake's per-component integral but only at
   the probe pixel).

### Estimate

~2 hours.

### Risks

- None.

---

## Combined estimate

Total: **~42–62 hours** depending on which items you take.

**Recommended order if all are in:**

1. 13.1 (AP LUT range) — quick, unlocks higher-altitude scenes.
2. 13.4 (material library) — quick, immediately visible.
3. 13.10 (optical-depth probe) — quick, dev ergonomics.
4. 13.3 (bloom) — small, completes the post-process chain.
5. 13.6 (windsock) — small, in-world cues for wind.
6. 13.9 (temporal jitter prep) — quick, unblocks TAA.
7. 13.5 (water) — medium, opens up coastal/lake scenes.
8. 13.7 (animation render) — medium, dev-ergonomics + content
   pipeline.
9. 13.8 (subsystem stress) — medium, hardening rather than feature.
10. 13.2 (cloud reprojection) — large, performance not correctness.

**Items that become obsolete if Phase 12 lands first:**

- None outright. 13.10 (probe optical depth) becomes more useful
  with Phase 12 RGB cloud transmittance (per-channel readouts make
  more sense).

---

## What Phase 13 explicitly does **not** include

- Real-data ingestion (parked).
- DEM-driven terrain (parked).
- Vehicle/canopy rain (plan §"Out of Scope").
- Networked multi-instance, VR (plan §"Out of Scope").
- Refractive water (no ground-under-water without DEM).
- Volumetric water (plan §"Out of Scope").
