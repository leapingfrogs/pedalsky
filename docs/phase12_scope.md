# Phase 12 — Scope Document

**Status:** draft. Implementation order, acceptance criteria, and
estimates inside each section. v2 work; v1 (plan Phases 0–11) is
complete and committed on `master`.

**Theme:** complete deferred v1 simplifications + add the most
visually-impactful renderable systems the plan listed as out-of-scope.

**Explicitly deferred** (not in this phase): real-data ingestion
(METAR / GRIB2), DEM-driven terrain, vehicle/canopy rain, VR,
networked multi-instance, TAA / temporal reprojection.

---

## 12.1 — Per-pixel cloud type blending

**Plan reference:** §3.2.3 (deferred from v1 with the comment "in
v1 the cloud type comes from the per-layer struct only").

### Current state

`ps_synthesis::weather_map` writes to a 128×128 RGBA16Float texture.
The `.r` channel is now a {0,1} spatial gate; `.g` is **reserved**
(unused, written as 0). The cloud march reads only `.r * layer.coverage`
to drive density and uses `layer.cloud_type` from the per-layer
struct as a *uniform-per-layer* type index.

`Scene.clouds.coverage_grid` already accepts a `type_path: Option<PathBuf>`
companion file in the schema (a u8-per-pixel type index). The runtime
loader for both `data_path` and `type_path` is unimplemented.

### Scope

1. **Synthesis: load coverage grid from disk.** When
   `coverage_grid.data_path` is set and points at a readable file of
   the right size (`size.0 * size.1 * 4` bytes for f32), populate
   weather_map.r with the gridded values (resampled bilinearly to the
   128×128 weather-map). Today the loader returns `None` and falls
   back to the layer's scalar coverage.
2. **Synthesis: load type grid.** When `coverage_grid.type_path` is
   also set, load it as `size.0 * size.1` u8 values and write them
   into weather_map.g (normalised to [0, 1] via `type_index / 7.0` so
   the eight `CloudType` enum values land at distinct fp16 values).
3. **Cloud march: read per-pixel type.** Rework
   `cloud_march.wgsl::sample_density` so the NDF lookup uses the
   weather_map.g type index when present, falling back to
   `layer.cloud_type` when the .g channel is zero or the layer has no
   coverage_grid configured. Add a sentinel value (e.g. .g = 0
   means "use layer default") so existing scenes keep working.
4. **Per-layer struct: signal "use grid"** — add a boolean
   `coverage_grid_active` on `CloudLayerGpu` (we already have one
   `_pad` slot); the shader switches behaviour on that.
5. **Test scene.** Bake a 128×128 type grid that paints cumulus over
   half the extent and stratus over the other half; render the scene
   and assert the cumulus half shows cumulus NDF profile (puffy bell)
   and the stratus half shows stratus NDF profile (top-heavy thin sheet)
   via SSIM against a blessed golden.

### Acceptance

- New scene `tests/scenes/cumulus_meets_stratus.toml` renders with
  visibly different cloud morphology in its two halves.
- Existing 8 reference scenes' goldens unchanged (regression).
- Coverage-grid unit test in `ps_synthesis` confirms a binary file
  round-trips through `synthesise → weather_map.r`.

### Estimate

~4–6 hours: 1h loader, 1h shader rework, 1h scene/golden, 1h tests,
buffer for naga's tendency to choke on switch/branch refactors.

### Risks

- Bilinear sampling of a *type-index* grid is wrong (would interpolate
  between cloud types, producing non-existent intermediate behaviour).
  Use `textureLoad` with nearest sampling for the type channel, only
  bilinear for coverage.
- The fp16 round-trip on the type index is lossy at the edges (7/7
  doesn't round to exactly 1.0). Consider a separate R8Uint texture
  for type if precision matters.

---

## 12.2 — RGB cloud transmittance

**Plan reference:** §6.6 (currently scalar in `.a`; documented as v2
work — "dual-target render or RGB-transmittance MRT").

### Current state

`cloud_march.wgsl` integrates per-channel `transmittance: vec3<f32>`
during the march, then **collapses** to luminance-weighted scalar at
output:

```wgsl
let t_lum = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
let cloud_alpha = 1.0 - t_lum;
return vec4<f32>(luminance, cloud_alpha);
```

The cloud composite uses standard `(One, OneMinusSrcAlpha)` blending
which only supports a scalar alpha. So warm sun seen through the
edge of a cumulus produces a grey fringe instead of a yellow one.

### Scope

1. **Render to two attachments.** The cloud march writes to MRT:
   - target 0 = luminance (existing)
   - target 1 = `vec3 transmittance` packed in RGB, alpha unused
2. **Composite with RGB blending.** A new composite shader reads both
   targets and writes
   `dst = src.luminance + dst * src.transmittance`
   via a custom blend state (`SrcFactor::ConstantColor`-style won't
   work; need a manual fullscreen pass that reads the existing HDR
   colour, multiplies by transmittance, adds luminance).
3. **Cloud-RT format.** Both targets are `Rgba16Float` so the layout
   doubles cloud RT memory (acceptable — it's already small relative
   to the HDR target).
4. **Visual test.** The current goldens assume scalar alpha;
   re-blessing will catch every cloud-bearing scene as a regression.
   Expect colour shifts at cloud edges.

### Acceptance

- Sunset-through-cirrus golden (high_cirrus_sunset) shows visible
  warm fringing on the cirrus edges where the sun is low and red-
  shifted.
- broken_cumulus_afternoon shows cumulus lit from the side with
  warmer back-side and cooler front-side rather than uniform grey.
- All 8 goldens re-blessed.

### Estimate

~6–8 hours: shader MRT rework, composite shader rewrite, blend state
plumbing in pipeline.rs, golden re-blessing + visual check loop.

### Risks

- Premultiplied-alpha blend semantics are different per-channel; the
  composite math needs to be derived carefully or the cloud edges
  will read wrong (dark halos or over-bright ringing).
- AP application in `cloud_march::fs_main` currently mixes scalar
  `cloud_alpha` with luminance. Per-channel transmittance lets us
  do per-channel AP attenuation too — worth a small additional
  rework while we're in the file.

---

## 12.3 — Lightning visuals

**Plan reference:** explicitly out-of-scope in v1 §"Out of Scope",
but the scene schema has `[lightning] strikes_per_min_per_km2` already
plumbed through `Lightning::default { strikes_per_min_per_km2: 0.0 }`.

### Current state

`Scene.lightning.strikes_per_min_per_km2` is parsed and validated.
Nothing reads it. No subsystem named "lightning" exists.

### Scope

1. **New crate `ps-lightning`.** Mirrors the precip subsystem
   structure: a `LightningSubsystem` implementing `RenderSubsystem`,
   factory wired into `AppBuilder` in `ps-app/src/main.rs`.
2. **Stochastic strike generator.** Per frame, sample a Poisson-process
   strike count from `strikes_per_min_per_km2 × visible_area_km2 ×
   dt_minutes`. Each strike picks a random origin under a cumulonimbus
   layer's footprint (use `top_down_density_mask` to bias spawn
   location toward dense cloud) and a random ground point within
   1–5 km of the origin's XZ.
3. **Geometry.** Generate a fractal lightning bolt as a polyline:
   start from the spawn origin, step toward the ground point with
   chaotic deflections (mid-point displacement at log-spaced
   subdivisions). Add 1–3 secondary forks branching off the main
   trunk. Output: a vertex buffer of line segments with per-vertex
   thickness.
4. **Rendering.** Two passes:
   - **Bolt pass:** instanced quads along each segment, additive
     blend, an emissive HDR colour ~10⁹ cd/m² for the trunk and 10⁸
     for forks. Quads face the camera (billboarded). Sub-ms render
     cost.
   - **Cloud illumination pass:** for the duration of the bolt's
     visible flash (~200 ms in two pulses), boost the cloud march's
     in-scatter contribution by a "lightning illuminance" uniform
     (Vec3 colour × intensity falloff from the bolt origin). Plumbed
     as a new `frame.lightning_illuminance: Vec4` field that the
     cloud shader reads when nonzero.
5. **Lifecycle.** Each strike has a 200 ms lifetime, two-pulse
   intensity envelope (fast attack, slow decay, gap, second peak,
   final decay). Stored in a small ring buffer of active strikes;
   the subsystem updates positions/intensities each frame.
6. **Audio is out of scope** (plan §"Out of Scope" explicitly excludes
   "lightning audio").

### Acceptance

- `tests/scenes/thunderstorm.toml` (already 0.5 strikes/min/km²) on
  next render shows occasional bright bolts during the camera's view
  window. Add a deterministic seed to make headless renders
  reproducible: `ps-app render --scene thunderstorm.toml --seed 42`
  produces identical bolts.
- Cloud above the bolt origin brightens visibly during the flash
  window in the EXR output (per-channel HDR luminance >2× ambient).
- Disabling the subsystem (`[render.subsystems].lightning = false`)
  removes both bolts and cloud illumination cleanly.

### Estimate

~10–14 hours. Largest single item in the scope. Most of the work is
in the geometry generator and the cloud-illumination uniform plumbing.

### Risks

- Cloud illumination requires the cloud march to read a per-frame
  uniform that varies independently of sun position. Adding a Vec4
  to `FrameUniforms` is a struct-layout change — must update the
  std140 linter test in `wgsl_layout.rs` and the matching WGSL
  struct in `common/uniforms.wgsl`.
- Bolts visible inside a thick cloud should be dimmed by the cloud's
  optical depth between bolt and viewer. v1 punt: render bolts as
  always-on-top emissive geometry (depth test off); the cloud
  illumination pass conveys the "felt" presence inside the cloud.
  Document this limitation; "bolts attenuated by cloud volume" is a
  v3 concern.

---

## 12.4 — Crepuscular rays / godrays

**Plan reference:** explicitly out-of-scope in v1 §"Out of Scope".

### Current state

No godray pass. The atmosphere subsystem produces sky brightness via
the sky-view LUT (no per-pixel directional information). Cloud edges
correctly self-shadow the cloud volume but cast no rays into the
atmosphere outside the cloud.

### Scope

1. **Screen-space radial blur** (Tatarchuk-style — referenced in plan
   reading list). After tone-map but before swapchain present:
   - Locate sun's screen-space NDC position from `frame.sun_direction`
     projected through `frame.view_proj`. Skip the pass when sun is
     off-screen.
   - Read the HDR target into a half-res scratch (cheap; 0.5–1 ms).
   - Run N=64 radial samples from each pixel toward the sun position,
     accumulating brightness and decaying with distance.
   - Add the result back to the HDR target (additive blend) so the
     ACES tone-mapper sees pre-tonemap luminance.
2. **Mask by occluders.** The radial blur amplifies bright pixels
   along the sun direction; clouds (high luminance near the sun) and
   bright ground patches get amplified. The classic cloud-edge
   crepuscular effect emerges naturally.
3. **UI exposure.** `[render.godrays]` block:
   - `enabled` (default true)
   - `intensity` slider 0..2
   - `decay` slider 0.9..1.0 per sample
   - `samples` slider 16..256

### Acceptance

- broken_cumulus_afternoon at the right pitch shows visible bright
  rays radiating out from cloud edges between the sun and the
  camera.
- Toggling the subsystem off removes them; the rest of the scene
  is unchanged.

### Estimate

~4–6 hours. Standard technique with abundant reference
implementations.

### Risks

- The radial-sample loop with N=64+ at 1080p costs 1–3 ms on a
  4070-class GPU. Budget impact, not a correctness risk.
- Sample-count vs banding tradeoff: low N produces visible spokes.
  Use a per-pixel jitter (the existing blue-noise tile) to break up
  the pattern.

---

## 12.5 — Auroras

**Plan reference:** explicitly out-of-scope in v1 §"Out of Scope".

### Current state

No aurora subsystem. The atmosphere model only has Rayleigh + Mie
+ ozone — auroras emit at specific O₂/N₂ excitation lines (557 nm
green, 630/636 nm red, 427/470 nm violet), unrelated to scattering.

### Scope

1. **New crate `ps-aurora`.** RenderSubsystem at PassStage::Translucent
   (after atmosphere sky, before clouds — auroras live above cloud
   altitude but below the sky-view LUT's far reaches).
2. **Geometry.** Auroras are vertical curtains of emission at altitude
   ~100–300 km. Generate as a sparse 3D scalar field (32×16×32
   Rgba16Float, the four channels being the 3 emission colours +
   density envelope) advected by a slow time-varying noise. The
   geometry is **not** a mesh — it's a fullscreen raymarch like the
   cloud system, but much shallower (4–8 steps, since auroras are
   optically thin).
3. **Geographic gating.** Auroras are visible only at latitudes near
   the auroral oval (60°–75° geomagnetic). Use a simple rule based
   on `[world] latitude_deg`: zero intensity below 50° abs latitude,
   ramp to full at 65°, decay above 80°.
4. **Solar activity input.** A new `[scene.aurora]` block:
   - `kp_index: f32` (planetary K-index 0–9; default 0)
   - `intensity: f32` (override scalar 0..1; default = derived from
     kp_index)
   - `predominant_colour: String` ("green" | "red" | "purple" |
     "mixed"; default "green")
5. **Render.** Raymarch the curtain density, accumulate emission
   weighted by colour. Output goes into the HDR target with additive
   blend.

### Acceptance

- New scene `tests/scenes/aurora_borealis.toml` at lat 65°N,
  midnight UTC, kp=5 produces visible green curtains in the sky.
- The same scene with kp=0 produces nothing visible.
- The same scene at lat 45°N produces nothing visible regardless
  of kp.

### Estimate

~8–12 hours: geometry generator (3D noise advection) is the bulk.
The shader is simpler than the cloud march (no light scattering,
just emission accumulation).

### Risks

- Colour calibration: real aurora emission is line-spectrum, not
  RGB. A naive RGB split looks "video-game green" rather than the
  delicate grey-green of weak displays. Reference: NORDLYS aurora
  Atlas at Tromsø. Plan to iterate on colour after first pass.
- Diffuse emission alone won't capture the fast-moving curtain
  motion that defines visual auroras. v2 punt: add a slow rotation
  to the noise advection; "fast curtain dynamics" is a v3 concern.

---

## 12.6 — Cloud-modulated overcast diffuse irradiance

**Plan reference:** Phase-11 followup #62 / #63 territory; identified
during golden tuning when winter snow under thick stratus rendered
as warm-tinted instead of overcast-white.

### Current state

The ground shader reads ambient sky illuminance via `sample_sky_at(p,
n, sun_dir)` which samples the sky-view LUT at the local zenith. The
sky-view LUT is built from the atmosphere model only — clouds are
**not** in the LUT. Result: under thick stratus, the ground sees the
clear-sky zenith brightness rather than the cloud-modulated overcast
diffuse that real scenes exhibit.

### Scope

1. **Compute a top-down cloud opacity scalar** during synthesis.
   Reuse the existing `top_down_density_mask` (already a 2D R8Unorm
   matching the weather-map extent — Phase 8 uses it for precip
   occlusion). Convert per-pixel density to a per-pixel
   transmittance via `T = exp(-k * density)` with k tuned so OVC
   cloud → T ≈ 0.05.
2. **Surface skylight modulation.** In `pbr.wgsl::sample_sky_at`,
   sample the top-down mask at the surface point's XZ, multiply
   the LUT-derived irradiance by `(1 - T_overcast)` (the diffuse
   contribution of an idealised "white overcast hemisphere") and
   then add `T_overcast * white_diffuse_overcast_intensity` to mix
   in the white component.
3. **Tune the white component.** Real overcast skies are 5–15k cd/m²
   uniform; pick a value that matches photographs (e.g. 8000 cd/m²
   for medium overcast, scaled by the sun's TOA illuminance to
   handle dawn/dusk).
4. **Apply to wet/snow paths too.** Both `wet_lit` and `snow_lit`
   compositing in `pbr.wgsl` use `sky_irr`; the change is in
   `sample_sky_at` so propagates automatically.

### Acceptance

- winter_overcast_snow: ground reads as **white snow**, not
  warm-tan, during the noon render.
- overcast_drizzle: ground reads as flat-grey overcast lighting, not
  blue-shifted clear-sky lighting.
- clear_summer_noon (no clouds → top-down mask is zero → no
  modulation): pixel-identical to current golden.

### Estimate

~4–5 hours: tweak synthesis to produce the overcast scalar, tweak
ground shader for the lookup + mix, retune the white-diffuse
constant against photograph references, re-bless winter scene.

### Risks

- The top-down density mask is computed at coarse vertical step;
  tall layers (thunderstorm, mountain wave) might over-attenuate.
  Add a separate "overcast scalar" texture with tuned vertical
  integration if mask reuse proves too coarse.
- Tuning the white-diffuse intensity is judgement-call territory —
  pick a value that looks right against three or four photos taken
  in known overcast conditions; keep the slider exposed in the UI.

---

## Combined estimate + ordering

Total work: **~36–51 hours** (4–7 person-days at quality-first pace).

**Recommended order:**

1. **12.1** (per-pixel cloud type) — small, completes a v1 deferral,
   touches few files.
2. **12.6** (overcast diffuse) — small, fixes a long-standing visual
   gap, no new subsystem.
3. **12.2** (RGB cloud transmittance) — medium, sits naturally
   alongside the per-pixel work since both touch the cloud march.
4. **12.4** (godrays) — medium, isolated post-process, no cross-cuts.
5. **12.3** (lightning) — large, new subsystem; do this when the
   cloud-rendering work is settled so the cloud-illumination uniform
   plumbing doesn't conflict.
6. **12.5** (auroras) — large, new subsystem; can be developed in
   parallel with lightning if you split the work.

After each item: re-bless the 8 reference scenes' goldens (some will
naturally drift) and verify the 134-test workspace stays green.

---

## What Phase 12 explicitly does **not** include

- Real-data ingestion (METAR / GRIB2 / NWP). Punted per scope brief;
  see Phase 13.
- DEM-driven terrain. Punted per scope brief.
- Vehicle/canopy rain shaders.
- TAA / temporal cloud-march reprojection.
- Cloud volume occluding lightning bolts (lightning bolts render
  as always-on-top emissive geometry; cloud-internal flash conveys
  the in-volume presence).
- Aurora curtain dynamics beyond slow rotation; "fast moving rays"
  v3.

---

## Out-of-band followups still tracked

These survived from Phase 11 and remain valid. Pick up alongside
12.x as convenient:

- **Side-by-side comparison mode** (plan §10.5 stretch). UI work,
  unrelated to renderable changes.
- **Graceful in-egui display of WGSL compile errors.** Today
  `App::reconfigure` panics through wgpu validation. Wrap shader
  module creation in an error scope, surface failures to a new
  `UiState.shader_error: Option<String>`.
