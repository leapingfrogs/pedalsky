# Comparison: Maxime Heckel's atmospheric series vs. PedalSky

A side-by-side review of three Maxime Heckel articles against the
current PedalSky implementation, with ranked recommendations.

Articles reviewed:
1. **Rendering the Sky, Sunsets and Planets** — sky/atmosphere LUTs,
   planetary scattering
2. **Real-time dreamy Cloudscapes with Volumetric Raymarching** —
   volumetric clouds in a WebGL fragment shader
3. **On Shaping Light** — post-process raymarched volumetric lighting
   with shadow maps

All three target WebGL via React Three Fiber / postprocessing.
PedalSky targets desktop wgpu with HDR + reverse-Z + photometric units.

---

## TL;DR

Across the three subsystems the picture is asymmetric:

| Subsystem | Verdict |
|---|---|
| **Sky / Atmosphere** | PedalSky is strictly more advanced than the article. No changes recommended. |
| **Volumetric Clouds** | Same lineage (Schneider/Hillaire), but PedalSky is several papers further down the road. The articles **do** show two performance patterns we don't currently exploit: half-res render with bicubic upsample, and frame-to-frame temporal reprojection. Those are the two opportunities worth pursuing. |
| **Volumetric Lighting** | Different design intent. Heckel raymarches through a shadow-mapped fog volume. PedalSky does Tatarchuk-style screen-space radial blur with a bright-pass mask. Neither is "better" in the abstract — PedalSky's approach is the right one for atmospheric-scene god-rays (sun streaks through cloud gaps); the article's approach is the right one for in-scene shaft-casting (sun through forest leaves, asteroids, etc.). The interesting gap is "in-cloud" shafts (the visible bright trail through a cloud body), which neither approach gives us today. |

The highest-leverage recommendations are concentrated in clouds:
adopt half-res march + bicubic upsample, then layer a TAA accumulator
on top. Together they can roughly halve cloud cost at parity quality.

---

## 1. Atmosphere / Sky

### Article approach (Heckel)

- Hillaire 2020 LUTs in WebGL fragment shaders.
- **Transmittance LUT**: 250 × 64, baked from Rayleigh + Mie + ozone.
- **Sky-view LUT**: azimuth × elevation, quadratic v parametrisation
  to concentrate samples near the horizon (`(uv.y² − 0.5) · π`).
- **Aerial-perspective**: 2D screen-space texture, depth-buffer aware,
  RGB inscatter + α transmittance.
- Phase functions: Rayleigh `3/(16π)·(1+μ²)`; Mie via Henyey-Greenstein,
  variable g.
- Primary march: 24 steps. Secondary (light) march: 6 steps.
- Atmosphere height 100 km (Kármán), view distance 200 km.
- Standard β coefficients: Rayleigh `(0.0058, 0.0135, 0.0331)`,
  Mie `0.003`, Ozone abs `(0.00065, 0.00188, 0.00008)`.
- ACES + gamma 2.2 at the end.
- Sun disk with eclipse blending; logarithmic depth buffer for
  planetary scale.

References cited: Hillaire EGSR 2020, sebh/UnrealEngineSkyAtmosphere,
Iñigo Quilez ray-sphere intersections, Three Geospatial.

### PedalSky approach

(`shaders/atmosphere/*`, `crates/ps-atmosphere/`)

- Hillaire 2020 LUTs in WGSL compute shaders.
- **Transmittance LUT**: 256 × 64 Rgba16Float, 40-step trapezoidal,
  `transmittance.comp.wgsl:7`.
- **Multi-scatter LUT**: 32 × 32 Rgba16Float, Fibonacci 8×8 = 64
  sphere directions, 20-step inner march, closed-form geometric
  series `L / (1 − f_ms)` (Hillaire 2020 Eq. 9). *Article does not
  bake an MS LUT.*
- **Sky-view LUT**: 192 × 108, 32-step march, quadratic v
  parametrisation matching the article but referenced to the
  geometric horizon at the camera's radius (not the flat-Earth
  horizon). Includes a ground-bounce term inside the sky-view bake.
- **Aerial-perspective LUT**: 32 × 32 × **64** **3D froxel** with
  **exponential** Z spacing 50 m → 100 km. *Article uses a 2D
  screen-space AP — PedalSky's 3D froxel resolves depth-varying
  inscatter in a way a 2D AP cannot, which matters for foreground
  geometry partially behind atmospheric haze.*
- Phase functions: identical Rayleigh + HG Mie formulas.
- Ozone: tent profile `max(0, 1 − |h − centre| / half_thickness)`
  (article also uses tent).
- Sun disk: analytic with Hillaire limb darkening
  (`SUN_DISK_LIMB_DARKEN = 0.6`), multiplied by transmittance LUT
  along the sun direction.
- **Energy-conserving step integral** `S_int = (S − S·T) / σ_t`
  (Hillaire 2020) — article's step integration is a simpler
  `transmittance += sample`.
- Numerically-stable height delta `(2t·r·cos_v + t²)/(r + |pi|)` to
  avoid `length(pi) − planet_radius_m` cancellation at planet scale.
  *This is something the article doesn't address; their per-pixel
  sky compositing in WebGL is more forgiving because it operates in
  smaller world units.*
- **Overcast modulation**: each sky pixel projects to a reference
  cloud altitude (1500 m), samples the synthesised top-down density
  mask, and mixes the clear-sky LUT toward an overcast-grey diffuse
  term (`sky.wgsl:120-180`). This gives the correct "white hemisphere
  under stratus" look that a pure atmosphere LUT cannot produce.
- HDR everywhere internal; ACES + EV100 in the dedicated
  `ps-postprocess` pass; reverse-Z `Depth32Float`.

### Differential summary

| Feature | Article | PedalSky | Gap |
|---|---|---|---|
| Transmittance LUT | 250×64 | 256×64 | – |
| Multi-scatter LUT | not present | 32×32 with 64-dir Fibonacci | **PedalSky ahead** |
| Sky-view LUT | flat-Earth horizon | geometric horizon at camera radius | **PedalSky ahead** |
| Aerial perspective | 2D screen-space | 3D froxel, exponential Z 50 m–100 km | **PedalSky ahead** |
| Ground bounce | not in LUT | folded into sky-view | **PedalSky ahead** |
| Numerical stability at planet scale | not addressed | algebraic identity for height delta | **PedalSky ahead** |
| Cloud-aware overcast | not present | density-mask modulation | **PedalSky ahead** |
| Phase functions | Rayleigh + HG | Rayleigh + HG | parity |
| Sun disk | analytic + eclipse | analytic + Hillaire limb darkening | parity |
| Energy-conserving step | no | yes | minor |
| Tonemap | ACES + γ 2.2 | ACES via dedicated subsystem | parity |

**Recommendation: no changes**. The article describes the technique
PedalSky already implemented in Phase 5 and the project has extended
it on every axis (MS LUT, 3D froxel AP, ground bounce, overcast
modulation, planet-scale numerical stability). The article is useful
as orientation reading for anyone new to the codebase, but it has
nothing to teach the current implementation.

---

## 2. Volumetric Clouds

### Article approach (Heckel)

- Texture-based 2D-slice 3D Perlin noise; 6-octave FBM with scaling
  factor 2.02, scale halving each octave.
- **No** Worley component, **no** weather map, **no** per-cloud-type
  density profile, **no** detail volume erosion. Clouds are SDF
  shapes (sphere, torus, capsule, cross) blended with FBM.
- Wind animation: time offset `q = p + t · 0.5 · vec3(1, -0.2, -1)`.
- Primary march: 100 steps `MARCH_SIZE = 0.08` (basic) →
  50 steps `MARCH_SIZE = 0.5` with blue-noise dither (optimised).
- Light march: 6 steps × march size 0.03 toward sun.
- Phase function: single Henyey-Greenstein.
- Beer's law absorption coefficient 0.9.
- "Powdered sugar" approximation not in the original, but Schneider's
  Beer-Powder is referenced.
- Fast diffuse via Iñigo Quilez's directional derivative trick:
  `diffuse = clamp((density(p) − density(p + 0.3·sunDir)) / 0.3, 0, 1)`.
- **Blue-noise dither** with 32-frame temporal rotation:
  `offset = fract(blueNoise + (frame % 32) / sqrt(0.5))`.
- **Half-resolution render** with bicubic upsampling. Author reports
  "not a lot of differences between 1x and 0.5x" after upscaling.

### PedalSky approach

(`shaders/clouds/cloud_march.wgsl`, `crates/ps-clouds/`)

- **128³ base shape volume** (Rgba8Unorm): R = Perlin-Worley,
  GBA = Worley FBM at three frequencies (2/8/14).
- **32³ detail volume** (Rgba8Unorm): Worley FBM at three frequencies
  (2/8/16).
- **128² 2D curl tile** (Rg8Unorm): Perlin curl, used to perturb
  detail lookup for swirling edges.
- **64² blue-noise tile**: CPU-baked void-and-cluster (Christensen &
  Kensler), 16 ms bake.
- **Weather map**: 32 km tile from `ps-synthesis`; per-pixel
  cloud-type override grid (128×128 R8Uint, sentinel 255).
- **3D wind-field**: synthesised RGBA16Float volume, channels
  `(u, v, w, turbulence)`, sampled per-layer at mid-altitude;
  per-sample sampling produced streaks so 14.F switched to one
  offset per layer (cloud_march.wgsl:432-466).
- **Schneider Nubis "skew with height"** (Phase 14.H): cloud tops
  offset downwind from bases, gives cumulus visible lean and anvil
  tilt under shear.
- **Diurnal modulation** (Phase 18): convective types' shape/detail
  bias scales with `smoothstep(-0.1, 0.4, sin(sun_altitude))`.
- 8 NDF profiles: `Cu`, `St`, `Sc`, `Ac`, `As`, `Ci`, `Cs`, `Cb`,
  each with a hand-tuned bell curve targeting ~0.78 peak.
- Schneider density remap: `lf_fbm` shifts base shape erosion;
  coverage remap; detail erosion with curl perturbation, height-
  dependent polarity (wispy top, fluffy base).
- **Phase function: Jendersie & d'Eon 2023** (SIGGRAPH "Approximate
  Mie Scattering Function for Fog and Cloud Rendering") — HG+Draine
  blend parameterised by droplet effective diameter (5–50 µm).
  Cumulonimbus mixed-phase: diameter blends water → ice across
  `h_norm ∈ [0.6, 0.85]`.
- **Ice halos** at 22° and 46° (cos peaks 0.92718 / 0.69466) as two
  narrow Gaussian-in-cos lobes, ramped in via ice fraction
  `smoothstep(35, 50, d_µm)`.
- **Per-channel chromatic Mie**: small-droplet warm-fringe bias,
  smoothly fades out by 20 µm (geometric-optics regime).
- **Hillaire 2016 multi-octave multiple-scattering**: 4 octaves
  scaling `energy×a`, `optical_depth×b`, `g×c` (defaults
  `a=b=c=0.5`).
- Beer-Powder (Schneider 2015 canonical form) lerped via
  `powder_strength`.
- Sky-view LUT sampled as ambient light at each cloud sample.
- AP LUT applied to cloud luminance at luminance-weighted depth
  along the ray.
- **Primary march: 192 steps** (vs article's 50).
- **Light march: 6 steps** (parity with article).
- **Spatial blue-noise jitter** with optional 16-frame XOR rotation
  (Phase 13.9), but no TAA accumulator — every frame computes the
  full 192-step march fresh.
- **Full-resolution render** — no half-res + upsample stage.
- Dual-source blending for per-channel RGB transmittance composite
  (the article uses single-channel α).
- Depth-aware termination clipping cloud march against ground.

### Differential summary

| Feature | Article | PedalSky | Verdict |
|---|---|---|---|
| Noise primitives | Perlin only, 6-octave FBM | Perlin-Worley + Worley FBM + curl + blue noise | **PedalSky much further along** |
| Density field | SDF + FBM blob | Schneider/Nubis remap with weather map | **PedalSky** |
| Cloud types | none (single blob) | 8 NDF profiles + per-pixel grid | **PedalSky** |
| Wind animation | time offset on FBM input | 3D wind volume + per-layer drift + height skew | **PedalSky** |
| Phase function | HG | HG + Draine (Jendersie & d'Eon 2023) | **PedalSky much ahead** |
| Multi-scatter | none | Hillaire 2016 multi-octave (4 octaves) | **PedalSky** |
| Beer/Powder | Beer + opt powder | Beer + Beer-Powder lerp | parity |
| Ambient term | constant `vec3(0.6, 0.6, 0.75)` | sky-view LUT at sample point | **PedalSky** |
| Chromatic transmittance | greyscale α | per-channel RGB via dual-source blend | **PedalSky** |
| Ice halos / fringes | none | 22° + 46° halo lobes + chromatic Mie | **PedalSky** |
| Aerial perspective on cloud | none | AP LUT at luminance-weighted depth | **PedalSky** |
| Diurnal evolution | none | convective-type bell on solar altitude | **PedalSky** |
| Lightning in-scatter | none | localised falloff from active strike | **PedalSky** |
| Primary march steps | 50 (with dither) | 192 | – (PedalSky higher quality) |
| Light march steps | 6 | 6 | parity |
| **Half-res + bicubic** | **yes** | **no** | **Article ahead — perf opportunity** |
| **TAA / temporal reprojection** | **yes (32-frame)** | **no (spatial dither only)** | **Article ahead — perf opportunity** |
| **Cone light sampling** | no (straight march) | no (straight march) | parity |

### Two real opportunities for PedalSky

#### A. Half-resolution cloud march + bicubic upsample

The cloud pass is the most expensive in the pipeline (192 march
steps × per-step 6-step light march × 4-octave multi-scatter loop).
Heckel reports that 0.5× render + bicubic upsample looks
indistinguishable from native — and his clouds are technically
simpler than ours, so the same trick should hold here, possibly
more so because edge erosion + curl already low-pass our detail.

**Estimated win**: ~4× cloud march throughput (1/4 fragment count),
allowing either ~2× FPS at current step counts or pushing
`cloud_steps` from 192 → 256+ within the same budget.

**Risk**:
- Bicubic upsample on premultiplied luminance + per-channel
  transmittance is tractable but needs care; the dual-source
  blending equation `final.rgb = src0.rgb + dst.rgb · src1.rgb`
  requires both attachments to be upsampled coherently.
- Edge ringing where a thin cumulus boundary spans only 1–2 half-res
  pixels. Frostbite / Hillaire 2020 §6 specifically calls out
  half-res cloud composition as a "needs care" zone.
- Conflicts with the per-pixel `cloud_type_grid` lookup, which
  expects to read at the full-res sample location. A half-res march
  would need to either snap to the 128×128 type grid (already
  coarser than our half-res), which is fine, or sample at the
  upsample stage (more expensive).

**Implementation sketch**:
- Add `cloud_render_scale` to `pedalsky.toml` (1.0, 0.5, 0.25).
- Allocate cloud RT at scaled size; cloud march writes to it.
- New composite pass does bicubic upsample (e.g. Catmull-Rom
  16-tap, like the Frostbite TAA paper §3.4) before dual-source
  blend into HDR.
- Gate behind a Debug-panel toggle for A/B testing.

#### B. TAA / temporal reprojection across frames

Currently PedalSky has spatial blue-noise jitter (great for hiding
step banding within a frame) plus an optional `temporal_jitter`
XOR that rotates the dither lookup across 16 frames. **But there
is no accumulation across frames** — every frame the march runs
fresh and the dither pattern decorrelates only spatially.

The article uses an exponential moving average on the cloud render
target, weighted by `fract(blueNoise + (frame%32) / sqrt(0.5))`,
which is effectively 1-tap TAA. With reprojection (reading the
previous frame from the previous-frame position derived from the
camera matrices), you get 4–16× temporal smoothing for free.

**Estimated win**: at 8-frame accumulation, the perceived sample
count climbs from 192 to ~1500 with no per-frame cost change.
Equivalently, you could drop `cloud_steps` from 192 to ~48 and
still match current quality.

**Risk**:
- Reprojection requires a prev-frame view-proj uniform, a history
  RT, and validity tests (clamp to AABB of recent samples to avoid
  ghosting on cloud-edge motion under camera rotation).
- Disabled for golden-test scenes (already gated by
  `freeze_time` / `wind_drift_strength = 0`).
- For a flying camera, occlusion-driven cloud-edge disocclusion
  needs the standard depth-rejection (Karis 2014) to avoid
  trailing.

**Implementation sketch**:
- Add prev-frame view/proj to FrameUniforms (already partially
  there for some passes).
- Add a `cloud_history` RT in `ps-clouds`.
- New compute pass between cloud march and composite that:
  reads current march output, reads history via reprojection,
  AABB-clamps to current neighbourhood (5×5 luminance min/max),
  EMAs at `1/N` weight.
- Disable on `freeze_time` and golden-bless paths.

This pairs naturally with the half-res march in (A): half-res +
TAA is the canonical AAA cloud configuration (Frostbite, Decima,
Horizon Zero Dawn).

#### C. Things from the article *not* worth adopting

- **Single HG phase**: Jendersie & d'Eon 2023 is strictly better.
- **Directional-derivative diffuse**: a clever cheap trick, but
  PedalSky already has multi-octave HG + Beer-Powder + sky-view
  ambient + lightning in-scatter. The directional derivative would
  be a regression.
- **SDF cloud shapes**: the article uses these because it has no
  weather map. PedalSky has synthesis, so this is a non-starter.
- **6-octave FBM in shader**: PedalSky bakes Perlin-Worley + Worley
  FBM into 128³/32³ volumes at startup. Per-frame FBM evaluation
  in shader would be a regression.

---

## 3. Volumetric Lighting / God Rays

### Article approach (Heckel, "On Shaping Light")

- Post-process raymarched volumetric light with **per-light shadow
  maps**.
- Each light has a virtual `PerspectiveCamera`, a render target
  (default 256×256, tested up to 1024×1024), and a `DepthTexture`.
- Pixel-side: reconstruct world ray from NDC, march N steps along
  the ray, at each sample:
  - Compute `lightClipPos` for this sample, look up shadow map,
    test occlusion.
  - Compute distance attenuation `exp(-0.05 · d)` and
    Henyey-Greenstein phase `HG(dot(rayDir, -lightDir))`.
  - Accumulate `luminance × transmittance × density × dt`.
  - Beer's law transmittance multiplied per step.
- Default parameters: 250 steps → 50 with blue-noise dither
  (5000 → 100 iterations equivalent quality).
- Step size 0.05 → 0.5 with dither.
- Fog density modulated by FBM (`NOISE_OCTAVES = 3`, frequency 0.5).
- SDF shape function for cylindrical / cone light volumes.
- Cube camera workaround for omnidirectional point lights.

References cited: hmans.dev, ShaderToy jackdavenport, Vlad
Ponomarenko (visual inspirations — no academic papers).

### PedalSky approach

(`crates/ps-godrays/`, `shaders/godrays/radial.wgsl`)

- **Screen-space radial blur** — Tatarchuk / Crytek style, *not*
  raymarched.
- Sun direction projected to screen NDC via `view_proj × vec4(dir, 0)`.
  When the projected w is positive and the NDC is within `|x,y| < 2`,
  the sun is "on screen" (or near-screen for inward rays).
- **Half-res scratch RT** (`HALF_RES_FACTOR = 2`, 1/4 area).
- For each output pixel: step toward `sun_uv` in N samples
  (configurable via `config.render.godrays.samples`); each sample
  reads HDR, applies a `bright_pass(rgb, threshold)` (per-channel
  luminance-mask soft threshold), accumulates with exponential
  decay weight.
- Final accumulation divided by `n_samples` to keep the integral
  sample-count-independent.
- **Additive composite** into HDR before tonemap.
- HDR copy needed because radial pass reads HDR while a separate
  pass writes it.
- No shadow map, no per-cloud-step volumetric integration, no
  in-scene volume.

### Why the two designs diverge

The article assumes a scene with **discrete occluders** (asteroids,
buildings, foliage) and wants the per-light shafts cast through that
geometry. Its shadow map is the right primitive.

PedalSky is a **weather/atmosphere renderer**: there are no scene
occluders other than clouds, and the cloud march has *already*
applied per-channel transmittance to the HDR target. The bright
pixels at the sun direction are exactly the pixels not occluded by
cloud; a radial blur from those bright pixels is mathematically
equivalent (within a constant) to a one-bounce volumetric
in-scatter integration where the volume is uniform and the occluder
is the cloud field.

So the radial-blur approach is correct *for what we're rendering*.

### Where the gap actually is

What PedalSky lacks is **in-cloud light shafts** — the visibly
bright trail you see when sun pierces a thick cloud and the shaft
of light is itself volumetric (the dust/moisture inside the cloud
scattering the sun back to the camera). The cloud march does the
single-scatter integration plus multi-octave HG, but:

- The shaft is *visible against the cloud body itself* and that
  requires high cross-cloud-step variance (which the multi-octave
  loop tends to flatten).
- Sun shafts piercing a *cloud gap* are the radial-blur case
  PedalSky already handles.

There are two ways to address in-cloud shafts:

1. **Add a "back-scatter brightness amplifier"** to the cloud march
   when `cos_theta` is near 1 (forward scatter from the sun). The
   existing `cloud_phase` already does this through HG+Draine, but
   the `multi_scatter_c` octave decay (default 0.5) means the
   second octave's `g` halves and the forward peak softens. A
   dedicated single-scatter-bias parameter on the cloud march
   could let the user dial in a stronger forward peak on the
   first octave without breaking the Jendersie & d'Eon fit.

2. **A second-pass volumetric light shaft** that raymarches through
   the cloud transmittance RT, accumulating in-scatter weighted by
   `(1 − T_lum) · HG(cos_theta, g_strong)`. This is essentially the
   article's design, but with the cloud transmittance buffer
   playing the role of the shadow map.

Option 1 is cheaper and reuses existing infrastructure; option 2 is
more flexible but adds a second cloud-traversal pass. For a weather
renderer, option 1 is probably the right starting point.

### Recommendation

**Keep the radial blur** — it is the correct technique for
sun-through-cloud-gap shafts. Consider exposing a per-octave
"forward bias" on the cloud phase function (option 1 above) to
strengthen in-cloud bright shafts without a second pass. Only
consider a true raymarched volumetric pass if scene-side geometry
(buildings, terrain features) ever gets added to PedalSky — at
that point the article's shadow-map design becomes load-bearing.

---

## 4. Performance optimisation cross-reference

Optimisations explicitly called out in the articles that PedalSky
already has:

- Multi-LUT decomposition of atmosphere (transmittance / MS /
  sky-view / AP) so the per-pixel sky pass is one lookup.
- Blue-noise spatial dither on cloud march.
- Bright-pass + half-res scratch on godrays.
- Per-channel transmittance composite (PedalSky goes further with
  dual-source blending).

Optimisations in the articles that PedalSky does **not** have:

| Optimisation | Source | Estimated win | Risk |
|---|---|---|---|
| Half-res cloud march + bicubic upsample | Cloud article | ~3-4× cloud cost | Medium — needs care on dual-source path |
| TAA temporal reprojection on clouds | Cloud + lighting articles | ~4-16× perceived samples | Medium — disocclusion handling |
| Cone-tap light sampling (Schneider Nubis) | not in articles, but referenced | Improves self-shadow quality at same step count | Low — drop-in replacement for straight light march |

Optimisations PedalSky has that the articles don't mention:

- Multi-scatter LUT closed-form geometric series.
- 3D froxel AP with exponential Z spacing.
- Energy-conserving step integration `(S − S·T)/σ_t`.
- Per-layer wind offset (vs per-sample, which causes vertical
  streaking — PedalSky learned this the hard way in Phase 14.F).
- Numerically-stable height delta at planet scale.

---

## 5. Ranked recommendations

Numbered by expected ROI (visual + perf vs implementation cost).

### Recommendation 1 — High value, medium effort
**Half-resolution cloud march + bicubic upsample.**

Add `cloud_render_scale` to config (1.0, 0.5, 0.25); allocate
cloud RT at scaled size; insert a Catmull-Rom bicubic upsample
stage before the dual-source blend. Gate behind a Debug-panel
toggle for A/B verification against goldens.

**Expected**: ~3–4× cloud throughput, freeing budget for either
higher step counts or higher resolution.

**Tradeoffs**: Edge ringing risk on thin cumulus boundaries. The
`cloud_type_grid` lookup is already at 128² so half-res march
won't disturb it. Dual-source blend needs both attachments
upsampled coherently — non-trivial but well-documented.

### Recommendation 2 — High value, higher effort
**TAA temporal reprojection on cloud render target.**

Add prev-frame view/proj to FrameUniforms, allocate a
`cloud_history` RT in `ps-clouds`, introduce a reprojection +
neighbourhood-clamp + EMA pass between cloud march and composite.
Disable on `freeze_time` / `wind_drift_strength = 0` / golden-bless.

**Expected**: 4–16× effective sample count, or equivalently drop
`cloud_steps` from 192 → 48 at parity.

**Tradeoffs**: Ghosting on disocclusion (camera rotation +
fast-moving cloud cells). Karis 2014 5×5 neighbourhood clamp
addresses this. Adds GPU memory (one extra HDR RT) and complicates
the `freeze_time` semantics.

**Synergy with R1**: best deployed together. Half-res march + TAA
is the canonical AAA cloud configuration.

### Recommendation 3 — Medium value, low effort
**Cone-tap light sampling for the cloud's `march_to_light`.**

The current straight light march samples 6 points along the sun
direction from the cloud sample. Schneider Nubis 2017 replaces
this with 5 forward samples + 1 wide-angle "anti-shadow" tap that
brightens the self-shadowed core, giving cumulus their
characteristic silver-lined edges without raising step count.

**Expected**: visibly better self-shadowing on cumulus tops at
no perf cost.

**Tradeoffs**: Schneider's exact tap pattern is well-documented;
risk is low. Adjust `light_steps` semantics in UI panel.

### Recommendation 4 — Speculative, look-and-feel
**Per-octave forward bias on the cloud phase function.**

Add a per-layer or global `forward_bias` parameter that biases the
first-octave HG `g` in the multi-octave multi-scatter loop without
touching subsequent octaves. Lets the user dial in stronger
"in-cloud sun shafts" — the visible bright trail when sun pierces
a cloud body — without a second raymarch pass.

**Expected**: better-looking sunbeams without architectural changes.

**Tradeoffs**: Breaks the Jendersie & d'Eon 2023 fit if pushed too
hard. Calibrate carefully against reference photography of
cumulonimbus side-lit by low sun.

### Recommendation 5 — Skip
**Switch godrays from radial blur to raymarched shadow-map
volumetric.**

The article's volumetric-light raymarch is designed for scenes
with discrete occluders (asteroids, etc.). PedalSky's occluders
are clouds, and clouds already write per-channel transmittance to
HDR before the godrays pass runs. The radial blur picks up
sun-shaft-through-cloud-gap exactly. Switching to a raymarched
pass would add cost (a second cloud traversal) without improving
the dominant case.

Reconsider if/when PedalSky adds scene geometry (terrain features,
buildings, foliage).

### Recommendation 6 — Skip
**Anything from the article's atmosphere model.**

PedalSky's atmosphere is the article's atmosphere, strictly
extended. No changes recommended.

---

## 6. Suggested next step

If you want to start with one piece, **Recommendation 1
(half-res cloud march + bicubic upsample)** is the most
self-contained: bounded blast radius, clear A/B test path via the
golden-image regression, and unlocks Recommendation 2 (TAA) which
otherwise has the right shape but pays off less without the
half-res cost reduction first.

Open questions before implementation:

- Should the cloud render-scale be exposed in the live UI panel
  (Debug section, alongside step counts) or only in the
  `pedalsky.toml` config?
- Do we want a single bicubic kernel (Catmull-Rom 16-tap) or to
  also support cheaper FSR-style upsampling for the 0.25× case?
- For TAA later: do we already track prev-frame view/proj
  anywhere? Some passes (postprocess auto-exposure) plausibly do.
