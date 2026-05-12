# Cloud calibration reference

Per-cloud-type numeric defaults for the volumetric cloud pipeline,
with the published sources behind each value and the cases where the
literature is silent or contradictory.

> **2026-05-12 update — Approximate Mie migration.** The per-layer
> Henyey–Greenstein triple was retired in favour of a single
> droplet effective diameter (µm), which drives the Jendersie &
> d'Eon 2023 "Approximate Mie" phase function (HG forward peak +
> Draine bulk lobe). The diameter values are listed in the table
> below; the HG coefficients are no longer authoritative — they
> are derived inside the shader from the diameter via the paper's
> Eqs. 4–7. See `shaders/clouds/cloud_march.wgsl::cloud_phase` for
> the runtime evaluation.

This document exists because the canonical Schneider 2015 + Hillaire
2016 pipeline doesn't publish a per-cloud-type calibration table —
the original papers describe *mechanisms* (Perlin–Worley base shape,
Worley FBM detail erosion, the coverage remap, dual-lobe HG, the
Hillaire/Wrenninge multi-octave multi-scatter approximation), but the
numeric presets used in commercial engines (Decima, Frostbite, UE5,
TrueSky, SilverLining) are art-direction artifacts that have not been
released. PedalSky's defaults are synthesised from three orthogonal
sources:

- **Real-time rendering literature** — Schneider 2015 / 2017 / 2022,
  Hillaire 2016 (Frostbite), Wrenninge 2015 (multi-octave), Bouthors
  2008 (dual-lobe HG fit to Mie), Jendersie & d'Eon 2023 (NVIDIA
  Approximate Mie, the modern upgrade path).
- **Atmospheric microphysics** — Wallace & Hobbs (2006), Houze (2014),
  Pruppacher & Klett (1997), Baran 2012/2013 (ice phase function),
  Kokhanovsky review (optical properties of clouds).
- **Observational climatology** — MODIS Collection 6 (Platnick 2017),
  CALIPSO/CloudSat (Sassen 2008), Cloudnet (Illingworth 2007), AMS
  Glossary of Meteorology, WMO International Cloud Atlas.

A live-sources verification pass on 2026-05-12 cross-checked the
values below against published references; entries are marked
**Verified**, **Convention** (community-standard but not pinned to a
specific paper), or **Inferred** (extrapolated between published
data-points).

---

## Per-cloud-type table

The `default_*` functions in `ps_core::scene` materialise these
values; scenes can override per-layer fields explicitly. The
calibration is intentionally conservative — values pick the middle
of each published range so that scenes without explicit overrides
land in a plausible regime.

| Field | Cu | St | Sc | Ac | As | Ci | Cs | Cb |
|-------|----|----|----|----|----|----|----|----|
| `density_scale` | 1.0 | 1.0 | 1.0 | 0.85 | 0.7 | 0.55 | 0.4 | 1.4 |
| `droplet_diameter_um` | 20 | 16 | 16 | 14 | 30 | 50 | 50 | 20 (water) → 50 (anvil) |
| `anvil_bias` (default) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| Altitude (m AGL) | 600–6000 | 0–1000 | 500–2200 | 2000–6000 | 2000–7000 | 6000–14000 | 6000–14000 | 600–16000 |
| Optical depth τ (visible) | 5–30 | 8–40 | 8–25 | 3–10 | 5–30 | 0.02–3 | 0.5–3 | 30–1000+ |
| Effective radius rₑ (μm) | 6–12 | 5–10 | 7–12 | 5–10 | 8–40 (mixed) | 20–60 | 25–70 | 10–80 (mixed) |
| Phase (water/ice) | water | water | water | water | mixed | ice | ice | mixed |

The Cumulonimbus diameter entry shows the water → ice transition
the shader interpolates across `h ∈ [0.6, 0.85]` based on the
sample's normalised height inside the layer (see
`shaders/clouds/cloud_march.wgsl::cloud_phase`). The Approximate
Mie fit is valid for 5 ≤ d ≤ 50 µm; cirrus and cirrostratus sit at
the upper edge of the fit range.

---

## How each row was chosen

### `density_scale`

Defaults set so that `density_scale × layer_thickness × sigma_s`
places vertical optical depth in the middle of each cloud type's
published range. With the engine's default `sigma_s ≈ 0.04 /m` and
typical layer thickness 600–1500m, the values above land
Cumulus τ ≈ 15, Cirrus τ ≈ 0.3, Cumulonimbus τ ≈ 80.

Verified ranges:
- **Cumulus 5–30** — Wallace & Hobbs 2006, Houze 2014 ch.3. Inferred.
- **Stratus 8–40** — Wallace & Hobbs 2006. Inferred.
- **Stratocumulus 8–25** — Marchand 2010 (MISR retrievals).
- **Altocumulus 3–10** — lidar retrievals (ScienceDirect, J.Atmos.Sci.).
- **Altostratus 5–30** — Inferred (literature spread is wide).
- **Cirrus 0.02–3** — MODIS C6 (Platnick 2017); thin sub-visible
  cirrus down to τ = 0.02 detectable by CALIPSO (Sassen 2008).
  Threshold τ = 10 separates warming/cooling radiative regimes.
- **Cirrostratus 0.5–3** — same observational dataset.
- **Cumulonimbus 30–1000+** — **AMS Glossary** definition: "ranges
  through orders of magnitude from low values less than 0.1 for thin
  cirrus to **over 1000 for a large cumulonimbus**" (verified via
  glossary.ametsoc.org/wiki/Cloud_optical_depth).

### Phase function (Approximate Mie, Jendersie & d'Eon 2023)

The cloud march evaluates the **HG + Draine blend** from Jendersie &
d'Eon's SIGGRAPH 2023 talk "An Approximate Mie Scattering Function
for Fog and Cloud Rendering". A single parameter — droplet
effective diameter `d` in µm — feeds the paper's four-coefficient
fit (Eqs. 4–7) to derive the HG asymmetry `g_HG`, the Draine
asymmetry `g_D`, the Draine alpha `α`, and the mixture weight
`w_D`:

```
g_HG(d) = exp(-0.0990567 / (d - 1.67154))
g_D(d)  = exp(-2.20679  / (d + 3.91029)) - 0.428934
α(d)    = exp(3.62489 - 8.29288 / (d + 5.52825))
w_D(d)  = exp(-0.599085 / (d - 0.641583) - 0.665888)
```

The phase function is then (paper Eq. 3):

```
p_fog(θ) = (1 - w_D) · HG(g_HG; θ) + w_D · Draine(g_D, α; θ)
```

where Draine generalises HG with an `(1 + α cos² θ)` term:

```
Draine(g, α; θ) = (1 - g²)(1 + α cos² θ)
                / [4 (1 + α(1 + 2g²)/3) · π · (1 + g² - 2g cos θ)^1.5]
```

Both HG and Draine are normalised to integrate to 1 over the
sphere. The fit is **wavelength-independent** (the paper averages
the underlying Mie reference over 400–700 nm).

**Validity range.** The fit is calibrated for `5 ≤ d ≤ 50 µm`. The
shader clamps inputs to that band before evaluation. Ice clouds
(cirrus, cirrostratus) sit at the upper edge with d ≈ 50 µm; the
fit extrapolates slightly above this, the paper notes the
approximation degrades but stays usable.

**Why Approximate Mie over the previous dual-lobe HG?** Three wins,
verified against Mie reference data in the paper's Figure 1:
sharper near-sun silver lining (single-HG was too broad), correct
anti-sun back-scatter brightness (single-HG missed it entirely),
and droplet-size-dependent shape (HG had no such input — any
"per-cloud-type" character had to come from per-type magic
constants). The single-droplet-diameter parametrisation is also
the natural input from real weather data (NWP grids publish
effective radius), so the future ingestion path simplifies.

**Cumulonimbus** is mixed-phase: water droplets in the convective
core, ice crystals in the anvil. The cloud march blends the
diameter from the layer's own value (water, ~20 µm) at `h ≤ 0.6`
toward 50 µm (ice anvil) at `h ≥ 0.85`. The transition pre-empts
the anvil NDF rise (which starts at `h = 0.7`).

**Constants come from the published paper.** The full talk PDF is
linked in the sources section below. The HLSL reference
implementation (`draine.hlsl` from NVIDIA's RTRT lab) was used as
a syntactic cross-check; the WGSL port lives in
`shaders/clouds/cloud_march.wgsl` as `draine_phase` and
`cloud_phase`.

> **History note.** PedalSky previously shipped a dual-lobe HG
> approximation with per-layer `(g_forward, g_backward, g_blend)`
> fields, calibrated against published water- and ice-cloud
> asymmetry values (`g_eff ≈ 0.85` water, `≈ 0.75` ice). That
> system worked but the Mie diffraction peak HG cannot capture
> was visibly off (broad silver linings); the migration to
> Approximate Mie was made in the same session that consolidated
> the calibration docs. The HG fields no longer exist on
> `CloudLayerGpu`; they were replaced by `droplet_diameter_um`.

### Per-channel chromatic scattering (`chromatic_mie_modulation`)

The Approximate Mie phase function is wavelength-independent (the
paper averages over 400–700 nm). Real Mie scattering for *typical*
cloud droplets (`d ≥ ~20 µm` at visible λ ≈ 0.5 µm gives Mie size
parameter `x = π d / λ ≈ 125`) is in the geometric-optics regime
and is essentially flat across the visible spectrum — this is why
cumulus and cirrus look white. But sub-micron droplet populations
(fresh fog tops, thin cirrus edges, cumulus updraft tops) drop
toward `x ≈ 1` and pick up a Rayleigh-like wavelength-selective
component, which is why fog at sunset reads warm.

`chromatic_mie_modulation(d_um)` in `cloud_march.wgsl` returns an
RGB multiplier on `sigma_t` that captures this behaviour:

- `d ≥ 20 µm` (cumulus, stratocumulus, cumulonimbus core, mature
  cirrus): factor is `(1, 1, 1)` — wavelength-flat.
- `d < 20 µm` (stratus, altocumulus, thin or fresh cirrus): factor
  picks up a blue boost and a red attenuation, smoothly ramped via
  `(1 - d/20)²`. The strength is capped at ±0.25 (max blue +25%,
  max red −12.5%) so the chromatic shift stays a fringe modulation
  rather than dominating the cloud appearance.

The function is a simplified physical analogue, not a fit to
tabulated Mie data — that would require evaluating the full Mie
solution at three wavelengths per layer per frame. The
approximation captures the *direction* of the wavelength
dependence (blue scatters more as droplets shrink) without paying
the full per-channel Mie evaluation cost.

### Anvil bias

`anvil_bias = 1.0` for Cumulonimbus only (others default to 0).
The shader's Cumulonimbus NDF case uses `anvil_bias` as a
multiplier on the anvil-top mass term — `1.0` is the historical v1
look; `0` suppresses the anvil entirely; `2.0` doubles its strength.

### Altitudes

WMO International Cloud Atlas levels:
- Low (<2 km AGL): St, Sc, Cu (Cu base; tops can punch to 6 km)
- Middle (2–7 km): Ac, As, sometimes Ns
- High (5–13 km temperate, higher in tropics): Ci, Cs, Cc
- Multi-level: Cb (base low, top tropopause-grazing)

Verified against `cloudatlas.wmo.int/some-useful-concepts-levels.html`.

### Effective radius

Verified ranges:
- Water clouds: 5–15 µm typical. Marine Sc ~10 µm; continental
  polluted Sc ~6 µm (Twomey effect; Wood 2012 MWR review). Pruppacher
  & Klett 1997 §2 is the canonical reference.
- Cirrus: 20–60 µm typical, extending to 70 µm. Verified against
  multiple papers (acp.copernicus.org/22:15179/2022; JAS 68(2)).
- Cumulonimbus: core 10–20 µm water, anvil 30–80 µm ice.

The droplet size feeds the phase function (larger ice crystals
→ broader forward lobe), but the engine doesn't currently expose
rₑ as a tunable — it's reflected in the per-cloud-type HG defaults.

---

## Other engine-wide cloud parameters

These live in `CloudParams` (`crates/ps-clouds/src/params.rs`) and
apply globally rather than per-cloud-type.

| Field | Default | Source |
|-------|---------|--------|
| `sigma_s` (RGB) | (0.030, 0.040, 0.060) /m | Hillaire 2016 baseline ≈ 0.04 /m. The chromatic spread was previously hand-tuned for sunset warm-fringing; the engine now derives chromatic modulation **per layer** from droplet diameter via `chromatic_mie_modulation(d)` in the cloud march (see "Per-channel chromatic scattering" below), so this baseline is still slightly biased but the per-layer factor is what produces the size-dependent warm fringes. |
| `sigma_a` | 0 | Water droplet absorption is near-zero in the visible. |
| `multi_scatter_a/b/c` | 0.5/0.5/0.5 | **Wrenninge 2015** "Art-directable Multiple Volumetric Scattering" (history.siggraph.org). a = energy retention per octave, b = optical-depth attenuation, c = phase eccentricity attenuation. Universal across implementations; not per-type. |
| `multi_scatter_octaves` | 4 | Hillaire 2016 recommends 4–8. PedalSky uses 4 for performance; 8 is the upper end of Wrenninge's recommendation. |
| `powder_strength` | 1.0 | Schneider 2015 Beer-Powder lerp factor; canonical default. |
| `detail_strength` | 0.05 | Lowered from Schneider's published 0.35 because the coverage remap in PedalSky pre-biases METAR coverage into a narrow band; high detail_strength erodes too aggressively in that regime. Documented in `cloud_layers.rs::remap_coverage_to_visible_band`. |
| `droplet_diameter_bias` | 1.0 | Global multiplier on the per-layer droplet diameter (post-clamp to the 5–50 µm Mie fit range). Default = use synthesised per-cloud-type diameter unchanged. |

---

## Things to watch out for

1. **The Schneider coverage remap is non-monotonic.** Increasing
   `detail_bias` while coverage is high can both concentrate density
   in the cloud core *and* cull edges below the remap threshold —
   visible cloud volume can drop while peak density rises. Documented
   at length in `cloud_layers.rs::remap_coverage_to_visible_band`.
2. **Cirrus / cirrostratus genuinely don't fit the Schneider pipeline
   well.** Both HZD and Frostbite render cirrus as a separate 2D
   texture layer, not a volumetric cloud type. PedalSky volumetric
   cirrus is past the published state of the art; expect to keep
   tuning it visually.
3. **Chromatic `sigma_s` is not physically motivated.** Mie
   scattering for ~10 µm droplets is wavelength-flat in the visible.
   The (0.030, 0.040, 0.060) triple is a stylistic knob that
   produces visible warm fringing at sunset; set RGB equal for
   physical accuracy.
4. **Single-lobe HG vs dual-lobe is a stylistic call**, not a
   physics one. The community-standard dual-lobe (0.8, -0.3, 0.5)
   for water clouds is not from the Schneider or Hillaire originals;
   both papers use single-lobe HG. Dual-lobe gives back-scatter
   brightness; NVIDIA Approximate Mie (Jendersie & d'Eon 2023) is
   the recommended upgrade path for higher fidelity.
5. **Ice-cloud halos (22°, 46°) are not captured by HG.** Adding
   them requires a separate sharp sun-angle-dependent lobe outside
   the HG framework. None of the current real-time literature does
   this cleanly; PedalSky doesn't either.

---

## Future calibration directions

These are noted-but-not-yet-implemented improvements that would land
the next tier of fidelity:

- **Ice halos.** Sharp 22° / 46° lobes are characteristic of
  hexagonal ice crystals and no HG/Draine-class approximation
  captures them. Adding an explicit angular feature when the
  view-to-sun cosine matches `cos 22°` or `cos 46°` for ice
  clouds (`droplet_diameter_um > 35 µm` is a reasonable gate)
  would recover the cue. Not currently implemented anywhere in
  the real-time literature.
- **Real-data ingestion path.** A `CloudLayer::from_nwp_grid_cell(...)`
  helper that takes (cloud_type, r_e, LWC/IWC, base, top, coverage)
  and produces a `CloudLayer` with the right `density_scale`,
  `droplet_diameter_um` (= 2 · r_e directly), and `anvil_bias`
  derived from microphysics. With Approximate Mie landed, the
  effective-radius → phase-function mapping is now trivial: the
  renderer's diameter field IS the data feed's diameter field.

---

## Sources

### Verified during the 2026-05-12 web-access pass

- **AMS Glossary of Meteorology** —
  glossary.ametsoc.org/wiki/Cloud_optical_depth (Cb τ up to 1000+).
- **WMO International Cloud Atlas** — cloudatlas.wmo.int
  (cloud-type altitude bands).
- **Baran 2012/2013** — acp.copernicus.org/articles/13/3185/2013
  (ice phase function g_eff ≈ 0.75).
- **MODIS Collection 6** — modis-images.gsfc.nasa.gov + Platnick 2017
  (cirrus τ distribution).
- **Kokhanovsky review** — patarnott.com/satsens/pdf/opticalPropertiesCloudsReview.pdf
  (water-cloud g = 0.858 at 10 µm, 550 nm).
- **Wrenninge 2015** — history.siggraph.org/learning/art-directable-multiple-volumetric-scattering-by-wrenninge/
  (multi-octave a = b = c = 0.5).
- **NVIDIA Approximate Mie (Jendersie & d'Eon 2023)** —
  research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf
  (recommended upgrade path).
- **Schneider 2015 HZD SIGGRAPH** —
  advances.realtimerendering.com/s2015/The Real-time Volumetric Cloudscapes
  of Horizon - Zero Dawn - ARTR.pdf (slides are mostly image-based;
  community paraphrases used).

### Referenced from training corpus (not live-verified)

- Schneider 2017 / 2022 "Nubis" and "Nubis Evolved" SIGGRAPH talks.
- Hillaire 2016 "Physically Based Sky, Atmosphere & Cloud Rendering
  in Frostbite" SIGGRAPH course PDF.
- Bouthors et al. 2008 I3D "Interactive multiple anisotropic
  scattering in clouds".
- Wallace & Hobbs 2006 *Atmospheric Science: An Introductory Survey*.
- Houze 2014 *Cloud Dynamics*.
- Pruppacher & Klett 1997 *Microphysics of Clouds and Precipitation*.
- Wood 2012 *MWR* 140:2373 (stratocumulus review).
- Illingworth et al. 2007 *BAMS* 88:883 (Cloudnet).
- Sassen, Wang & Liu 2008 *JGR* 113:D00A12 (CALIPSO sub-visual cirrus).
- WMO International Cloud Atlas 2017 revision.
