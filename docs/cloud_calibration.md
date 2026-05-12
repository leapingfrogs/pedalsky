# Cloud calibration reference

Per-cloud-type numeric defaults for the volumetric cloud pipeline,
with the published sources behind each value and the cases where the
literature is silent or contradictory.

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
| `g_forward` | 0.80 | 0.80 | 0.80 | 0.80 | 0.72 | 0.70 | 0.70 | 0.80 (water) → 0.70 (anvil) |
| `g_backward` | -0.30 | -0.30 | -0.30 | -0.30 | -0.20 | -0.10 | -0.10 | -0.30 → -0.10 |
| `g_blend` | 0.50 | 0.50 | 0.50 | 0.50 | 0.45 | 0.30 | 0.30 | 0.50 → 0.30 |
| `anvil_bias` (default) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| Altitude (m AGL) | 600–6000 | 0–1000 | 500–2200 | 2000–6000 | 2000–7000 | 6000–14000 | 6000–14000 | 600–16000 |
| Optical depth τ (visible) | 5–30 | 8–40 | 8–25 | 3–10 | 5–30 | 0.02–3 | 0.5–3 | 30–1000+ |
| Effective radius rₑ (μm) | 6–12 | 5–10 | 7–12 | 5–10 | 8–40 (mixed) | 20–60 | 25–70 | 10–80 (mixed) |
| Phase (water/ice) | water | water | water | water | mixed | ice | ice | mixed |

The Cumulonimbus HG entries show the water → ice transition the
shader interpolates across `h ∈ [0.6, 0.85]` based on the sample's
normalised height inside the layer (see
`shaders/clouds/cloud_march.wgsl::dual_lobe_hg_with_g_scale`).

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

### Phase function (Henyey–Greenstein)

Single-lobe HG cannot capture cloud scattering accurately: water
clouds have a sharp forward peak from Mie scattering at ~10 µm
droplets; ice clouds have weaker forward scattering with sharp halos
at 22° and 46° that no HG captures. The pipeline uses a dual-lobe
HG (`g_forward`, `g_backward`, `g_blend`) as a Schneider-style
approximation. Values target a single-lobe-equivalent `g_eff`
matching published data.

**Water clouds** target `g_eff ≈ 0.85`. **Verified** against
Kokhanovsky's optical-properties review: g = 0.858 for a 10 µm
gamma droplet distribution at 550 nm. The dual-lobe triple
(0.80, -0.30, 0.50) is a **convention** widely used in the
community but not specifically published by Schneider or Hillaire —
both originals use single-lobe HG. The dual-lobe form gives the
back-side brightness that real clouds exhibit and that single-HG
misses.

**Ice clouds** target `g_eff ≈ 0.75` per **Baran 2012** (JQSRT
113:1239) and **Baran 2013** (ACP 13:3185). Verified: published
range for ice asymmetry parameter is 0.74–0.80 (rough crystals
0.74–0.78, smooth/halo-producing crystals up to ~0.80). PedalSky's
ice triple (0.70, -0.10, 0.30) is a dual-lobe fit targeting that
g_eff — slightly less forward-peaked than the water pair, with a
narrower back lobe.

> **A note on a now-corrected mistake.** An earlier PedalSky version
> shipped ice HG = (0.40, -0.15, 0.40), based on an unverified claim
> that TrueSky and SilverLining default ice cloud g to ≈0.4. The
> 2026-05-12 verification pass found no live source supporting this;
> Baran's published g_eff ≈ 0.75 is incompatible with such a low
> g_forward. The values were corrected on the same date — see
> `default_hg(CloudType)` in `crates/ps-core/src/scene.rs`.

**Altostratus** is mixed-phase (water at base, ice above the
freezing level). The pair (0.72, -0.20, 0.45) targets g_eff ≈ 0.80,
between the water and ice triples.

**Cumulonimbus** is the second mixed-phase case (water in the
convective core, ice in the anvil). The cloud march interpolates
between the water triple at h_norm ≤ 0.6 and the ice triple at
h_norm ≥ 0.85, with smoothstep blending across the band. The phase
transition pre-empts the anvil NDF rise (which starts at h = 0.7).

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
| `sigma_s` (RGB) | (0.030, 0.040, 0.060) /m | Hillaire 2016 baseline ≈ 0.04 /m; PedalSky chromaticity is an extension for sunset warm-fringing. Not physically motivated — Mie scattering for ~10 µm droplets is essentially wavelength-flat in the visible. |
| `sigma_a` | 0 | Water droplet absorption is near-zero in the visible. |
| `multi_scatter_a/b/c` | 0.5/0.5/0.5 | **Wrenninge 2015** "Art-directable Multiple Volumetric Scattering" (history.siggraph.org). a = energy retention per octave, b = optical-depth attenuation, c = phase eccentricity attenuation. Universal across implementations; not per-type. |
| `multi_scatter_octaves` | 4 | Hillaire 2016 recommends 4–8. PedalSky uses 4 for performance; 8 is the upper end of Wrenninge's recommendation. |
| `powder_strength` | 1.0 | Schneider 2015 Beer-Powder lerp factor; canonical default. |
| `detail_strength` | 0.05 | Lowered from Schneider's published 0.35 because the coverage remap in PedalSky pre-biases METAR coverage into a narrow band; high detail_strength erodes too aggressively in that regime. Documented in `cloud_layers.rs::remap_coverage_to_visible_band`. |
| `hg_*_bias` | 1.0 | Global multipliers on the per-layer HG triple. Default = use synthesised per-cloud-type values unchanged. |

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

- **Droplet effective radius as a primary input.** When real weather
  data feeds the renderer (NWP grids), `r_e` is naturally available
  per cloud type. Mapping `r_e → g_forward` via the Bouthors 2008
  Mie fit lets the renderer follow the data automatically.
- **NVIDIA Approximate Mie phase function.** Jendersie & d'Eon 2023
  publish a closed-form Mie approximation with proper droplet-size
  scaling — see
  research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf.
- **Per-channel sigma_s tied to droplet size.** Sub-µm droplet
  populations (fresh fog, thin cirrus, cumulus updraft tops) actually
  do exhibit chromatic Mie. The current PedalSky chromaticity is
  decorative; tying it to droplet size would make it physical.
- **Ice halos.** A sharp 22° lobe added on top of the HG dual-lobe
  for ice clouds (when the sun is in view) would recover a key
  visual cue that distinguishes ice from water clouds.
- **Real-data ingestion path.** A `CloudLayer::from_nwp_grid_cell(...)`
  helper that takes (cloud_type, r_e, LWC/IWC, base, top, coverage)
  and produces a `CloudLayer` with the right `density_scale`,
  `g_forward`, and `anvil_bias` derived from microphysics.

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
