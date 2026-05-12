# PedalSky

A composable, dependency-injected wgpu-based weather renderer test harness.

PedalSky takes a weather scene description (cloud layers, surface conditions,
precipitation, time and location) and renders a physically based, photometric
sky + clouds + ground + precipitation scene from a fly camera. Every tunable
parameter is exposed as a UI slider; every renderable is a separately
toggleable subsystem.

The full architectural specification is in
`Pedalsky corrected implementation plan.md`. Phases 0 through 11 are
complete; a per-phase audit lives at the bottom of this file.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Workspace, GPU init, HDR + reverse-Z, fly camera, ACES tone-mapper | Done |
| 1 | Config schema, `RenderSubsystem` trait, `AppBuilder`, hot-reload | Done |
| 2 | `WorldClock`, NREL SPA sun, Meeus 47 moon, TOA solar illuminance | Done |
| 3 | Weather synthesis pipeline (atmosphere, weather map, wind, mask, layers) | Done |
| 4 | Per-frame uniforms, render-graph executor, shared shader helpers | Done |
| 5 | Hillaire 2020 atmosphere — 4 LUTs, sky raymarch, AP application | Done |
| 6 | Schneider/Hillaire volumetric clouds — noise volumes, march, composite | Done |
| 7 | PBR ground + Lagarde wet surface + snow + Halton-spawned ripples | Done |
| 8 | Precipitation — particle compute + far rain streaks + Marshall-Palmer | Done |
| 9 | Composition, ACES, EV100 exposure, debug auto-exposure | Done |
| 10 | egui panel system: world / render / atmosphere / clouds / wet / precip / debug | Done |
| 11 | 8 reference scenes, headless `render` subcommand, golden-image regression | Done |

All plan-mandated cross-cutting items are wired (commits b on master):
`--seed` and `--gpu-trace` CLI flags, WGSL shader hot-reload,
GPU timestamp queries surfacing in the Debug panel, naga std140
build-time linter, and camera fov/near/speed UI sliders. The plan's
explicitly-stretch side-by-side comparison mode (§10.5) is the only
remaining gap and is tracked as a "stretch" goal in the plan itself.

## Install and build

Requires Rust 1.85+ and a wgpu-compatible GPU (the project is developed
against Vulkan on NVIDIA but should run on any back-end wgpu supports).

```sh
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

134 tests run across the workspace; all should pass.

## Running the live app

```sh
cargo run --release --bin ps-app
```

The window opens at the resolution set in `pedalsky.toml [window]`. The
left side has the egui control panel; the rest is the render viewport.

Camera bindings:

| Key | Action |
|---|---|
| W / A / S / D | Forward / strafe |
| Space / Ctrl  | Up / down |
| Q / E         | Roll left / right |
| Shift         | Sprint (×4) |
| LMB-drag      | Mouse-look |
| Esc           | Release mouse |
| F4            | Quit |

The fly camera's position, pitch, and yaw are persistent for the session.

## Running the test harness

The headless render subcommand drives the same render path as the live
app but without a window. It produces four artifacts per invocation:
the tonemapped PNG, the linear HDR EXR, a JSON dump of the synthesised
weather state, and a TOML log of every config value used.

```sh
cargo run --bin ps-app -- render \
    --scene tests/scenes/broken_cumulus_afternoon.toml \
    --time 2026-05-10T14:30:00Z \
    --output out/broken_cumulus_afternoon
```

Optional `--width <px>` and `--height <px>` (default 1280×720).

The eight reference scenes in `tests/scenes/` are exercised by
`cargo test -p ps-app --test golden`, which renders each headlessly and
compares to `tests/golden/<scene>.png` via SSIM (tolerance ≥ 0.99).

When a deliberate visual change lands and you want to update the
goldens:

```sh
cargo run --bin ps-bless          # regenerates every golden under tests/golden/
git diff tests/golden/             # inspect the diffs before committing
```

The per-scene status of the regression goldens lives at
`tests/golden/README.md`.

## Authoring a scene

A scene is a self-contained TOML file describing a meteorological
situation. Schema is pinned by `schema_version = 1`; `Scene::load`
rejects unknown fields. See Appendix B of the plan or any of
`tests/scenes/*.toml` for working examples.

Minimum scene:

```toml
schema_version = 1

[surface]
visibility_m   = 30000.0
temperature_c  = 17.0
dewpoint_c     = 7.0
pressure_hpa   = 1018.0
wind_dir_deg   = 240.0
wind_speed_mps = 5.0

[surface.wetness]
ground_wetness  = 0.0
puddle_coverage = 0.0
puddle_start    = 0.6
snow_depth_m    = 0.0

[[clouds.layers]]
type               = "Cumulus"
base_m             = 1500.0
top_m              = 2300.0
coverage           = 0.5
density_scale      = 1.0
shape_octave_bias  = 0.0
detail_octave_bias = 0.0

[precipitation]
type               = "None"
intensity_mm_per_h = 0.0

[lightning]
strikes_per_min_per_km2 = 0.0
```

Notes:

- `coverage` is METAR-natural: 0.25 ≈ FEW, 0.5 ≈ BKN, 1.0 ≈ OVC. The
  synthesis pipeline pre-biases this onto the cloud march's Schneider
  visible band so authors don't need to know the underlying remap.
- `cloud_type` is one of `Cumulus`, `Stratus`, `Stratocumulus`,
  `Altocumulus`, `Altostratus`, `Cirrus`, `Cirrostratus`, `Cumulonimbus`.
  Per-cloud-type defaults (density, phase function, altitude bands,
  optical-depth targets) are documented with their published sources
  in `docs/cloud_calibration.md`.
- Multiple `[[clouds.layers]]` blocks must be **vertically disjoint**;
  `Scene::validate` rejects overlap with a structured error naming the
  offending pair.
- `snow_depth_m > 0` together with `temperature_c < 0.5` switches the
  ground material from PBR to snow. This is independent of the
  `wet_surface` master toggle in `pedalsky.toml`.
- `[clouds.coverage_grid]` lets a scene attach a per-pixel coverage
  grid. The `data_path` is a row-major little-endian f32 binary; a
  reference generator lives at
  `crates/ps-app/examples/generate_mountain_wave.rs`. (Runtime loading
  of the grid is deferred to v2; today the layer's scalar `coverage`
  is the fallback.)

To use a scene with the live app, point `pedalsky.toml`'s
`[paths] weather` at it. To use a scene with the headless renderer,
pass `--scene <path>` directly.

## Noise baking

The cloud march samples four noise textures (Schneider/Nubis):

- 128³ base shape (Perlin–Worley + Worley FBM at three frequencies)
- 32³ detail (Worley FBM at three frequencies)
- 128² 2D curl noise (Perlin curl)
- 64² 2D blue noise (void-and-cluster, spatial only — no temporal jitter)

These are baked on the GPU at first launch and cached to
`assets/noise/*.bin` keyed by content hash. Subsequent launches skip
the bake and load the cache.

To force a re-bake, delete `assets/noise/`. (No bin files are shipped;
making the bake reproducible is part of the test harness.)

## UI panels

The egui overlay is docked on the left edge. Each section is a
collapsible accordion.

- **World** — date/time controls; equinox/solstice/dawn/dusk shortcuts;
  lat/lon override; pause; time scale; read-only sun/moon altitude +
  azimuth + Julian day.
- **Render** — EV100, tone-mapper choice, HDR/PNG screenshot buttons.
  Screenshots write to `[paths] screenshot_dir`.
- **Subsystems** — checkboxes that enable/disable each render subsystem
  on the fly.
- **Atmosphere** — every field of `AtmosphereParams` as a slider with
  numeric input. Per-field reset button (↺) and a "Reset to Earth
  defaults" button.
- **Clouds** — engine-side `CloudParams` (cloud_steps, light_steps,
  multi-scatter octaves, detail strength, powder strength, freeze
  time…); per-layer accordions for `CloudLayerGpu` fields (base, top,
  coverage, density_scale, biases).
- **Wet surface** — wetness, puddle_coverage, puddle_start,
  snow_depth_m.
- **Precipitation** — kind (None/Rain/Snow/Sleet) and
  intensity_mm_per_h.
- **Debug** — fullscreen LUT viewers (transmittance / multi-scatter /
  sky-view as 2D, AP with a depth slice slider 0..32 km); per-pixel
  probe readouts (transmittance, optical depth at the cursor).

Slider edits route through the same `subsystem.reconfigure(&config,
&gpu)` path as file-watcher hot-reload. Editing `pedalsky.toml` while
the app runs has the same effect as moving the equivalent slider.

## Producing HDR EXR output

Two paths:

1. **From the live app**: open the Render panel, click "Screenshot HDR
   (EXR)". The file lands in `[paths] screenshot_dir` (default
   `screenshots/`).
2. **From the headless render**: `cargo run --bin ps-app -- render
   --scene <…> --output <base>` always emits `<base>.exr` alongside
   the PNG.

EXRs are linear-space `Rgba16Float` straight from the HDR target — no
tone-map applied. Use them for downstream colour science, comparison
against a reference, or pulling raw luminance values via
`crates/ps-app/examples/dump_exr_centre.rs`:

```sh
cargo run --example dump_exr_centre -p ps-app -- screenshots/foo.exr
```

## Repository layout

```
pedalsky/
├── Cargo.toml                          # workspace
├── pedalsky.toml                       # engine root config (live app)
├── Pedalsky corrected implementation plan.md
├── README.md
├── scenes/                             # live-app scene fixtures
│   └── broken_cumulus_afternoon.toml
├── tests/
│   ├── scenes/                         # 8 reference scenes (Phase 11.1)
│   ├── scenes/presets/                 # binary coverage grids
│   └── golden/                         # blessed PNGs + status README
├── crates/
│   ├── ps-core/                # config, scene, traits, world, gpu, framebuffer, camera, hot-reload
│   ├── ps-synthesis/           # weather scene → GPU resources
│   ├── ps-atmosphere/          # Hillaire 2020 sky/atmosphere LUTs + sky shader
│   ├── ps-clouds/              # Schneider/Nubis volumetric clouds
│   ├── ps-ground/              # PBR ground + Lagarde wet + snow + Halton ripples
│   ├── ps-precip/              # particle rain/snow + far streaks
│   ├── ps-postprocess/         # ACES filmic + auto-exposure
│   ├── ps-backdrop/            # Phase 1 demo: HDR clear at SkyBackdrop
│   ├── ps-tint/                # Phase 1 demo: fullscreen RGB multiply
│   ├── ps-ui/                  # egui panels + screenshot writers
│   └── ps-app/                 # winit window + render loop + headless test harness + ps-bless
└── shaders/                            # WGSL by subsystem
```

## Hot reload

`ps_core::HotReload` watches `pedalsky.toml` and the active scene file
via `notify`, debouncing changes (200 ms default) into
`WatchEvent::ConfigChanged` / `WatchEvent::SceneChanged` events on a
crossbeam channel. The `ps-app` binary polls this channel every frame
and either calls `App::reconfigure(&new_config, &gpu)` to apply the
change in place or drops and rebuilds affected subsystems via their
factories.

```rust
use ps_core::{HotReload, WatchEvent, DEFAULT_DEBOUNCE};

let watcher = HotReload::watch(&config_path, &scene_path, DEFAULT_DEBOUNCE)?;
for event in watcher.events() {
    match event {
        WatchEvent::ConfigChanged(_) => /* re-load, validate, app.reconfigure(...) */ (),
        WatchEvent::SceneChanged(_)  => /* re-load and re-synthesise */ (),
        WatchEvent::Error(msg)       => tracing::warn!(%msg, "hot-reload error"),
    }
}
```

WGSL hot-reload (per plan §Cross-Cutting/Shader hot-reload) is parsed
from `[debug] shader_hot_reload` but not yet wired — see "Outstanding
cross-cutting items" below.

## Tests

```sh
cargo test --workspace
```

131 tests across the workspace. The headline categories:

- `ps-core` — config, scene, subsystem trait, app, hot-reload.
- `ps-synthesis` — weather map, wind field, density mask, layer envelopes.
- `ps-atmosphere` — pipeline, sky pass output, ground bounce.
- `ps-clouds` — pipeline, noise volume content, cumulus visibility.
- `ps-ground` — pipeline, wet/snow paths, wet flag gating.
- `ps-precip` — pipeline, rain/snow distinguishability, cloud-mask occlusion, ripple effect.
- `ps-postprocess` — tone-mapper, EV scale, auto-exposure.
- `ps-ui` — state round-trip, slider routing, GPU-timestamp drain.
- `ps-app` — Phase 5–11 integration tests including the golden-image
  regression `cargo test -p ps-app --test golden`.

A diagnostic `dump_exr_centre` example reads the centre column of any
EXR (useful for verifying HDR luminance values against expected
photometric calculations).

## Outstanding cross-cutting items

Only one plan item remains, and the plan itself marks it as
"stretch":

- **Side-by-side comparison mode** (plan §10.5, stretch) — load two
  scene configs into one window split by a draggable bar.

CLI flags wired today:

  --gpu-trace <dir>   wgpu API call trace dump (requires the trace
                      feature on wgpu, enabled in the workspace)
  --seed <u64>        determinism seed for precipitation particle
                      spawning (the cloud march's blue-noise jitter
                      is spatial-only and already deterministic)
  --lut-overlay       force-on the atmosphere LUT 2x2 overlay

Other in-tree mechanisms:

- **WGSL shader hot-reload** is on by default (gated by
  `[debug] shader_hot_reload = true`). Edit any file under
  `shaders/`; the next frame loads it and rebuilds affected
  pipelines. Syntax errors currently propagate through wgpu's
  validation as a panic — graceful in-egui error display is a
  followup.
- **GPU timestamps** appear in the Debug panel's GPU timings
  section. Requires `TIMESTAMP_QUERY_INSIDE_ENCODERS` (NVIDIA's
  Vulkan driver supports this; integrated GPUs may not).
- **Camera UI** (fov, near, speed) sits between the World and
  Render panels in the egui overlay.
- **Naga std140 layout linter** runs as part of `cargo test`
  via `crates/ps-core/tests/uniform_layout.rs` (FrameUniforms,
  WorldUniforms, SurfaceParams) and `crates/ps-app/tests/wgsl_layout.rs`
  (CloudParams, CloudLayerGpu, PrecipUniforms). Both check per-field
  byte offsets, not just total size.

## Implementation principles

These are taken verbatim from the plan and apply to all code:

1. **Physical correctness first.** Use the equations, units, and constants
   from the cited papers; no "looks close" approximations.
2. **HDR everywhere internal.** Render targets are `Rgba16Float`. The
   tone-mapper is the very last stage.
3. **Photometric units.** Sun illuminance ≈ 127 500 lux at TOA; sky
   luminance in cd/m².
4. **Linear-space throughout.** sRGB → linear at sample time; linear
   → sRGB only at swapchain present.
5. **Reverse-Z depth.** `Depth32Float`, near plane → 1.0, far → 0.0.
6. **Right-handed Y-up world coordinates.** +X east, +Y up, +Z south.
   Planet centre at world `(0, -ground_radius_m, 0)`.
7. **Composable subsystems.** Every renderable implements
   `RenderSubsystem`. Disabling one in `[render.subsystems]` removes
   it from the render graph cleanly.
8. **Every tunable shader uniform is a UI slider.** No magic numbers
   in shader bodies.
9. **Deterministic for given inputs.** Same config + same time produces
   the same image (modulo the determinism gaps noted above).
10. **Testability.** Every subsystem has a headless render path.

## License

MIT OR Apache-2.0.
