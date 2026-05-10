# PedalSky — wgpu Weather Renderer Test Harness
## Corrected Implementation Plan

> This document supersedes the original `pedalsky-implementation-plan.md`. All
> issues identified by the plan self-review have been applied directly into the
> phase bodies. The original "Plan Self-Review & Identified Gaps" section has
> been removed; sample fixtures are retained in Appendices A and B.

**Project goal.** A composable, dependency-injected wgpu-based sample renderer
that consumes pre-ingested weather data (gridded fields + point observations),
synthesises it into a standard GPU-consumable format, and renders the resulting
atmosphere, clouds, precipitation, and ground using the Hillaire 2020
sky/atmosphere technique and the Schneider 2015 / Hillaire 2016 volumetric
cloud technique. **Quality over performance** at every junction. The
application must expose a runtime overlay for adjusting world time/date and
tuning all key shader control variables via sliders with numeric readouts.

---

## Top-Level Principles (Read Before Starting)

The implementing agent must follow these without exception:

1. **No shortcuts on physical correctness.** Use the equations, units, and
   constants given in this plan. Do not substitute "looks close enough"
   approximations unless the plan explicitly authorises a fast path. When in
   doubt, reach back to the SIGGRAPH papers cited.
2. **HDR everywhere internal.** Render targets are `Rgba16Float`. Tone-mapping
   is the very last stage. Do not clamp colour values prior to tone-map.
3. **Physical (photometric) units.** Sun illuminance ≈ 127 500 lux at top of
   atmosphere; sky luminance values in cd/m². Do not rescale to 0–1 ranges
   except in the final tone-map.
4. **Linear-space throughout.** Inputs are sRGB → linear at sample time.
   Outputs are linear → sRGB only at the swapchain present.
5. **Reverse-Z depth.** Use a `Depth32Float` reverse-Z buffer (near plane →
   1.0, far → 0.0) for horizon precision.
6. **Right-handed Y-up world coordinates.** +X east, +Y up, +Z south. The
   planet centre is at world `(0, -ground_radius_m, 0)`. Verified
   right-handed: `cross(east, up) = south`. **All projection matrices use the
   `_rh` (right-handed) variants in `glam`.**
7. **Composable subsystems.** Every renderable (atmosphere, clouds, ground,
   precip, wet surface) implements a common `RenderSubsystem` trait and is
   wired through dependency injection so `[render.subsystems].atmosphere = false`
   removes it from the render graph cleanly.
8. **Every shader uniform that is tunable in the spec is exposed as a slider in
   the UI.** No hard-coded magic numbers in shader bodies.
9. **Deterministic for given inputs.** The same config + time produces the same
   image bit-for-bit (modulo TAA jitter, which must be controllable from the
   UI). Blue-noise jitter is **spatial only** (frame-deterministic) so the
   cloud march does not shimmer when paused.
10. **Testability.** Every subsystem has a "headless render to PNG" path so
    test scenes can be regression-tested.

---

## Crate / Workspace Layout

The project is a Cargo workspace. Each subsystem crate implements a trait;
`ps-app` wires them up.

```
pedalsky/
├── Cargo.toml                  # workspace
├── README.md                   # required deliverable; see "Cross-Cutting"
├── crates/
│   ├── ps-core/                # traits, configs, world context, math, GPU primitives
│   ├── ps-synthesis/           # weather data → GPU-ready resources
│   ├── ps-atmosphere/          # Hillaire 2020 sky/atmosphere
│   ├── ps-clouds/              # volumetric clouds (Schneider/Hillaire)
│   ├── ps-ground/              # ground plane + wet surface BRDF
│   ├── ps-precip/              # rain / snow
│   ├── ps-postprocess/         # tone-map, exposure, optional bloom
│   ├── ps-ui/                  # egui overlay
│   └── ps-app/                 # main binary
├── shaders/
│   ├── common/                 # math.wgsl, units.wgsl, view_ray.wgsl, ndf.wgsl
│   ├── atmosphere/
│   ├── clouds/
│   ├── ground/
│   ├── precip/
│   └── postprocess/
├── assets/
│   ├── presets/
│   ├── noise/
│   └── lookup/
├── tests/
│   ├── golden/
│   └── scenes/
└── tools/
    └── ps-noise-baker/
```

### Workspace dependencies

Pin in workspace `Cargo.toml`:

- `wgpu = "0.20"` — verify against the current crate when starting (APIs have
  drifted across versions; if a newer release is current, update the pin and
  audit breaking changes before any other work).
- `winit = "0.30"`
- `egui = "0.27"`, `egui-wgpu = "0.27"`, `egui-winit = "0.27"`
- `glam = { version = "0.27", features = ["serde", "bytemuck"] }`
- `bytemuck = { version = "1", features = ["derive"] }`
- `serde = { version = "1", features = ["derive"] }`
- `toml = "0.8"`
- `chrono = "0.4"`
- `image = "0.25"`
- `image-compare = "0.4"` — perceptual diff (RMS + SSIM) for golden-image regression
- `exr = "1"` — HDR EXR screenshot export
- `crossbeam = "0.8"` — synthesis worker channels
- `notify = "6"` — config & shader hot-reload
- `anyhow = "1"`, `thiserror = "1"`
- `tracing = "0.1"`, `tracing-subscriber = "0.3"`
- `pollster = "0.3"` — `block_on` for wgpu's async startup (winit 0.30 is sync)
- `ndarray = "0.15"`
- `half = "2"`

The `tools/ps-noise-baker` crate may use `noise = "0.9"` only for reference
comparison; the runtime path generates noise on the GPU.

---

## Phase 0 — Workspace, Foundation, Render Loop Skeleton

**Goal:** A black window with a textured ground plane and a flying camera,
rendered through an HDR pipeline that ends in a working tone-mapper. No
weather yet.

### 0.1 Crate skeleton & build
Set up the workspace per the layout above. Each subsystem crate compiles to a
no-op stub satisfying its trait. CI runs `cargo check`, `clippy`, `fmt`,
`test`.

### 0.2 wgpu device init (`ps-core::gpu`)
- `Instance` with all backends.
- Adapter: `HighPerformance` power preference, surface-compatible.
- Device: required features `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`,
  `TEXTURE_BINDING_ARRAY`, `FLOAT32_FILTERABLE`. Bump
  `max_storage_buffer_binding_size` and `max_sampled_textures_per_shader_stage`
  to 32+; `max_color_attachments` to 8.
- **Surface format selection.** Query
  `surface.get_capabilities(&adapter).formats` and pick the first sRGB-suffixed
  format; fall back to `Bgra8UnormSrgb` only if it is in the list.
- `PresentMode::AutoVsync`, toggleable to `Immediate`.
- Resize handling recreates surface + HDR target + depth buffer.

Use `pollster::block_on` to bridge wgpu's async startup into winit 0.30's sync
event loop.

### 0.3 HDR offscreen target + depth (`ps-core::framebuffer`)
- `Rgba16Float` colour target sized to swapchain.
- `Depth32Float` reverse-Z depth.
- `HdrFramebuffer { color, depth, size }`.

### 0.4 Camera (`ps-core::camera`)
- `FlyCamera`: WASD + mouse look + Space/Ctrl up/down + Q/E roll.
- Stored as world position (m), pitch, yaw.
- **Right-handed reverse-Z infinite-far perspective:**
  ```rust
  let proj = Mat4::perspective_infinite_reverse_rh(fovy_rad, aspect, near_m);
  ```
  (`glam` provides this; it produces `near → 1.0`, `far → 0.0` in NDC z.)
- UI controls for fov, near, speed.

### 0.5 Ground plane (`ps-ground` v0)
- Single quad, 200 km × 200 km, axis-aligned in the XZ plane, centred under
  camera.
- Procedural checker pattern fragment shader. No PBR yet.

**Known limitation.** A 200 km flat plane ignores planet curvature. Once
Phase 5 introduces the spherical atmosphere the plane will visibly extend past
the horizon if the camera is high. For a stationary ground-level test harness
this is acceptable; if the camera ever needs to fly above ~5 km, replace with
a curved spherical cap or use a far-clip horizon trick.

### 0.6 Tone-mapping pass (`ps-postprocess` v0)
- Fullscreen-triangle pass reads HDR target, writes swapchain.
- ACES Filmic (Narkowicz 2015 fit) and a passthrough/clamp option for
  debugging.
- Exposure as a uniform (`ev100`), default EV100 = 14.

### 0.7 Render graph
For Phase 0:
```
clear HDR (sky-black) → ground pass → tone-map → swapchain → egui (added Phase 10)
```
Subsequent phases insert passes between ground and tone-map. The render graph
is data-driven from the registered subsystem passes (Phase 1).

**Acceptance:** flying camera over checker ground plane, FPS counter in window
title.

---

## Phase 1 — Configuration, Subsystem Trait, Dependency Injection

**Goal:** A robust config system with hot-reload, a `RenderSubsystem` trait,
and an `AppBuilder` that constructs only the subsystems the config asks for.

### 1.1 Top-level config
The root `pedalsky.toml` is given in Appendix A; the schema is defined in
`ps-core::config`.

### 1.2 Schema (`ps-core::config`)
- All structs `#[derive(Deserialize, Serialize, Clone, Debug)]` with
  `#[serde(deny_unknown_fields)]`.
- `Default` impl per struct so partial files work.
- `Config::load(path: &Path) -> Result<Config>` with rich `thiserror` errors
  reporting offending field+line.
- `validate(&self)` checks lat/lon ranges, time consistency, file existence.
  Run before any GPU resources are constructed.

### 1.3 Subsystem trait (`ps-core::subsystem`)

```rust
pub trait RenderSubsystem: Send + Sync {
    fn name(&self) -> &'static str;

    /// Called once per frame, before any render passes. May write to GPU
    /// buffers and rebuild bind groups. Subsystems MUST rebuild bind groups
    /// here rather than caching across frames; another subsystem may have
    /// been recreated by hot-reload, invalidating cached references.
    fn prepare(&mut self, ctx: &mut PrepareContext);

    /// Render-graph passes this subsystem contributes. Called once at
    /// registration; the executor flattens and sorts these into the per-frame
    /// command sequence.
    fn register_passes(&self) -> Vec<RegisteredPass>;

    /// Optional UI panel. Slider edits feed back through `reconfigure()`
    /// using the same code path as file-watcher changes, so behaviour is
    /// identical regardless of source.
    fn ui(&mut self, _ui: &mut egui::Ui) {}

    /// Re-apply changed config values without dropping the subsystem. Default
    /// is a no-op; subsystems that hold heavy GPU resources implement this to
    /// avoid full recreation when only tunable parameters change.
    fn reconfigure(&mut self, _config: &Config, _ctx: &GpuContext)
        -> anyhow::Result<()> { Ok(()) }

    fn enabled(&self) -> bool;
    fn set_enabled(&mut self, e: bool);
}

/// A single registered render-graph pass. A subsystem may register many.
pub struct RegisteredPass {
    pub name: &'static str,
    pub stage: PassStage,
    pub run: Box<dyn Fn(&mut wgpu::CommandEncoder, &RenderContext) + Send + Sync>,
}

/// Coarse ordering of passes. Within a stage, registration order is preserved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassStage {
    Compute,           // LUT bakes etc.
    SkyBackdrop,       // sky / atmosphere fullscreen, far depth
    Opaque,            // ground, depth-tested
    Translucent,       // clouds, precipitation
    PostProcess,       // pre tone-map (bloom, etc.)
    ToneMap,           // exactly one pass (the tone-mapper)
    Overlay,           // post tone-map (egui)
}
```

This replaces the original `GraphSlot` enum. The atmosphere subsystem
registers three passes (LUT compute → SkyBackdrop → Opaque (AP-on-ground
applies inside ground via LUT samples)).

### 1.4 Render contexts

```rust
pub struct PrepareContext<'a> {
    pub device:           &'a wgpu::Device,
    pub queue:            &'a wgpu::Queue,
    pub world:            &'a WorldState,
    pub weather:          &'a WeatherState,
    pub frame_uniforms:   &'a FrameUniforms,
    pub atmosphere_luts:  Option<&'a AtmosphereLuts>,
    pub dt_seconds:       f32,
}

pub struct RenderContext<'a> {
    pub device:            &'a wgpu::Device,
    pub queue:             &'a wgpu::Queue,
    pub framebuffer:       &'a HdrFramebuffer,
    pub frame_bind_group:  &'a wgpu::BindGroup,    // group 0
    pub world_bind_group:  &'a wgpu::BindGroup,    // group 1
    pub luts_bind_group:   Option<&'a wgpu::BindGroup>,  // group 3
    pub frame_uniforms:    &'a FrameUniforms,
}
```

### 1.5 AppBuilder + dependency injection
The `AppBuilder` reads the config, instantiates only enabled subsystems via
factory functions, calls `register_passes()` on each, and stores both the
subsystem list (for `prepare()`/`ui()`) and a flattened sorted pass list (for
per-frame execution).

`prepare()` is called on subsystems in the order
`prepare_passes_sorted_by(|p| p.stage)`. This guarantees the atmosphere
subsystem prepares its LUTs before the cloud subsystem reads them.

```rust
pub struct App {
    gpu: GpuContext,
    world: WorldState,
    weather: WeatherState,
    subsystems: Vec<Box<dyn RenderSubsystem>>,
    passes: Vec<RegisteredPass>,                  // sorted by PassStage
    framebuffer: HdrFramebuffer,
    tonemap: Tonemap,
    ui: UiState,
}
```

**No `Default::new()` constructors are permitted** for subsystems — every
subsystem is constructed from a `&Config` slice and a `&GpuContext`.

### 1.6 Hot reload
- `notify` watcher on `pedalsky.toml` and the weather scene file.
- On change: re-load, re-validate, diff against the live config; for each
  affected subsystem call `reconfigure()`. If `reconfigure()` returns an error
  (because a structural change is required), drop and recreate the subsystem.
- The UI overlay (Phase 10) routes slider edits through the same
  `reconfigure()` path, so behaviour is unified.

**Acceptance:** Toggling `[render.subsystems].atmosphere = false` while the
app is running removes the subsystem cleanly with no GPU validation errors.

---

## Phase 2 — World Context, Time, and Sun Direction

**Goal:** A `WorldState` that holds the current simulated date/time and
computes sun and moon directions plus TOA solar illuminance. Drives everything
downstream.

### 2.1 `WorldClock` (`ps-core::time`)
- `epoch: DateTime<Utc>`
- `time_scale: f64`
- `paused: bool`
- `simulated_seconds: f64` — accumulated only when not paused; this is what
  noise-time in cloud and rain shaders uses, so pausing freezes evolution.
- `current_utc(&self) -> DateTime<Utc>`

### 2.2 Sun and moon position (`ps-core::astro`)

Implement the **NREL Solar Position Algorithm (SPA)** by Reda & Andreas
(2003), accurate to ±0.0003° from year −2000 to +6000. Reference:
<https://midcdmz.nrel.gov/spa/> and `NREL/TP-560-34302`.

Steps (do not abbreviate):

1. Calculate Julian Day (JD) and Julian Ephemeris Day (JDE) from UTC. Apply
   `TT = UTC + ΔT`. Use a polynomial fit for ΔT (Espenak & Meeus 2006); for
   2026, ΔT ≈ 70 s. Document the fit chosen.
2. Earth heliocentric longitude L, latitude B, radius R via the periodic
   terms in SPA tables A4.1–A4.6.
3. Geocentric: Θ = L + 180°, β = -B.
4. Nutation Δψ, Δε; true obliquity ε.
5. Aberration correction Δτ = -20.4898″/(3600·R).
6. Apparent sun longitude λ = Θ + Δψ + Δτ.
7. Geocentric right ascension α and declination δ.
8. Observer parallax in α and δ.
9. Local hour angle H from apparent sidereal time at Greenwich + observer
   longitude − α.
10. Topocentric zenith and azimuth.

Do **not** use SPA's abridged form. Use the full SPA. Performance is fine —
this runs once per frame on the CPU.

Provide
```rust
pub struct SunPosition {
    pub altitude_rad: f32,
    pub azimuth_rad: f32,    // clockwise from north
    pub distance_au: f32,
}
```

For the moon, implement Meeus chapter 47.

### 2.3 World-space conversion
With `+X east, +Y up, +Z south` (right-handed) and azimuth measured clockwise
from north:

```rust
let sun_dir = Vec3::new(
    cos(altitude) * sin(azimuth),
    sin(altitude),
    -cos(altitude) * cos(azimuth),
);
```

### 2.4 TOA solar illuminance

```rust
let e_toa_lux = 127_500.0 * (1.0 / sun.distance_au).powi(2);
```

127 500 lx = solar constant 1361 W/m² × CIE photopic luminous efficacy for
the solar spectrum (~93.7 lm/W). The surface illuminance falls out of the
Phase 5 transmittance LUT.

### 2.5 UI inputs (forward reference to Phase 10)
- Year/month/day/hour/minute/second editable.
- Time-of-year shortcuts: spring equinox, summer solstice, autumn equinox,
  winter solstice.
- Time-of-day shortcuts: dawn, midday, dusk, midnight.
- Lat/lon override sliders.
- Read-only display of computed sun/moon altitude/azimuth and Julian day.

**Acceptance:** Sweeping the time slider through 24 hours produces sun
positions verifiable against <https://www.suncalc.org/> for the configured
lat/lon, accurate to better than 0.1°.

---

## Phase 3 — Weather Data Synthesis

**Goal:** Take the raw configured weather inputs and produce a single
`WeatherState` of GPU-ready resources used by all downstream subsystems.

### 3.1 Weather scene config
A scene config (loaded via `paths.weather`) is defined by Appendix B's
schema; the schema is implemented in `ps-synthesis::scene`.

### 3.2 The synthesis pipeline (`ps-synthesis`)

```rust
pub struct WeatherState {
    pub atmosphere: AtmosphereParams,        // Phase 5
    pub weather_map: GpuTexture2d,           // 2D RGBA16Float
    pub cloud_layers: GpuStorageBuffer<CloudLayerGpu>,
    pub cloud_layer_count: u32,
    pub wind_field: GpuTexture3d,            // 32×32×16 RGBA16Float
    pub top_down_density_mask: GpuTexture2d, // 2D R8Unorm: cloud cover projected onto ground
    pub sun_direction: Vec3,
    pub sun_illuminance: Vec3,
    pub surface: SurfaceParams,
    pub haze_extinction_per_m: Vec3,
}

#[repr(C)] #[derive(Pod, Zeroable, Copy, Clone)]
pub struct CloudLayerGpu {
    pub base_m: f32,
    pub top_m: f32,
    pub coverage: f32,
    pub density_scale: f32,
    pub cloud_type: u32,         // matches CloudType repr(u8) widened
    pub shape_bias: f32,
    pub detail_bias: f32,
    pub anvil_bias: f32,
}

#[repr(C)] #[derive(Pod, Zeroable, Copy, Clone)]
pub struct SurfaceParams {
    pub visibility_m: f32,
    pub temperature_c: f32,
    pub dewpoint_c: f32,
    pub pressure_hpa: f32,
    pub wind_dir_deg: f32,
    pub wind_speed_mps: f32,
    pub ground_wetness: f32,
    pub puddle_coverage: f32,
    pub snow_depth_m: f32,
    pub puddle_start: f32,       // wetness threshold for puddles, default 0.6
    pub _pad: [f32; 2],
}
```

#### 3.2.1 Visibility → Mie haze coefficient
Koschmieder: `β_haze = 3.912 / V_m`. Distribute uniformly across RGB
(chromatic Mie is future work).

#### 3.2.2 Cloud layer envelope synthesis
For each `[[clouds.layers]]`:
- Look up the cloud type's NDF (vertical density profile, Phase 6.4).
- Synthesise the `CloudLayerGpu` struct.
- **Validate non-overlap.** If two layers overlap in altitude, the synthesis
  stage rejects the scene with a `thiserror` error pointing at the offending
  pair. For v1, layers must be vertically disjoint.

#### 3.2.3 Weather map texture
A 2D texture **anchored to world origin** (the camera barely moves in this
test harness, so a 32 km extent centred on the origin is sufficient).
128×128 RGBA16Float channels:
- **R** = coverage in [0, 1]; the union of (a) gridded NWP coverage and
  (b) point-observation kernels (METAR cloud-cover code splatted with a
  Gaussian σ ≈ station radius).
- **G** = reserved (was per-pixel cloud type index — deferred to v2; in v1 the
  cloud type comes from the per-layer struct only).
- **B** = relative height offset of the cloud base in [-1, 1] of the layer
  thickness.
- **A** = local precipitation intensity / ground wetness scalar.

The synthesis stage builds this on the CPU once per weather update (a few Hz)
and uploads via `queue.write_texture`.

#### 3.2.4 Wind field texture
3D RGBA16Float, 32 (X) × 32 (Z) × 16 (Y). Channels (u, v, w, turbulence).

The Y axis maps `[0, wind.top_m]` to `v ∈ [0, 1]`. Sampled in shaders as:

```wgsl
fn wind_at(p_world: vec3<f32>) -> vec3<f32> {
    let uvw = vec3<f32>(
        p_world.x / wind.extent_m + 0.5,
        clamp(p_world.y / wind.top_m, 0.0, 1.0),
        p_world.z / wind.extent_m + 0.5,
    );
    return textureSampleLevel(wind_field, samp, uvw, 0.0).rgb;
}
```

If no gridded data is provided, synthesise:
- `u(z) = u_surf · (z / z_ref)^α` with α ≈ 0.143 (1/7-power-law, neutral
  stability).
- Ekman spiral rotation up to 30° to the right (NH) at gradient height.
- `w` zero to first order; small thermals (Gaussian bumps) under cumulus.
- `turbulence` increases below cloud bases, near terrain, and in CB layers.

#### 3.2.5 Top-down density mask
For Phase 8 precipitation occlusion. A 2D R8Unorm texture matching the
weather map's spatial extent. Computed by integrating the cloud density column
above each ground sample using the same density function the renderer uses,
at a coarse vertical step. Updated whenever cloud layers or weather map
change.

#### 3.2.6 Cloud type enum

```rust
#[repr(u8)] #[derive(Serialize, Deserialize, Clone, Copy)]
pub enum CloudType {
    Cumulus = 0,
    Stratus = 1,
    Stratocumulus = 2,
    Altocumulus = 3,
    Altostratus = 4,
    Cirrus = 5,
    Cirrostratus = 6,
    Cumulonimbus = 7,
}
```

Per-type defaults:

| Type | Base (m) | Top (m) | Density | NDF profile | Detail erosion |
|---|---|---|---|---|---|
| Cumulus | 1200–1800 | base+800 | 1.0 | Bell, peak ~0.4 | High |
| Stratus | 200–600 | base+400 | 0.7 | Top-heavy | Low |
| Stratocumulus | 600–1500 | base+700 | 0.85 | Mid-heavy | Medium |
| Altocumulus | 3500–5500 | base+500 | 0.6 | Mid-heavy | Medium |
| Altostratus | 3000–5000 | base+1500 | 0.5 | Top-heavy | Low |
| Cirrus | 7000–11000 | base+800 | 0.3 | Top-heavy | Very low (wispy) |
| Cirrostratus | 8000–11000 | base+500 | 0.3 | Top-heavy | Very low |
| Cumulonimbus | 600–2000 | base+10000 | 1.2 | Mushroom + anvil | Very high |

All tunable via UI (Phase 10).

### 3.3 Acceptance for Phase 3
- A `headless_dump` CLI subcommand of `ps-app` writes:
  - `weather_map.png` (2×2 grid of channel visualisations: R as greyscale,
    G as a per-type palette stripe, B as bias greyscale, A as blue gradient).
  - `wind_field_xz_slices.png` (3 horizontal slices at low/mid/upper altitude).
  - `top_down_density.png`.
  - `weather_dump.json` of the full `WeatherState` scalars (textures
    summarised by min/max/mean).

---

## Phase 4 — Render Loop, Frame Resources, Uniform Layout

**Goal:** Lock down per-frame resource layout used by all later subsystems.

### 4.1 `FrameUniforms`

Bind group 0, visible to every pass:

```rust
#[repr(C)] #[derive(Pod, Zeroable, Copy, Clone)]
pub struct FrameUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub camera_position_world: Vec4,    // .w unused
    pub sun_direction: Vec4,             // .w = sun_angular_radius_rad
    pub sun_illuminance: Vec4,           // RGB cd/m²·sr, .w = lux at TOA
    pub viewport_size: Vec4,             // w, h, 1/w, 1/h
    pub time_seconds: f32,               // wall time since start (UI use only)
    pub simulated_seconds: f32,          // pause-aware; cloud/rain shaders use this
    pub frame_index: u32,
    pub ev100: f32,
}
```

A second uniform `WorldUniforms` carries planet/atmosphere constants (Phase 5).

**std140 cross-check.** Run `naga`'s struct-layout linter on `FrameUniforms`,
`WorldUniforms`, `CloudLayerGpu`, `SurfaceParams` to catch alignment
mismatches between Rust `#[repr(C)]` and WGSL `var<uniform>`. Fix mismatches
by inserting explicit `_pad` fields rather than reordering domain fields.

### 4.2 Bind group conventions
- **Group 0**: `FrameUniforms`
- **Group 1**: `WorldUniforms` (atmosphere + planet)
- **Group 2**: subsystem-specific resources
- **Group 3**: shared atmosphere LUTs (transmittance, multi-scatter, sky-view,
  AP)

Group 3 contains 4 textures + 1 sampler — well within wgpu defaults.

### 4.3 Shared shader helpers (`shaders/common/`)

`shaders/common/math.wgsl`:

```wgsl
fn remap(v: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
    return new_min + (v - old_min) * (new_max - new_min) / max(old_max - old_min, 1e-5);
}

fn max3(v: vec3<f32>) -> f32 { return max(v.x, max(v.y, v.z)); }

struct ViewRay { origin: vec3<f32>, dir: vec3<f32>, };

/// Build a world-space ray from a screen-space fragment coordinate, honouring
/// the reverse-Z infinite-far perspective convention. depth=1 is the near
/// plane, depth=0 is at infinity.
fn compute_view_ray(frag_xy: vec2<f32>) -> ViewRay {
    let viewport = frame.viewport_size.xy;
    let ndc = vec2<f32>(
        (frag_xy.x / viewport.x) * 2.0 - 1.0,
        1.0 - (frag_xy.y / viewport.y) * 2.0
    );
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far_h  = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let near_p = near_h.xyz / near_h.w;
    let far_p  = far_h.xyz  / far_h.w;
    return ViewRay(near_p, normalize(far_p - near_p));
}

/// Returns true and the two intersection distances along `rd` if the ray
/// intersects the sphere; the smaller is `t.x`, the larger is `t.y`.
fn ray_sphere_intersect(ro: vec3<f32>, rd: vec3<f32>,
                        center: vec3<f32>, radius: f32,
                        t: ptr<function, vec2<f32>>) -> bool {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return false; }
    let sq = sqrt(disc);
    *t = vec2<f32>(-b - sq, -b + sq);
    return true;
}

fn world_to_weather_uv(p_xz: vec2<f32>) -> vec2<f32> {
    return p_xz / weather.weather_extent_m + vec2<f32>(0.5);
}
```

`shaders/common/ndf.wgsl` — see Phase 6.4.

`shaders/common/atmosphere.wgsl` — atmospheric helpers used by both sky and
clouds (transmittance LUT sampling etc., Phase 5).

### 4.4 Render graph executor
After all `prepare()`s have run (in `PassStage` order), the executor walks the
flattened pass list also in `PassStage` order and invokes each pass's `run`
closure with the encoder and `RenderContext`.

**Acceptance:** Empty render graph still presents a valid frame. Frame
uniforms are populated correctly; a debug panel in Phase 10 displays them.

---

## Phase 5 — Atmosphere & Sky (Hillaire 2020)

**Goal:** A physically based sky and aerial-perspective system implementing
Hillaire 2020, *"A Scalable and Production Ready Sky and Atmosphere Rendering
Technique"* (CGF 39:13). Reference: <https://sebh.github.io/publications/>.
Open implementation: <https://github.com/sebh/UnrealEngineSkyAtmosphere>.

### 5.1 Atmosphere parameters (`WorldUniforms`)

**Internal unit convention: metres throughout the shaders.** Convert from
config-friendly km → m at uniform-load time.

```rust
#[repr(C)] #[derive(Pod, Zeroable, Copy, Clone)]
pub struct AtmosphereParams {
    pub planet_radius_m: f32,          // 6_360_000
    pub atmosphere_top_m: f32,         // 6_460_000
    pub rayleigh_scale_height_m: f32,  // 8_000
    pub mie_scale_height_m: f32,       // 1_200

    pub rayleigh_scattering: Vec4,     // (5.802, 13.558, 33.100) × 1e-6 /m
    pub mie_scattering: Vec4,          // (3.996, 3.996, 3.996)  × 1e-6 /m
    pub mie_absorption: Vec4,          // (4.4,   4.4,   4.4)    × 1e-6 /m
    pub mie_g: f32,                    // 0.8

    pub ozone_absorption: Vec4,        // (0.650, 1.881, 0.085) × 1e-6 /m
    pub ozone_center_m: f32,           // 25_000
    pub ozone_thickness_m: f32,        // 30_000

    pub ground_albedo: Vec4,
    pub haze_extinction_per_m: Vec4,
}
```

All exposed as sliders. "Reset to physical Earth defaults" button.

### 5.2 LUTs (compute shaders)

Sizes match Hillaire's reference implementation.

#### 5.2.1 Transmittance LUT — 256 × 64, `Rgba16Float`
Each texel encodes `(view_zenith_angle, altitude)` and stores `vec3 T = exp(-Σ σ_t·ds)`. 40 trapezoidal steps. Built once and rebuilt only when atmosphere
parameters change.

#### 5.2.2 Multi-scattering LUT — 32 × 32, `Rgba16Float`
Hillaire 2020 §5.2: isotropic phase + 8×8 sample directions per texel + 20
march steps. Closed-form geometric series for higher orders:
`L_full = L_2 · 1 / (1 − f_ms)` where `f_ms` is the integrated single-scatter
contribution over the sphere.

Implementation reference: `MultiScatLutPS.usf` in
<https://github.com/sebh/UnrealEngineSkyAtmosphere>. Re-read before writing
the WGSL.

#### 5.2.3 Sky-view LUT — 192 × 108, `Rgba16Float`
Non-linear latitude parametrisation (concentrates samples around the horizon
where Rayleigh detail matters):

```
v = 0.5 + 0.5 · sign(lat) · sqrt(|lat| / (PI/2))
u = wrap((azimuth − sun_azimuth) / (2·PI))
```

Sample via the transmittance + multi-scattering LUTs; 32 march steps.
Rebuilt every frame.

#### 5.2.4 Aerial perspective LUT — 32 × 32 × 32 (3D), `Rgba16Float`
Camera-relative froxel volume, slices linearly out to ~32 km along the
camera frustum. RGB = inscattered luminance; A = transmittance. Used by ground
and cloud composition. Rebuilt every frame.

**Limitation.** 32 km linear is fine for ground-level viewing. For high-
altitude camera flights the far slice does not cover space-to-ground viewing.
Documented for v2; out of scope for v1's stationary test harness.

### 5.3 Sky raymarch shader
A fullscreen-triangle pass at `PassStage::SkyBackdrop`. It writes into the
HDR target at the **far** depth value (0.0 in reverse-Z), so subsequent
opaque passes naturally win the depth test where they have geometry.

For each pixel: sample the sky-view LUT at the view direction. Composite
analytic sun disk with limb darkening:

```
L_disk = L_sun · (1 − u·(1 − cos θ_disk)),  u = 0.6
```

The sky pass also includes the ground-bounce term per Hillaire 2020 §6
(multiply the integrated bounce radiance by `ground_albedo / pi`).

### 5.4 Aerial perspective application
**Applied inside the ground pass**, not as a separate pass, so the order is
correct under depth-tested rendering:

```wgsl
// Inside the ground fragment shader, after lit colour is computed
let ndc_xy = (frag_xy / frame.viewport_size.xy) * 2.0 - 1.0;
let lin_depth = linear_depth(scene_depth, frame.proj);
let ap_uvw = vec3<f32>(ndc_xy * 0.5 + 0.5, saturate(lin_depth / 32_000.0));
let ap = textureSampleLevel(aerial_perspective_lut, samp_lut, ap_uvw, 0.0);
let final_color = lit_color * ap.a + ap.rgb;   // apply transmittance + add inscatter
```

### 5.5 Multiple-scattering toggle
Disable via UI for didactic comparison. Default on.

### 5.6 UI exposure
Sliders for sun angular radius, mie g, every scattering coefficient, scale
heights, ozone parameters, ground albedo. "Reset to physical Earth defaults"
button.

**Acceptance:** Daytime clear sky matches Hillaire's published screenshots
qualitatively. Dawn/dusk produces correct horizon reddening. With sun at
zenith, clear sky, EV100 = 15, the ground-irradiance debug readout falls
within [90 000, 110 000] lux (after atmospheric transmittance through a
clear column).

---

## Phase 6 — Volumetric Clouds (Schneider + Hillaire)

**Goal:** A ray-marched volumetric cloud system implementing Schneider Nubis
(2015/2017/2022) with Hillaire 2016 energy-conserving integration and
multi-octave multiple-scattering. Driven by the synthesised weather state.

References (re-read before implementing):
- Schneider & Vos 2015, *Real-time Volumetric Cloudscapes of Horizon Zero
  Dawn*, SIGGRAPH 2015 — slides on `schneidervfx.com`.
- Schneider 2017, *Nubis: Authoring Real-Time Volumetric Cloudscapes with
  the Decima Engine*, SIGGRAPH 2017.
- Schneider 2022, *Nubis, Evolved*, SIGGRAPH 2022.
- Hillaire 2016, *Physically Based Sky, Atmosphere & Cloud Rendering in
  Frostbite*, SIGGRAPH 2016.
- Toft, Bowles, Zimmermann 2016, arXiv:1609.05344.

### 6.1 Noise volumes (`tools/ps-noise-baker` + GPU prebake)

Generated via wgpu compute shaders at first launch; cached to
`assets/noise/*.bin` keyed by content hash. **Do not** ship pre-baked
binaries — making the bake reproducible is part of the harness.

- **Base shape** — 128³ `Rgba8Unorm`. R = Perlin–Worley; G/B/A = Worley FBM at
  freq 2/8/14.
- **Detail** — 32³ `Rgba8Unorm`. R/G/B = Worley FBM at 2/8/16; A spare.
- **Curl** — 128² 2D `Rg8Unorm`. Curl of a 2D Perlin field.
- **Blue noise** — 64×64 2D `R8Unorm`, void-and-cluster (Christensen &
  Kensler). Used to dither the cloud march start; **spatial only** so the
  pattern is frame-deterministic and does not shimmer when paused.

### 6.2 Atmosphere geometry
Each cloud layer occupies the spherical shell
`[planet_r + base_m, planet_r + top_m]`. The march iterates the layers in
**ray-entry order** (sorted per pixel), not in altitude order — this is
correct for any camera viewpoint, including downward views from above the
layer.

### 6.3 Cloud march — params and helpers

```wgsl
struct CloudParams {
    sigma_s: f32,                    // 0.04 default, slider
    sigma_a: f32,                    // 0.0 default, slider
    g_forward: f32,                  // 0.8
    g_backward: f32,                 // -0.3
    g_blend: f32,                    // 0.5
    light_steps: u32,                // 6, slider 1..16
    cloud_steps: u32,                // 192, slider 32..256
    detail_strength: f32,            // 0.35
    curl_strength: f32,              // 0.1
    powder_strength: f32,            // 1.0  — lerp between Beer-only and Beer-Powder
    multi_scatter_octaves: u32,      // 4
    multi_scatter_a: f32,            // 0.5  — energy attenuation per octave
    multi_scatter_b: f32,            // 0.5  — optical depth attenuation per octave
    multi_scatter_c: f32,            // 0.5  — phase anisotropy attenuation per octave
    base_scale_m: f32,               // 4500
    detail_scale_m: f32,             // 800
    weather_scale_m: f32,            // 32000
    ambient_strength: f32,           // 1.0
};

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(max(denom, 1e-4), 1.5));
}

/// Dual-lobe HG with anisotropy g scaled by `g_scale`. Used for the
/// multi-octave multiple-scattering approximation: cos_theta is geometric
/// and is NOT scaled; only g is.
fn dual_lobe_hg_with_g_scale(cos_theta: f32, p: CloudParams, g_scale: f32) -> f32 {
    let gf = p.g_forward * g_scale;
    let gb = p.g_backward * g_scale;
    return mix(henyey_greenstein(cos_theta, gf),
               henyey_greenstein(cos_theta, gb),
               p.g_blend);
}

fn dual_lobe_hg(cos_theta: f32, p: CloudParams) -> f32 {
    return dual_lobe_hg_with_g_scale(cos_theta, p, 1.0);
}
```

### 6.4 Density-height gradient (NDF) — full WGSL

`shaders/common/ndf.wgsl`:

```wgsl
fn ndf(h: f32, t: u32) -> f32 {
    switch t {
        case 0u: { // Cumulus: bell, peak ~0.4
            return smoothstep(0.0, 0.07, h) * smoothstep(1.0, 0.2, h);
        }
        case 1u: { // Stratus: top-heavy, low and thin
            return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.6, 1.0, h));
        }
        case 2u: { // Stratocumulus: mid-heavy
            return smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.4, h);
        }
        case 3u: { // Altocumulus: mid-heavy, thinner overall
            return smoothstep(0.0, 0.20, h) * smoothstep(1.0, 0.3, h) * 0.8;
        }
        case 4u: { // Altostratus: top-heavy sheet
            return smoothstep(0.0, 0.30, h) * (1.0 - smoothstep(0.7, 1.0, h)) * 0.6;
        }
        case 5u: { // Cirrus: thin, top-heavy, wispy
            return smoothstep(0.0, 0.40, h) * (1.0 - smoothstep(0.6, 1.0, h)) * 0.4;
        }
        case 6u: { // Cirrostratus: very thin sheet, highest
            return smoothstep(0.0, 0.50, h) * (1.0 - smoothstep(0.8, 1.0, h)) * 0.3;
        }
        case 7u: { // Cumulonimbus: mushroom (bottom-heavy + anvil)
            let base  = smoothstep(0.0, 0.05, h);
            let mid   = smoothstep(0.95, 0.5, h);
            let anvil = smoothstep(0.7, 0.9, h) * 1.5;
            let mix_t = smoothstep(0.65, 0.8, h);
            return base * mix(mid, anvil, mix_t);
        }
        default: { return 0.0; }
    }
}

/// Used for ambient (sky) light contribution; biases ambient brightness up
/// near the cloud top (more direct sky exposure) and down near the base
/// (occluded by the cloud above).
fn ambient_height_gradient(h: f32) -> f32 {
    return mix(0.3, 1.0, h);
}
```

### 6.5 Density and light-march helpers

```wgsl
fn compute_altitude_m(p_world: vec3<f32>) -> f32 {
    let planet_center = vec3<f32>(0.0, -world.planet_radius_m, 0.0);
    return length(p_world - planet_center) - world.planet_radius_m;
}

fn sample_density(p_world: vec3<f32>, layer: CloudLayerGpu,
                  weather: vec4<f32>, params: CloudParams) -> f32 {
    let altitude_m = compute_altitude_m(p_world);
    let h = (altitude_m - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
    if (h < 0.0 || h > 1.0) { return 0.0; }

    let base_uv = p_world.xyz / params.base_scale_m;
    let base = textureSampleLevel(noise_base, samp, base_uv, 0.0);
    let lf_fbm = base.g * 0.625 + base.b * 0.25 + base.a * 0.125;
    let base_cloud = remap(base.r, -(1.0 - lf_fbm), 1.0, 0.0, 1.0);

    let profile = ndf(h, layer.cloud_type);
    var cloud = base_cloud * profile;

    let coverage = weather.r * layer.coverage;
    cloud = remap(cloud, 1.0 - coverage, 1.0, 0.0, 1.0);
    cloud = cloud * coverage;

    // Curl-perturbed detail erosion at the boundary
    let curl = textureSampleLevel(
        noise_curl, samp, p_world.xz / params.detail_scale_m, 0.0
    ).rg;
    let detail_uv = (p_world + vec3<f32>(curl.x, 0.0, curl.y) * params.curl_strength)
                  / params.detail_scale_m;
    let detail = textureSampleLevel(noise_detail, samp, detail_uv, 0.0);
    let hf_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
    let detail_mod = mix(hf_fbm, 1.0 - hf_fbm, saturate(h * 10.0));
    cloud = remap(cloud, detail_mod * params.detail_strength, 1.0, 0.0, 1.0);

    return saturate(cloud) * layer.density_scale;
}

fn march_to_light(p: vec3<f32>, sun_dir: vec3<f32>, layer: CloudLayerGpu,
                  params: CloudParams) -> f32 {
    let altitude_m = compute_altitude_m(p);
    var od = 0.0;
    var pos = p;
    let dist_to_top = max((layer.top_m - altitude_m) / max(sun_dir.y, 0.05), 1.0);
    let step = dist_to_top / f32(params.light_steps);
    for (var i = 0u; i < params.light_steps; i = i + 1u) {
        pos = pos + sun_dir * step;
        let weather = textureSampleLevel(weather_map, samp,
                                         world_to_weather_uv(pos.xz), 0.0);
        od = od + sample_density(pos, layer, weather, params) * step;
    }
    return od;
}

fn integrate_step(S: vec3<f32>, sigma_t: f32, ds: f32,
                  trans: ptr<function, vec3<f32>>) -> vec3<f32> {
    let Tr = exp(-vec3<f32>(sigma_t) * ds);
    let Sint = (S - S * Tr) / max(vec3<f32>(sigma_t), vec3<f32>(1e-5));
    let result = (*trans) * Sint;
    *trans = (*trans) * Tr;
    return result;
}

/// Per-layer ray/shell intersection. Handles camera below, inside, or above
/// the layer. Returns false if the ray misses the layer entirely.
fn ray_layer_intersect(ray: ViewRay, layer: CloudLayerGpu,
                       t0: ptr<function, f32>, t1: ptr<function, f32>) -> bool {
    let planet_center = vec3<f32>(0.0, -world.planet_radius_m, 0.0);
    let r_bottom = world.planet_radius_m + layer.base_m;
    let r_top    = world.planet_radius_m + layer.top_m;
    var t_top: vec2<f32>;
    var t_bot: vec2<f32>;
    let hit_top = ray_sphere_intersect(ray.origin, ray.dir, planet_center, r_top, &t_top);
    let hit_bot = ray_sphere_intersect(ray.origin, ray.dir, planet_center, r_bottom, &t_bot);
    if (!hit_top) { return false; }

    // Determine [t0, t1] interval where the ray is inside the shell.
    // Fast path: clamp top intersection by bottom intersection.
    *t0 = max(t_top.x, 0.0);
    *t1 = max(t_top.y, 0.0);
    if (hit_bot && t_bot.x > 0.0) {
        *t1 = min(*t1, t_bot.x);
    }
    return *t1 > *t0;
}
```

### 6.6 Cloud march — fragment shader

```wgsl
struct LayerHit { idx: u32, t0: f32, t1: f32, hit: bool, };

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let ray = compute_view_ray(frag_coord.xy);
    let cos_theta = dot(ray.dir, frame.sun_direction.xyz);
    let jitter = textureLoad(blue_noise,
                              vec2<i32>(frag_coord.xy) % 64, 0).r;

    // Compute per-layer intervals and sort by ray-entry distance.
    var hits: array<LayerHit, 8>;
    for (var l = 0u; l < weather.cloud_layer_count; l = l + 1u) {
        var t0: f32; var t1: f32;
        let h = ray_layer_intersect(ray, cloud_layers.layers[l], &t0, &t1);
        hits[l] = LayerHit(l, t0, t1, h);
    }
    // Insertion sort by t0 ascending.
    for (var i = 1u; i < weather.cloud_layer_count; i = i + 1u) {
        var j = i;
        while (j > 0u && hits[j - 1u].t0 > hits[j].t0) {
            let tmp = hits[j];
            hits[j] = hits[j - 1u];
            hits[j - 1u] = tmp;
            j = j - 1u;
        }
    }

    var luminance = vec3<f32>(0.0);
    var transmittance = vec3<f32>(1.0);
    let sigma_t = params.sigma_s + params.sigma_a;

    for (var i = 0u; i < weather.cloud_layer_count; i = i + 1u) {
        let lh = hits[i];
        if (!lh.hit) { continue; }
        let layer = cloud_layers.layers[lh.idx];
        let step = (lh.t1 - lh.t0) / f32(params.cloud_steps);
        var t = lh.t0 + jitter * step;

        for (var s = 0u; s < params.cloud_steps; s = s + 1u) {
            let p = ray.origin + ray.dir * t;
            let weather_sample = textureSampleLevel(
                weather_map, samp, world_to_weather_uv(p.xz), 0.0
            );
            let density = sample_density(p, layer, weather_sample, params);
            if (density > 1e-3) {
                let od_to_sun = march_to_light(p, frame.sun_direction.xyz, layer, params);

                // Schneider 2015 Beer-Powder energy term (canonical form).
                let beer        = exp(-sigma_t * od_to_sun);
                let powder      = 1.0 - exp(-2.0 * sigma_t * od_to_sun);
                let beer_powder = beer * powder * 2.0;
                // The slider lerps between pure Beer (no dark-edge brighten) and
                // full Beer-Powder for didactic comparison.
                let energy = mix(beer, beer_powder, params.powder_strength);

                // Hillaire 2016 multi-octave multiple-scattering approximation.
                // Each octave: energy *= a, optical depth *= b, anisotropy *= c.
                // cos_theta is geometric and never scaled.
                var sun_in = vec3<f32>(0.0);
                var a = 1.0; var b = 1.0; var c = 1.0;
                for (var n = 0u; n < params.multi_scatter_octaves; n = n + 1u) {
                    sun_in = sun_in + a * frame.sun_illuminance.rgb
                            * dual_lobe_hg_with_g_scale(cos_theta, params, c)
                            * exp(-sigma_t * od_to_sun * b);
                    a = a * params.multi_scatter_a;
                    b = b * params.multi_scatter_b;
                    c = c * params.multi_scatter_c;
                }

                let altitude_m = compute_altitude_m(p);
                let h = (altitude_m - layer.base_m) / max(layer.top_m - layer.base_m, 1.0);
                let ambient = sample_sky_ambient(p) * params.ambient_strength
                              * ambient_height_gradient(h);

                let S = density * params.sigma_s * (sun_in * energy + ambient);
                luminance = luminance + integrate_step(S, density * sigma_t,
                                                      step, &transmittance);
            }
            if (max3(transmittance) < 0.01) { break; }
            t = t + step;
        }
    }

    // Composition target: RGB = premultiplied luminance, A = 1 - T_lum where
    // T_lum is the luminance-weighted scalar transmittance. The composition
    // pass uses standard `One, OneMinusSrcAlpha` blending. This loses
    // chromatic transmittance — acceptable for v1; documented as v2 work
    // (dual-target render or RGB-transmittance MRT).
    let t_lum = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec4<f32>(luminance, 1.0 - t_lum);
}
```

`sample_sky_ambient(p)` reads the Phase 5 sky-view LUT at the up vector
through `p`; defined in `shaders/common/atmosphere.wgsl`.

### 6.7 Cloud composition into the HDR target
A separate `Translucent`-stage pass blits the cloud target over the HDR
target with `Blend::OneMinusSrcAlpha` (premultiplied alpha). The cloud pass
also samples the AP LUT along the cloud's median t and applies aerial
perspective to the cloud luminance before blending.

### 6.8 No checkerboard reprojection in v1
Full resolution. A reprojection path is reserved as future optimisation
behind `[render.clouds].reprojection`, but only `"off"` is supported.

### 6.9 UI exposure
Every `CloudParams` field as a slider with numeric box. Per-layer overrides
in a "Cloud Layer N" panel. "Freeze time" toggle stops `simulated_seconds`
advancing for screenshots.

**Acceptance:** A single cumulus layer with coverage 0.4 produces visibly
cumulus-shaped clouds that cast convincing self-shadows when the sun is low.
The "silver lining" appears with the sun behind a cloud (dual-lobe HG
verified). Cumulonimbus produces a clear anvil top.

---

## Phase 7 — Ground & Wet Surface

**Goal:** Replace Phase 0's placeholder ground with a physically-based ground
plane that responds to the synthesised wetness/snow state and applies aerial
perspective.

### 7.1 Ground BRDF
Standard GGX/Smith specular + Lambertian diffuse over a tiled albedo.

**Procedural pattern.** Low-frequency Voronoi tile (cell size ~5 m), with each
cell coloured from a 3-entry palette:
```
palette = [
    vec3<f32>(0.18, 0.18, 0.18),   // mid grey
    vec3<f32>(0.22, 0.20, 0.16),   // warm tan
    vec3<f32>(0.14, 0.16, 0.18),   // cool grey
]
cell_color = palette[cell_id_hash % 3];
```
This gives shadows enough local contrast to read against without introducing
a regular grid.

### 7.2 Wet surface (Lagarde 2013)
References:
- <https://seblagarde.wordpress.com/2013/03/19/water-drop-3a-physically-based-wet-surfaces/>
- <https://seblagarde.wordpress.com/2013/01/03/water-drop-2b-dynamic-rain-and-its-effects/>

Three regimes blended by `wetness` ∈ [0, 1]:

```wgsl
struct WetMaterial { albedo: vec3<f32>, roughness: f32, };

fn wet_material(albedo: vec3<f32>, roughness: f32, wetness: f32) -> WetMaterial {
    let dark_albedo = pow(albedo, vec3<f32>(mix(1.0, 3.0, wetness)));
    let wet_rough   = mix(roughness, max(roughness * 0.5, 0.05), wetness);
    return WetMaterial(dark_albedo, wet_rough);
}
```

For `wetness > surface.puddle_start` (default 0.6, exposed in
`[surface.wetness] puddle_start = ...`), blend in a thin water layer (Weidlich
& Wilkie 2007) with `F0 = 0.02`, modulated by a noise mask that respects
surface normal: `puddle_mask = noise(p.xz) * step(0.95, dot(N, up))`.

### 7.3 Snow
Snow rendering is gated:
```
snow_visible = (surface.temperature_c < 0.5) && (surface.snow_depth_m > 0.0)
```

When visible, blend a snow layer with albedo ~0.9 and roughness 0.85 over
the wet/dry composite. Snow distribution uses the inverse of the puddle mask
(snow accumulates on flat surfaces, doesn't accumulate where puddles form).

### 7.4 Aerial perspective on ground
Applied **inside** the ground fragment shader using the AP LUT (see Phase 5.4).
Do not compute AP in a separate pass; the depth-tested ground pass writes the
final ground colour with AP already applied.

**Acceptance:** Toggling `wetness` from 0 to 1 produces visibly darker,
glossier ground with correct Fresnel at grazing angles. Snow at 0.05 m depth
produces near-white ground when temperature is below freezing.

---

## Phase 8 — Precipitation

**Goal:** Toggleable rain and snow rendering. Driven by
`WeatherState.precipitation`.

### 8.1 Rain — particle emission

A compute shader maintains a storage buffer of particles
`{ position: vec3, age: f32, velocity: vec3, kind: u32 }` of fixed length
`render.precip.near_particle_count` (default 8000). Each frame:

1. Compute pass advances each particle: integrate position by velocity ·
   dt; resample wind via `wind_at(p)`; apply wind drift; age increments;
   particles exiting a 50 m cylinder around the camera respawn at random top
   positions seeded from the frame index.
2. Render pass instances one oriented quad per particle, draw size proportional
   to (velocity · exposure_time) for motion-blur streaks.
3. `draw_indirect` reads the live count from the compute shader's atomic
   counter.

Density is `intensity_mm_per_h` mapped through Marshall–Palmer:
`N(D) = 8000 · exp(-4.1 · I^{-0.21} · D)` for drop diameters in mm.

### 8.2 Far rain — screen-space streaks
Three layered scrolling streak textures at depths 50, 200, 1000 m. Each
scrolled by `wind − camera_velocity` projected into screen space. Depth-tested
against the scene depth buffer.

### 8.3 Cloud occlusion mask
Both layers are modulated by sampling
`weather.top_down_density_mask` at the particle's XZ position. **No top-down
ray traced through the cloud field at runtime** — the Phase 3.2.5 mask
already encodes the top-down density for the entire scene and is updated
whenever cloud parameters change. This is correct regardless of camera
attitude.

### 8.4 Snow
Same particle architecture as near rain but with terminal velocity ~1 m/s,
softer round splat, stronger wind influence, and `kind = 1` so the render
shader picks a different sprite.

### 8.5 Surface ripples
Animated normal-map ripples on the wet ground when `wetness > 0.5` and
`intensity > 0`. 2–4 spawners per m² per second from a Halton sequence indexed
by `simulated_seconds` so the pattern is deterministic over time and respects
the time-pause flag.

### 8.6 No vehicle/canopy rain in v1
Out of scope. Documented as future work.

**Acceptance:** Setting `intensity_mm_per_h = 5` while clouds are present
produces visible rain streaks; turning off clouds zeroes the top-down density
mask and the rain disappears.

---

## Phase 9 — Composition, Tone Mapping, Photometric Exposure

**Goal:** Correct end-to-end HDR composite with photometric exposure.

### 9.1 Render order

```
1.  transmittance LUT      [Compute, on AtmosphereParams change]
2.  multi-scatter LUT      [Compute, on AtmosphereParams change]
3.  sky-view LUT           [Compute, every frame]
4.  aerial-perspective LUT [Compute, every frame]
5.  sky pass               [SkyBackdrop, fullscreen, depth = far value]
6.  ground pass            [Opaque, depth-tested, AP applied in-shader]
7.  cloud pass + composite [Translucent, depth-aware march termination]
8.  precipitation          [Translucent]
9.  tone-map               [ToneMap → swapchain]
10. egui overlay           [Overlay, → swapchain after tone-map]
```

The sky writes at the far depth value (0.0 in reverse-Z) so the depth test
naturally lets opaque geometry win. Aerial perspective is applied **inside**
the ground shader by sampling the AP LUT — never as a separate post-pass over
the already-mixed sky+ground colour. Cloud march termination uses the per-pixel
linear scene depth as `t_max` so clouds correctly clip behind opaque geometry.

### 9.2 Auto-exposure
Disabled by default. User sets `ev100` via slider. A debug auto-exposure mode
(compute average log-luminance of the HDR target, pick EV100 to centre it) is
behind `[debug] auto_exposure = true`.

### 9.3 Tone mapper
ACES Filmic (Narkowicz fit):

```wgsl
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}
```

Followed by linear-to-sRGB encoding (or write to a `*Srgb` swapchain texture
and let the GPU encode).

### 9.4 EV100 → exposure scale
`exposure = 1.0 / (1.2 * pow(2.0, ev100))`. Multiply HDR luminance (cd/m²)
by exposure before the ACES curve.

**Acceptance:** Mid-day clear sky tones to a believable photographic image at
EV100 = 15. Sunset works at EV100 = 12. Twilight at EV100 = 8.

---

## Phase 10 — UI Overlay (egui)

**Goal:** A docked, collapsible panel system exposing every tunable parameter
and the world clock.

### 10.1 egui-wgpu integration
Standard `egui_wgpu_winit` setup. Renders into the swapchain at
`PassStage::Overlay`, after tone-map.

### 10.2 Panel structure
- **World** — date/time controls, lat/lon override, time scale, pause,
  equinox/solstice/dawn/dusk shortcuts. Read-only display of: sun
  altitude/azimuth, moon altitude/azimuth, Julian day.
- **Render** — EV100, tone mapper, vsync, "Screenshot tonemapped (PNG)" and
  "Screenshot HDR (EXR)" buttons. Both write to `paths.screenshot_dir`.
- **Subsystem toggles** — checkboxes that enable/disable each subsystem.
- **Atmosphere** — every field of `AtmosphereParams` as a slider with numeric
  input. "Reset" per parameter, "Reset all to Earth" button.
- **Clouds** — every field of `CloudParams` plus per-layer accordions for the
  layer's `CloudLayerGpu` fields. "Freeze time" toggle.
- **Wet surface** — `wetness`, `puddle_coverage`, `puddle_start`,
  `snow_depth_m`.
- **Precipitation** — `type`, `intensity_mm_per_h`.
- **Debug** — toggleable fullscreen LUT viewers (transmittance, multi-scatter,
  sky-view as 2D; aerial-perspective with a depth-slice slider 0..32 km).
  Probe-point readouts for transmittance/optical-depth at a selected screen
  pixel. FPS, frame time, GPU timestamps for each pass.

Every slider shows its numeric value with at least 4 significant figures and
accepts numeric input directly.

### 10.3 Slider edits → reconfigure
All slider edits write to the live `Config` object in memory and call
`subsystem.reconfigure(&config, &gpu)` — the same path used by file-watch
hot-reload. No special "applied from UI" code path exists; both sources are
unified.

### 10.4 Preset loader
Top-bar "Load scene…" and "Save scene…" buttons. "Save" serialises the
current full state (engine + scene + per-subsystem overrides) to a TOML pair
matching the format in Appendices A/B.

### 10.5 Comparison mode (stretch)
Side-by-side mode loading two scene configs into one window, split by a
draggable vertical bar. Useful for A/B comparing weather states.

**Acceptance:** Every uniform mentioned in Phases 5–9 is reachable via a
slider in the overlay and updates the next frame with no recompile.

---

## Phase 11 — Test Harness, Reference Scenes, Validation

**Goal:** A library of reference weather scenarios that exercise every code
path, plus golden-image regression.

### 11.1 Reference scenes (`tests/scenes/`)
Ship as actual TOML fixtures from day one:

1. **`clear_summer_noon.toml`** — Dunblane, 2026-06-21 12:00 BST, no clouds,
   vis 30 km. Validates pure atmosphere.
2. **`broken_cumulus_afternoon.toml`** — 2026-05-10 15:00 UTC, single cumulus
   layer, coverage 0.5, base 1500 m. (See Appendix B.)
3. **`overcast_drizzle.toml`** — Single stratus layer base 400 m top 800 m,
   coverage 1.0, light rain 1 mm/h, wetness 0.8.
4. **`thunderstorm.toml`** — Cumulonimbus base 800 m top 11000 m, coverage
   0.8, heavy rain 20 mm/h.
5. **`high_cirrus_sunset.toml`** — 2026-09-22 19:00 UTC, single cirrus layer
   9000 m, coverage 0.4.
6. **`winter_overcast_snow.toml`** — Stratocumulus, snow on ground 0.1 m,
   light snow falling.
7. **`twilight_civil.toml`** — Sun at altitude −5°.
8. **`mountain_wave_clouds.toml`** — Synthesised altocumulus lenticularis
   using a custom coverage grid file (lozenge-shaped patches).

### 11.2 Headless renderer
A subcommand `ps-app render --scene <toml> --time <ISO8601> --output <png>`
initialises the app without a window:

```rust
let adapter = instance.request_adapter(&RequestAdapterOptions {
    power_preference: PowerPreference::HighPerformance,
    compatible_surface: None,           // <-- no surface
    force_fallback_adapter: false,
}).await?;
```

The framebuffer is a `Texture` with `COPY_SRC` usage; a staging buffer reads
back to host, then `image::save_buffer_*` writes the PNG (or `exr` writes
EXR for HDR).

### 11.3 Golden-image regression
A `cargo test --test golden` target that headlessly renders all eight
reference scenes at 1280×720 and a fixed time, then compares to
`tests/golden/*.png` via `image-compare = "0.4"`.

**Tolerance:** SSIM ≥ 0.99. Bit-exact comparison is not feasible because
different GPU vendors round fp16 differently.

`cargo run --bin ps-bless` writes the current renders as the new goldens.

### 11.4 Documentation outputs
The harness produces, per scene:
- `<scene>.png` — tonemapped final image.
- `<scene>.exr` — HDR offscreen target.
- `<scene>.weather_dump.json` — synthesised state.
- `<scene>.parameter_log.toml` — every slider value at render time.

**Acceptance:** All eight reference scenes render headlessly with no warnings,
produce visually convincing images, and pass the golden-image regression test.

---

## Cross-Cutting Concerns

### Logging
`tracing` with per-module level filters. Default `INFO`; per-frame at `TRACE`.

### Errors
Top-level `anyhow::Result<()>` from `main`. Library crates use their own
`thiserror` enums, mapped to `anyhow` at crate boundaries. **No `unwrap()`**
outside test code.

### GPU debugging
- `wgpu::DebugLabel` on every pipeline, bind group, and texture.
- Validation enabled in debug builds (`[debug] gpu_validation = true`).
- `--gpu-trace <dir>` flag writes a wgpu trace.

### Shader hot-reload
WGSL files in `shaders/` watched by `notify`. On change, affected pipelines
rebuild. Compilation errors surface in the egui overlay rather than crashing.

### Determinism
- `--seed <u64>` flag controls every RNG path (jitter, particle spawning,
  noise). Default seeded from a config field.
- Blue-noise jitter is **spatial only** and frame-deterministic so the cloud
  march does not shimmer when paused.
- Cloud and precipitation animations use `world.simulated_seconds` (pause-
  aware), not wall-clock `frame.time_seconds`, so pausing the world clock
  freezes evolution.

### Threading
- Render loop on the main thread (winit requirement).
- Synthesis runs on a `crossbeam` worker thread when weather data is large;
  results published via a channel and uploaded on the next `prepare()`.

### Anti-aliasing
- MSAA off in v1 (volumetric clouds and atmosphere don't benefit from MSAA;
  ground geometry edges are tolerable at 1080p).
- TAA off in v1; if added later, blue-noise jitter changes from spatial-only
  to temporally varying — but this is a v2 concern.

### README deliverable
`README.md` at the workspace root must cover: install, run-the-test-harness,
how to author a scene, how to bake noise, how to interpret the UI panels, how
to update goldens, and how to produce HDR EXR output.

---

## What Is Explicitly Out of Scope (v1)

- Data ingestion (METAR parsing, GRIB2 reading, etc.).
- Vehicle/aircraft models or windscreen rain shaders.
- Realistic terrain (DEM-driven). Ground is a flat plane.
- Crepuscular rays / godrays.
- Lightning visuals + audio.
- Auroras.
- VR rendering.
- Networked multi-instance.
- Per-pixel cloud type blending (deferred from §3.2.3).
- RGB cloud transmittance (scalar in `.a` for v1).
- Temporal reprojection / TAA.

---

## Suggested Phase Order

The phases are also the implementation order. **Do not skip ahead.** Each
phase produces a runnable build that demonstrably works before the next
begins.

- Phases 0–4 must complete and the empty render graph must run cleanly
  before any shader work.
- Phase 5 (atmosphere) must complete and visually validate against Hillaire
  reference screenshots before Phase 6 (clouds).
- Phase 7 (ground) and 8 (precipitation) can be developed in parallel after
  Phase 6.
- Phase 10 (UI) is developed in lockstep — add panels for each subsystem as
  it lands.
- Phase 11 (test harness) is the final acceptance phase.

---

## Reference Reading List

Pin these tabs during implementation:

- Hillaire 2020, *A Scalable and Production Ready Sky and Atmosphere Rendering
  Technique*: <https://sebh.github.io/publications/>
- Hillaire reference impl: <https://github.com/sebh/UnrealEngineSkyAtmosphere>
- Bruneton 2008 *Precomputed Atmospheric Scattering* + 2017 reimpl:
  <https://ebruneton.github.io/precomputed_atmospheric_scattering/>
- Schneider & Vos 2015, *Real-time Volumetric Cloudscapes of Horizon Zero
  Dawn*: <https://www.schneidervfx.com/>
- Schneider 2017 Nubis & 2022 Nubis Evolved:
  <https://www.guerrilla-games.com/read/>
- Hillaire 2016, *Physically Based Sky, Atmosphere & Cloud Rendering in
  Frostbite*:
  <https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf>
- Toft, Bowles, Zimmermann 2016, *Optimisations for Real-Time Volumetric
  Cloudscapes*: <https://arxiv.org/abs/1609.05344>
- NREL Solar Position Algorithm: <https://midcdmz.nrel.gov/spa/>
- Lagarde 2013, *Water drop* series 1–3:
  <https://seblagarde.wordpress.com/>
- Tatarchuk 2006, *Artist-Directable Real-Time Rain Rendering*:
  <https://advances.realtimerendering.com/s2006/Tatarchuk-Rain.pdf>
- pbr-book.org volume scattering chapters:
  <https://pbr-book.org/3ed-2018/Volume_Scattering>
- The full pipeline survey delivered alongside this plan.

---

## Final Note to the Implementing Agent

This is a quality-first test harness for an experienced engineer who will
scrutinise the visual results against reference imagery and against
real-world weather observations. **Do not optimise for performance until
correctness is achieved.** If a simpler approximation is tempting, prefer the
slower, physically grounded version and leave the optimisation as a clearly-
flagged TODO with a citation to the technique that would replace it. When in
doubt, ask for clarification before making a quality-affecting trade-off.

---

## Appendix A — Sample `pedalsky.toml`

Engine root configuration. Belongs at the workspace root. A copy is provided
alongside this plan as `pedalsky.toml`.

```toml
[window]
width  = 1920
height = 1080
title  = "PedalSky"
vsync  = true

[world]
latitude_deg       = 56.1922       # Dunblane, Scotland
longitude_deg      = -3.9645
ground_elevation_m = 60.0
ground_albedo      = [0.18, 0.18, 0.18]
ground_radius_m    = 6_360_000.0
atmosphere_top_m   =   100_000.0

[time]
year   = 2026
month  = 5
day    = 10
hour   = 14
minute = 30
second = 0
timezone_offset_hours = 1.0
auto_advance          = false
time_scale            = 1.0

[render]
hdr_format   = "Rgba16Float"
depth_format = "Depth32Float"
ev100        = 14.0
tone_mapper  = "ACESFilmic"
clear_color  = [0.0, 0.0, 0.0, 1.0]

[render.subsystems]
ground        = true
atmosphere    = true
clouds        = true
precipitation = false
wet_surface   = false

[render.atmosphere]
multi_scattering        = true
sun_disk                = true
sun_angular_radius_deg  = 0.27
ozone_enabled           = true

[render.clouds]
cloud_steps           = 192
light_steps           = 6
multi_scatter_octaves = 4
detail_strength       = 0.35
powder_strength       = 1.0
reprojection          = "off"
freeze_time           = false

[render.precip]
near_particle_count = 8000
far_layers          = 3

[paths]
weather        = "scenes/broken_cumulus_afternoon.toml"
noise_cache    = "assets/noise"
screenshot_dir = "screenshots"

[debug]
gpu_validation    = true
shader_hot_reload = true
log_level         = "info"
auto_exposure     = false
```

---

## Appendix B — Sample scene `scenes/broken_cumulus_afternoon.toml`

Equivalent METAR (synthetic): `EGPN 101430Z 24010KT 9999 BKN050 17/07 Q1018 NOSIG`.
A copy is provided alongside this plan as `broken_cumulus_afternoon.toml`.

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

The seven other reference scenes from Phase 11.1 follow this same shape with
different layer counts, types, and surface conditions.
