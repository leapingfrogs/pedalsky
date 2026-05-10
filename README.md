# PedalSky

A composable, dependency-injected wgpu-based weather renderer test harness.
See `Pedalsky corrected implementation plan.md` for the full architectural
specification.

## Status

This codebase is being built phase by phase against the plan.

| Phase | Scope | Status |
|---|---|---|
| 0 | Cargo workspace, GPU init, HDR framebuffer + reverse-Z depth, fly camera, ACES tone-mapper, checker ground, winit window | **Done** |
| 1 | Config + scene schemas, `RenderSubsystem` trait, contexts, `AppBuilder` + factories, hot-reload watcher, demo Backdrop / Tint subsystems, headless integration tests | **Done** |
| 2+ | World/sun, weather synthesis, atmosphere, clouds, ground (PBR), precipitation, post-process, UI, golden-image regression | Not started |

## Repository layout

```
pedalsky/
├── Cargo.toml                  # workspace
├── pedalsky.toml               # engine root config
├── scenes/                     # weather scene fixtures
│   └── broken_cumulus_afternoon.toml
├── crates/
│   ├── ps-core/                # config, scene, traits, contexts, app, hot-reload, gpu, framebuffer, camera
│   ├── ps-postprocess/         # ACES Filmic / Passthrough tone-mapper
│   ├── ps-ground/              # Phase 0 procedural checker plane (Phase 7 replaces with PBR)
│   ├── ps-backdrop/            # Phase 1 demo: HDR clear at SkyBackdrop
│   ├── ps-tint/                # Phase 1 demo: fullscreen RGB multiply at PostProcess
│   └── ps-app/                 # winit window + render loop + headless test harness
├── shaders/
│   ├── ground/                 # Phase 0 checker shader
│   ├── postprocess/            # tone-map shader
│   └── tint/                   # Phase 1 demo tint shader
└── assets/                     # noise volumes (Phase 6+)
```

## Building and running

Requires a recent Rust toolchain (`rust-version = 1.85`).

```sh
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run --bin ps-app
```

`cargo run --bin ps-app` currently:

1. Locates `pedalsky.toml` at the workspace root.
2. Loads and validates it, logging a structured summary at `info` level.
3. Loads the scene file pointed to by `[paths] weather`, validates it.
4. Opens a window per `[window]`, initialises wgpu, allocates an HDR
   `Rgba16Float` colour target plus a reverse-Z `Depth32Float` depth
   buffer, builds the ACES Filmic tone-mapper and the placeholder ground
   plane.
5. Drives a fly camera (WASD, mouse-look while LMB-held, Space/Ctrl to
   ascend/descend, Q/E to roll, Shift to sprint) and renders the scene at
   the surface format's native rate.

The window stays open until you close it; press **Esc** to release the
mouse, **F4** to exit.

## Configuration

`pedalsky.toml` at the workspace root is the engine config. It uses
`#[serde(deny_unknown_fields)]` everywhere — typos are surfaced as load
errors. Every block has sensible defaults so partial configs work.

Subsystems are toggled via `[render.subsystems]`:

```toml
[render.subsystems]
ground        = true
atmosphere    = true
clouds        = true
precipitation = false
wet_surface   = false
backdrop      = true   # Phase 1 demo subsystem
tint          = false  # Phase 1 demo subsystem
```

Setting a flag to `false` removes the subsystem from the render graph
entirely — its factory is never invoked, no GPU resources are allocated.

## Scene files

`[paths] weather` selects the scene config (default
`scenes/broken_cumulus_afternoon.toml`). Each scene is a self-contained TOML
file — see Appendix B of the plan for the schema.

Cloud layers are validated for non-overlap:

```toml
[[clouds.layers]]
type = "Cumulus"
base_m = 1500.0
top_m  = 2300.0
```

A scene with two layers covering overlapping altitude ranges is rejected at
load time with a `SceneError::OverlappingCloudLayers` naming both indices.

## Hot reload

`ps_core::HotReload` watches `pedalsky.toml` and the scene file via `notify`
and emits debounced (`200 ms` default) `WatchEvent::ConfigChanged` /
`WatchEvent::SceneChanged` events on a crossbeam channel. The watcher
itself does not parse — the caller re-loads, re-validates, and either calls
`App::reconfigure(&new_config, &gpu)` to apply the change in place or drops
and rebuilds affected subsystems via their factories.

```rust
use ps_core::{HotReload, WatchEvent, DEFAULT_DEBOUNCE};

let watcher = HotReload::watch(&config_path, &scene_path, DEFAULT_DEBOUNCE)?;
for event in watcher.events() {
    match event {
        WatchEvent::ConfigChanged(_) => /* re-load, validate, app.reconfigure(...) */ (),
        WatchEvent::SceneChanged(_) => /* re-load and re-synthesise */ (),
        WatchEvent::Error(msg) => tracing::warn!(%msg, "hot-reload error"),
    }
}
```

The `ps-app` binary wires this into the winit main loop; editing
`pedalsky.toml` while the app is running calls `App::reconfigure(...)`
on the next frame.

## Testing

```sh
cargo test --workspace
```

The `crates/ps-core/tests/` integration tests cover:

- **`config.rs`** (9 tests) — round-trip of the workspace `pedalsky.toml`,
  defaults for partial configs, rejection of unknown fields, range checks,
  schema evolution guards, file-existence validation via
  `validate_with_base`.
- **`scene.rs`** (5 tests) — round-trip of
  `scenes/broken_cumulus_afternoon.toml`, PascalCase-serialised cloud type
  round-trip across all eight variants, vertical-overlap rejection,
  `deny_unknown_fields` enforcement, future-schema-version rejection.
- **`subsystem.rs`** (2 tests) — `PassStage` ordering, the trait derive
  set, a compile-only smoke test that exercises every field of
  `PrepareContext` / `RenderContext`.
- **`app.rs`** (7 tests) — disabled-subsystem factories are not invoked,
  enabled factories are, multi-stage pass flattening sorts by `PassStage`,
  `prepare()` runs in the order given by each subsystem's minimum
  `PassStage`, behavioural test that drives `App::frame` end-to-end and
  asserts on the actual call order, `reconfigure()` adds/drops subsystems
  on the fly, duplicate factory names are rejected.
- **`hot_reload.rs`** (4 tests) — config change emits `ConfigChanged`,
  scene change emits `SceneChanged`, a burst of writes within the debounce
  window collapses to one event, invalid TOML still emits an event.

The `crates/ps-app/tests/integration.rs` file adds 4 headless-render
integration tests covering Backdrop, Tint, runtime reconfigure, and a
boot-from-real-`pedalsky.toml` smoke test. Total: **31 tests**.

## Known cross-cutting gaps (Phase 2+)

The following plan-mandated cross-cutting items are intentionally deferred
because they need code from later phases or runtime infrastructure outside
the Phase 0/1 scope. Each is captured here so a future implementer can
pick them up:

- **`--seed <u64>` CLI flag** (plan §Cross-Cutting/Determinism) — needs RNG
  paths to seed first; Phase 6 owns blue-noise jitter and Phase 8 owns
  particle spawning, both of which thread `seed` through their state.
- **`--gpu-trace <dir>` flag** (plan §GPU debugging) — wgpu trace API
  changed considerably between releases; will land alongside Phase 5
  when there's something non-trivial to trace.
- **Shader hot-reload** (plan §Cross-Cutting/Shader hot-reload) —
  `[debug] shader_hot_reload` is parsed today but unused. Wire alongside
  Phase 5 when WGSL pipelines start landing.
- **Camera UI sliders for fov / near / speed** (plan §0.4) — Phase 10
  egui overlay owns this.
- **README depth on noise baking, UI panels, golden-image updates, EXR
  output** (plan §README deliverable) — these document features owned by
  Phase 6 (noise bake), Phase 10 (UI), Phase 11 (goldens, EXR). Will land
  with each phase.
- **`ps-app render` headless subcommand** writing PNG/EXR to
  `paths.screenshot_dir` — Phase 11.
- **Reconfigure-only-affected-subsystems** (plan §1.6) — current
  implementation calls `reconfigure()` on every live subsystem on any
  config change. Subsystems internally no-op when their slice of the
  config hasn't changed, so the spirit is preserved; the letter ("diff
  against the live config; for each affected subsystem") wants a
  per-subsystem diff that's more useful when Phase 10's UI starts firing
  reconfigures at slider rates.

## License

MIT OR Apache-2.0.
