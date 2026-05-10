# PedalSky

A composable, dependency-injected wgpu-based weather renderer test harness.
See `Pedalsky corrected implementation plan.md` for the full architectural
specification.

## Status

This codebase is being built phase by phase against the plan.

| Phase | Scope | Status |
|---|---|---|
| 0 | Cargo workspace, GPU init, HDR framebuffer, fly camera, tone-map placeholder, ground placeholder | **Partial** — workspace + headless GPU init + HDR framebuffer + fly camera struct landed; windowed render pipeline (winit main loop, tone-map pass, checker ground shader) deferred |
| 1 | Config + scene schemas, `RenderSubsystem` trait, contexts, `AppBuilder` + factories, hot-reload watcher | **Done** for non-GPU work — demo subsystems (Backdrop / Tint) and the headless-render integration tests deferred |
| 2+ | World/sun, weather synthesis, atmosphere, clouds, ground, precipitation, post-process, UI, golden-image regression | Not started |

The deferred Phase 0 windowed pipeline and Phase 1 demo subsystems are
intentionally held back: they need a working winit main loop and a
fullscreen-triangle pipeline that the schedule for this session did not
permit landing safely.

## Repository layout

```
pedalsky/
├── Cargo.toml                  # workspace
├── pedalsky.toml               # engine root config
├── scenes/                     # weather scene fixtures
│   └── broken_cumulus_afternoon.toml
├── crates/
│   ├── ps-core/                # config, scene, traits, contexts, app, hot-reload
│   ├── ps-postprocess/         # tone-map (placeholder)
│   └── ps-app/                 # main binary
├── shaders/                    # WGSL (Phase 4+)
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
3. Loads the scene file pointed to by `[paths] weather`, validates it,
   logs a summary.
4. Exits.

Once Phase 0's windowed pipeline lands, the same binary will open a window
and render the demo subsystems.

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

The `ps-app` binary will wire this in once Phase 0's windowed render loop
lands; today the watcher is library-level only.

## Testing

```sh
cargo test --workspace
```

The `crates/ps-core/tests/` integration tests cover:

- **`config.rs`** — round-trip of the workspace `pedalsky.toml`, defaults
  for partial configs, rejection of unknown fields, range checks, schema
  evolution guards.
- **`scene.rs`** — round-trip of `scenes/broken_cumulus_afternoon.toml`,
  PascalCase-serialised cloud type round-trip across all eight variants,
  vertical-overlap rejection, `deny_unknown_fields` enforcement, future-
  schema-version rejection.
- **`subsystem.rs`** — `PassStage` ordering, the trait derive set, a
  compile-only smoke test that exercises every field of `PrepareContext` /
  `RenderContext`.
- **`app.rs`** — disabled-subsystem factories are not invoked, enabled
  factories are, multi-stage pass flattening sorts by `PassStage`,
  `prepare()` runs in the order given by each subsystem's minimum
  `PassStage`, `reconfigure()` adds/drops subsystems on the fly, duplicate
  factory names are rejected.
- **`hot_reload.rs`** — config change emits `ConfigChanged`, scene change
  emits `SceneChanged`, a burst of writes within the debounce window
  collapses to one event, invalid TOML still emits an event (the caller
  handles the parse error rather than crashing the watcher).

## License

MIT OR Apache-2.0.
