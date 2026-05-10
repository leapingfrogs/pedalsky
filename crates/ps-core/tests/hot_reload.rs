//! Phase 1 Group B tests for the hot-reload watcher.
//!
//! These tests touch the filesystem and depend on `notify` event timing,
//! which is platform- and load-dependent. Tests use generous timeouts and
//! deliberately do not assert on event ordering across kinds.

use std::io::Write;
use std::time::Duration;

use ps_core::{HotReload, WatchEvent};

/// Minimum-content valid pedalsky.toml stub.
const CONFIG_STUB: &str = r#"
[window]
width = 800
height = 600
title = "test"
vsync = true
"#;

const SCENE_STUB: &str = r#"
schema_version = 1
[surface]
[[clouds.layers]]
type = "Cumulus"
base_m = 1000.0
top_m = 2000.0
coverage = 0.5
density_scale = 1.0
shape_octave_bias = 0.0
detail_octave_bias = 0.0
[precipitation]
type = "None"
intensity_mm_per_h = 0.0
[lightning]
strikes_per_min_per_km2 = 0.0
"#;

fn write_atomic(path: &std::path::Path, content: &str) {
    // Write to a temp file in the same dir then rename, to make notify see a
    // single Rename(To) on Windows rather than several partial writes.
    let dir = path.parent().unwrap();
    let tmp = dir.join(format!(
        ".tmp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).unwrap();
}

fn collect_for(rx: &crossbeam_channel::Receiver<WatchEvent>, dur: Duration) -> Vec<WatchEvent> {
    let mut out = Vec::new();
    let deadline = std::time::Instant::now() + dur;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(ev) => out.push(ev),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
    out
}

#[test]
fn config_change_emits_event() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pedalsky.toml");
    let scn = dir.path().join("scene.toml");
    std::fs::write(&cfg, CONFIG_STUB).unwrap();
    std::fs::write(&scn, SCENE_STUB).unwrap();

    let watcher =
        HotReload::watch(&cfg, &scn, Duration::from_millis(120)).expect("watcher created");
    // Give the watcher a moment to subscribe.
    std::thread::sleep(Duration::from_millis(150));

    write_atomic(&cfg, "[window]\nwidth = 1024\n");

    let events = collect_for(watcher.events(), Duration::from_millis(2000));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, WatchEvent::ConfigChanged(_))),
        "expected ConfigChanged in {events:?}"
    );
}

#[test]
fn scene_change_emits_event() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pedalsky.toml");
    let scn = dir.path().join("scene.toml");
    std::fs::write(&cfg, CONFIG_STUB).unwrap();
    std::fs::write(&scn, SCENE_STUB).unwrap();

    let watcher = HotReload::watch(&cfg, &scn, Duration::from_millis(120)).unwrap();
    std::thread::sleep(Duration::from_millis(150));

    write_atomic(&scn, SCENE_STUB);
    write_atomic(&scn, SCENE_STUB);

    let events = collect_for(watcher.events(), Duration::from_millis(2000));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, WatchEvent::SceneChanged(_))),
        "expected SceneChanged in {events:?}"
    );
}

#[test]
fn debounce_collapses_burst() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pedalsky.toml");
    let scn = dir.path().join("scene.toml");
    std::fs::write(&cfg, CONFIG_STUB).unwrap();
    std::fs::write(&scn, SCENE_STUB).unwrap();

    let watcher = HotReload::watch(&cfg, &scn, Duration::from_millis(250)).unwrap();
    std::thread::sleep(Duration::from_millis(150));

    // Three rapid writes well inside the 250 ms window.
    write_atomic(&cfg, "[window]\nwidth = 100\n");
    std::thread::sleep(Duration::from_millis(40));
    write_atomic(&cfg, "[window]\nwidth = 200\n");
    std::thread::sleep(Duration::from_millis(40));
    write_atomic(&cfg, "[window]\nwidth = 300\n");

    let events = collect_for(watcher.events(), Duration::from_millis(2000));
    let config_events = events
        .iter()
        .filter(|e| matches!(e, WatchEvent::ConfigChanged(_)))
        .count();
    assert_eq!(
        config_events, 1,
        "expected exactly one ConfigChanged after debounce, got {config_events}: {events:?}"
    );
}

#[test]
fn invalid_toml_after_change_does_not_panic() {
    // The watcher itself doesn't parse — this test verifies the watcher
    // emits a `ConfigChanged` event even when the new content is invalid
    // TOML. The CALLER is responsible for handling the parse error.
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("pedalsky.toml");
    let scn = dir.path().join("scene.toml");
    std::fs::write(&cfg, CONFIG_STUB).unwrap();
    std::fs::write(&scn, SCENE_STUB).unwrap();

    let watcher = HotReload::watch(&cfg, &scn, Duration::from_millis(120)).unwrap();
    std::thread::sleep(Duration::from_millis(150));

    write_atomic(&cfg, "this is = ][ not valid toml");

    let events = collect_for(watcher.events(), Duration::from_millis(2000));
    let any_config = events
        .iter()
        .any(|e| matches!(e, WatchEvent::ConfigChanged(_)));
    assert!(
        any_config,
        "watcher should emit despite invalid content: {events:?}"
    );
}
