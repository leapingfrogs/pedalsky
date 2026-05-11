//! Filesystem watcher that emits debounced `ConfigChanged` / `SceneChanged`
//! events on a crossbeam channel.
//!
//! See plan §1.6.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, Sender};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{debug, error, warn};

/// Default debounce window. A burst of writes within this period collapses to
/// a single emitted event.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(200);

/// Events emitted by the watcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEvent {
    /// The engine config (`pedalsky.toml`) changed.
    ConfigChanged(PathBuf),
    /// The active scene file changed.
    SceneChanged(PathBuf),
    /// `notify` reported an error or the file is currently invalid TOML.
    Error(String),
}

/// Events emitted by the shader watcher (plan §Cross-Cutting/Shader
/// hot-reload). Each `Changed` carries the path under `shaders/` that
/// triggered the event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderWatchEvent {
    /// A WGSL file under `shaders/` was created/modified/removed.
    Changed(PathBuf),
    /// `notify` reported an error.
    Error(String),
}

/// File classification used internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WatchedKind {
    Config,
    Scene,
}

/// A live hot-reload watcher. Drop this to stop watching and join the
/// debounce thread.
pub struct HotReload {
    rx: Receiver<WatchEvent>,
    /// Holding this keeps `notify` alive for as long as the watcher is in scope.
    _watcher: RecommendedWatcher,
    /// Holding this lets the debounce thread know we want it to exit.
    stop: Arc<Mutex<bool>>,
    /// Joined on drop so the thread is cleaned up deterministically.
    debounce_thread: Option<thread::JoinHandle<()>>,
}

impl HotReload {
    /// Watch the given config and scene paths. Returns the watcher and a
    /// receiver of debounced events.
    pub fn watch(
        config_path: &Path,
        scene_path: &Path,
        debounce: Duration,
    ) -> notify::Result<Self> {
        let (raw_tx, raw_rx) = unbounded::<(WatchedKind, PathBuf)>();
        let (tx, rx) = unbounded::<WatchEvent>();

        let config_path = config_path.to_path_buf();
        let scene_path = scene_path.to_path_buf();
        let config_path_for_closure = config_path.clone();
        let scene_path_for_closure = scene_path.clone();
        let raw_tx_for_watcher = raw_tx.clone();
        let tx_for_watcher = tx.clone();

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                match res {
                    Ok(event) => {
                        if !is_relevant_event(&event.kind) {
                            return;
                        }
                        for path in event.paths {
                            // Use canonicalize-equivalent tail comparison: notify
                            // sometimes reports the parent directory, which we
                            // ignore. We classify by basename match.
                            let kind =
                                classify(&path, &config_path_for_closure, &scene_path_for_closure);
                            if let Some(k) = kind {
                                if let Err(e) = raw_tx_for_watcher.send((k, path)) {
                                    warn!("watcher channel send failed: {e}");
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx_for_watcher.send(WatchEvent::Error(err.to_string()));
                    }
                }
            })?;

        // Watch the parent directories of each path. Some platforms (notably
        // Windows) report renames better when the parent dir is watched.
        for p in [&config_path, &scene_path] {
            if let Some(parent) = p.parent() {
                watcher.watch(parent, RecursiveMode::NonRecursive)?;
            } else {
                watcher.watch(p, RecursiveMode::NonRecursive)?;
            }
        }

        let stop = Arc::new(Mutex::new(false));
        let stop_for_thread = stop.clone();
        let tx_for_thread = tx.clone();
        let debounce_thread = thread::spawn(move || {
            debounce_loop(raw_rx, tx_for_thread, debounce, stop_for_thread);
        });

        Ok(Self {
            rx,
            _watcher: watcher,
            stop,
            debounce_thread: Some(debounce_thread),
        })
    }

    /// Receiver of debounced events. Clone-friendly via `crossbeam`.
    pub fn events(&self) -> &Receiver<WatchEvent> {
        &self.rx
    }
}

impl Drop for HotReload {
    fn drop(&mut self) {
        if let Ok(mut s) = self.stop.lock() {
            *s = true;
        }
        if let Some(handle) = self.debounce_thread.take() {
            // Best-effort: if the thread has panicked, log and continue.
            if let Err(e) = handle.join() {
                error!("hot-reload debounce thread panicked: {e:?}");
            }
        }
    }
}

/// Recursive watcher over a `shaders/` directory. Emits debounced
/// [`ShaderWatchEvent::Changed`] events on any `.wgsl` change. The host
/// receives, calls `App::reconfigure(...)` (or a focused
/// `App::rebuild_pipelines`) and the subsystems pick up the new source
/// via [`crate::shaders::load_shader`].
pub struct ShaderHotReload {
    rx: Receiver<ShaderWatchEvent>,
    _watcher: RecommendedWatcher,
    stop: Arc<Mutex<bool>>,
    debounce_thread: Option<thread::JoinHandle<()>>,
}

impl ShaderHotReload {
    /// Watch the given `shaders/` directory recursively.
    pub fn watch(shaders_root: &Path, debounce: Duration) -> notify::Result<Self> {
        let (raw_tx, raw_rx) = unbounded::<PathBuf>();
        let (tx, rx) = unbounded::<ShaderWatchEvent>();
        let tx_for_watcher = tx.clone();

        let mut watcher = notify::recommended_watcher(
            move |res: notify::Result<notify::Event>| match res {
                Ok(event) => {
                    if !is_relevant_event(&event.kind) {
                        return;
                    }
                    for path in event.paths {
                        // Only emit for .wgsl files; the watcher fires for
                        // editor temp files (`.swp`, `~`-suffixed) too.
                        if path.extension().is_some_and(|e| e == "wgsl") {
                            if let Err(e) = raw_tx.send(path) {
                                warn!("shader watcher channel send failed: {e}");
                            }
                        }
                    }
                }
                Err(err) => {
                    let _ = tx_for_watcher.send(ShaderWatchEvent::Error(err.to_string()));
                }
            },
        )?;
        watcher.watch(shaders_root, RecursiveMode::Recursive)?;

        let stop = Arc::new(Mutex::new(false));
        let stop_for_thread = stop.clone();
        let tx_for_thread = tx.clone();
        let debounce_thread = thread::spawn(move || {
            shader_debounce_loop(raw_rx, tx_for_thread, debounce, stop_for_thread);
        });

        Ok(Self {
            rx,
            _watcher: watcher,
            stop,
            debounce_thread: Some(debounce_thread),
        })
    }

    /// Receiver of debounced shader events.
    pub fn events(&self) -> &Receiver<ShaderWatchEvent> {
        &self.rx
    }
}

impl Drop for ShaderHotReload {
    fn drop(&mut self) {
        if let Ok(mut s) = self.stop.lock() {
            *s = true;
        }
        if let Some(handle) = self.debounce_thread.take() {
            if let Err(e) = handle.join() {
                error!("shader hot-reload debounce thread panicked: {e:?}");
            }
        }
    }
}

fn shader_debounce_loop(
    raw_rx: Receiver<PathBuf>,
    tx: Sender<ShaderWatchEvent>,
    debounce: Duration,
    stop: Arc<Mutex<bool>>,
) {
    // Single per-shader-file pending slot; collapse the bursts that
    // editors emit on save (write + truncate + rename can fire 3-4
    // events within milliseconds).
    use std::collections::HashMap;
    let mut pending: HashMap<PathBuf, Instant> = HashMap::new();
    loop {
        match stop.lock() {
            Ok(g) if *g => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let next_deadline = pending.values().min().copied();
        let timeout = match next_deadline {
            Some(d) => d.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(50),
        };
        match raw_rx.recv_timeout(timeout) {
            Ok(path) => {
                debug!(?path, "raw shader watcher event");
                pending.insert(path, Instant::now() + debounce);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                let now = Instant::now();
                let ready: Vec<PathBuf> = pending
                    .iter()
                    .filter_map(|(p, d)| (now >= *d).then(|| p.clone()))
                    .collect();
                for p in ready {
                    pending.remove(&p);
                    let event = ShaderWatchEvent::Changed(p);
                    debug!(?event, "emitting debounced shader event");
                    if let Err(e) = tx.send(event) {
                        warn!("shader debounce send failed: {e}");
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn is_relevant_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    )
}

fn classify(path: &Path, config_path: &Path, scene_path: &Path) -> Option<WatchedKind> {
    fn same_tail(a: &Path, b: &Path) -> bool {
        match (a.canonicalize(), b.canonicalize()) {
            (Ok(ca), Ok(cb)) => ca == cb,
            _ => a.file_name() == b.file_name() && a.parent_eq(b),
        }
    }
    if same_tail(path, config_path) {
        Some(WatchedKind::Config)
    } else if same_tail(path, scene_path) {
        Some(WatchedKind::Scene)
    } else {
        None
    }
}

trait ParentEq {
    fn parent_eq(&self, other: &Path) -> bool;
}
impl ParentEq for Path {
    fn parent_eq(&self, other: &Path) -> bool {
        match (self.parent(), other.parent()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

fn debounce_loop(
    raw_rx: Receiver<(WatchedKind, PathBuf)>,
    tx: Sender<WatchEvent>,
    debounce: Duration,
    stop: Arc<Mutex<bool>>,
) {
    // Pending state per kind: most-recent path + the deadline at which we
    // emit the collapsed event.
    let mut pending: [Option<(PathBuf, Instant)>; 2] = [None, None];

    fn idx(k: WatchedKind) -> usize {
        match k {
            WatchedKind::Config => 0,
            WatchedKind::Scene => 1,
        }
    }
    fn from_idx(i: usize) -> WatchedKind {
        match i {
            0 => WatchedKind::Config,
            _ => WatchedKind::Scene,
        }
    }

    loop {
        // Treat a poisoned stop-flag (a panicking thread holding the lock) as
        // an implicit stop signal — better than panicking the debounce thread
        // too.
        match stop.lock() {
            Ok(g) if *g => break,
            Ok(_) => {}
            Err(_) => break,
        }
        // Compute the next deadline; if nothing pending, just block on raw_rx.
        let next_deadline = pending
            .iter()
            .filter_map(|p| p.as_ref().map(|(_, d)| *d))
            .min();

        let timeout = match next_deadline {
            Some(d) => d.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(50),
        };

        match raw_rx.recv_timeout(timeout) {
            Ok((kind, path)) => {
                debug!(?kind, ?path, "raw watcher event");
                pending[idx(kind)] = Some((path, Instant::now() + debounce));
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                let now = Instant::now();
                for (i, slot) in pending.iter_mut().enumerate() {
                    let take = matches!(slot, Some((_, deadline)) if now >= *deadline);
                    if take {
                        if let Some((path, _)) = slot.take() {
                            let event = match from_idx(i) {
                                WatchedKind::Config => WatchEvent::ConfigChanged(path),
                                WatchedKind::Scene => WatchEvent::SceneChanged(path),
                            };
                            debug!(?event, "emitting debounced event");
                            if let Err(e) = tx.send(event) {
                                warn!("debounce send failed: {e}");
                            }
                        }
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}
