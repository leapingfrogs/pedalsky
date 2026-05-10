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

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            match res {
                Ok(event) => {
                    if !is_relevant_event(&event.kind) {
                        return;
                    }
                    for path in event.paths {
                        // Use canonicalize-equivalent tail comparison: notify
                        // sometimes reports the parent directory, which we
                        // ignore. We classify by basename match.
                        let kind = classify(&path, &config_path_for_closure, &scene_path_for_closure);
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
        if *stop.lock().unwrap() {
            break;
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
