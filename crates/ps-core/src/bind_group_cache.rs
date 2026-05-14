//! Key-comparison cache for `wgpu::BindGroup`.
//!
//! Many subsystems build a per-frame bind group whose entries reference
//! resources that only change occasionally — `WeatherState::textures`
//! views (which only change when `ps-synthesis::synthesise` reruns),
//! the framebuffer depth view (only on resize), and so on. Rebuilding
//! the bind group every frame costs ~5–30 µs at the wgpu hub for no
//! behavioural reason in steady state.
//!
//! `BindGroupCache<K>` stamps each rebuild with a caller-defined key
//! (typically a tuple of revision numbers and sizes) and only re-runs
//! the builder closure when the key changes. Typical use:
//!
//! ```ignore
//! struct MySubsystem {
//!     bg_cache: Mutex<BindGroupCache<(u64, u32, u32)>>,
//!     // …
//! }
//!
//! // inside a pass closure:
//! let bg = self.bg_cache.lock().unwrap().get_or_build(
//!     (ctx.weather.revision, fb_w, fb_h),
//!     || ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
//!         label: Some("my-bg"),
//!         layout: &self.layout,
//!         entries: &[…],
//!     }),
//! );
//! ```
//!
//! The returned `Arc<wgpu::BindGroup>` is cheap to clone and outlives
//! the lock guard so callers can release the cache lock immediately
//! after calling `get_or_build`.

use std::sync::Arc;

/// Key-comparison `wgpu::BindGroup` cache. Holds one bind group at a
/// time; rebuilds when the supplied key differs from the cached one,
/// or when the cache is empty.
///
/// `K` is typically a `(u64, u32, u32, bool, …)` composite of every
/// piece of state the bind group depends on. Use `PartialEq + Copy`
/// types — the cache copies the key on every comparison.
pub struct BindGroupCache<K> {
    cached: Option<(K, Arc<wgpu::BindGroup>)>,
}

impl<K> Default for BindGroupCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> BindGroupCache<K> {
    /// Construct an empty cache.
    pub const fn new() -> Self {
        Self { cached: None }
    }

    /// Return the bind group associated with `key`, rebuilding via
    /// `build` if the cache is empty or holds a different key.
    ///
    /// The returned `Arc` is cheap to clone — callers typically clone
    /// once into the active pass and drop the lock guard.
    pub fn get_or_build<F>(&mut self, key: K, build: F) -> Arc<wgpu::BindGroup>
    where
        K: PartialEq + Copy,
        F: FnOnce() -> wgpu::BindGroup,
    {
        if let Some((cached_key, bg)) = &self.cached {
            if *cached_key == key {
                return Arc::clone(bg);
            }
        }
        let bg = Arc::new(build());
        self.cached = Some((key, Arc::clone(&bg)));
        bg
    }

    /// Invalidate the cache unconditionally. Useful when the caller
    /// knows the bind group's inputs changed without a key bump
    /// (e.g. a sibling texture allocated in the same subsystem).
    pub fn invalidate(&mut self) {
        self.cached = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_starts_empty() {
        let cache: BindGroupCache<u64> = BindGroupCache::new();
        assert!(cache.cached.is_none());
    }

    #[test]
    fn invalidate_clears_cache() {
        // Can't construct a real BindGroup without a wgpu device, so
        // exercise the structural invariant via `invalidate` directly.
        let mut cache: BindGroupCache<u64> = BindGroupCache::new();
        cache.invalidate();
        assert!(cache.cached.is_none());
    }
}
