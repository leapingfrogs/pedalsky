//! Binary disk cache for satellite imagery tiles.
//!
//! Same shape as `ps_terrain::BinaryCache` but keyed on
//! `(source, z, x, y)` — the standard slippy-map tile coordinate
//! tuple. TTL is long (30 days) because the EOX cloudless composite
//! is a yearly product.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Cache handle bound to a directory.
#[derive(Debug, Clone)]
pub struct BinaryCache {
    root: PathBuf,
}

/// Result of a cache lookup.
#[derive(Debug)]
pub enum BinaryCacheHit {
    /// Fresh hit within TTL.
    Fresh(Vec<u8>),
    /// File exists but older than TTL.
    Stale(Vec<u8>),
    /// No file.
    Miss,
}

impl BinaryCache {
    /// Create a cache anchored at `root`. The directory is created on
    /// demand inside [`Self::write`].
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Filename for an XYZ tile. Example:
    /// `eox-s2cloudless-2024-z12-x2002-y1271.jpg`.
    fn key(source: &str, zoom: u32, x: u32, y: u32) -> String {
        format!("{source}-z{zoom}-x{x}-y{y}.jpg")
    }

    fn path_for(&self, source: &str, zoom: u32, x: u32, y: u32) -> PathBuf {
        self.root.join(Self::key(source, zoom, x, y))
    }

    /// Look up a cache entry; returns `Fresh` within TTL, `Stale`
    /// otherwise, `Miss` if absent.
    pub fn read_any(
        &self,
        source: &str,
        zoom: u32,
        x: u32,
        y: u32,
        ttl: Duration,
    ) -> io::Result<BinaryCacheHit> {
        let path = self.path_for(source, zoom, x, y);
        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(BinaryCacheHit::Miss),
            Err(e) => return Err(e),
        };
        let age = metadata
            .modified()
            .ok()
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .unwrap_or(Duration::MAX);
        let body = fs::read(&path)?;
        if age <= ttl {
            Ok(BinaryCacheHit::Fresh(body))
        } else {
            Ok(BinaryCacheHit::Stale(body))
        }
    }

    /// Look up a cache entry; only returns `Some(body)` if fresh.
    pub fn read_fresh(
        &self,
        source: &str,
        zoom: u32,
        x: u32,
        y: u32,
        ttl: Duration,
    ) -> io::Result<Option<Vec<u8>>> {
        match self.read_any(source, zoom, x, y, ttl)? {
            BinaryCacheHit::Fresh(b) => Ok(Some(b)),
            BinaryCacheHit::Stale(_) | BinaryCacheHit::Miss => Ok(None),
        }
    }

    /// Write a cache entry, creating the directory tree on demand.
    pub fn write(&self, source: &str, zoom: u32, x: u32, y: u32, body: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let path = self.path_for(source, zoom, x, y);
        fs::write(&path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format() {
        assert_eq!(
            BinaryCache::key("eox-s2cloudless-2024", 12, 2002, 1271),
            "eox-s2cloudless-2024-z12-x2002-y1271.jpg"
        );
    }

    #[test]
    fn round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let cache = BinaryCache::new(dir.path());
        let body = b"\xff\xd8\xff\xe0fake-jpeg";
        cache
            .write("eox-s2cloudless-2024", 12, 2002, 1271, body)
            .unwrap();
        let got = cache
            .read_fresh(
                "eox-s2cloudless-2024",
                12,
                2002,
                1271,
                Duration::from_secs(60),
            )
            .unwrap();
        assert_eq!(got.as_deref(), Some(&body[..]));
    }
}
