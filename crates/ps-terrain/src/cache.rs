//! Binary disk cache for terrain tiles.
//!
//! Parallel to `ps_weather_feed::Cache` but writes raw bytes instead
//! of JSON strings — DEM tiles are GeoTIFF binaries. Key shape mirrors
//! ps-weather-feed: `(source, lat_floor, lon_floor)` with a long TTL
//! because DEM tiles don't change over time.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// Cache handle bound to a directory.
#[derive(Debug, Clone)]
pub struct BinaryCache {
    root: PathBuf,
}

/// Result of a cache lookup. Mirrors `ps_weather_feed::cache::CacheHit`.
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

    /// Canonical filename for a 1° DEM tile: `<source>-<NS><lat>_<EW><lon>.tif`.
    /// `lat_floor`/`lon_floor` are the integer 1° corners (south-west of
    /// the tile per the Copernicus convention).
    fn key(source: &str, lat_floor: i32, lon_floor: i32) -> String {
        let (ns, lat) = if lat_floor >= 0 { ('N', lat_floor) } else { ('S', -lat_floor) };
        let (ew, lon) = if lon_floor >= 0 { ('E', lon_floor) } else { ('W', -lon_floor) };
        format!("{source}-{ns}{lat:02}_{ew}{lon:03}.tif")
    }

    fn path_for(&self, source: &str, lat_floor: i32, lon_floor: i32) -> PathBuf {
        self.root.join(Self::key(source, lat_floor, lon_floor))
    }

    /// Look up a cache entry; returns `Fresh` within TTL, `Stale`
    /// otherwise, `Miss` if absent.
    pub fn read_any(
        &self,
        source: &str,
        lat_floor: i32,
        lon_floor: i32,
        ttl: Duration,
    ) -> io::Result<BinaryCacheHit> {
        let path = self.path_for(source, lat_floor, lon_floor);
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
        lat_floor: i32,
        lon_floor: i32,
        ttl: Duration,
    ) -> io::Result<Option<Vec<u8>>> {
        match self.read_any(source, lat_floor, lon_floor, ttl)? {
            BinaryCacheHit::Fresh(b) => Ok(Some(b)),
            BinaryCacheHit::Stale(_) | BinaryCacheHit::Miss => Ok(None),
        }
    }

    /// Write a cache entry, creating the directory tree on demand.
    pub fn write(
        &self,
        source: &str,
        lat_floor: i32,
        lon_floor: i32,
        body: &[u8],
    ) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let path = self.path_for(source, lat_floor, lon_floor);
        fs::write(&path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format_north_east() {
        assert_eq!(
            BinaryCache::key("copernicus-glo30", 56, 12),
            "copernicus-glo30-N56_E012.tif"
        );
    }

    #[test]
    fn key_format_south_west() {
        assert_eq!(
            BinaryCache::key("copernicus-glo30", -34, -58),
            "copernicus-glo30-S34_W058.tif"
        );
    }

    #[test]
    fn round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let cache = BinaryCache::new(dir.path());
        let body = b"\x00\x01\x02fake-tiff";
        cache.write("copernicus-glo30", 56, -4, body).unwrap();
        let got = cache
            .read_fresh("copernicus-glo30", 56, -4, Duration::from_secs(60))
            .unwrap();
        assert_eq!(got.as_deref(), Some(&body[..]));
    }

    #[test]
    fn miss_when_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = BinaryCache::new(dir.path());
        let hit = cache
            .read_any("copernicus-glo30", 0, 0, Duration::from_secs(60))
            .unwrap();
        assert!(matches!(hit, BinaryCacheHit::Miss));
    }
}
