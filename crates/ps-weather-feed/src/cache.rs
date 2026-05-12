//! Disk cache for weather-API responses, keyed by (source, lat,
//! lon, UTC hour).
//!
//! Why we cache. Open-Meteo's free tier publishes ~10 000 calls/day
//! per IP; the METAR endpoint has even tighter limits. Without a
//! cache, every click of the "Fetch real weather" button would burn
//! a request — easy to hit a rate limit by accident while
//! iterating on UI changes. The cache layer makes a click cheap.
//!
//! Design:
//!
//! - Cache directory configurable; default sits under the workspace
//!   so it's discoverable (and gitignored).
//! - Filename includes the source name + a coarse lat/lon (3
//!   decimal places ≈ 110 m precision) + the UTC hour-aligned
//!   timestamp. That keys the cache by the same axes the
//!   underlying API does, so repeated fetches at the same place
//!   and hour are byte-identical hits.
//! - TTL is per-source: forecasts are valid for ~1 hour (a new
//!   hour brings a new forecast row); METARs refresh ~every
//!   30 minutes (`fmh` issuance). Caller supplies the TTL.
//! - On HTTP failure the caller can still fall back to an expired
//!   cache file — `read_any()` returns regardless of age. This
//!   keeps the feature usable offline once the cache has been
//!   warmed.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Timelike, Utc};

/// Result of a cache lookup.
#[derive(Debug)]
pub enum CacheHit {
    /// A fresh cache file (within TTL).
    Fresh(String),
    /// A cache file exists but is older than the TTL. Returned by
    /// `read_any`; `read_fresh` reports Miss for the same state.
    Stale(String),
    /// No cache file at all.
    Miss,
}

/// Cache handle bound to a directory.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Create a cache anchored at `root`. The directory is created
    /// on demand inside `write`; `Cache::new` itself is infallible
    /// so callers can pass paths that don't exist yet.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Build the canonical filename for a (source, lat, lon, hour)
    /// tuple. The latitude / longitude are rounded to three
    /// decimal places (~110 m precision) so neighbouring spots
    /// share a cache entry. Hour is UTC, second-zeroed.
    fn key(source: &str, lat: f64, lon: f64, time: DateTime<Utc>) -> String {
        // Round the time to the previous hour boundary so that two
        // requests within the same hour collide.
        let hour = time
            .with_minute(0).expect("0 minutes valid")
            .with_second(0).expect("0 seconds valid")
            .with_nanosecond(0).expect("0 nanos valid");
        // 3-dp lat/lon; clamp the precision *output* (not the input
        // values) so callers don't have to round before calling.
        format!(
            "{source}-{lat:.3}_{lon:.3}_{}.json",
            hour.format("%Y-%m-%dT%H"),
        )
    }

    fn path_for(&self, source: &str, lat: f64, lon: f64, time: DateTime<Utc>) -> PathBuf {
        self.root.join(Self::key(source, lat, lon, time))
    }

    /// Look up a cache entry. Returns `Fresh` if the file exists
    /// and its mtime is within `ttl`; `Stale` if it exists but is
    /// older; `Miss` if it doesn't exist.
    pub fn read_any(
        &self,
        source: &str,
        lat: f64,
        lon: f64,
        time: DateTime<Utc>,
        ttl: Duration,
    ) -> io::Result<CacheHit> {
        let path = self.path_for(source, lat, lon, time);
        let metadata = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(CacheHit::Miss),
            Err(e) => return Err(e),
        };
        let age = metadata
            .modified()
            .ok()
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .unwrap_or(Duration::MAX);
        let body = fs::read_to_string(&path)?;
        if age <= ttl {
            Ok(CacheHit::Fresh(body))
        } else {
            Ok(CacheHit::Stale(body))
        }
    }

    /// Look up a cache entry, returning `Some(body)` only if it
    /// is fresh. Stale and missing both map to `None`.
    pub fn read_fresh(
        &self,
        source: &str,
        lat: f64,
        lon: f64,
        time: DateTime<Utc>,
        ttl: Duration,
    ) -> io::Result<Option<String>> {
        match self.read_any(source, lat, lon, time, ttl)? {
            CacheHit::Fresh(body) => Ok(Some(body)),
            CacheHit::Stale(_) | CacheHit::Miss => Ok(None),
        }
    }

    /// Write a cache entry. Creates the directory tree on demand.
    pub fn write(
        &self,
        source: &str,
        lat: f64,
        lon: f64,
        time: DateTime<Utc>,
        body: &str,
    ) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let path = self.path_for(source, lat, lon, time);
        fs::write(&path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 12, 13, 30, 45).unwrap()
    }

    #[test]
    fn key_collapses_within_the_same_hour() {
        let t1 = Utc.with_ymd_and_hms(2026, 5, 12, 13, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 5, 12, 13, 59, 59).unwrap();
        let k1 = Cache::key("openmeteo", 56.1922, -3.9645, t1);
        let k2 = Cache::key("openmeteo", 56.1922, -3.9645, t2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_splits_across_hours() {
        let t1 = Utc.with_ymd_and_hms(2026, 5, 12, 13, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2026, 5, 12, 14, 0, 0).unwrap();
        let k1 = Cache::key("openmeteo", 56.0, -3.0, t1);
        let k2 = Cache::key("openmeteo", 56.0, -3.0, t2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_collapses_within_lat_lon_precision() {
        // 110 m precision: two coords within ~0.0005° of each other
        // round to the same string.
        let t = ts();
        let k1 = Cache::key("openmeteo", 56.19220, -3.96450, t);
        let k2 = Cache::key("openmeteo", 56.19233, -3.96456, t);
        assert_eq!(k1, k2);
    }

    #[test]
    fn round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path());
        let body = r#"{"hello": "world"}"#;
        cache.write("openmeteo", 56.0, -3.0, ts(), body).unwrap();
        let hit = cache
            .read_any("openmeteo", 56.0, -3.0, ts(), Duration::from_secs(60))
            .unwrap();
        match hit {
            CacheHit::Fresh(got) => assert_eq!(got, body),
            other => panic!("expected Fresh, got {other:?}"),
        }
    }

    #[test]
    fn miss_when_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path());
        let hit = cache
            .read_any("openmeteo", 56.0, -3.0, ts(), Duration::from_secs(60))
            .unwrap();
        assert!(matches!(hit, CacheHit::Miss));
    }

    #[test]
    fn read_fresh_returns_none_for_stale() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path());
        cache.write("openmeteo", 56.0, -3.0, ts(), "old").unwrap();
        // TTL 0 ⇒ everything is stale immediately.
        let got = cache
            .read_fresh("openmeteo", 56.0, -3.0, ts(), Duration::from_secs(0))
            .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn read_any_returns_stale_for_old_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::new(dir.path());
        cache.write("openmeteo", 56.0, -3.0, ts(), "old").unwrap();
        let hit = cache
            .read_any("openmeteo", 56.0, -3.0, ts(), Duration::from_secs(0))
            .unwrap();
        match hit {
            CacheHit::Stale(got) => assert_eq!(got, "old"),
            other => panic!("expected Stale, got {other:?}"),
        }
    }
}
