//! NOAA SWPC planetary Kp index ingestion (Phase 15.A).
//!
//! Two endpoints, both free, no auth, no per-key quota:
//!
//! - `services.swpc.noaa.gov/products/noaa-planetary-k-index.json` —
//!   observed history, 3-hourly cadence, ~8-day rolling window.
//! - `services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json`
//!   — observed → ~3-day forecast, 3-hourly cadence, ~11 days total.
//!   Each entry carries an `observed` field indicating whether the
//!   value is measured ("observed"), nowcast ("estimated"), or
//!   forecast ("predicted").
//!
//! The forecast endpoint actually contains the observed history too,
//! so we only need to fetch one URL in the common case. We provide
//! both fetch functions for flexibility (a future "purely historical"
//! consumer might want the dedicated observed endpoint).
//!
//! Cache TTL: 1 hour. The forecast updates a few times per day; a 1h
//! TTL keeps the renderer responsive without hammering the upstream.
//! Cache key uses lat=0, lon=0 as a sentinel because the data is
//! global (Kp is a planetary index — same value worldwide).
//!
//! Response schema (verified 2026-05-13 against the live endpoint):
//!
//! ```json
//! [
//!   {
//!     "time_tag": "2026-05-13T12:00:00",
//!     "kp": 2.33,
//!     "observed": "observed",
//!     "noaa_scale": null
//!   },
//!   ...
//! ]
//! ```

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::cache::{Cache, CacheHit};

/// Default cache TTL for the Kp forecast. NOAA updates the file a
/// few times per day; 1 h keeps us fresh without burning quota.
pub const DEFAULT_TTL: Duration = Duration::from_secs(3600);

/// User-agent string for SWPC requests. NOAA's API doesn't strictly
/// require one but identifying ourselves is polite and helps if they
/// ever need to debug rate-limit hits.
const USER_AGENT: &str = "PedalSky-WeatherFeed/0.1 (https://github.com/anthropic/pedalsky)";

/// Combined observed + forecast endpoint. The historical observed
/// values are included alongside the predictions in this one URL.
const FORECAST_URL: &str = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json";

/// One row of the SWPC Kp series. Time is naive ISO 8601 in UTC;
/// the API doesn't append a Z but the documentation states UTC.
#[derive(Debug, Clone, Deserialize)]
pub struct KpRow {
    /// Naive ISO 8601 timestamp (UTC, no timezone suffix).
    pub time_tag: String,
    /// Kp index value in the canonical 0..9 range (fractional —
    /// e.g. 2.33 ≈ "2+", 2.67 ≈ "3-").
    pub kp: f32,
    /// One of `"observed"`, `"estimated"`, `"predicted"`. Useful for
    /// UI labelling (so the user knows whether they're looking at a
    /// measurement or a forecast).
    pub observed: String,
    /// NOAA G-scale storm level when applicable, e.g. `"G2"`. Null
    /// for low Kp values.
    pub noaa_scale: Option<String>,
}

/// Parsed Kp series — the full window from the forecast endpoint.
#[derive(Debug, Clone, Default)]
pub struct KpSeries {
    /// Rows ordered earliest → latest as the API returns them.
    pub rows: Vec<KpRow>,
}

impl KpSeries {
    /// Find the entry whose `time_tag` is closest to `target`.
    /// Returns `None` only if the series is empty.
    pub fn nearest(&self, target: DateTime<Utc>) -> Option<&KpRow> {
        let mut best: Option<(i64, &KpRow)> = None;
        for row in &self.rows {
            let Some(t) = parse_swpc_ts(&row.time_tag) else { continue };
            let diff_secs = (target - t).num_seconds().abs();
            if best.map_or(true, |(d, _)| diff_secs < d) {
                best = Some((diff_secs, row));
            }
        }
        best.map(|(_, row)| row)
    }
}

/// Parse SWPC's naive ISO 8601 timestamp (e.g. `"2026-05-13T12:00:00"`)
/// as UTC. Returns `None` on malformed input.
fn parse_swpc_ts(raw: &str) -> Option<DateTime<Utc>> {
    // SWPC omits the timezone suffix; append `Z` to make it RFC 3339.
    let canonical = if raw.ends_with('Z') {
        raw.to_string()
    } else {
        format!("{raw}Z")
    };
    DateTime::parse_from_rfc3339(&canonical)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Fetch the SWPC Kp series covering the observed history plus the
/// 3-day forecast. The cache key uses lat=0 lon=0 because the data
/// is global (Kp is a planetary index).
pub fn fetch(cache_root: &PathBuf, target: DateTime<Utc>, ttl: Duration) -> anyhow::Result<KpSeries> {
    let cache = Cache::new(cache_root);

    // Try fresh cache. The hour-truncated `target` is what the cache
    // key bucket is anchored on — within the same UTC hour we serve
    // identical bytes.
    if let Some(body) = cache.read_fresh("swpc-kp", 0.0, 0.0, target, ttl)? {
        tracing::debug!(target: "ps_weather_feed::kp_index", "cache hit (fresh)");
        return parse(&body);
    }

    tracing::info!(target: "ps_weather_feed::kp_index", url = FORECAST_URL, "fetching");
    let http_result = ureq::get(FORECAST_URL)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(15))
        .call();

    match http_result {
        Ok(resp) => {
            let body = resp.into_string()?;
            cache.write("swpc-kp", 0.0, 0.0, target, &body)?;
            parse(&body)
        }
        Err(http_err) => {
            // Stale cache fallback so offline / rate-limited users
            // still get *some* Kp value.
            match cache.read_any("swpc-kp", 0.0, 0.0, target, ttl)? {
                CacheHit::Fresh(body) | CacheHit::Stale(body) => {
                    tracing::warn!(
                        target: "ps_weather_feed::kp_index",
                        error = %http_err,
                        "live fetch failed; serving stale cache"
                    );
                    parse(&body)
                }
                CacheHit::Miss => Err(anyhow::anyhow!(
                    "SWPC Kp live fetch failed and no cache fallback: {http_err}"
                )),
            }
        }
    }
}

/// Parse the SWPC JSON body. Public so tests can exercise the
/// parser without an HTTP round-trip.
pub fn parse(body: &str) -> anyhow::Result<KpSeries> {
    let rows: Vec<KpRow> = serde_json::from_str(body)?;
    Ok(KpSeries { rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    const SAMPLE: &str = r#"[
        {"time_tag": "2026-05-13T00:00:00", "kp": 1.33, "observed": "observed", "noaa_scale": null},
        {"time_tag": "2026-05-13T03:00:00", "kp": 1.67, "observed": "observed", "noaa_scale": null},
        {"time_tag": "2026-05-13T06:00:00", "kp": 2.33, "observed": "observed", "noaa_scale": null},
        {"time_tag": "2026-05-13T09:00:00", "kp": 2.67, "observed": "estimated", "noaa_scale": null},
        {"time_tag": "2026-05-13T12:00:00", "kp": 5.67, "observed": "predicted", "noaa_scale": "G2"},
        {"time_tag": "2026-05-13T15:00:00", "kp": 4.33, "observed": "predicted", "noaa_scale": null}
    ]"#;

    #[test]
    fn parses_sample_response() {
        let series = parse(SAMPLE).unwrap();
        assert_eq!(series.rows.len(), 6);
        assert_eq!(series.rows[0].kp, 1.33);
        assert_eq!(series.rows[4].observed, "predicted");
        assert_eq!(series.rows[4].noaa_scale.as_deref(), Some("G2"));
        assert!(series.rows[0].noaa_scale.is_none());
    }

    #[test]
    fn nearest_picks_closest_time() {
        let series = parse(SAMPLE).unwrap();
        // 11:30 UTC is closer to 12:00 than 09:00.
        let t = Utc.with_ymd_and_hms(2026, 5, 13, 11, 30, 0).unwrap();
        assert_eq!(series.nearest(t).unwrap().kp, 5.67);
        // 10:00 is also closer to 09:00 than 12:00.
        let t = Utc.with_ymd_and_hms(2026, 5, 13, 10, 0, 0).unwrap();
        assert_eq!(series.nearest(t).unwrap().kp, 2.67);
    }

    #[test]
    fn nearest_handles_empty_series() {
        let series = KpSeries::default();
        let t = Utc.with_ymd_and_hms(2026, 5, 13, 12, 0, 0).unwrap();
        assert!(series.nearest(t).is_none());
    }

    #[test]
    fn parse_swpc_ts_round_trip() {
        let dt = parse_swpc_ts("2026-05-13T12:34:56").unwrap();
        assert_eq!(dt.format("%Y-%m-%dT%H:%M:%S").to_string(), "2026-05-13T12:34:56");
    }
}
