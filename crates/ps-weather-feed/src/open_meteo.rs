//! Open-Meteo client. Public API: `fetch(...)` returns a parsed
//! `OpenMeteoResponse` from cache or live HTTP.
//!
//! Open-Meteo's free hourly forecast endpoint:
//! <https://api.open-meteo.com/v1/forecast> with the parameters
//! enumerated in [`build_url`]. We request the surface vars
//! PedalSky needs for `Scene.surface` (temperature_2m,
//! dew_point_2m, surface_pressure, visibility, wind_*,
//! precipitation, snowfall) plus cloud cover at six pressure
//! levels (1000, 925, 850, 700, 500, 300 hPa) which give us the
//! per-layer cover for the six altitude bands PedalSky's cloud
//! pipeline runs through.
//!
//! Response schema (hourly arrays):
//!
//! - `time` — array of ISO-8601 strings, one per forecast hour.
//! - `temperature_2m`, `dew_point_2m` (°C)
//! - `surface_pressure` (hPa)
//! - `visibility` (m)
//! - `wind_speed_10m` (km/h)  ← convert to m/s
//! - `wind_direction_10m` (°, meteorological — direction FROM)
//! - `precipitation`, `rain` (mm/h), `snowfall` (cm/h)
//! - `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`,
//!   `cloud_cover_high` (%)
//! - `cape`, `convective_inhibition` (J/kg)
//! - `cloud_cover_<level>hPa` (%) for level ∈ {1000, 925, 850,
//!   700, 500, 300}
//!
//! Cache TTL: 1 hour (matches the API's hourly cadence).

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Timelike, Utc};
use serde::Deserialize;

use crate::cache::{Cache, CacheHit};

/// Default cache TTL for the hourly forecast — one hour matches
/// the API's row cadence.
pub const DEFAULT_TTL: Duration = Duration::from_secs(3600);

/// HTTP user-agent string. APIs frown on un-identified traffic.
const USER_AGENT: &str = "PedalSky-WeatherFeed/0.1 (https://github.com/anthropic/pedalsky)";

/// Pressure levels (hPa) we fetch cloud cover at. Each maps to a
/// PedalSky cloud layer in the mapping pass.
pub const PRESSURE_LEVELS_HPA: &[u32] = &[1000, 925, 850, 700, 500, 300];

/// Pressure levels (hPa) we fetch upper-air **wind** at. Drives the
/// `Surface.winds_aloft` profile that the wind-field synthesis
/// interpolates instead of the synthetic 1/7 power law. We skip
/// 1000 (close to the 10 m surface anchor and noisy) and 925 (often
/// inside the boundary layer where the power-law extrapolation is
/// already adequate); 200 hPa sits in the stratosphere and doesn't
/// influence troposphere clouds. The remaining four levels cover
/// low / mid / high cloud altitudes (~1.5 / 3.0 / 5.5 / 9.0 km).
pub const WIND_LEVELS_HPA: &[u32] = &[850, 700, 500, 300];

/// One hour's worth of forecast values. Vec lengths all match the
/// `time` array.
#[derive(Debug, Clone, Deserialize)]
pub struct Hourly {
    /// ISO-8601 timestamp strings, one per forecast hour.
    pub time: Vec<String>,
    /// Air temperature at 2 m AGL (°C).
    pub temperature_2m: Vec<f32>,
    /// Dewpoint at 2 m AGL (°C).
    pub dew_point_2m: Vec<f32>,
    /// Surface pressure (hPa).
    pub surface_pressure: Vec<f32>,
    /// Visibility (m).
    pub visibility: Vec<f32>,
    /// Wind speed at 10 m AGL (km/h).
    pub wind_speed_10m: Vec<f32>,
    /// Wind direction at 10 m AGL (°, meteorological).
    pub wind_direction_10m: Vec<f32>,
    /// Total precipitation rate (mm/h).
    pub precipitation: Vec<f32>,
    /// Rain rate (mm/h, distinct from `snowfall`).
    pub rain: Vec<f32>,
    /// Snow rate (cm/h).
    pub snowfall: Vec<f32>,
    /// Total cloud cover (%).
    pub cloud_cover: Vec<f32>,
    /// Convective available potential energy (J/kg). Drives
    /// thunderstorm gating in the scene mapping.
    pub cape: Vec<f32>,

    // Per-pressure-level cloud cover (%). Each Vec matches the
    // `time` array's length.
    /// 1000 hPa cloud cover (%). Surface / fog band.
    #[serde(rename = "cloud_cover_1000hPa")]
    pub cloud_cover_1000hpa: Vec<f32>,
    /// 925 hPa cloud cover (%). Low stratus / shallow cumulus.
    #[serde(rename = "cloud_cover_925hPa")]
    pub cloud_cover_925hpa: Vec<f32>,
    /// 850 hPa cloud cover (%). Stratocumulus / cumulus base.
    #[serde(rename = "cloud_cover_850hPa")]
    pub cloud_cover_850hpa: Vec<f32>,
    /// 700 hPa cloud cover (%). Altocumulus.
    #[serde(rename = "cloud_cover_700hPa")]
    pub cloud_cover_700hpa: Vec<f32>,
    /// 500 hPa cloud cover (%). Altostratus.
    #[serde(rename = "cloud_cover_500hPa")]
    pub cloud_cover_500hpa: Vec<f32>,
    /// 300 hPa cloud cover (%). Cirrus / cirrostratus.
    #[serde(rename = "cloud_cover_300hPa")]
    pub cloud_cover_300hpa: Vec<f32>,

    // Phase 14 — winds aloft. Each Vec matches the `time` array; the
    // serde `default` makes the field tolerant of older cached
    // responses (or APIs that drop a level), so cached payloads from
    // before this commit deserialise to empty Vecs and degrade
    // gracefully to the surface-only path.
    /// 850 hPa wind speed (km/h).
    #[serde(rename = "wind_speed_850hPa", default)]
    pub wind_speed_850hpa: Vec<f32>,
    /// 700 hPa wind speed (km/h).
    #[serde(rename = "wind_speed_700hPa", default)]
    pub wind_speed_700hpa: Vec<f32>,
    /// 500 hPa wind speed (km/h).
    #[serde(rename = "wind_speed_500hPa", default)]
    pub wind_speed_500hpa: Vec<f32>,
    /// 300 hPa wind speed (km/h).
    #[serde(rename = "wind_speed_300hPa", default)]
    pub wind_speed_300hpa: Vec<f32>,
    /// 850 hPa wind direction (°, meteorological).
    #[serde(rename = "wind_direction_850hPa", default)]
    pub wind_direction_850hpa: Vec<f32>,
    /// 700 hPa wind direction (°, meteorological).
    #[serde(rename = "wind_direction_700hPa", default)]
    pub wind_direction_700hpa: Vec<f32>,
    /// 500 hPa wind direction (°, meteorological).
    #[serde(rename = "wind_direction_500hPa", default)]
    pub wind_direction_500hpa: Vec<f32>,
    /// 300 hPa wind direction (°, meteorological).
    #[serde(rename = "wind_direction_300hPa", default)]
    pub wind_direction_300hpa: Vec<f32>,
}

impl Hourly {
    /// Find the index of the forecast row closest to `target`.
    /// Open-Meteo returns hour-aligned UTC timestamps; the closest
    /// row is the one whose `time` value differs by the smallest
    /// amount in absolute terms.
    pub fn nearest_index(&self, target: DateTime<Utc>) -> Option<usize> {
        let mut best: Option<(i64, usize)> = None;
        for (i, raw) in self.time.iter().enumerate() {
            let parsed = parse_open_meteo_ts(raw)?;
            let diff_secs = (target - parsed).num_seconds().abs();
            if best.map_or(true, |(d, _)| diff_secs < d) {
                best = Some((diff_secs, i));
            }
        }
        best.map(|(_, i)| i)
    }

    /// Per-pressure-level cloud cover at row `i` as `(level_hpa,
    /// cover_pct)` tuples ordered from lowest pressure (highest
    /// altitude) downward to surface. Returns an empty iterator
    /// when the index is out of range.
    pub fn cloud_cover_by_level(&self, i: usize) -> Vec<(u32, f32)> {
        if i >= self.time.len() {
            return Vec::new();
        }
        vec![
            (300, self.cloud_cover_300hpa[i]),
            (500, self.cloud_cover_500hpa[i]),
            (700, self.cloud_cover_700hpa[i]),
            (850, self.cloud_cover_850hpa[i]),
            (925, self.cloud_cover_925hpa[i]),
            (1000, self.cloud_cover_1000hpa[i]),
        ]
    }

    /// Phase 14 — per-pressure-level winds at row `i` as
    /// `(level_hpa, speed_kmh, dir_deg)` tuples. Levels are taken
    /// from [`WIND_LEVELS_HPA`] and the result is sorted by pressure
    /// **descending** (i.e. lowest altitude first), matching the
    /// `Surface.winds_aloft` storage order. Missing or NaN samples
    /// (e.g. when a cached response predates the wind-aloft
    /// addition, or when a model hour drops a level) are filtered
    /// out so the consumer sees a clean profile.
    pub fn winds_aloft_by_level(&self, i: usize) -> Vec<(u32, f32, f32)> {
        if i >= self.time.len() {
            return Vec::new();
        }
        let mut out: Vec<(u32, f32, f32)> = Vec::with_capacity(WIND_LEVELS_HPA.len());
        for &level in WIND_LEVELS_HPA {
            let (speed_vec, dir_vec) = match level {
                850 => (&self.wind_speed_850hpa, &self.wind_direction_850hpa),
                700 => (&self.wind_speed_700hpa, &self.wind_direction_700hpa),
                500 => (&self.wind_speed_500hpa, &self.wind_direction_500hpa),
                300 => (&self.wind_speed_300hpa, &self.wind_direction_300hpa),
                _ => continue,
            };
            // Either vec may be empty (older cached response) or
            // shorter than `time` (rare API edge case) — guard both.
            let Some(speed) = speed_vec.get(i).copied() else { continue };
            let Some(dir) = dir_vec.get(i).copied() else { continue };
            if !speed.is_finite() || !dir.is_finite() {
                continue;
            }
            out.push((level, speed, dir));
        }
        // Lowest pressure (highest altitude) sits last so the caller
        // can append in altitude order if it walks the vec
        // forward. We use descending pressure here (≡ ascending
        // altitude).
        out.sort_by(|a, b| b.0.cmp(&a.0));
        out
    }
}

/// Top-level Open-Meteo forecast response. Only the fields we use
/// are parsed; serde drops the rest.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenMeteoResponse {
    /// Latitude the API resolved the query to (may differ slightly
    /// from the request after grid snap).
    pub latitude: f64,
    /// Longitude the API resolved the query to.
    pub longitude: f64,
    /// Surface elevation at the resolved grid point (m).
    pub elevation: f32,
    /// Hourly forecast block.
    pub hourly: Hourly,
}

/// Build the full Open-Meteo URL for the variables PedalSky needs.
pub fn build_url(lat: f64, lon: f64) -> String {
    let surface_vars = [
        "temperature_2m",
        "dew_point_2m",
        "surface_pressure",
        "visibility",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "cape",
    ];
    let level_vars: Vec<String> = PRESSURE_LEVELS_HPA
        .iter()
        .map(|p| format!("cloud_cover_{p}hPa"))
        .collect();
    let wind_level_vars: Vec<String> = WIND_LEVELS_HPA
        .iter()
        .flat_map(|p| {
            [
                format!("wind_speed_{p}hPa"),
                format!("wind_direction_{p}hPa"),
            ]
        })
        .collect();
    let mut hourly: Vec<String> = surface_vars.iter().map(|s| (*s).to_string()).collect();
    hourly.extend(level_vars);
    hourly.extend(wind_level_vars);
    format!(
        "https://api.open-meteo.com/v1/forecast\
         ?latitude={lat:.4}&longitude={lon:.4}\
         &hourly={}&forecast_days=2&timezone=UTC",
        hourly.join(","),
    )
}

/// Parse Open-Meteo's timestamp format. They return naive ISO-8601
/// (no timezone suffix) but document the values as UTC.
fn parse_open_meteo_ts(raw: &str) -> Option<DateTime<Utc>> {
    // The API returns strings like "2026-05-12T13:00" — append
    // ":00Z" if there's no second field, then parse as RFC3339.
    let needs_suffix = raw.len() == 16; // "YYYY-MM-DDTHH:MM"
    let canonical = if needs_suffix {
        format!("{raw}:00Z")
    } else if !raw.ends_with('Z') {
        format!("{raw}Z")
    } else {
        raw.to_string()
    };
    DateTime::parse_from_rfc3339(&canonical)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Fetch a forecast for `(lat, lon)` covering `target`. Returns
/// the parsed response. Tries the cache first (within `ttl`); on
/// miss issues an HTTP request; on HTTP failure falls back to the
/// most recent stale cache entry (if any).
pub fn fetch(
    cache_root: &PathBuf,
    lat: f64,
    lon: f64,
    target: DateTime<Utc>,
    ttl: Duration,
) -> anyhow::Result<OpenMeteoResponse> {
    let cache = Cache::new(cache_root);
    let hour = target
        .with_minute(0).expect("0 minutes valid")
        .with_second(0).expect("0 seconds valid")
        .with_nanosecond(0).expect("0 nanos valid");

    // Try fresh cache.
    if let Some(body) = cache.read_fresh("openmeteo", lat, lon, hour, ttl)? {
        tracing::debug!(target: "ps_weather_feed::open_meteo", "cache hit (fresh)");
        return Ok(serde_json::from_str(&body)?);
    }

    // Live fetch.
    let url = build_url(lat, lon);
    tracing::info!(target: "ps_weather_feed::open_meteo", %url, "fetching");
    let http_result = ureq::get(&url)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(15))
        .call();

    match http_result {
        Ok(resp) => {
            let body = resp.into_string()?;
            cache.write("openmeteo", lat, lon, hour, &body)?;
            Ok(serde_json::from_str(&body)?)
        }
        Err(http_err) => {
            // Fall back to any cache hit — fresh or stale — so the
            // feature works in the air-gapped / rate-limited case.
            match cache.read_any("openmeteo", lat, lon, hour, ttl)? {
                CacheHit::Fresh(body) | CacheHit::Stale(body) => {
                    tracing::warn!(
                        target: "ps_weather_feed::open_meteo",
                        error = %http_err,
                        "live fetch failed; serving stale cache"
                    );
                    Ok(serde_json::from_str(&body)?)
                }
                CacheHit::Miss => Err(anyhow::anyhow!(
                    "open-meteo live fetch failed and no cache fallback: {http_err}"
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn url_includes_all_pressure_levels() {
        let url = build_url(56.1922, -3.9645);
        for p in PRESSURE_LEVELS_HPA {
            assert!(url.contains(&format!("cloud_cover_{p}hPa")), "missing {p}");
        }
        for p in WIND_LEVELS_HPA {
            assert!(
                url.contains(&format!("wind_speed_{p}hPa")),
                "missing wind_speed_{p}hPa",
            );
            assert!(
                url.contains(&format!("wind_direction_{p}hPa")),
                "missing wind_direction_{p}hPa",
            );
        }
        assert!(url.contains("latitude=56.1922"));
        assert!(url.contains("longitude=-3.9645"));
        assert!(url.contains("timezone=UTC"));
    }

    #[test]
    fn parse_timestamp_short_form() {
        let dt = parse_open_meteo_ts("2026-05-12T13:00").unwrap();
        assert_eq!(dt.format("%Y-%m-%dT%H:%M:%S").to_string(), "2026-05-12T13:00:00");
    }

    /// Recorded slice of a real Open-Meteo response. Exercises the
    /// serde deserialisation path including all the
    /// `cloud_cover_<level>hPa` rename attributes. If Open-Meteo
    /// renames a field, this test breaks first.
    const SAMPLE_RESPONSE: &str = r#"{
        "latitude": 56.19,
        "longitude": -3.96,
        "elevation": 83.0,
        "hourly": {
            "time": ["2026-05-12T12:00", "2026-05-12T13:00", "2026-05-12T14:00"],
            "temperature_2m": [13.4, 14.1, 14.8],
            "dew_point_2m": [9.0, 9.2, 9.4],
            "surface_pressure": [1015.0, 1015.2, 1015.5],
            "visibility": [30000.0, 30000.0, 28000.0],
            "wind_speed_10m": [12.0, 14.5, 16.0],
            "wind_direction_10m": [220.0, 225.0, 230.0],
            "precipitation": [0.0, 0.0, 0.2],
            "rain": [0.0, 0.0, 0.2],
            "snowfall": [0.0, 0.0, 0.0],
            "cloud_cover": [40.0, 55.0, 65.0],
            "cape": [100.0, 200.0, 150.0],
            "cloud_cover_1000hPa": [0.0, 0.0, 5.0],
            "cloud_cover_925hPa": [10.0, 15.0, 25.0],
            "cloud_cover_850hPa": [40.0, 55.0, 60.0],
            "cloud_cover_700hPa": [30.0, 35.0, 40.0],
            "cloud_cover_500hPa": [20.0, 25.0, 30.0],
            "cloud_cover_300hPa": [10.0, 15.0, 20.0]
        }
    }"#;

    #[test]
    fn deserialise_sample_response() {
        let resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_RESPONSE).unwrap();
        assert_eq!(resp.hourly.time.len(), 3);
        assert_eq!(resp.hourly.cloud_cover_850hpa[1], 55.0);
        assert_eq!(resp.hourly.cloud_cover_300hpa[2], 20.0);
        assert_eq!(resp.hourly.wind_direction_10m[0], 220.0);
    }

    #[test]
    fn nearest_index_picks_closest_hour() {
        let resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_RESPONSE).unwrap();
        // 13:25 is closer to 13:00 than to 14:00.
        let t = Utc.with_ymd_and_hms(2026, 5, 12, 13, 25, 0).unwrap();
        assert_eq!(resp.hourly.nearest_index(t), Some(1));
        // 13:31 tips over to 14:00.
        let t = Utc.with_ymd_and_hms(2026, 5, 12, 13, 31, 0).unwrap();
        assert_eq!(resp.hourly.nearest_index(t), Some(2));
    }

    #[test]
    fn cloud_cover_by_level_returns_high_to_low() {
        let resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_RESPONSE).unwrap();
        let cov = resp.hourly.cloud_cover_by_level(1);
        let levels: Vec<u32> = cov.iter().map(|(l, _)| *l).collect();
        assert_eq!(levels, vec![300, 500, 700, 850, 925, 1000]);
        assert_eq!(cov[0].1, 15.0); // 300 hPa @ row 1
        assert_eq!(cov[5].1, 0.0);  // 1000 hPa @ row 1
    }

    /// Older cached responses (predating the wind-aloft addition)
    /// must still deserialise: the rename-default attributes leave
    /// `wind_*_<lvl>hpa` as empty Vecs, and `winds_aloft_by_level`
    /// returns an empty slice rather than panicking.
    #[test]
    fn missing_winds_aloft_deserialises_clean() {
        let resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_RESPONSE).unwrap();
        assert!(resp.hourly.wind_speed_850hpa.is_empty());
        assert!(resp.hourly.winds_aloft_by_level(0).is_empty());
    }

    const SAMPLE_WITH_WINDS_ALOFT: &str = r#"{
        "latitude": 56.19,
        "longitude": -3.96,
        "elevation": 83.0,
        "hourly": {
            "time": ["2026-05-12T13:00"],
            "temperature_2m": [14.1],
            "dew_point_2m": [9.2],
            "surface_pressure": [1015.2],
            "visibility": [30000.0],
            "wind_speed_10m": [14.5],
            "wind_direction_10m": [225.0],
            "precipitation": [0.0],
            "rain": [0.0],
            "snowfall": [0.0],
            "cloud_cover": [55.0],
            "cape": [200.0],
            "cloud_cover_1000hPa": [0.0],
            "cloud_cover_925hPa": [15.0],
            "cloud_cover_850hPa": [55.0],
            "cloud_cover_700hPa": [35.0],
            "cloud_cover_500hPa": [25.0],
            "cloud_cover_300hPa": [15.0],
            "wind_speed_850hPa": [25.0],
            "wind_speed_700hPa": [45.0],
            "wind_speed_500hPa": [80.0],
            "wind_speed_300hPa": [130.0],
            "wind_direction_850hPa": [230.0],
            "wind_direction_700hPa": [245.0],
            "wind_direction_500hPa": [260.0],
            "wind_direction_300hPa": [275.0]
        }
    }"#;

    #[test]
    fn winds_aloft_by_level_sorted_low_altitude_first() {
        let resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_WITH_WINDS_ALOFT).unwrap();
        let winds = resp.hourly.winds_aloft_by_level(0);
        assert_eq!(winds.len(), 4);
        // Sort order is descending pressure ⇔ ascending altitude.
        assert_eq!(
            winds.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(),
            vec![850, 700, 500, 300],
        );
        // Speeds returned in source units (km/h) — the mapping pass
        // does the m/s conversion.
        assert_eq!(winds[0].1, 25.0);
        assert_eq!(winds[3].1, 130.0);
        // Directions kept as-supplied (meteorological).
        assert_eq!(winds[2].2, 260.0);
    }

    #[test]
    fn winds_aloft_filters_nan() {
        // Force a NaN on one of the levels by editing the parsed
        // struct — simulates an API row that drops a level.
        let mut resp: OpenMeteoResponse = serde_json::from_str(SAMPLE_WITH_WINDS_ALOFT).unwrap();
        resp.hourly.wind_speed_500hpa[0] = f32::NAN;
        let winds = resp.hourly.winds_aloft_by_level(0);
        // 500 hPa should be dropped; the remaining three levels keep
        // their order.
        assert_eq!(winds.len(), 3);
        assert_eq!(
            winds.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(),
            vec![850, 700, 300],
        );
    }
}
