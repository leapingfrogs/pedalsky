//! NOAA SWPC OVATION aurora nowcast ingestion (Phase 15.B).
//!
//! Endpoint: `services.swpc.noaa.gov/json/ovation_aurora_latest.json`.
//! Updates every 5 minutes; published as a 30-minute nowcast of the
//! aurora oval intensity over a global 1°×1° grid. ~65 000 grid
//! cells (360 × 181), ~900 KB JSON.
//!
//! Schema (verified 2026-05-13 against the live endpoint):
//!
//! ```json
//! {
//!   "Observation Time": "2026-05-13T11:30:00Z",
//!   "Forecast Time":    "2026-05-13T12:00:00Z",
//!   "Data Format":      "[Longitude, Latitude, Aurora]",
//!   "coordinates": [[0, -90, 5], [0, -89, 0], ...],
//!   "type": "..."
//! }
//! ```
//!
//! Longitude is integer 0..359, latitude integer -90..90. The aurora
//! value is the OVATION-model intensity — empirically 0..~17 today;
//! NOAA documents it as a 0..100 scale but the live data tends to
//! peak in the low 20s during strong storms. We treat it as a
//! generic intensity and let the consumer map to whatever range it
//! needs.
//!
//! Cache TTL: 5 minutes. The endpoint updates that often and the
//! aurora oval can shift noticeably over a 30-minute window during
//! active storms.

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::cache::{Cache, CacheHit};

/// Cache TTL — matches the upstream's 5-minute update cadence.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);

const USER_AGENT: &str = "PedalSky-WeatherFeed/0.1 (https://github.com/anthropic/pedalsky)";
const OVATION_URL: &str = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json";

/// Grid dimensions — longitude is 360 cells (0..359), latitude is
/// 181 cells (-90..90 inclusive).
pub const LON_CELLS: usize = 360;
/// Latitude cell count.
pub const LAT_CELLS: usize = 181;

/// Raw deserialised SWPC response. We rename to handle the
/// space-containing JSON keys.
#[derive(Debug, Clone, Deserialize)]
struct OvationResponse {
    #[serde(rename = "Observation Time")]
    observation_time: String,
    #[serde(rename = "Forecast Time")]
    forecast_time: String,
    /// Each entry is `[longitude, latitude, aurora_value]`.
    coordinates: Vec<[i32; 3]>,
}

/// Parsed OVATION grid. The `intensities` field is a flat `[lon, lat]`
/// array with `index = lat_idx * LON_CELLS + lon_idx` where `lat_idx
/// = lat_deg + 90` (so lat -90 → idx 0, lat +90 → idx 180).
#[derive(Debug, Clone)]
pub struct OvationGrid {
    /// ISO 8601 timestamp the source magnetometer data was sampled.
    pub observation_time: String,
    /// ISO 8601 timestamp the nowcast is *valid for* (typically 30
    /// minutes after the observation).
    pub forecast_time: String,
    /// 360 × 181 row-major intensity grid, stored as `u8` to halve
    /// the memory footprint vs `f32`. Values in [0, 255] but the
    /// upstream tops out around the low 20s even during strong
    /// storms — calibration is documented in `intensity_at`.
    pub intensities: Vec<u8>,
}

impl OvationGrid {
    /// Sample the grid at `(lon_deg, lat_deg)`. Longitudes wrap
    /// modulo 360; latitudes clamp to [-90, 90]. Returns the raw
    /// integer intensity at the nearest grid cell.
    pub fn raw_intensity_at(&self, lon_deg: f64, lat_deg: f64) -> u8 {
        let lon_wrapped = lon_deg.rem_euclid(360.0);
        let lat_clamped = lat_deg.clamp(-90.0, 90.0);
        let lon_idx = (lon_wrapped.round() as usize) % LON_CELLS;
        let lat_idx = ((lat_clamped + 90.0).round() as usize).min(LAT_CELLS - 1);
        self.intensities[lat_idx * LON_CELLS + lon_idx]
    }

    /// Convenience: sample the grid and rescale to `[0, 1]` using a
    /// calibration appropriate for the aurora subsystem's
    /// `intensity_override` field.
    ///
    /// Raw OVATION values during strong storms peak in the low 20s.
    /// Rescaling by `/ 30.0` gives a slider that hits 1.0 at "very
    /// strong aurora" without saturating during typical activity.
    /// Values are then clamped to `[0, 1]`.
    pub fn intensity_at(&self, lon_deg: f64, lat_deg: f64) -> f32 {
        const SATURATE_AT: f32 = 30.0;
        let raw = self.raw_intensity_at(lon_deg, lat_deg) as f32;
        (raw / SATURATE_AT).clamp(0.0, 1.0)
    }
}

/// Parse the SWPC JSON body into an `OvationGrid`. The raw response
/// is a list of `[lon, lat, value]` triples — we rasterise it into
/// the row-major grid layout described on `OvationGrid`.
pub fn parse(body: &str) -> anyhow::Result<OvationGrid> {
    let resp: OvationResponse = serde_json::from_str(body)?;
    let mut intensities = vec![0u8; LON_CELLS * LAT_CELLS];
    let mut filled = 0_usize;
    for tuple in &resp.coordinates {
        let lon = tuple[0];
        let lat = tuple[1];
        let value = tuple[2];
        if !(0..360).contains(&lon) || !(-90..=90).contains(&lat) {
            continue;
        }
        let idx = ((lat + 90) as usize) * LON_CELLS + (lon as usize);
        intensities[idx] = value.clamp(0, 255) as u8;
        filled += 1;
    }
    if filled < LON_CELLS * LAT_CELLS / 2 {
        anyhow::bail!(
            "OVATION grid is suspiciously sparse: {filled} / {} cells filled",
            LON_CELLS * LAT_CELLS,
        );
    }
    Ok(OvationGrid {
        observation_time: resp.observation_time,
        forecast_time: resp.forecast_time,
        intensities,
    })
}

/// Fetch the OVATION nowcast. Cache key uses lat=0, lon=0 (the data
/// is global; per-location sampling happens on the parsed grid).
pub fn fetch(cache_root: &PathBuf, target: DateTime<Utc>, ttl: Duration) -> anyhow::Result<OvationGrid> {
    let cache = Cache::new(cache_root);

    if let Some(body) = cache.read_fresh("swpc-ovation", 0.0, 0.0, target, ttl)? {
        tracing::debug!(target: "ps_weather_feed::ovation", "cache hit (fresh)");
        return parse(&body);
    }

    tracing::info!(target: "ps_weather_feed::ovation", url = OVATION_URL, "fetching");
    let http_result = ureq::get(OVATION_URL)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(30))
        .call();

    match http_result {
        Ok(resp) => {
            let body = resp.into_string()?;
            cache.write("swpc-ovation", 0.0, 0.0, target, &body)?;
            parse(&body)
        }
        Err(http_err) => match cache.read_any("swpc-ovation", 0.0, 0.0, target, ttl)? {
            CacheHit::Fresh(body) | CacheHit::Stale(body) => {
                tracing::warn!(
                    target: "ps_weather_feed::ovation",
                    error = %http_err,
                    "live fetch failed; serving stale cache"
                );
                parse(&body)
            }
            CacheHit::Miss => Err(anyhow::anyhow!(
                "SWPC OVATION live fetch failed and no cache fallback: {http_err}"
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but complete OVATION payload covering the
    /// full 360 × 181 grid so the sparsity check passes. Strong
    /// auroral oval near lat ±70°, zero everywhere else.
    fn synthetic_payload() -> String {
        let mut coords: Vec<String> = Vec::with_capacity(LON_CELLS * LAT_CELLS);
        for lat in -90..=90 {
            for lon in 0..360 {
                // Gaussian-ish ring at |lat| ≈ 70°.
                let dist = ((lat as f32).abs() - 70.0).abs();
                let v = if dist < 3.0 { 25 } else { 0 };
                coords.push(format!("[{lon},{lat},{v}]"));
            }
        }
        format!(
            r#"{{
                "Observation Time": "2026-05-13T11:30:00Z",
                "Forecast Time": "2026-05-13T12:00:00Z",
                "Data Format": "[Longitude, Latitude, Aurora]",
                "coordinates": [{}]
            }}"#,
            coords.join(","),
        )
    }

    #[test]
    fn parses_grid() {
        let body = synthetic_payload();
        let grid = parse(&body).unwrap();
        assert_eq!(grid.intensities.len(), LON_CELLS * LAT_CELLS);
        assert_eq!(grid.observation_time, "2026-05-13T11:30:00Z");
    }

    #[test]
    fn samples_at_auroral_oval() {
        let grid = parse(&synthetic_payload()).unwrap();
        // 70°N over Norway — should hit the ring.
        assert_eq!(grid.raw_intensity_at(20.0, 70.0), 25);
        // 70°S over the southern ocean — same magnitude.
        assert_eq!(grid.raw_intensity_at(180.0, -70.0), 25);
        // The equator — well outside the oval.
        assert_eq!(grid.raw_intensity_at(0.0, 0.0), 0);
    }

    #[test]
    fn longitude_wraps_modulo_360() {
        let grid = parse(&synthetic_payload()).unwrap();
        // -10° east of London → equivalent to 350°. Both should
        // index the same cell on the oval ring.
        let a = grid.raw_intensity_at(-10.0, 70.0);
        let b = grid.raw_intensity_at(350.0, 70.0);
        assert_eq!(a, b);
    }

    #[test]
    fn latitude_clamps_at_poles() {
        let grid = parse(&synthetic_payload()).unwrap();
        // Beyond +90° clamps to +90 (which is in the oval at our
        // synthetic ring? no — 90 is 20° from 70, well outside).
        let polar = grid.raw_intensity_at(0.0, 95.0);
        assert_eq!(polar, 0);
    }

    #[test]
    fn intensity_at_rescales_to_unit_range() {
        let grid = parse(&synthetic_payload()).unwrap();
        // Raw 25 / 30 = 0.833.
        let val = grid.intensity_at(20.0, 70.0);
        assert!((val - 0.833).abs() < 0.01, "got {val}");
        // Off-oval is 0.0.
        assert_eq!(grid.intensity_at(0.0, 0.0), 0.0);
    }

    #[test]
    fn refuses_sparse_payload() {
        // Empty coordinates → must fail the sparsity check.
        let body = r#"{
            "Observation Time": "2026-05-13T11:30:00Z",
            "Forecast Time":    "2026-05-13T12:00:00Z",
            "Data Format":      "[Longitude, Latitude, Aurora]",
            "coordinates":      []
        }"#;
        assert!(parse(body).is_err());
    }
}
