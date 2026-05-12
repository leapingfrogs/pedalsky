//! METAR client. Public API: `fetch_nearest(lat, lon)` returns the
//! nearest aviation-weather station report from the Aviation
//! Weather Center API.
//!
//! Why we use this for enrichment, not as the primary source:
//! METARs are accurate observations but very sparse (one station
//! every ~50–200 km, almost all at airports). For everywhere
//! except the immediate vicinity of an airport the forecast grid
//! (Open-Meteo) is what we have. When a station happens to be
//! close to the requested lat/lon, METAR gives:
//!
//! - Authoritative visibility (NWP often overestimates this).
//! - Explicit cloud groups (FEW/SCT/BKN/OVC at specific feet AGL),
//!   which can split / correct Open-Meteo's pressure-level
//!   buckets.
//! - Present-weather signals like TSRA (thunderstorm with rain)
//!   that reliably tag cumulonimbus + lightning.
//!
//! API: `aviationweather.gov/api/data/metar` — free, JSON,
//! ~30-minute issue cadence.

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::cache::{Cache, CacheHit};

/// Cache TTL — METARs typically issue every 30 minutes (or sooner
/// on rapid weather changes), so cache responses for 15 minutes to
/// stay close to the source cadence.
pub const DEFAULT_TTL: Duration = Duration::from_secs(900);

const USER_AGENT: &str = "PedalSky-WeatherFeed/0.1 (https://github.com/anthropic/pedalsky)";

/// Bounding-box half-width in degrees around the requested
/// (lat, lon) when scanning for nearby stations. 1° ≈ 111 km at
/// the equator, less at higher latitudes. Wide enough that
/// Dunblane (56.19°N, -3.96°E) reaches stations in central /
/// southern Scotland — Glasgow (EGPF), Edinburgh (EGPH),
/// Prestwick (EGPK), Leuchars (EGQL).
pub const SEARCH_RADIUS_DEG: f64 = 1.0;

/// One cloud-group entry from the METAR `clouds` array.
#[derive(Debug, Clone, Deserialize)]
pub struct MetarCloudLayer {
    /// One of "FEW", "SCT", "BKN", "OVC", "SKC", "CLR", "NCD".
    /// "CB" or "TCU" may follow as a separate suffix in raw METAR
    /// — that's captured in the `rawOb` string, not here.
    pub cover: String,
    /// Base altitude AGL in feet. Missing for SKC/CLR (no clouds).
    #[serde(default)]
    pub base: Option<i32>,
}

/// One METAR record.
#[derive(Debug, Clone, Deserialize)]
pub struct MetarRecord {
    /// ICAO station identifier (e.g. "EGPN").
    #[serde(rename = "icaoId")]
    pub icao_id: String,
    /// Station latitude (degrees).
    pub lat: f64,
    /// Station longitude (degrees).
    pub lon: f64,
    /// Station elevation AMSL (m).
    pub elev: f32,
    /// Observation time, RFC3339 UTC.
    #[serde(rename = "reportTime")]
    pub report_time: String,
    /// Air temperature (°C). Optional because malformed/special
    /// METARs occasionally omit this.
    #[serde(default)]
    pub temp: Option<f32>,
    /// Dewpoint (°C).
    #[serde(default)]
    pub dewp: Option<f32>,
    /// Wind direction (°, meteorological — FROM).
    #[serde(default)]
    pub wdir: Option<f32>,
    /// Wind speed (knots).
    #[serde(default)]
    pub wspd: Option<f32>,
    /// Visibility string. Free-form: "6+", "5SM", "9999", "10"…
    #[serde(default)]
    pub visib: Option<String>,
    /// QNH altimeter (hPa).
    #[serde(default)]
    pub altim: Option<f32>,
    /// The original raw METAR text. Useful for present-weather
    /// detection (e.g. "+TSRA", "VC FG").
    #[serde(rename = "rawOb", default)]
    pub raw_ob: Option<String>,
    /// Parsed cloud groups.
    #[serde(default)]
    pub clouds: Vec<MetarCloudLayer>,
}

impl MetarRecord {
    /// Distance in degrees (Euclidean over lat/lon) from the
    /// requested point. Cheap rather than haversine — at the
    /// scales we care about the ranking is the same.
    pub fn distance_deg_from(&self, lat: f64, lon: f64) -> f64 {
        let dlat = self.lat - lat;
        let dlon = self.lon - lon;
        (dlat * dlat + dlon * dlon).sqrt()
    }

    /// Parse the `reportTime` field as a UTC `DateTime`.
    pub fn parsed_time(&self) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(&self.report_time)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }

    /// Visibility in metres, parsed from the `visib` string. The
    /// API mixes "9999" (metres, ICAO convention), "10" (statute
    /// miles, US convention), "6+" (greater than 6 SM), and other
    /// forms. We default to 30 km when parsing fails.
    pub fn visibility_m(&self) -> f32 {
        let Some(v) = self.visib.as_ref() else {
            return 30_000.0;
        };
        // Drop the leading '+' from "6+" etc.
        let trimmed = v.trim_end_matches('+');
        // "9999" → 9999 m direct.
        if let Ok(metres) = trimmed.parse::<f32>() {
            if metres > 100.0 {
                return metres;
            }
            // Small value → assume statute miles.
            return metres * 1609.344;
        }
        // "5SM" → 5 statute miles. Strip the suffix.
        let no_unit = trimmed
            .trim_end_matches("SM")
            .trim_end_matches("KM")
            .trim();
        if let Ok(miles) = no_unit.parse::<f32>() {
            return miles * 1609.344;
        }
        30_000.0
    }

    /// Does the raw METAR text indicate a thunderstorm? `TS` is
    /// the ICAO present-weather code.
    pub fn is_thunderstorm(&self) -> bool {
        self.raw_ob
            .as_deref()
            .map(|s| s.contains(" TS") || s.contains("TSRA") || s.contains("TSGR"))
            .unwrap_or(false)
    }

    /// Does the raw METAR text indicate snow?
    pub fn is_snow(&self) -> bool {
        self.raw_ob
            .as_deref()
            .map(|s| {
                s.contains(" SN") || s.contains(" -SN") || s.contains(" +SN") || s.contains("SHSN")
            })
            .unwrap_or(false)
    }

    /// Does the raw METAR text indicate rain (any intensity)?
    /// Matches the ICAO present-weather codes RA (rain), DZ
    /// (drizzle), SHRA (rain showers), TSRA (thunderstorm with
    /// rain), and the intensity-prefixed forms.
    pub fn is_rain(&self) -> bool {
        let Some(s) = self.raw_ob.as_deref() else {
            return false;
        };
        // Look for any of the rain codes preceded by a space,
        // intensity prefix (+/-), or thunderstorm/shower prefix.
        s.contains("RA")
            && (s.contains(" RA")
                || s.contains("-RA")
                || s.contains("+RA")
                || s.contains("SHRA")
                || s.contains("TSRA"))
    }
}

/// Fetch the nearest METAR station to `(lat, lon)`. Returns `None`
/// when no station reports within the search radius (rare in
/// inhabited regions, common over open ocean).
pub fn fetch_nearest(
    cache_root: &PathBuf,
    lat: f64,
    lon: f64,
    target: DateTime<Utc>,
    ttl: Duration,
) -> anyhow::Result<Option<MetarRecord>> {
    let cache = Cache::new(cache_root);

    // Build the bounding-box query. The METAR API expects
    // `bbox=lat_min,lon_min,lat_max,lon_max`. We use a wide box
    // and then filter to the nearest station client-side.
    let lat_min = lat - SEARCH_RADIUS_DEG;
    let lat_max = lat + SEARCH_RADIUS_DEG;
    let lon_min = lon - SEARCH_RADIUS_DEG;
    let lon_max = lon + SEARCH_RADIUS_DEG;
    let url = format!(
        "https://aviationweather.gov/api/data/metar\
         ?bbox={lat_min:.4},{lon_min:.4},{lat_max:.4},{lon_max:.4}\
         &format=json&taf=false"
    );

    let body = match cache.read_fresh("metar", lat, lon, target, ttl)? {
        Some(b) => {
            tracing::debug!(target: "ps_weather_feed::metar", "cache hit (fresh)");
            b
        }
        None => {
            tracing::info!(target: "ps_weather_feed::metar", %url, "fetching");
            let http_result = ureq::get(&url)
                .set("User-Agent", USER_AGENT)
                .timeout(Duration::from_secs(15))
                .call();
            match http_result {
                Ok(resp) => {
                    let body = resp.into_string()?;
                    cache.write("metar", lat, lon, target, &body)?;
                    body
                }
                Err(http_err) => match cache.read_any("metar", lat, lon, target, ttl)? {
                    CacheHit::Fresh(b) | CacheHit::Stale(b) => {
                        tracing::warn!(
                            target: "ps_weather_feed::metar",
                            error = %http_err,
                            "live fetch failed; serving stale cache"
                        );
                        b
                    }
                    CacheHit::Miss => {
                        return Err(anyhow::anyhow!(
                            "METAR live fetch failed and no cache fallback: {http_err}"
                        ));
                    }
                },
            }
        }
    };

    let records: Vec<MetarRecord> = serde_json::from_str(&body)?;
    // Pick the nearest station that has a usable report. Stations
    // without temp/dewp/clouds are practically useless for
    // enrichment — skip them.
    Ok(records
        .into_iter()
        .filter(|r| r.temp.is_some() && r.dewp.is_some())
        .min_by(|a, b| {
            a.distance_deg_from(lat, lon)
                .partial_cmp(&b.distance_deg_from(lat, lon))
                .unwrap_or(std::cmp::Ordering::Equal)
        }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real METAR-array sample (one station). Verifies our serde
    /// deserialisation against the live shape including the
    /// `icaoId` → `icao_id` rename and the embedded clouds array.
    const SAMPLE: &str = r#"[
        {
            "icaoId": "EGPK",
            "lat": 55.509,
            "lon": -4.587,
            "elev": 14.0,
            "reportTime": "2026-05-12T22:20:00.000Z",
            "temp": 8.0,
            "dewp": 5.0,
            "wdir": 260.0,
            "wspd": 12.0,
            "visib": "6+",
            "altim": 1003.0,
            "rawOb": "METAR EGPK 122220Z 26012KT 9999 SCT010CB BKN028 08/05 Q1003",
            "clouds": [
                {"cover": "SCT", "base": 1000},
                {"cover": "BKN", "base": 2800}
            ]
        },
        {
            "icaoId": "EGPF",
            "lat": 55.871,
            "lon": -4.434,
            "elev": 8.0,
            "reportTime": "2026-05-12T22:20:00.000Z",
            "temp": 9.0,
            "dewp": 6.0,
            "wdir": 240.0,
            "wspd": 10.0,
            "visib": "9999",
            "altim": 1003.0,
            "rawOb": "METAR EGPF 122220Z 24010KT 9999 BKN030 09/06 Q1003",
            "clouds": [
                {"cover": "BKN", "base": 3000}
            ]
        }
    ]"#;

    #[test]
    fn deserialise_array() {
        let records: Vec<MetarRecord> = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].icao_id, "EGPK");
        assert_eq!(records[0].clouds.len(), 2);
        assert_eq!(records[0].clouds[0].base, Some(1000));
    }

    #[test]
    fn distance_ranks_correctly() {
        // Dunblane is ~56.19°N, -3.96°E. EGPF (Glasgow, ~55.87, -4.43)
        // is closer than EGPK (Prestwick, ~55.51, -4.59).
        let records: Vec<MetarRecord> = serde_json::from_str(SAMPLE).unwrap();
        let dunblane = (56.1922, -3.9645);
        let d0 = records[0].distance_deg_from(dunblane.0, dunblane.1);
        let d1 = records[1].distance_deg_from(dunblane.0, dunblane.1);
        assert!(d1 < d0, "EGPF should be closer to Dunblane than EGPK");
    }

    #[test]
    fn visibility_parsing() {
        let mut r: MetarRecord = serde_json::from_str(
            r#"{"icaoId":"X","lat":0.0,"lon":0.0,"elev":0.0,
                "reportTime":"2026-01-01T00:00:00.000Z","clouds":[]}"#,
        )
        .unwrap();
        r.visib = Some("9999".into());
        assert_eq!(r.visibility_m(), 9999.0);
        r.visib = Some("6+".into());
        assert!((r.visibility_m() - 6.0 * 1609.344).abs() < 0.5);
        r.visib = Some("5SM".into());
        assert!((r.visibility_m() - 5.0 * 1609.344).abs() < 0.5);
        r.visib = None;
        assert_eq!(r.visibility_m(), 30_000.0);
    }

    #[test]
    fn present_weather_detection() {
        let mut r: MetarRecord = serde_json::from_str(
            r#"{"icaoId":"X","lat":0.0,"lon":0.0,"elev":0.0,
                "reportTime":"2026-01-01T00:00:00.000Z","clouds":[]}"#,
        )
        .unwrap();
        r.raw_ob = Some("METAR X 010000Z 24010KT 9999 +TSRA OVC020CB 18/15 Q1003".into());
        assert!(r.is_thunderstorm());
        assert!(r.is_rain());
        assert!(!r.is_snow());
        r.raw_ob = Some("METAR X 010000Z 24010KT 2000 -SN OVC005 -02/-04 Q1010".into());
        assert!(!r.is_thunderstorm());
        assert!(!r.is_rain());
        assert!(r.is_snow());
    }
}
