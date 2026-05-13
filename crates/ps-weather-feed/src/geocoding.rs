//! Open-Meteo geocoding API ingestion (Phase 16.A).
//!
//! Endpoint: `geocoding-api.open-meteo.com/v1/search?name=<query>`.
//! Free, no key, same fair-use policy as the weather endpoints.
//! Returns up to `count` matching places with coordinates, elevation,
//! country / admin1 / admin2, timezone, and population. Forward
//! geocoding only — reverse geocoding ("what city is at this
//! lat/lon?") is not exposed by Open-Meteo as of mid-2026.
//!
//! Schema (verified live 2026-05-13):
//!
//! ```json
//! {
//!   "results": [
//!     {
//!       "id": 2650769,
//!       "name": "Dunblane",
//!       "latitude": 56.188, "longitude": -3.964, "elevation": 64.0,
//!       "country": "United Kingdom",
//!       "admin1": "Scotland", "admin2": "Stirling",
//!       "timezone": "Europe/London",
//!       "population": 9310,
//!       "feature_code": "PPL", "country_code": "GB"
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! Empty queries (or no matches) return `{}` with no `results` key
//! at all — the deserialiser uses `#[serde(default)]` to map that to
//! an empty Vec.
//!
//! Cache: 1-day TTL. Coordinates of "Dunblane" don't drift, so we
//! can afford to bucket aggressively. Key is the normalised query
//! string (lowercase, underscores for separators) so two clicks of
//! "Search" on the same query are byte-identical cache hits.

use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

use crate::cache::Cache;

/// Default cache TTL. Place coordinates don't move, so a day is a
/// reasonable bucket — long enough to make repeated searches free,
/// short enough that the occasional Open-Meteo dataset refresh
/// (e.g. a new admin boundary) propagates within a day.
pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);

const USER_AGENT: &str = "PedalSky-WeatherFeed/0.1 (https://github.com/anthropic/pedalsky)";
const SEARCH_URL: &str = "https://geocoding-api.open-meteo.com/v1/search";

/// One result row from the Open-Meteo geocoding API. We deserialise
/// the fields PedalSky actually uses — the API returns more
/// (admin1_id, admin2_id, country_id, etc.) which serde drops.
#[derive(Debug, Clone, Deserialize)]
pub struct GeocodeResult {
    /// Open-Meteo / GeoNames place identifier — stable across calls,
    /// useful as a dedup key when the UI shows a result list.
    pub id: u64,
    /// Place name (localised to the API's default — English at the
    /// time of writing; can be controlled with `&language=`).
    pub name: String,
    /// Latitude in degrees north.
    pub latitude: f64,
    /// Longitude in degrees east.
    pub longitude: f64,
    /// Surface elevation in metres AMSL. Useful to seed
    /// `pedalsky.toml::[world].ground_elevation_m` when the user
    /// warps somewhere new.
    #[serde(default)]
    pub elevation: f32,
    /// Full country name (e.g. `"United Kingdom"`).
    #[serde(default)]
    pub country: String,
    /// First-level administrative division (state / province /
    /// country-subdivision). E.g. `"Scotland"`, `"Texas"`.
    #[serde(default)]
    pub admin1: String,
    /// Second-level administrative division (county / district).
    /// Optional in the upstream data and often blank.
    #[serde(default)]
    pub admin2: String,
    /// IANA timezone name (e.g. `"Europe/London"`). Empty when
    /// upstream omits it.
    #[serde(default)]
    pub timezone: String,
    /// Resident population. `0` when upstream omits it (often for
    /// very small or recent places).
    #[serde(default)]
    pub population: u64,
    /// GeoNames feature code (e.g. `"PPL"` = populated place,
    /// `"PPLA"` = administrative capital, `"PPLC"` = country
    /// capital). Useful for filtering / icon choice in the UI.
    #[serde(default)]
    pub feature_code: String,
    /// ISO 3166-1 alpha-2 country code.
    #[serde(default)]
    pub country_code: String,
}

/// Raw deserialised search response. Open-Meteo omits `results`
/// entirely when there are no matches, hence the `default`.
#[derive(Debug, Deserialize, Default)]
struct GeocodeResponse {
    #[serde(default)]
    results: Vec<GeocodeResult>,
}

/// Normalise a free-form query string into a safe cache-filename
/// bucket: lowercase, ASCII-only by lossy substitution, path
/// separators and unusual chars replaced with underscores. Length
/// is capped so an absurdly long query doesn't produce an absurdly
/// long filename.
fn normalise_query(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    for ch in q.chars() {
        let safe = match ch {
            'a'..='z' | '0'..='9' | '-' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => '_',
        };
        out.push(safe);
        if out.len() >= 64 {
            break;
        }
    }
    out
}

/// Fetch up to `count` matching places for `query`. Returns an empty
/// Vec for no-match queries (Open-Meteo's `{}` response). Result
/// rows are returned in the order Open-Meteo provides them
/// (population-sorted in practice, but callers shouldn't rely on
/// that; sort yourselves if the order matters).
pub fn search(
    cache_root: &PathBuf,
    query: &str,
    count: usize,
    ttl: Duration,
) -> anyhow::Result<Vec<GeocodeResult>> {
    let q = query.trim();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let bucket = format!("{}-n{count}", normalise_query(q));
    let cache = Cache::new(cache_root);

    if let Some(body) = cache.read_fresh_by_key("geocoding", &bucket, ttl)? {
        tracing::debug!(target: "ps_weather_feed::geocoding", "cache hit (fresh)");
        return parse(&body);
    }

    let url = format!(
        "{SEARCH_URL}?name={}&count={count}&language=en&format=json",
        urlencoding::encode(q),
    );
    tracing::info!(target: "ps_weather_feed::geocoding", %url, "fetching");
    let resp = ureq::get(&url)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(10))
        .call()?;
    let body = resp.into_string()?;
    cache.write_by_key("geocoding", &bucket, &body)?;
    parse(&body)
}

/// Parse the JSON body. Public so tests can exercise the parser
/// without an HTTP round trip.
pub fn parse(body: &str) -> anyhow::Result<Vec<GeocodeResult>> {
    let resp: GeocodeResponse = serde_json::from_str(body)?;
    Ok(resp.results)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "results": [
            {
                "id": 2650769,
                "name": "Dunblane",
                "latitude": 56.18843, "longitude": -3.96417,
                "elevation": 64.0,
                "feature_code": "PPL",
                "country_code": "GB",
                "timezone": "Europe/London",
                "population": 9310,
                "country": "United Kingdom",
                "admin1": "Scotland",
                "admin2": "Stirling"
            },
            {
                "id": 5943853,
                "name": "Dunblane",
                "latitude": 51.18337, "longitude": -106.86797,
                "elevation": 576.0,
                "feature_code": "PPL",
                "country_code": "CA",
                "timezone": "America/Regina",
                "country": "Canada",
                "admin1": "Saskatchewan",
                "admin2": "Coteau No. 255"
            }
        ]
    }"#;

    #[test]
    fn parses_sample_response() {
        let results = parse(SAMPLE).unwrap();
        assert_eq!(results.len(), 2);
        let r = &results[0];
        assert_eq!(r.name, "Dunblane");
        assert_eq!(r.country, "United Kingdom");
        assert_eq!(r.admin1, "Scotland");
        assert!((r.latitude - 56.188).abs() < 0.001);
        assert_eq!(r.population, 9310);
    }

    /// Open-Meteo returns `{}` (literally an empty object) when the
    /// query matches nothing. The parser must accept that and yield
    /// an empty Vec without panicking.
    #[test]
    fn parses_empty_response() {
        let results = parse("{}").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn parses_response_with_missing_optional_fields() {
        // A row that omits everything except the required coordinate
        // / id / name. Open-Meteo's docs say empty fields aren't
        // returned, so we have to tolerate missing admin1/admin2/
        // timezone/population.
        let body = r#"{
            "results": [
                { "id": 1, "name": "Nowhere",
                  "latitude": 0.0, "longitude": 0.0 }
            ]
        }"#;
        let results = parse(body).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].country, "");
        assert_eq!(results[0].population, 0);
        assert_eq!(results[0].elevation, 0.0);
    }

    #[test]
    fn normalise_query_replaces_unsafe_chars() {
        assert_eq!(normalise_query("Dunblane"), "dunblane");
        // "São" — `ã` is one Unicode scalar value so it maps to one
        // underscore, not two. The space then becomes another.
        assert_eq!(normalise_query("São Paulo"), "s_o_paulo");
        assert_eq!(normalise_query("New York, NY"), "new_york__ny");
        assert_eq!(normalise_query("../../etc/passwd"), "______etc_passwd");
    }

    #[test]
    fn normalise_query_truncates_long_input() {
        let huge = "a".repeat(300);
        let normalised = normalise_query(&huge);
        assert_eq!(normalised.len(), 64);
    }

    #[test]
    fn search_returns_empty_for_blank_query() {
        let dir = tempfile::tempdir().unwrap();
        let results = search(
            &dir.path().to_path_buf(),
            "   ",
            5,
            Duration::from_secs(60),
        )
        .unwrap();
        assert!(results.is_empty());
    }
}
