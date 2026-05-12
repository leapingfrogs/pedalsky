//! Real-weather ingestion for PedalSky.
//!
//! Fetches forecast + observed weather data and maps it into a
//! `ps_core::Scene` so the renderer can show the actual sky over a
//! configured lat/lon at a chosen time.
//!
//! Sources:
//!
//! - **Open-Meteo** (`api.open-meteo.com`) — free, no API key, no
//!   strict per-key quota. Publishes hourly forecast values
//!   including cloud cover at six pressure levels (1000, 925, 850,
//!   700, 500, 300 hPa) which map cleanly to PedalSky's altitude-
//!   banded cloud types, plus surface temperature, dewpoint,
//!   pressure, wind, visibility, precipitation, CAPE.
//! - **METAR** (Aviation Weather Center,
//!   `aviationweather.gov/api/data/metar`) — optional enrichment.
//!   Real observations at nearby airports correct the surface
//!   conditions when an authoritative datapoint exists.
//!
//! Every response is cached on disk so repeated fetches at the
//! same place + hour don't hit the upstream APIs. See `cache.rs`.

#![deny(missing_docs)]

pub mod cache;
pub mod mapping;
pub mod metar;
pub mod open_meteo;

pub use cache::Cache;
pub use mapping::{enrich_with_metar, open_meteo_to_scene};
pub use metar::MetarRecord;
pub use open_meteo::OpenMeteoResponse;

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use ps_core::Scene;

/// Options for [`fetch_scene`] — the public top-level call.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Observer latitude (degrees north).
    pub lat: f64,
    /// Observer longitude (degrees east).
    pub lon: f64,
    /// Target time. Open-Meteo data is hour-aligned; we pick the
    /// nearest forecast row.
    pub time: DateTime<Utc>,
    /// If true, also fetch the nearest METAR station and apply
    /// any surface / present-weather enrichment.
    pub enrich_with_metar: bool,
    /// Cache directory. Created on demand.
    pub cache_dir: PathBuf,
}

impl FetchOptions {
    /// Defaults: Dunblane, Scotland, the current UTC hour,
    /// METAR enrichment on, cache under `./cache/weather`.
    pub fn dunblane_now() -> Self {
        Self {
            lat: 56.1922,
            lon: -3.9645,
            time: Utc::now(),
            enrich_with_metar: true,
            cache_dir: PathBuf::from("cache").join("weather"),
        }
    }
}

/// Errors raised by [`fetch_scene`].
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    /// Open-Meteo fetch failed and no cache fallback existed.
    #[error("open-meteo: {0}")]
    OpenMeteo(#[source] anyhow::Error),
    /// Resulting scene failed `Scene::validate()`. This should be
    /// rare — usually means we generated overlapping cloud layers.
    #[error("synthesised scene failed validation: {0}")]
    SceneInvalid(#[source] ps_core::SceneError),
}

/// Top-level: fetch live weather and return a populated `Scene`.
///
/// Steps: pull Open-Meteo for the configured (lat, lon, hour),
/// optionally fetch the nearest METAR station and apply surface
/// + present-weather enrichment, and synthesise a `Scene` with
/// up to six cloud layers + surface conditions + precipitation +
/// lightning.
///
/// All HTTP calls go through the disk cache; if a request fails
/// the cache's last-known good response is served as a fallback.
pub fn fetch_scene(opts: &FetchOptions) -> Result<Scene, FetchError> {
    let resp = open_meteo::fetch(
        &opts.cache_dir,
        opts.lat,
        opts.lon,
        opts.time,
        open_meteo::DEFAULT_TTL,
    )
    .map_err(FetchError::OpenMeteo)?;

    let mut scene = open_meteo_to_scene(&resp, opts.time);

    if opts.enrich_with_metar {
        // METAR fetch is best-effort — a failure here just means
        // we skip enrichment.
        match metar::fetch_nearest(
            &opts.cache_dir,
            opts.lat,
            opts.lon,
            opts.time,
            metar::DEFAULT_TTL,
        ) {
            Ok(Some(m)) => {
                tracing::info!(
                    target: "ps_weather_feed",
                    icao = %m.icao_id,
                    dist_deg = m.distance_deg_from(opts.lat, opts.lon),
                    "applying METAR enrichment"
                );
                enrich_with_metar(&mut scene, opts.lat, opts.lon, &m);
            }
            Ok(None) => {
                tracing::info!(
                    target: "ps_weather_feed",
                    "no METAR station within search radius — skipping enrichment"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "ps_weather_feed",
                    error = %e,
                    "METAR fetch failed — skipping enrichment"
                );
            }
        }
    }

    scene.validate().map_err(FetchError::SceneInvalid)?;
    Ok(scene)
}

#[doc(hidden)]
pub fn _ttl_for_tests(d: Duration) -> Duration {
    d
}
