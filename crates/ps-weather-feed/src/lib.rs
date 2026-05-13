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
//! - **NOAA SWPC** (`services.swpc.noaa.gov`, Phase 15) — planetary
//!   Kp index for the aurora subsystem. See `kp_index.rs` (3-hourly
//!   observed + 3-day forecast) and `ovation.rs` (per-location
//!   aurora intensity from the 30-minute OVATION nowcast).
//!
//! Every response is cached on disk so repeated fetches at the
//! same place + hour don't hit the upstream APIs. See `cache.rs`.

#![deny(missing_docs)]

pub mod cache;
pub mod geocoding;
pub mod kp_index;
pub mod mapping;
pub mod metar;
pub mod open_meteo;
pub mod ovation;

pub use cache::Cache;
pub use geocoding::GeocodeResult;
pub use kp_index::{KpRow, KpSeries};
pub use mapping::{enrich_with_metar, open_meteo_to_scene};
pub use metar::MetarRecord;
pub use open_meteo::OpenMeteoResponse;
pub use ovation::OvationGrid;

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
    /// Phase 15.A — if true, also fetch the NOAA SWPC planetary Kp
    /// index and write it onto `scene.aurora.kp_index`. Cheap (one
    /// global file, cached hourly) and best-effort: failures fall
    /// back to leaving `kp_index = 0`, which is the pre-Phase-15
    /// default and disables the aurora subsystem.
    pub fetch_kp_index: bool,
    /// Phase 15.B — if true, also fetch the NOAA SWPC OVATION
    /// aurora nowcast and sample it at `(lon, lat)`. The result
    /// drives `scene.aurora.intensity_override`, giving the aurora
    /// subsystem a physically-grounded "how bright at this exact
    /// location" answer that respects the auroral-oval shape (it's
    /// offset toward the magnetic pole and rotates with UTC).
    /// Best-effort: failures leave the override at `-1.0` (the
    /// "derive from Kp" sentinel).
    pub fetch_ovation: bool,
    /// Cache directory. Created on demand.
    pub cache_dir: PathBuf,
}

impl FetchOptions {
    /// Defaults: Dunblane, Scotland, the current UTC hour,
    /// METAR enrichment on, Kp + OVATION fetched, cache under
    /// `./cache/weather`.
    pub fn dunblane_now() -> Self {
        Self {
            lat: 56.1922,
            lon: -3.9645,
            time: Utc::now(),
            enrich_with_metar: true,
            fetch_kp_index: true,
            fetch_ovation: true,
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

    if opts.fetch_kp_index {
        // Phase 15.A — Kp fetch is best-effort. On failure we leave
        // `scene.aurora.kp_index = 0` and the aurora subsystem stays
        // dark, matching the pre-15 behaviour.
        match kp_index::fetch(&opts.cache_dir, opts.time, kp_index::DEFAULT_TTL) {
            Ok(series) => match series.nearest(opts.time) {
                Some(row) => {
                    tracing::info!(
                        target: "ps_weather_feed",
                        kp = row.kp,
                        observed = %row.observed,
                        time_tag = %row.time_tag,
                        "applying SWPC Kp index"
                    );
                    scene.aurora.kp_index = row.kp;
                }
                None => {
                    tracing::warn!(
                        target: "ps_weather_feed",
                        "SWPC Kp series empty — aurora disabled"
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    target: "ps_weather_feed",
                    error = %e,
                    "SWPC Kp fetch failed — aurora left at default kp_index = 0"
                );
            }
        }
    }

    if opts.fetch_ovation {
        // Phase 15.B — OVATION fetch is best-effort. The 30-minute
        // nowcast is only useful near `opts.time = now`; for
        // historical or far-future renders the data is irrelevant
        // (it's a measurement, not a forecast), so we still fetch
        // it but the result is just "what the aurora oval looks
        // like *right now*" overlaid on a possibly-different
        // simulated time. Users wanting future-projected aurora
        // need the Kp forecast path (Phase 15.A) which derives
        // intensity from the global scalar.
        match ovation::fetch(&opts.cache_dir, opts.time, ovation::DEFAULT_TTL) {
            Ok(grid) => {
                let intensity = grid.intensity_at(opts.lon, opts.lat);
                tracing::info!(
                    target: "ps_weather_feed",
                    intensity,
                    lon = opts.lon,
                    lat = opts.lat,
                    obs_time = %grid.observation_time,
                    "applying SWPC OVATION intensity"
                );
                // The aurora subsystem treats `intensity_override`
                // values >= 0 as "use this directly"; negative
                // values mean "derive from Kp". So we only set the
                // override when OVATION reports non-zero — a 0
                // intensity here is meaningful (no aurora) but the
                // Kp-derived path may still have something to say
                // for very weak storms. The threshold is loose.
                if intensity > 0.02 {
                    scene.aurora.intensity_override = intensity;
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "ps_weather_feed",
                    error = %e,
                    "SWPC OVATION fetch failed — using Kp-derived intensity"
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
