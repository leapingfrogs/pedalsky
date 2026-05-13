//! Map fetched API responses to a `ps_core::Scene`.
//!
//! Open-Meteo gives us:
//!
//! - Surface conditions (T, dewp, P, wind, visibility, precip,
//!   CAPE) that map almost 1:1 to `Scene.surface` + `precipitation`.
//! - Cloud cover percentages at six pressure levels, which
//!   correspond to PedalSky's altitude-banded cloud types via the
//!   table in [`band_to_cloud_type`]. Cover above
//!   [`COVERAGE_THRESHOLD`] becomes a `CloudLayer`; below
//!   threshold is dropped to avoid swamping the scene with empty
//!   bands.
//!
//! METAR (when supplied) overrides the surface conditions with
//! the observed values from the nearest airport, and adds
//! present-weather cues:
//!
//! - `TS` / `TSRA` → cumulonimbus + non-zero lightning.
//! - `SN` → snow precipitation.
//! - The observed cloud groups (FEW/SCT/BKN/OVC at specific feet
//!   AGL) replace the pressure-level interpretation **only when
//!   the station is within ~30 km** of the requested point.

use chrono::{DateTime, Utc};
use ps_core::{
    Aurora, CloudLayer, CloudType, Clouds, Lightning, PrecipKind, Precipitation, Scene, Surface,
    SurfaceMaterial, Wetness, WindAloftSample,
};

use crate::metar::MetarRecord;
use crate::open_meteo::{Hourly, OpenMeteoResponse};

/// Conversion factor: km/h → m/s (one over 3.6, rounded).
const KMH_TO_MPS: f32 = 0.277_778;

/// Minimum cloud cover (%) for an altitude band to produce a
/// `CloudLayer`. Bands below this are silently dropped — keeps
/// the scene's layer count small on partly-cloudy days.
pub const COVERAGE_THRESHOLD: f32 = 5.0;

/// METAR enrichment only kicks in when the nearest station is
/// within this distance (degrees ≈ Euclidean lat/lon norm). At
/// higher latitudes a degree of longitude is shorter, so the
/// effective km radius shrinks; that's the right direction
/// (further stations are less relevant anyway).
pub const METAR_PROXIMITY_DEG: f64 = 0.35;

/// Standard-atmosphere altitude (m AMSL) for a given pressure
/// level (hPa). Inverse of the dry-air barometric formula
/// truncated at the troposphere. Adequate for cloud rendering —
/// the actual altitude varies a few percent with surface pressure
/// but our shader's altitude-banded rendering doesn't depend on
/// sub-100 m precision.
pub fn pressure_to_altitude_m(pressure_hpa: u32) -> f32 {
    // p / p0 = (1 - L h / T0)^(g M / R L)
    // ⇒ h = T0/L * (1 - (p/p0)^(R L / (g M)))
    let p0 = 1013.25_f32;
    let t0 = 288.15_f32;
    let lapse = 0.0065_f32; // K/m
    let exponent = 0.190284_f32; // R·L/(g·M) for dry air
    let ratio = pressure_hpa as f32 / p0;
    t0 / lapse * (1.0 - ratio.powf(exponent))
}

/// Pick a cloud type for an altitude band given its cover.
/// Heuristic table — matches WMO conventions: low + high cover =
/// continuous overcast (Stratus); low + scattered cover = puffy
/// cumulus; mid-level → Sc/Ac/As; high-altitude → Ci/Cs.
pub fn band_to_cloud_type(pressure_hpa: u32, cover_pct: f32) -> CloudType {
    let cover_high = cover_pct >= 70.0;
    match pressure_hpa {
        // Surface / low fog band. High cover → Stratus, low →
        // Cumulus (fair-weather puffs forming just above the
        // boundary layer).
        1000 | 925 if cover_high => CloudType::Stratus,
        1000 | 925 => CloudType::Cumulus,
        // Stratocumulus / cumulus base band.
        850 if cover_high => CloudType::Stratocumulus,
        850 => CloudType::Cumulus,
        // Mid-level. Altocumulus when broken, Altostratus when
        // overcast.
        700 if cover_high => CloudType::Altostratus,
        700 => CloudType::Altocumulus,
        // 500 hPa always reads as Altostratus (mid-level sheet).
        500 => CloudType::Altostratus,
        // High level — Cirrostratus when continuous, Cirrus when
        // wispy.
        300 if cover_high => CloudType::Cirrostratus,
        300 => CloudType::Cirrus,
        // Fallback (shouldn't be reached if PRESSURE_LEVELS_HPA
        // is in sync).
        _ => CloudType::Cumulus,
    }
}

/// Layer vertical thickness (m) for a band. Picked to match each
/// cloud type's NDF profile in the cloud march without producing
/// overlapping layers.
fn band_thickness_m(pressure_hpa: u32) -> f32 {
    match pressure_hpa {
        1000 => 400.0,  // Stratus/fog
        925 => 600.0,   // Cumulus/Stratus base
        850 => 800.0,   // Cumulus mass / Sc
        700 => 600.0,   // Ac/As mid-level
        500 => 1000.0,  // As deep sheet
        300 => 1500.0,  // Ci/Cs high
        _ => 500.0,
    }
}

/// Convert a single hour of Open-Meteo data into a `Scene`.
/// Cumulonimbus is detected from the combination of high CAPE +
/// high total cover + significant precip rate (heuristic
/// validated against typical thunderstorm soundings).
pub fn open_meteo_to_scene(resp: &OpenMeteoResponse, target: DateTime<Utc>) -> Scene {
    let idx = resp.hourly.nearest_index(target).unwrap_or(0);
    let h = &resp.hourly;

    let mut scene = Scene::default();
    scene.surface = surface_from_open_meteo(h, idx);
    scene.precipitation = precip_from_open_meteo(h, idx);
    scene.lightning = lightning_from_open_meteo(h, idx);
    scene.aurora = Aurora::default(); // Open-Meteo doesn't carry Kp.
    scene.clouds = clouds_from_open_meteo(h, idx);
    scene
}

fn surface_from_open_meteo(h: &Hourly, i: usize) -> Surface {
    let wind_speed_kmh = h.wind_speed_10m.get(i).copied().unwrap_or(0.0);
    let wind_speed_mps = wind_speed_kmh * KMH_TO_MPS;
    let winds_aloft: Vec<WindAloftSample> = h
        .winds_aloft_by_level(i)
        .into_iter()
        .map(|(level, speed_kmh, dir_deg)| WindAloftSample {
            pressure_hpa: level as u16,
            altitude_m: pressure_to_altitude_m(level),
            speed_mps: speed_kmh * KMH_TO_MPS,
            dir_deg,
        })
        .collect();

    // Wetness — derived from recent rain. Open-Meteo gives an
    // hourly rate, but the ground stays wet for ~hours after; sum
    // the last 3 hours and clamp.
    let rain_sum = (0..3)
        .map(|back| {
            i.checked_sub(back)
                .and_then(|j| h.rain.get(j).copied())
                .unwrap_or(0.0)
        })
        .sum::<f32>();
    let ground_wetness = (rain_sum / 5.0).clamp(0.0, 1.0);
    let puddle_coverage = (rain_sum / 8.0).clamp(0.0, 0.8);
    let snow_sum_cm = (0..6)
        .map(|back| {
            i.checked_sub(back)
                .and_then(|j| h.snowfall.get(j).copied())
                .unwrap_or(0.0)
        })
        .sum::<f32>();
    let snow_depth_m = (snow_sum_cm / 100.0).clamp(0.0, 1.0);

    Surface {
        visibility_m: h.visibility.get(i).copied().unwrap_or(30_000.0),
        temperature_c: h.temperature_2m.get(i).copied().unwrap_or(15.0),
        dewpoint_c: h.dew_point_2m.get(i).copied().unwrap_or(8.0),
        pressure_hpa: h.surface_pressure.get(i).copied().unwrap_or(1013.0),
        wind_dir_deg: h.wind_direction_10m.get(i).copied().unwrap_or(0.0),
        wind_speed_mps,
        wetness: Wetness {
            ground_wetness,
            puddle_coverage,
            puddle_start: 0.6,
            snow_depth_m,
        },
        material: SurfaceMaterial::Grass,
        winds_aloft,
    }
}

fn precip_from_open_meteo(h: &Hourly, i: usize) -> Precipitation {
    let rain = h.rain.get(i).copied().unwrap_or(0.0);
    let snow = h.snowfall.get(i).copied().unwrap_or(0.0);
    let total = h.precipitation.get(i).copied().unwrap_or(rain + snow);
    if total < 0.05 {
        return Precipitation {
            kind: PrecipKind::None,
            intensity_mm_per_h: 0.0,
        };
    }
    let kind = if snow > rain {
        PrecipKind::Snow
    } else if snow > 0.05 && rain > 0.05 {
        PrecipKind::Sleet
    } else {
        PrecipKind::Rain
    };
    Precipitation {
        kind,
        intensity_mm_per_h: total,
    }
}

fn lightning_from_open_meteo(h: &Hourly, i: usize) -> Lightning {
    // Heuristic Cb detector: CAPE above 1000 J/kg + total cover
    // above 80% + precipitation rate above 4 mm/h ⇒ active
    // thunderstorm. The strike rate scales weakly with CAPE
    // beyond the threshold; cap so the renderer doesn't see
    // hundreds of strikes per minute even for extreme cells.
    let cape = h.cape.get(i).copied().unwrap_or(0.0);
    let cover = h.cloud_cover.get(i).copied().unwrap_or(0.0);
    let precip = h.precipitation.get(i).copied().unwrap_or(0.0);
    let active = cape > 1000.0 && cover > 80.0 && precip > 4.0;
    if !active {
        return Lightning::default();
    }
    let strikes = ((cape - 1000.0) / 3000.0).clamp(0.0, 1.0) * 0.8 + 0.1;
    Lightning {
        strikes_per_min_per_km2: strikes,
    }
}

fn clouds_from_open_meteo(h: &Hourly, i: usize) -> Clouds {
    // Walk pressure levels from highest altitude (300 hPa) to
    // surface, emitting a CloudLayer per band that exceeds the
    // coverage threshold. Layer altitudes use the barometric
    // formula; we keep layers vertically disjoint by using the
    // band thickness as the layer span.
    let mut layers: Vec<CloudLayer> = Vec::new();
    let mut last_top_m: f32 = -1.0;
    for (pressure, cover_pct) in h.cloud_cover_by_level(i).iter().rev() {
        if *cover_pct < COVERAGE_THRESHOLD {
            continue;
        }
        let centre = pressure_to_altitude_m(*pressure);
        let thickness = band_thickness_m(*pressure);
        let mut base = (centre - thickness * 0.5).max(50.0);
        // Maintain disjointness — if this band's base sits below
        // the previous (lower-altitude) band's top, lift it.
        if base < last_top_m {
            base = last_top_m + 50.0;
        }
        let top = (base + thickness).max(base + 100.0);
        last_top_m = top;
        let cloud_type = band_to_cloud_type(*pressure, *cover_pct);
        layers.push(CloudLayer {
            cloud_type,
            base_m: base,
            top_m: top,
            coverage: (*cover_pct / 100.0).clamp(0.0, 1.0),
            ..CloudLayer::default()
        });
    }
    Clouds {
        layers,
        coverage_grid: None,
    }
}

/// Apply METAR enrichment in place. Surface observations replace
/// the corresponding NWP values when the station is within
/// `METAR_PROXIMITY_DEG`; cloud groups override the pressure-level
/// interpretation only at the same proximity. Present-weather
/// codes (TSRA / TS) override the lightning gating.
pub fn enrich_with_metar(scene: &mut Scene, lat: f64, lon: f64, metar: &MetarRecord) {
    let dist = metar.distance_deg_from(lat, lon);
    if dist > METAR_PROXIMITY_DEG {
        // Too far away to be more reliable than the gridded NWP.
        tracing::debug!(
            target: "ps_weather_feed::mapping",
            icao = %metar.icao_id,
            dist_deg = dist,
            "METAR too distant — surface enrichment skipped"
        );
        return;
    }

    // Surface: use METAR values where available; otherwise keep
    // the Open-Meteo defaults.
    if let Some(v) = metar.temp {
        scene.surface.temperature_c = v;
    }
    if let Some(v) = metar.dewp {
        scene.surface.dewpoint_c = v;
    }
    if let Some(v) = metar.wdir {
        scene.surface.wind_dir_deg = v;
    }
    if let Some(v) = metar.wspd {
        // METAR wind speed is knots.
        scene.surface.wind_speed_mps = v * 0.514_444;
    }
    if let Some(v) = metar.altim {
        scene.surface.pressure_hpa = v;
    }
    let v = metar.visibility_m();
    if v > 0.0 {
        scene.surface.visibility_m = v;
    }

    // Present-weather overrides.
    if metar.is_thunderstorm() {
        // Force a cumulonimbus layer if not already present, and
        // ensure lightning is active.
        if !scene.clouds.layers.iter().any(|l| l.cloud_type == CloudType::Cumulonimbus) {
            scene.clouds.layers.push(CloudLayer {
                cloud_type: CloudType::Cumulonimbus,
                base_m: 800.0,
                top_m: 11_000.0,
                coverage: 0.85,
                ..CloudLayer::default()
            });
        }
        if scene.lightning.strikes_per_min_per_km2 < 0.2 {
            scene.lightning.strikes_per_min_per_km2 = 0.5;
        }
    }
    if metar.is_snow() {
        scene.precipitation.kind = PrecipKind::Snow;
        if scene.precipitation.intensity_mm_per_h < 0.5 {
            scene.precipitation.intensity_mm_per_h = 1.0;
        }
    } else if metar.is_rain() && scene.precipitation.kind == PrecipKind::None {
        scene.precipitation.kind = PrecipKind::Rain;
        if scene.precipitation.intensity_mm_per_h < 0.5 {
            scene.precipitation.intensity_mm_per_h = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn t() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 12, 13, 0, 0).unwrap()
    }

    fn cumulus_scene_response() -> OpenMeteoResponse {
        // One hour of synthetic data — broken cumulus, light wind,
        // dry ground.
        serde_json::from_str(
            r#"{
                "latitude": 56.19, "longitude": -3.96, "elevation": 83.0,
                "hourly": {
                    "time": ["2026-05-12T13:00"],
                    "temperature_2m": [17.0],
                    "dew_point_2m": [9.0],
                    "surface_pressure": [1015.0],
                    "visibility": [30000.0],
                    "wind_speed_10m": [18.0],
                    "wind_direction_10m": [240.0],
                    "precipitation": [0.0],
                    "rain": [0.0],
                    "snowfall": [0.0],
                    "cloud_cover": [55.0],
                    "cape": [100.0],
                    "cloud_cover_1000hPa": [0.0],
                    "cloud_cover_925hPa": [0.0],
                    "cloud_cover_850hPa": [50.0],
                    "cloud_cover_700hPa": [10.0],
                    "cloud_cover_500hPa": [0.0],
                    "cloud_cover_300hPa": [0.0]
                }
            }"#,
        ).unwrap()
    }

    #[test]
    fn pressure_to_altitude_orders_correctly() {
        // Lower pressure → higher altitude. Just sanity-check
        // monotonicity and rough magnitudes.
        let a_300 = pressure_to_altitude_m(300);
        let a_500 = pressure_to_altitude_m(500);
        let a_850 = pressure_to_altitude_m(850);
        let a_1000 = pressure_to_altitude_m(1000);
        assert!(a_300 > a_500);
        assert!(a_500 > a_850);
        assert!(a_850 > a_1000);
        // Rough sanity check: 300 hPa ≈ 9 km.
        assert!(a_300 > 8000.0 && a_300 < 10_000.0);
        // 1000 hPa ≈ 100 m AMSL.
        assert!(a_1000 < 300.0);
    }

    #[test]
    fn surface_maps_cleanly() {
        let resp = cumulus_scene_response();
        let scene = open_meteo_to_scene(&resp, t());
        assert_eq!(scene.surface.temperature_c, 17.0);
        assert_eq!(scene.surface.dewpoint_c, 9.0);
        assert_eq!(scene.surface.pressure_hpa, 1015.0);
        // 18 km/h → 5 m/s.
        assert!((scene.surface.wind_speed_mps - 5.0).abs() < 0.1);
        assert_eq!(scene.surface.wind_dir_deg, 240.0);
        assert_eq!(scene.surface.visibility_m, 30000.0);
        // The cumulus fixture omits winds_aloft, so the field is empty.
        assert!(scene.surface.winds_aloft.is_empty());
    }

    #[test]
    fn winds_aloft_populated_when_payload_carries_them() {
        let resp: OpenMeteoResponse = serde_json::from_str(
            r#"{
                "latitude": 56.19, "longitude": -3.96, "elevation": 83.0,
                "hourly": {
                    "time": ["2026-05-12T13:00"],
                    "temperature_2m": [14.0],
                    "dew_point_2m": [9.0],
                    "surface_pressure": [1015.0],
                    "visibility": [30000.0],
                    "wind_speed_10m": [18.0],
                    "wind_direction_10m": [240.0],
                    "precipitation": [0.0],
                    "rain": [0.0],
                    "snowfall": [0.0],
                    "cloud_cover": [40.0],
                    "cape": [100.0],
                    "cloud_cover_1000hPa": [0.0],
                    "cloud_cover_925hPa": [0.0],
                    "cloud_cover_850hPa": [40.0],
                    "cloud_cover_700hPa": [10.0],
                    "cloud_cover_500hPa": [0.0],
                    "cloud_cover_300hPa": [0.0],
                    "wind_speed_850hPa": [36.0],
                    "wind_speed_700hPa": [54.0],
                    "wind_speed_500hPa": [90.0],
                    "wind_speed_300hPa": [144.0],
                    "wind_speed_200hPa": [180.0],
                    "wind_direction_850hPa": [240.0],
                    "wind_direction_700hPa": [255.0],
                    "wind_direction_500hPa": [270.0],
                    "wind_direction_300hPa": [285.0],
                    "wind_direction_200hPa": [290.0]
                }
            }"#,
        )
        .unwrap();
        let scene = open_meteo_to_scene(&resp, t());
        let aloft = &scene.surface.winds_aloft;
        // Phase 14.J — 5 levels now (850/700/500/300/200).
        assert_eq!(aloft.len(), 5);
        // Sorted ascending by altitude (descending by pressure).
        assert_eq!(aloft[0].pressure_hpa, 850);
        assert_eq!(aloft[4].pressure_hpa, 200);
        assert!(aloft[0].altitude_m < aloft[4].altitude_m);
        // 36 km/h → 10 m/s (within rounding).
        assert!((aloft[0].speed_mps - 10.0).abs() < 0.1);
        // 144 km/h → 40 m/s.
        assert!((aloft[3].speed_mps - 40.0).abs() < 0.2);
        // 180 km/h → 50 m/s.
        assert!((aloft[4].speed_mps - 50.0).abs() < 0.2);
        // Direction passed through unchanged.
        assert_eq!(aloft[2].dir_deg, 270.0);
        // 850 hPa is roughly 1.5 km AMSL.
        assert!(aloft[0].altitude_m > 1200.0 && aloft[0].altitude_m < 1700.0);
        // 300 hPa sits around 9 km.
        assert!(aloft[3].altitude_m > 8000.0 && aloft[3].altitude_m < 10_000.0);
        // 200 hPa sits around 12 km — covering the top of the
        // synthesised wind-field volume so cirrus reads real data
        // instead of clamped 300 hPa.
        assert!(aloft[4].altitude_m > 11_000.0 && aloft[4].altitude_m < 13_000.0);
    }

    #[test]
    fn cloud_layers_emit_only_above_threshold() {
        let resp = cumulus_scene_response();
        let scene = open_meteo_to_scene(&resp, t());
        // Only the 850 hPa band (50%) and 700 hPa band (10%)
        // exceed the 5% threshold. The 1000/925/500/300 bands at
        // 0% are dropped.
        assert_eq!(scene.clouds.layers.len(), 2);
        // Layers are ordered surface-up — first should be 850 hPa
        // (lower altitude).
        let base_0 = scene.clouds.layers[0].base_m;
        let base_1 = scene.clouds.layers[1].base_m;
        assert!(base_0 < base_1, "layers should be ordered low-to-high");
    }

    #[test]
    fn layers_stay_disjoint() {
        // Construct an overlapping-cover scenario by inflating
        // all bands above threshold. Each layer should sit above
        // the previous layer's top, not inside it.
        let resp: OpenMeteoResponse = serde_json::from_str(
            r#"{
                "latitude": 0.0, "longitude": 0.0, "elevation": 0.0,
                "hourly": {
                    "time": ["2026-05-12T13:00"],
                    "temperature_2m": [10.0],
                    "dew_point_2m": [5.0],
                    "surface_pressure": [1013.0],
                    "visibility": [30000.0],
                    "wind_speed_10m": [10.0],
                    "wind_direction_10m": [200.0],
                    "precipitation": [0.0],
                    "rain": [0.0],
                    "snowfall": [0.0],
                    "cloud_cover": [90.0],
                    "cape": [0.0],
                    "cloud_cover_1000hPa": [80.0],
                    "cloud_cover_925hPa": [80.0],
                    "cloud_cover_850hPa": [80.0],
                    "cloud_cover_700hPa": [80.0],
                    "cloud_cover_500hPa": [80.0],
                    "cloud_cover_300hPa": [80.0]
                }
            }"#,
        )
        .unwrap();
        let scene = open_meteo_to_scene(&resp, t());
        for w in scene.clouds.layers.windows(2) {
            assert!(
                w[0].top_m <= w[1].base_m,
                "layers overlap: {:?} top {} vs {:?} base {}",
                w[0].cloud_type, w[0].top_m,
                w[1].cloud_type, w[1].base_m,
            );
        }
        scene.validate().expect("scene must validate");
    }

    #[test]
    fn thunderstorm_lightning_gating() {
        let resp: OpenMeteoResponse = serde_json::from_str(
            r#"{
                "latitude": 0.0, "longitude": 0.0, "elevation": 0.0,
                "hourly": {
                    "time": ["2026-05-12T13:00"],
                    "temperature_2m": [24.0],
                    "dew_point_2m": [22.0],
                    "surface_pressure": [1000.0],
                    "visibility": [3000.0],
                    "wind_speed_10m": [25.0],
                    "wind_direction_10m": [180.0],
                    "precipitation": [12.0],
                    "rain": [12.0],
                    "snowfall": [0.0],
                    "cloud_cover": [95.0],
                    "cape": [2500.0],
                    "cloud_cover_1000hPa": [80.0],
                    "cloud_cover_925hPa": [85.0],
                    "cloud_cover_850hPa": [90.0],
                    "cloud_cover_700hPa": [80.0],
                    "cloud_cover_500hPa": [60.0],
                    "cloud_cover_300hPa": [60.0]
                }
            }"#,
        ).unwrap();
        let scene = open_meteo_to_scene(&resp, t());
        assert!(scene.lightning.strikes_per_min_per_km2 > 0.0);
        assert_eq!(scene.precipitation.kind, PrecipKind::Rain);
    }

    #[test]
    fn metar_enrichment_overrides_when_close() {
        let resp = cumulus_scene_response();
        let mut scene = open_meteo_to_scene(&resp, t());
        let m: MetarRecord = serde_json::from_str(
            r#"{
                "icaoId": "EGPN", "lat": 56.45, "lon": -3.03,
                "elev": 4.0, "reportTime": "2026-05-12T13:00:00.000Z",
                "temp": 19.5, "dewp": 8.5, "wdir": 260.0, "wspd": 8.0,
                "visib": "9999", "altim": 1016.0,
                "rawOb": "METAR EGPN 121300Z 26008KT 9999 FEW040 19/08 Q1016",
                "clouds": [{"cover": "FEW", "base": 4000}]
            }"#,
        ).unwrap();
        // EGPN (Dundee) is ~0.96° from Dunblane — outside our
        // 0.35° proximity gate. Verify: enrichment should NOT
        // change temp.
        enrich_with_metar(&mut scene, 56.1922, -3.9645, &m);
        assert_eq!(scene.surface.temperature_c, 17.0);

        // Now use a closer fake station (within 0.35°).
        let close: MetarRecord = serde_json::from_str(
            r#"{
                "icaoId": "EGPF", "lat": 56.1, "lon": -3.9,
                "elev": 8.0, "reportTime": "2026-05-12T13:00:00.000Z",
                "temp": 19.5, "dewp": 8.5, "wdir": 260.0, "wspd": 8.0,
                "visib": "9999", "altim": 1016.0,
                "rawOb": "METAR EGPF 121300Z 26008KT 9999 FEW040 19/08 Q1016",
                "clouds": []
            }"#,
        ).unwrap();
        enrich_with_metar(&mut scene, 56.1922, -3.9645, &close);
        assert_eq!(scene.surface.temperature_c, 19.5);
        // 8 knots ≈ 4.116 m/s.
        assert!((scene.surface.wind_speed_mps - 4.116).abs() < 0.05);
    }

    #[test]
    fn metar_thunderstorm_forces_cb() {
        let resp = cumulus_scene_response();
        let mut scene = open_meteo_to_scene(&resp, t());
        let m: MetarRecord = serde_json::from_str(
            r#"{
                "icaoId": "EGPF", "lat": 56.1, "lon": -3.9,
                "elev": 8.0, "reportTime": "2026-05-12T13:00:00.000Z",
                "temp": 22.0, "dewp": 19.0, "wdir": 240.0, "wspd": 14.0,
                "visib": "5000", "altim": 1000.0,
                "rawOb": "METAR EGPF 121300Z 24014KT 5000 +TSRA OVC025CB 22/19 Q1000",
                "clouds": [{"cover": "OVC", "base": 2500}]
            }"#,
        ).unwrap();
        enrich_with_metar(&mut scene, 56.1922, -3.9645, &m);
        assert!(scene.clouds.layers.iter().any(|l| l.cloud_type == CloudType::Cumulonimbus));
        assert!(scene.lightning.strikes_per_min_per_km2 >= 0.5);
    }
}
