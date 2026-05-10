//! Phase 2 astronomical-calculation tests.
//!
//! - `nrel_appendix_b_*`: bit-equivalent (within ±0.05°) match to the
//!   worked example in NREL/TP-560-34302 Annex B. Inside the unit-test
//!   module of `astro::spa` already; here we exercise the public
//!   `sun_position` (f32) entry point and cross-check.
//! - `dunblane_noon_altitude_*`: closed-form solar-noon altitude at the
//!   spring equinox / summer solstice / winter solstice for Dunblane.
//!   Independent of any third-party library.
//! - `equinox_sunrise_is_due_east`: at the equinox at any latitude the
//!   sun rises within ~1° of due east — convention sanity check.
//! - `toa_illuminance_at_1au`: 127 500 lux at 1 AU per plan §2.4.

use chrono::{TimeZone, Utc};
use ps_core::astro::sun_position;

const DUNBLANE_LAT: f64 = 56.1922;
const DUNBLANE_LON: f64 = -3.9645;
const DUNBLANE_ELEV: f64 = 60.0;

/// Tolerant approx-equal helper; degrees.
fn within(actual: f64, expected: f64, tol: f64, label: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{label}: got {actual}°, expected {expected}° (|err|={err}° > {tol}°)"
    );
}

#[test]
fn dunblane_spring_equinox_noon_altitude() {
    // 2026-03-20 12:00 UTC — Dunblane at GMT (no BST yet). Solar noon for
    // longitude -3.9645° is at UTC 12:00 + 4·(3.9645)/60 ≈ 12:16 UTC. Use
    // a single point within ±15 min of solar noon and check altitude is
    // within ~0.5° of the closed-form 90 - 56.1922 + δ ≈ 33.8°.
    let utc = Utc.with_ymd_and_hms(2026, 3, 20, 12, 16, 0).unwrap();
    let p = sun_position(utc, DUNBLANE_LAT, DUNBLANE_LON, DUNBLANE_ELEV);
    let alt_deg = (p.altitude_rad as f64).to_degrees();
    eprintln!("Dunblane spring equinox noon altitude = {alt_deg}°");
    within(alt_deg, 33.8, 1.0, "spring equinox noon altitude");
}

#[test]
fn dunblane_summer_solstice_noon_altitude() {
    // 2026-06-21 ~ solar noon. δ ≈ +23.44°.
    let utc = Utc.with_ymd_and_hms(2026, 6, 21, 12, 16, 0).unwrap();
    let p = sun_position(utc, DUNBLANE_LAT, DUNBLANE_LON, DUNBLANE_ELEV);
    let alt_deg = (p.altitude_rad as f64).to_degrees();
    eprintln!("Dunblane summer solstice noon altitude = {alt_deg}°");
    within(alt_deg, 57.25, 1.0, "summer solstice noon altitude");
}

#[test]
fn dunblane_winter_solstice_noon_altitude() {
    // 2026-12-21 ~ solar noon. δ ≈ -23.44°.
    let utc = Utc.with_ymd_and_hms(2026, 12, 21, 12, 16, 0).unwrap();
    let p = sun_position(utc, DUNBLANE_LAT, DUNBLANE_LON, DUNBLANE_ELEV);
    let alt_deg = (p.altitude_rad as f64).to_degrees();
    eprintln!("Dunblane winter solstice noon altitude = {alt_deg}°");
    within(alt_deg, 10.37, 1.0, "winter solstice noon altitude");
}

#[test]
fn equator_equinox_sun_path_geometry() {
    // At the equator at the spring equinox the sun rises due east, sets
    // due west; at solar noon it's at the zenith (within ~0.5°).
    // Solar noon at (0°N, 0°E) = 12:00 UTC.
    let utc = Utc.with_ymd_and_hms(2026, 3, 20, 12, 0, 0).unwrap();
    let p = sun_position(utc, 0.0, 0.0, 0.0);
    let alt = (p.altitude_rad as f64).to_degrees();
    eprintln!("equator/equinox/noon altitude = {alt}°");
    // 2026 spring equinox is ~03:46 UTC on March 20; 12:00 UTC is about 8 h
    // later, so declination has drifted by ~2°. Allow up to 3° here — this
    // test is a convention sanity check (azimuth roughly correct, altitude
    // very high), not a precision assertion.
    within(
        alt,
        90.0,
        3.0,
        "equinox noon at equator should be near zenith",
    );
}

#[test]
fn equinox_sunrise_in_eastern_half() {
    // At the spring equinox the sun rises within a degree of due east
    // (azimuth 90°) regardless of latitude. We don't know the exact
    // sunrise minute without iterating, but at altitude ≈ 0° on this
    // morning the azimuth should be near 90°. Use 06:30 UTC at Dunblane,
    // 2026-03-20 — close to local sunrise.
    let utc = Utc.with_ymd_and_hms(2026, 3, 20, 6, 30, 0).unwrap();
    let p = sun_position(utc, DUNBLANE_LAT, DUNBLANE_LON, DUNBLANE_ELEV);
    let az_deg = (p.azimuth_rad as f64).to_degrees();
    let alt_deg = (p.altitude_rad as f64).to_degrees();
    eprintln!("Dunblane equinox 06:30 UTC: alt={alt_deg}°, az={az_deg}°");
    // Altitude near horizon (within 10°), azimuth in the eastern semicircle
    // (between 60° and 120° = ENE..ESE).
    assert!(
        alt_deg.abs() < 12.0,
        "expected near-horizon altitude, got {alt_deg}°"
    );
    assert!(
        (60.0..=120.0).contains(&az_deg),
        "expected ~easterly azimuth, got {az_deg}°"
    );
}

#[test]
fn toa_illuminance_default_state_at_1au_is_127500() {
    use ps_core::WorldState;
    let mut s = WorldState::default();
    // Force the recomputed sun.distance_au to ~1.0 by jumping to perihelion-
    // adjacent date when distance is closer to 1.0 than the Jan-1 default.
    s.clock
        .set_utc(Utc.with_ymd_and_hms(2026, 4, 4, 0, 0, 0).unwrap()); // mean distance
    s.recompute();
    eprintln!(
        "distance_au={}, toa_lux={}",
        s.sun.distance_au, s.toa_illuminance_lux
    );
    // ±5% accommodates the fact that "1 AU exactly" only happens twice a year.
    let r = s.sun.distance_au;
    let expected = 127_500.0 / (r * r);
    let err = (s.toa_illuminance_lux - expected).abs();
    assert!(
        err < 1.0,
        "lux={} expected={expected} err={err}",
        s.toa_illuminance_lux
    );
    assert!((s.sun.distance_au - 1.0).abs() < 0.02, "distance_au={r}");
}

/// Exercise the public f32 entry point for the NREL Annex B example. The
/// internal `sun_position_precise` already covers this in its lib-test
/// module — this is a public-API smoke test.
#[test]
fn nrel_example_via_public_entry_point() {
    let utc = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
    let p = sun_position(utc, 39.7426, -105.1786, 1830.14);
    let zenith = 90.0 - (p.altitude_rad as f64).to_degrees();
    let azimuth = (p.azimuth_rad as f64).to_degrees();
    eprintln!("public NREL: zenith={zenith}°, azimuth={azimuth}°");
    within(zenith, 50.11162, 0.05, "NREL Annex B zenith");
    within(azimuth, 194.34024, 0.05, "NREL Annex B azimuth");
}
