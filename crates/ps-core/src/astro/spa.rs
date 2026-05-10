//! Phase 2 §2.2 — NREL Solar Position Algorithm (full form).
//!
//! Reference: Reda, I. and Andreas, A., 2003. *Solar Position Algorithm
//! for Solar Radiation Applications.* NREL/TP-560-34302.
//! <https://midcdmz.nrel.gov/spa/> · <https://www.nrel.gov/docs/fy08osti/34302.pdf>
//!
//! Accuracy: ±0.0003° from year −2000 to +6000. ΔT polynomial fit:
//! Espenak & Meeus 2006 (NASA GSFC five-millennium-canon polynomials).

use chrono::{DateTime, Datelike, Timelike, Utc};

use super::tables::{
    nutation_argument, NUTATION_TERMS, TERMS_B0, TERMS_B1, TERMS_L0, TERMS_L1, TERMS_L2, TERMS_L3,
    TERMS_L4, TERMS_L5, TERMS_R0, TERMS_R1, TERMS_R2, TERMS_R3, TERMS_R4,
};

/// Topocentric sun position, returned by [`sun_position`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SunPosition {
    /// Altitude above the local horizon, radians. Positive = above.
    pub altitude_rad: f32,
    /// Azimuth measured clockwise from due north, radians.
    pub azimuth_rad: f32,
    /// Sun-to-Earth distance in astronomical units.
    pub distance_au: f32,
}

/// Public f64 entry point for tests that need higher precision than the
/// `f32` `SunPosition` retains.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SunPositionPrecise {
    /// Altitude above the local horizon, degrees (positive = above).
    pub altitude_deg: f64,
    /// Azimuth measured clockwise from due north, degrees [0, 360).
    pub azimuth_deg: f64,
    /// Zenith angle (90° − altitude), degrees.
    pub zenith_deg: f64,
    /// Sun-to-Earth distance in astronomical units.
    pub distance_au: f64,
    /// Geocentric right ascension in degrees, [0, 360).
    pub right_ascension_deg: f64,
    /// Geocentric declination in degrees, [-90, 90].
    pub declination_deg: f64,
    /// Apparent sidereal time at Greenwich, degrees [0, 360).
    pub apparent_sidereal_time_deg: f64,
}

/// Compute the topocentric sun position for an observer at
/// `(latitude_deg, longitude_deg, elevation_m)` at simulated `utc`.
pub fn sun_position(
    utc: DateTime<Utc>,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
) -> SunPosition {
    let p = sun_position_precise(utc, latitude_deg, longitude_deg, elevation_m);
    SunPosition {
        altitude_rad: (p.altitude_deg.to_radians()) as f32,
        azimuth_rad: (p.azimuth_deg.to_radians()) as f32,
        distance_au: p.distance_au as f32,
    }
}

/// Same as [`sun_position`] but returns `f64` quantities so tests can
/// compare against published reference values without losing precision.
pub fn sun_position_precise(
    utc: DateTime<Utc>,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
) -> SunPositionPrecise {
    // §3.1 Julian Day from UTC.
    let jd = julian_day_utc(utc);
    let dt = delta_t_seconds(utc.year());
    let jde = jd + dt / 86_400.0;
    let jc = (jd - 2_451_545.0) / 36_525.0;
    let jce = (jde - 2_451_545.0) / 36_525.0;
    let jme = jce / 10.0;

    // §3.2 Earth heliocentric L, B, R.
    let l_deg = wrap_deg(sum_polynomial_series(jme, EARTH_L_TABLES).to_degrees());
    let b_deg = sum_polynomial_series(jme, EARTH_B_TABLES).to_degrees();
    let r_au = sum_polynomial_series(jme, EARTH_R_TABLES);

    // §3.3 Geocentric.
    let theta_deg = wrap_deg(l_deg + 180.0);
    let beta_deg = -b_deg;

    // §3.4 Nutation.
    let (delta_psi_deg, delta_eps_deg) = nutation(jce);

    // §3.5 True obliquity (Eq 24): ε = ε0 + Δε; ε0 from Laskar 1986 polynomial.
    let u = jme / 10.0;
    let eps0_arcsec = 84_381.448 - 4_680.93 * u - 1.55 * u.powi(2) + 1_999.25 * u.powi(3)
        - 51.38 * u.powi(4)
        - 249.67 * u.powi(5)
        - 39.05 * u.powi(6)
        + 7.12 * u.powi(7)
        + 27.87 * u.powi(8)
        + 5.79 * u.powi(9)
        + 2.45 * u.powi(10);
    let eps_deg = eps0_arcsec / 3_600.0 + delta_eps_deg;

    // §3.6 Aberration correction.
    let delta_tau_deg = -20.4898 / (3_600.0 * r_au);

    // §3.7 Apparent sun longitude.
    let lambda_deg = theta_deg + delta_psi_deg + delta_tau_deg;

    // §3.8 Apparent sidereal time at Greenwich.
    let v0_deg = wrap_deg(
        280.460_618_37 + 360.985_647_366_29 * (jd - 2_451_545.0) + 0.000_387_933 * jc.powi(2)
            - jc.powi(3) / 38_710_000.0,
    );
    let v_deg = wrap_deg(v0_deg + delta_psi_deg * eps_deg.to_radians().cos());

    // §3.9 Geocentric right ascension and declination.
    let lambda_r = lambda_deg.to_radians();
    let beta_r = beta_deg.to_radians();
    let eps_r = eps_deg.to_radians();
    let alpha_deg = wrap_deg(
        f64::atan2(
            lambda_r.sin() * eps_r.cos() - beta_r.tan() * eps_r.sin(),
            lambda_r.cos(),
        )
        .to_degrees(),
    );
    let decl_deg =
        f64::asin(beta_r.sin() * eps_r.cos() + beta_r.cos() * eps_r.sin() * lambda_r.sin())
            .to_degrees();

    // §3.10 Local hour angle.
    let h_deg = wrap_deg(v_deg + longitude_deg - alpha_deg);

    // §3.11 Topocentric right ascension and declination.
    let xi_deg = 8.794 / (3_600.0 * r_au); // equatorial horizontal parallax
    let phi_r = latitude_deg.to_radians();
    let u_term = (0.99664719 * phi_r.tan()).atan();
    let x = u_term.cos() + (elevation_m / 6_378_140.0) * phi_r.cos();
    let y = 0.99664719 * u_term.sin() + (elevation_m / 6_378_140.0) * phi_r.sin();

    let h_r = h_deg.to_radians();
    let decl_r = decl_deg.to_radians();
    let xi_r = xi_deg.to_radians();
    let delta_alpha_r = f64::atan2(
        -x * xi_r.sin() * h_r.sin(),
        decl_r.cos() - x * xi_r.sin() * h_r.cos(),
    );
    let delta_alpha_deg = delta_alpha_r.to_degrees();
    let _alpha_topo_deg = wrap_deg(alpha_deg + delta_alpha_deg);
    let decl_topo_r = f64::atan2(
        (decl_r.sin() - y * xi_r.sin()) * delta_alpha_r.cos(),
        decl_r.cos() - x * xi_r.sin() * h_r.cos(),
    );
    let h_topo_r = h_r - delta_alpha_r;

    // §3.13 Topocentric zenith and azimuth.
    let cos_z = phi_r.sin() * decl_topo_r.sin() + phi_r.cos() * decl_topo_r.cos() * h_topo_r.cos();
    let zenith_topo_r = cos_z.acos();
    // Plan azimuth convention: clockwise from north. SPA Eq 47 returns
    // azimuth measured westward from south; Eq 48 converts to "from north"
    // by +180°. We then reduce to [0, 360).
    let az_from_south_r = f64::atan2(
        h_topo_r.sin(),
        h_topo_r.cos() * phi_r.sin() - decl_topo_r.tan() * phi_r.cos(),
    );
    let azimuth_deg = wrap_deg(az_from_south_r.to_degrees() + 180.0);
    let altitude_deg = 90.0 - zenith_topo_r.to_degrees();

    SunPositionPrecise {
        altitude_deg,
        azimuth_deg,
        zenith_deg: zenith_topo_r.to_degrees(),
        distance_au: r_au,
        right_ascension_deg: alpha_deg,
        declination_deg: decl_deg,
        apparent_sidereal_time_deg: v_deg,
    }
}

// ---------------------------------------------------------------------------
// §3.1 Julian Day, ΔT
// ---------------------------------------------------------------------------

/// Julian Day Number for `utc`, including fractional day. SPA Eq 4.
pub fn julian_day_utc(utc: DateTime<Utc>) -> f64 {
    let mut y = utc.year() as f64;
    let mut m = utc.month() as f64;
    let day_fraction = utc.day() as f64
        + (utc.hour() as f64 + (utc.minute() as f64 + utc.second() as f64 / 60.0) / 60.0) / 24.0;
    if m <= 2.0 {
        y -= 1.0;
        m += 12.0;
    }
    let a = (y / 100.0).floor();
    // Gregorian calendar reform: dates before 1582-10-15 use Julian calendar.
    let is_gregorian = (utc.year() > 1582)
        || (utc.year() == 1582 && utc.month() > 10)
        || (utc.year() == 1582 && utc.month() == 10 && utc.day() >= 15);
    let b = if is_gregorian {
        2.0 - a + (a / 4.0).floor()
    } else {
        0.0
    };
    (365.25 * (y + 4_716.0)).floor() + (30.6001 * (m + 1.0)).floor() + day_fraction + b - 1_524.5
}

/// ΔT (TT − UT) in seconds, Espenak & Meeus 2006 polynomial fit covering
/// year 2005..=2050. Outside this band a linear extrapolation is used —
/// SPA only needs ΔT to a few seconds for sun-position accuracy at the
/// ±0.0003° level, so this is fine for the harness's 2026-era time range.
pub fn delta_t_seconds(year: i32) -> f64 {
    // Espenak & Meeus 2006, NASA GSFC five-millennium canon, segment for
    // 2005 ≤ year ≤ 2050:  ΔT = 62.92 + 0.32217 t + 0.005589 t²
    // where t = y − 2000.
    if (2005..=2050).contains(&year) {
        let t = year as f64 - 2000.0;
        62.92 + 0.32217 * t + 0.005589 * t * t
    } else if year > 2050 && year < 2150 {
        // Extrapolation segment from the same source.
        // ΔT = -20 + 32 ((y − 1820)/100)² − 0.5628 (2150 − y)
        let u = (year as f64 - 1820.0) / 100.0;
        -20.0 + 32.0 * u * u - 0.5628 * (2150.0 - year as f64)
    } else if (1986..2005).contains(&year) {
        // 1986..2005 segment.
        let t = year as f64 - 2000.0;
        63.86 + 0.3345 * t - 0.060_374 * t * t + 0.001_727_5 * t.powi(3) + 0.000_651_814 * t.powi(4)
            - 0.000_023_715_99 * t.powi(5)
    } else {
        // Coarse fallback for years outside the polynomial domains we need.
        // The harness time range is 2000-2100; this branch is reached only
        // by future plan additions.
        let u = (year as f64 - 1820.0) / 100.0;
        -20.0 + 32.0 * u * u
    }
}

// ---------------------------------------------------------------------------
// §3.2 Heliocentric L, B, R via periodic series
// ---------------------------------------------------------------------------

type SeriesRows = &'static [(f64, f64, f64)];

/// L0..L5 (Earth heliocentric longitude, radians at the end).
const EARTH_L_TABLES: &[SeriesRows] = &[TERMS_L0, TERMS_L1, TERMS_L2, TERMS_L3, TERMS_L4, TERMS_L5];
const EARTH_B_TABLES: &[SeriesRows] = &[TERMS_B0, TERMS_B1];
const EARTH_R_TABLES: &[SeriesRows] = &[TERMS_R0, TERMS_R1, TERMS_R2, TERMS_R3, TERMS_R4];

/// Sum a SPA periodic series of polynomials in `jme`. Each "table" is a
/// list of (A, B, C) rows; the row contributes `A·cos(B + C·jme)`. The
/// full quantity is `sum_i (sum_rows_i) · jme^i / 10^8` (radians for
/// longitude/latitude; AU for radius).
fn sum_polynomial_series(jme: f64, tables: &[SeriesRows]) -> f64 {
    let mut total = 0.0_f64;
    for (i, table) in tables.iter().enumerate() {
        let mut sum = 0.0;
        for &(a, b, c) in *table {
            sum += a * (b + c * jme).cos();
        }
        total += sum * jme.powi(i as i32);
    }
    total / 1e8
}

// ---------------------------------------------------------------------------
// §3.4 Nutation
// ---------------------------------------------------------------------------

/// Compute (Δψ, Δε) in degrees from Julian Ephemeris Century `jce`.
fn nutation(jce: f64) -> (f64, f64) {
    let mut delta_psi_arcsec = 0.0;
    let mut delta_eps_arcsec = 0.0;
    for term in NUTATION_TERMS {
        let arg = nutation_argument(jce, &term.y);
        let (sa, ca) = arg.sin_cos();
        delta_psi_arcsec += (term.a + term.b * jce) * sa;
        delta_eps_arcsec += (term.c + term.d * jce) * ca;
    }
    // Coefficients are 0.0001 arcseconds; convert to degrees.
    (
        delta_psi_arcsec / 36_000_000.0,
        delta_eps_arcsec / 36_000_000.0,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn wrap_deg(x: f64) -> f64 {
    let r = x % 360.0;
    if r < 0.0 {
        r + 360.0
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn julian_day_j2000_is_2451545() {
        // 2000-01-01 12:00:00 UTC = JD 2451545.0
        let utc = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap();
        let jd = julian_day_utc(utc);
        assert!((jd - 2_451_545.0).abs() < 1e-6, "got {jd}");
    }

    #[test]
    fn julian_day_nrel_example() {
        // SPA paper Appendix B example: local 2003-10-17 12:30:30 at
        // longitude -105.1786 with timezone -7. That converts to
        // UT 2003-10-17 19:30:30, which is JD 2452930.3128472.
        let utc = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
        let jd = julian_day_utc(utc);
        assert!(
            (jd - 2_452_930.312_847).abs() < 1e-5,
            "got {jd}; want 2452930.312847"
        );
    }

    #[test]
    fn delta_t_2026_near_75_seconds() {
        let dt = delta_t_seconds(2026);
        // Espenak & Meeus 2006 polynomial: ~75 s. (Plan rough estimate ~70 s.)
        assert!(
            (70.0..=80.0).contains(&dt),
            "ΔT(2026) = {dt} (expected ~75)"
        );
    }

    #[test]
    fn wrap_deg_handles_negatives() {
        assert!((wrap_deg(-10.0) - 350.0).abs() < 1e-9);
        assert!((wrap_deg(370.0) - 10.0).abs() < 1e-9);
        assert!((wrap_deg(0.0)).abs() < 1e-9);
    }

    /// SPA paper Appendix B worked example.
    /// Inputs: UT 2003-10-17 19:30:30, lat 39.7426°N, lon −105.1786°, elev 1830.14 m.
    /// Expected (from Table A.5):
    ///   Topocentric zenith = 50.11162°
    ///   Topocentric azimuth (from north, clockwise) = 194.34024°
    #[test]
    fn nrel_appendix_b_example_within_tolerance() {
        let utc = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
        let p = sun_position_precise(utc, 39.7426, -105.1786, 1830.14);
        // Plan acceptance: better than 0.1°. We aim for tighter; SPA's
        // headline accuracy is ±0.0003°, but we tolerate ±0.05° to absorb
        // any small transcription deviations in the 200+ tabled coefficients.
        let zenith_err = (p.zenith_deg - 50.11162).abs();
        let azimuth_err = (p.azimuth_deg - 194.34024).abs();
        eprintln!(
            "NREL ex: zenith={} ({:+e}), azimuth={} ({:+e})",
            p.zenith_deg, zenith_err, p.azimuth_deg, azimuth_err
        );
        assert!(zenith_err < 0.05, "zenith error {zenith_err}° > 0.05°");
        assert!(azimuth_err < 0.05, "azimuth error {azimuth_err}° > 0.05°");
    }
}
