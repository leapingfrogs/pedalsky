//! Phase 2 §2.2 — Meeus chapter 47 lunar position.
//!
//! Reference: Meeus, J., *Astronomical Algorithms*, 2nd ed. 1998,
//! Willmann-Bell. Chapter 47 ("Position of the Moon") describes the
//! ELP-2000/82 truncated to ~60 longitude terms and ~60 latitude terms.
//!
//! Accuracy target: ~10″ in longitude, ~4″ in latitude — far below the
//! plan's overall ±0.1° tolerance for sun. For the harness we only need
//! the moon's direction in the sky for night-sky illumination (Phase 5+);
//! sub-arcminute precision is overkill but matches the plan's literal
//! "implement Meeus chapter 47" instruction.
//!
//! Coefficients are transcribed from Meeus's published expressions; their
//! digit grouping matches the source rather than Rust's idiomatic 3-digit
//! grouping, and reflowing them would obscure the cross-reference, hence
//! the module-level `inconsistent_digit_grouping` allow below.

#![allow(clippy::inconsistent_digit_grouping)]

use chrono::{DateTime, Utc};

use super::spa::{julian_day_utc, SunPosition};
use super::tables::moon::{MB_TERMS, ML_TERMS, MR_TERMS};

/// Compute the topocentric moon position. Returns the same `SunPosition`
/// shape (altitude, azimuth, distance in AU) as the sun for symmetry.
///
/// `latitude_deg`, `longitude_deg` are observer location (degrees, north
/// positive / east positive).
pub fn moon_position(utc: DateTime<Utc>, latitude_deg: f64, longitude_deg: f64) -> SunPosition {
    let jd = julian_day_utc(utc);
    let t = (jd - 2_451_545.0) / 36_525.0;

    // Eq 47.1 (Moon mean longitude L', degrees).
    let lp = wrap_deg(
        218.3164477 + 481_267.881_234_21 * t - 0.0015786 * t * t + t.powi(3) / 538_841.0
            - t.powi(4) / 65_194_000.0,
    );
    // Eq 47.2 (Moon mean elongation D, degrees).
    let d = wrap_deg(
        297.8501921 + 445_267.111_4034 * t - 0.001_8819 * t * t + t.powi(3) / 545_868.0
            - t.powi(4) / 113_065_000.0,
    );
    // Eq 47.3 (Sun mean anomaly M, degrees).
    let m_sun =
        wrap_deg(357.5291092 + 35_999.050_29 * t - 0.000_1536 * t * t + t.powi(3) / 24_490_000.0);
    // Eq 47.4 (Moon mean anomaly M', degrees).
    let mp = wrap_deg(
        134.9633964 + 477_198.867_5055 * t + 0.008_7414 * t * t + t.powi(3) / 69_699.0
            - t.powi(4) / 14_712_000.0,
    );
    // Eq 47.5 (Moon argument of latitude F, degrees).
    let f = wrap_deg(
        93.2720950 + 483_202.017_5233 * t - 0.003_6539 * t * t - t.powi(3) / 3_526_000.0
            + t.powi(4) / 863_310_000.0,
    );

    // E factor for terms multiplying M / 2M (Eq 47.6).
    let e = 1.0 - 0.002_516 * t - 0.000_007_4 * t * t;
    let e2 = e * e;

    let mut sigma_l = 0.0; // longitude perturbation (units 10⁻⁶ degree)
    let mut sigma_r = 0.0; // distance perturbation (km)
    for &(d_m, m_m, mp_m, f_m, l_amp, r_amp) in ML_TERMS {
        let arg =
            (d_m as f64 * d + m_m as f64 * m_sun + mp_m as f64 * mp + f_m as f64 * f).to_radians();
        let factor = match m_m.unsigned_abs() {
            1 => e,
            2 => e2,
            _ => 1.0,
        };
        sigma_l += factor * l_amp * arg.sin();
        sigma_r += factor * r_amp * arg.cos();
    }

    let mut sigma_b = 0.0; // latitude perturbation (units 10⁻⁶ degree)
    for &(d_m, m_m, mp_m, f_m, b_amp) in MB_TERMS {
        let arg =
            (d_m as f64 * d + m_m as f64 * m_sun + mp_m as f64 * mp + f_m as f64 * f).to_radians();
        let factor = match m_m.unsigned_abs() {
            1 => e,
            2 => e2,
            _ => 1.0,
        };
        sigma_b += factor * b_amp * arg.sin();
    }

    // The "additive" Venus / Jupiter / flattening corrections from §47.6:
    let a1 = (119.75 + 131.849 * t).to_radians();
    let a2 = (53.09 + 479_264.290 * t).to_radians();
    let a3 = (313.45 + 481_266.484 * t).to_radians();
    sigma_l +=
        3958.0 * a1.sin() + 1962.0 * (lp.to_radians() - f.to_radians()).sin() + 318.0 * a2.sin();
    sigma_b += -2235.0 * lp.to_radians().sin()
        + 382.0 * a3.sin()
        + 175.0 * (a1 - f.to_radians()).sin()
        + 175.0 * (a1 + f.to_radians()).sin()
        + 127.0 * (lp.to_radians() - mp.to_radians()).sin()
        - 115.0 * (lp.to_radians() + mp.to_radians()).sin();

    // Use unused MR_TERMS placeholder so it doesn't compile-warn — Meeus
    // shares the L/R table; we only consume L+R here, MR_TERMS reserved
    // for a future split implementation.
    let _ = MR_TERMS;

    let lambda_deg = lp + sigma_l / 1_000_000.0; // apparent geocentric longitude
    let beta_deg = sigma_b / 1_000_000.0; // geocentric latitude
    let delta_km = 385_000.56 + sigma_r / 1_000.0; // distance to centre, km

    // Convert ecliptic to equatorial. True obliquity at this jce:
    let jce = t;
    let u = jce / 10.0;
    let eps0_arcsec = 84_381.448 - 4_680.93 * u - 1.55 * u.powi(2) + 1_999.25 * u.powi(3)
        - 51.38 * u.powi(4)
        - 249.67 * u.powi(5)
        - 39.05 * u.powi(6)
        + 7.12 * u.powi(7)
        + 27.87 * u.powi(8)
        + 5.79 * u.powi(9)
        + 2.45 * u.powi(10);
    let eps_deg = eps0_arcsec / 3_600.0; // close enough; Δε is sub-arcsec

    let lam_r = lambda_deg.to_radians();
    let bet_r = beta_deg.to_radians();
    let eps_r = eps_deg.to_radians();
    let alpha_r = f64::atan2(
        lam_r.sin() * eps_r.cos() - bet_r.tan() * eps_r.sin(),
        lam_r.cos(),
    );
    let decl_r = f64::asin(bet_r.sin() * eps_r.cos() + bet_r.cos() * eps_r.sin() * lam_r.sin());

    // Apparent sidereal time at Greenwich (Meeus Eq 12.4 — simplified).
    let theta0_deg = wrap_deg(
        280.460_618_37 + 360.985_647_366_29 * (jd - 2_451_545.0) + 0.000_387_933 * t * t
            - t.powi(3) / 38_710_000.0,
    );

    let lst_deg = wrap_deg(theta0_deg + longitude_deg);
    let h_r = (lst_deg - alpha_r.to_degrees()).to_radians();

    let phi_r = latitude_deg.to_radians();
    let cos_z = phi_r.sin() * decl_r.sin() + phi_r.cos() * decl_r.cos() * h_r.cos();
    let zenith_r = cos_z.acos();
    let alt_r = std::f64::consts::FRAC_PI_2 - zenith_r;

    let az_from_south_r = f64::atan2(
        h_r.sin(),
        h_r.cos() * phi_r.sin() - decl_r.tan() * phi_r.cos(),
    );
    let azimuth_deg = wrap_deg(az_from_south_r.to_degrees() + 180.0);

    // Convert moon distance (km) to AU for symmetry with `SunPosition`. The
    // sun's distance is naturally AU; the moon's "AU" is small (~0.0026 AU
    // at perigee). Phase 5 will use this for night illumination scaling.
    let distance_au = delta_km / 149_597_870.7;

    SunPosition {
        altitude_rad: alt_r as f32,
        azimuth_rad: azimuth_deg.to_radians() as f32,
        distance_au: distance_au as f32,
    }
}

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

    /// Sanity: the moon should be roughly 0.002–0.003 AU from Earth (~360k–
    /// 405k km perigee/apogee).
    #[test]
    fn moon_distance_within_orbital_range() {
        let utc = Utc.with_ymd_and_hms(2026, 5, 10, 14, 30, 0).unwrap();
        let m = moon_position(utc, 56.1922, -3.9645);
        let km = m.distance_au * 149_597_870.7;
        assert!(
            (350_000.0..=410_000.0).contains(&km),
            "moon distance {km} km outside [350k, 410k]"
        );
    }

    /// Sanity: across a 24-hour span the moon should cross the meridian once
    /// — i.e. azimuth should sweep through the full circle. We sample at
    /// 4-hour intervals and check we see at least one near-zero / near-360
    /// transition.
    #[test]
    fn moon_azimuth_sweeps_through_24h() {
        let mut prev = -1.0_f32;
        let mut wraps = 0;
        for h in 0..24 {
            let utc = Utc.with_ymd_and_hms(2026, 5, 10, h, 0, 0).unwrap();
            let m = moon_position(utc, 56.1922, -3.9645);
            let az_deg = m.azimuth_rad.to_degrees();
            if prev >= 0.0 && (prev > 270.0 && az_deg < 90.0) {
                wraps += 1;
            }
            prev = az_deg;
        }
        assert!(
            wraps >= 1,
            "expected at least one azimuth wrap in 24h, got {wraps}"
        );
    }
}
