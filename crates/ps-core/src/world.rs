//! Phase 2 §2.3/2.4 — `WorldState`: clock + observer + sun + moon
//! + TOA solar illuminance.
//!
//! Replaces the Phase 1 placeholder unit struct in `contexts.rs`. Updated
//! once per frame from ps-app's main loop.

use chrono::{DateTime, TimeZone, Utc};
use glam::Vec3;

use crate::astro::{moon_position, sun_position, SunPosition};
use crate::time::WorldClock;

/// Solar illuminance at the top of Earth's atmosphere, at 1 AU, in lux.
///
/// `1361 W/m² × 93.7 lm/W` (CIE photopic luminous efficacy for the solar
/// spectrum). Plan §2.4.
pub const TOA_ILLUMINANCE_AT_1AU_LUX: f32 = 127_500.0;

/// Bundled simulated world state passed through `PrepareContext`.
#[derive(Debug, Clone)]
pub struct WorldState {
    /// Pause-aware simulated clock.
    pub clock: WorldClock,
    /// Observer latitude in degrees (north positive).
    pub latitude_deg: f64,
    /// Observer longitude in degrees (east positive).
    pub longitude_deg: f64,
    /// Observer elevation above sea level in metres.
    pub elevation_m: f64,
    /// Topocentric sun position recomputed each frame.
    pub sun: SunPosition,
    /// World-space unit vector pointing toward the sun.
    pub sun_direction_world: Vec3,
    /// TOA solar illuminance (lux) at the current sun-Earth distance.
    pub toa_illuminance_lux: f32,
    /// Topocentric moon position recomputed each frame.
    pub moon: SunPosition,
    /// World-space unit vector pointing toward the moon.
    pub moon_direction_world: Vec3,
}

impl Default for WorldState {
    fn default() -> Self {
        // J2000.0 epoch is the conventional default; observer at the equator
        // on the prime meridian. ps-app overwrites these with values from
        // pedalsky.toml at startup.
        let epoch = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap();
        let mut s = Self {
            clock: WorldClock::new(epoch),
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            elevation_m: 0.0,
            sun: SunPosition::default(),
            sun_direction_world: Vec3::Y,
            toa_illuminance_lux: TOA_ILLUMINANCE_AT_1AU_LUX,
            moon: SunPosition::default(),
            moon_direction_world: Vec3::Y,
        };
        s.recompute();
        s
    }
}

impl WorldState {
    /// Construct from initial `utc`, observer location, and elevation.
    pub fn new(
        initial_utc: DateTime<Utc>,
        latitude_deg: f64,
        longitude_deg: f64,
        elevation_m: f64,
    ) -> Self {
        let mut s = Self {
            clock: WorldClock::new(initial_utc),
            latitude_deg,
            longitude_deg,
            elevation_m,
            sun: SunPosition::default(),
            sun_direction_world: Vec3::Y,
            toa_illuminance_lux: TOA_ILLUMINANCE_AT_1AU_LUX,
            moon: SunPosition::default(),
            moon_direction_world: Vec3::Y,
        };
        s.recompute();
        s
    }

    /// Advance the clock by `real_dt_secs` and recompute sun/moon state.
    pub fn tick(&mut self, real_dt_secs: f64) {
        self.clock.tick(real_dt_secs);
        self.recompute();
    }

    /// Recompute sun/moon position + illuminance from the current
    /// simulated UTC and observer location.
    pub fn recompute(&mut self) {
        let utc = self.clock.current_utc();
        self.sun = sun_position(utc, self.latitude_deg, self.longitude_deg, self.elevation_m);
        self.sun_direction_world = altaz_to_world(self.sun.altitude_rad, self.sun.azimuth_rad);
        let inv_d = 1.0 / self.sun.distance_au.max(1e-6);
        self.toa_illuminance_lux = TOA_ILLUMINANCE_AT_1AU_LUX * inv_d * inv_d;

        self.moon = moon_position(utc, self.latitude_deg, self.longitude_deg);
        self.moon_direction_world = altaz_to_world(self.moon.altitude_rad, self.moon.azimuth_rad);
    }
}

/// Convert (altitude, azimuth-clockwise-from-north) to a world-space
/// direction in PedalSky's `+X east, +Y up, +Z south` right-handed frame.
/// Plan §2.3.
pub fn altaz_to_world(altitude_rad: f32, azimuth_rad: f32) -> Vec3 {
    let (sa, ca) = altitude_rad.sin_cos();
    let (sz, cz) = azimuth_rad.sin_cos();
    Vec3::new(ca * sz, sa, -ca * cz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altaz_north_horizon_is_minus_z() {
        let v = altaz_to_world(0.0, 0.0);
        assert!((v - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
    }

    #[test]
    fn altaz_east_horizon_is_plus_x() {
        let v = altaz_to_world(0.0, std::f32::consts::FRAC_PI_2);
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn altaz_zenith_is_plus_y() {
        let v = altaz_to_world(std::f32::consts::FRAC_PI_2, 0.0);
        assert!((v - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn toa_illuminance_at_1au_is_127500() {
        let mut s = WorldState::default();
        // Force sun.distance_au = 1.0 then recompute the derived value.
        s.sun.distance_au = 1.0;
        let inv_d = 1.0 / s.sun.distance_au;
        s.toa_illuminance_lux = TOA_ILLUMINANCE_AT_1AU_LUX * inv_d * inv_d;
        assert!((s.toa_illuminance_lux - 127_500.0).abs() < 1.0);
    }
}
