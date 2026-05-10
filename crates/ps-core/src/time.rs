//! Phase 2 §2.1 — `WorldClock`: simulated wall-clock + pause-aware
//! `simulated_seconds` accumulator.
//!
//! `current_utc()` returns the simulated UTC. `simulated_seconds` advances
//! only when not paused; this is what cloud and rain shaders use, so
//! pausing the clock freezes evolution while still letting the renderer
//! redraw at the same UTC.

use chrono::{DateTime, Duration, Utc};

/// World clock: a simulated UTC clock plus a pause-aware second accumulator.
///
/// `time_scale` multiplies real wall-clock seconds when `tick()` advances
/// the simulated time. Setting `paused = true` freezes both the simulated
/// UTC and `simulated_seconds`.
#[derive(Debug, Clone)]
pub struct WorldClock {
    /// Current simulated UTC.
    epoch: DateTime<Utc>,
    /// Multiplier applied to real wall-clock seconds when `tick()` is called.
    /// `1.0` = real-time; `60.0` = 1 simulated minute per real second.
    time_scale: f64,
    /// When `true`, `tick()` does nothing.
    paused: bool,
    /// Pause-aware accumulated simulated seconds since this clock was created.
    simulated_seconds: f64,
}

impl WorldClock {
    /// Construct a clock starting at `initial_utc`.
    pub fn new(initial_utc: DateTime<Utc>) -> Self {
        Self {
            epoch: initial_utc,
            time_scale: 1.0,
            paused: false,
            simulated_seconds: 0.0,
        }
    }

    /// Advance the clock by `real_dt_secs` of wall-clock time.
    ///
    /// If `paused`, this is a no-op. Otherwise advances both the simulated
    /// UTC and `simulated_seconds` by `real_dt_secs * time_scale`.
    pub fn tick(&mut self, real_dt_secs: f64) {
        if self.paused {
            return;
        }
        let scaled = real_dt_secs * self.time_scale;
        self.simulated_seconds += scaled;
        // Convert to nanoseconds for chrono Duration; clamp to i64 range
        // (a single tick of >292 years would overflow but isn't reachable
        // from any sensible time_scale).
        let nanos = (scaled * 1_000_000_000.0) as i64;
        self.epoch += Duration::nanoseconds(nanos);
    }

    /// Set the simulated UTC directly. Does not change `simulated_seconds`
    /// (which represents elapsed simulated time, not wall-clock UTC).
    pub fn set_utc(&mut self, utc: DateTime<Utc>) {
        self.epoch = utc;
    }

    /// Set the time-scale multiplier.
    pub fn set_time_scale(&mut self, scale: f64) {
        self.time_scale = scale;
    }

    /// Pause / unpause.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Whether the clock is currently paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Current time-scale multiplier.
    pub fn time_scale(&self) -> f64 {
        self.time_scale
    }

    /// Current simulated UTC.
    pub fn current_utc(&self) -> DateTime<Utc> {
        self.epoch
    }

    /// Pause-aware accumulated simulated seconds.
    pub fn simulated_seconds(&self) -> f64 {
        self.simulated_seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn tick_advances_when_not_paused() {
        let mut c = WorldClock::new(Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 0).unwrap());
        c.tick(1.0);
        c.tick(0.5);
        assert!((c.simulated_seconds() - 1.5).abs() < 1e-9);
        assert_eq!(
            c.current_utc(),
            Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 1).unwrap()
                + Duration::nanoseconds(500_000_000)
        );
    }

    #[test]
    fn pause_freezes_both_utc_and_simulated_seconds() {
        let start = Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 0).unwrap();
        let mut c = WorldClock::new(start);
        c.tick(2.0);
        c.set_paused(true);
        c.tick(10.0);
        assert!((c.simulated_seconds() - 2.0).abs() < 1e-9);
        assert_eq!(
            c.current_utc(),
            Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 2).unwrap()
        );
        c.set_paused(false);
        c.tick(1.0);
        assert!((c.simulated_seconds() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn time_scale_multiplies_advance() {
        let mut c = WorldClock::new(Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 0).unwrap());
        c.set_time_scale(60.0); // 1 real second = 1 simulated minute
        c.tick(1.0);
        assert!((c.simulated_seconds() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn set_utc_does_not_reset_simulated_seconds() {
        let mut c = WorldClock::new(Utc.with_ymd_and_hms(2026, 5, 10, 12, 0, 0).unwrap());
        c.tick(5.0);
        c.set_utc(Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap()); // jump
        assert!((c.simulated_seconds() - 5.0).abs() < 1e-9);
    }
}
