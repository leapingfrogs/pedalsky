//! Active-strike state + the two-pulse intensity envelope.

use glam::{Vec2, Vec3};

use crate::bolt::Bolt;
use crate::rng::Pcg32;

/// One active strike: where it lives in the world + its current age.
/// Bolt geometry is generated at spawn and cached here.
#[derive(Clone, Debug)]
pub struct ActiveStrike {
    /// World-space spawn point (cloud-base XZ, altitude ≈ cloud
    /// reference altitude).
    pub origin: Vec3,
    /// World-space ground attach point (y = 0).
    pub attach: Vec3,
    /// Seconds elapsed since the strike fired.
    pub age_s: f32,
    /// Pre-generated polyline (trunk + forks).
    pub bolt: Bolt,
}

/// Container for active strikes + the deterministic RNG used to spawn
/// and shape them.
pub struct StrikeStore {
    rng: Pcg32,
    active: Vec<ActiveStrike>,
    capacity: u32,
}

impl StrikeStore {
    /// Construct empty.
    pub fn new(seed: u64, capacity: u32) -> Self {
        Self {
            rng: Pcg32::new(seed),
            active: Vec::with_capacity(capacity as usize),
            capacity,
        }
    }

    /// Reseed the RNG (e.g. on reconfigure when the user changes
    /// the seed slider).
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Pcg32::new(seed);
    }

    /// Tick all active strikes forward by `dt` seconds and evict
    /// any past their lifetime.
    pub fn advance(&mut self, dt_s: f32) {
        for s in &mut self.active {
            s.age_s += dt_s;
        }
        self.active
            .retain(|s| s.age_s <= crate::STRIKE_LIFETIME_S);
    }

    /// Sample N from a Poisson with mean λ. Forwarded to the RNG.
    pub fn poisson_sample(&mut self, lambda: f32) -> u32 {
        self.rng.poisson(lambda)
    }

    /// Uniform XZ point inside the centred mask extent.
    pub fn uniform_xz_in_extent(&mut self, extent_m: f32) -> Vec2 {
        self.rng.xz_in_extent(extent_m)
    }

    /// Uniform XZ offset between two horizontal radii (used for the
    /// origin → ground attach offset).
    pub fn uniform_attach_offset(&mut self, r_min: f32, r_max: f32) -> Vec2 {
        self.rng.ring_xz(r_min, r_max)
    }

    /// Push a new strike. Evicts the oldest if at capacity.
    pub fn push(&mut self, s: ActiveStrike) {
        if self.active.len() as u32 >= self.capacity {
            self.active.remove(0);
        }
        self.active.push(s);
    }

    /// Borrow the active list.
    pub fn active(&self) -> &[ActiveStrike] {
        &self.active
    }

    /// Borrow the RNG so the bolt generator can use the same
    /// deterministic stream.
    pub fn rng_mut(&mut self) -> &mut Pcg32 {
        &mut self.rng
    }
}

/// Two-pulse intensity envelope. `t = age_s`, normalised against
/// `lifetime_s`. Returns a unitless envelope in `[0, 1]` driven by
/// fast attack → first peak → quick decay → quiet → second peak →
/// final decay (the canonical CG bolt waveform).
pub fn pulse_envelope(t: f32, lifetime_s: f32) -> f32 {
    if t < 0.0 || t > lifetime_s {
        return 0.0;
    }
    let u = t / lifetime_s;
    // First pulse centred at u = 0.10, σ = 0.04, height 1.0.
    // Second pulse centred at u = 0.55, σ = 0.10, height 0.55
    // (slower fade-out). Sum gives the two-bump waveform.
    let p1_arg = (u - 0.10) / 0.04;
    let p1 = (-(p1_arg * p1_arg)).exp();
    let p2_arg = (u - 0.55) / 0.10;
    let p2 = 0.55 * (-(p2_arg * p2_arg)).exp();
    (p1 + p2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_envelope_zero_outside_window() {
        assert_eq!(pulse_envelope(-0.1, 0.2), 0.0);
        assert_eq!(pulse_envelope(0.3, 0.2), 0.0);
    }

    #[test]
    fn pulse_envelope_has_two_peaks() {
        // First peak near t = 0.10 * lifetime
        let l = 0.2;
        let p1 = pulse_envelope(0.10 * l, l);
        assert!(p1 > 0.95, "first peak too low: {p1}");
        // Second peak near t = 0.55 * lifetime, smaller
        let p2 = pulse_envelope(0.55 * l, l);
        assert!(p2 > 0.45 && p2 < 0.65, "second peak unexpected: {p2}");
        // Trough between them
        let p_trough = pulse_envelope(0.30 * l, l);
        assert!(p_trough < 0.05, "trough too high: {p_trough}");
    }

    #[test]
    fn store_evicts_oldest_at_capacity() {
        let mut s = StrikeStore::new(1, 2);
        for i in 0..5 {
            s.push(ActiveStrike {
                origin: Vec3::splat(i as f32),
                attach: Vec3::ZERO,
                age_s: 0.0,
                bolt: crate::bolt::Bolt::default(),
            });
        }
        assert_eq!(s.active().len(), 2);
        // Last two pushed kept.
        assert_eq!(s.active()[0].origin.x, 3.0);
        assert_eq!(s.active()[1].origin.x, 4.0);
    }
}
