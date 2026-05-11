//! Deterministic seedable PRNG. PCG XSH-RR (32-bit output, 64-bit
//! state) — small, fast, good enough quality for stochastic strike
//! placement and bolt geometry. Not cryptographic.

use glam::{Vec2, Vec3};

/// PCG-XSH-RR 64→32 generator.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    /// Construct from a seed. Zero seed is allowed; we mix in a
    /// non-zero stream constant so the first output isn't all-zero.
    pub fn new(seed: u64) -> Self {
        let mut s = Self {
            state: seed.wrapping_add(0x6A09E667F3BCC908),
        };
        // Discard a few outputs — PCG's first output after seeding
        // tends to correlate weakly with the seed.
        s.next_u32();
        s.next_u32();
        s
    }

    /// 32-bit unsigned uniform.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        // PCG multiplier (LCG step).
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // XSH-RR output.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// `[0, 1)` uniform.
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits → exact f32 mantissa width.
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform in `[lo, hi)`.
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Standard normal via Box–Muller. Cheaper than rejection given
    /// we only need a few per strike.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1.0e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        r * theta.cos()
    }

    /// Poisson sample with mean λ. Knuth's algorithm — fine for the
    /// small λ this subsystem uses (typical: 0..2 strikes per frame).
    pub fn poisson(&mut self, lambda: f32) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        let l = (-lambda).exp();
        let mut k = 0_u32;
        let mut p = 1.0_f32;
        loop {
            k += 1;
            p *= self.next_f32();
            if p <= l {
                return k - 1;
            }
            // Safety bail — Knuth's algorithm degrades for large λ
            // and we don't expect λ > 100 for any sane scene rate.
            if k > 1000 {
                return k - 1;
            }
        }
    }

    /// Uniform XZ point inside a centred extent.
    pub fn xz_in_extent(&mut self, extent_m: f32) -> Vec2 {
        let half = extent_m * 0.5;
        Vec2::new(self.range_f32(-half, half), self.range_f32(-half, half))
    }

    /// Uniform XZ offset between two horizontal radii.
    pub fn ring_xz(&mut self, r_min: f32, r_max: f32) -> Vec2 {
        let theta = self.range_f32(0.0, std::f32::consts::TAU);
        // Area-uniform: r = sqrt(uniform(r_min², r_max²))
        let r2 = self.range_f32(r_min * r_min, r_max * r_max);
        let r = r2.sqrt();
        Vec2::new(r * theta.cos(), r * theta.sin())
    }

    /// Random unit vector orthogonal to `n`. Used when displacing a
    /// midpoint perpendicular to a polyline segment.
    pub fn orthogonal_unit(&mut self, n: Vec3) -> Vec3 {
        // Pick any axis not parallel to n.
        let helper = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let perp = n.cross(helper).normalize_or_zero();
        // Rotate `perp` by a random angle around `n`.
        let theta = self.range_f32(0.0, std::f32::consts::TAU);
        let (s, c) = theta.sin_cos();
        let perp2 = n.cross(perp);
        (perp * c + perp2 * s).normalize_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_is_deterministic() {
        let mut a = Pcg32::new(42);
        let mut b = Pcg32::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn pcg_next_f32_in_unit_range() {
        let mut r = Pcg32::new(7);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn poisson_zero_lambda_returns_zero() {
        let mut r = Pcg32::new(1);
        for _ in 0..50 {
            assert_eq!(r.poisson(0.0), 0);
        }
    }

    #[test]
    fn poisson_mean_matches_lambda() {
        let mut r = Pcg32::new(1);
        let lambda = 2.5_f32;
        let n = 5_000;
        let total: u32 = (0..n).map(|_| r.poisson(lambda)).sum();
        let mean = total as f32 / n as f32;
        // Generous tolerance — 5000 samples → SE ≈ √(2.5/5000) ≈ 0.022
        assert!((mean - lambda).abs() < 0.1, "mean={mean} expected ≈ {lambda}");
    }
}
