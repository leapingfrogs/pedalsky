//! Phase 3 §3.2.1 — visibility → Mie haze coefficient (Koschmieder).

use glam::Vec3;

/// Koschmieder constant ln(50) ≈ 3.912 (the "5 % contrast" threshold the
/// meteorological visibility V is defined against).
const KOSCHMIEDER: f32 = 3.912;

/// Compute the Mie haze extinction coefficient (per metre) from the
/// configured meteorological visibility. Distributes uniformly across RGB
/// — chromatic Mie is future work (plan §3.2.1).
///
/// Returns 0 if `visibility_m <= 0`.
pub fn haze_extinction_per_m(visibility_m: f32) -> Vec3 {
    if visibility_m <= 0.0 {
        return Vec3::ZERO;
    }
    Vec3::splat(KOSCHMIEDER / visibility_m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_km_visibility_gives_known_beta() {
        // β = 3.912 / 10000 ≈ 3.912e-4 per metre.
        let b = haze_extinction_per_m(10_000.0);
        assert!((b.x - 3.912e-4).abs() < 1e-9);
        assert_eq!(b.x, b.y);
        assert_eq!(b.x, b.z);
    }

    #[test]
    fn zero_or_negative_visibility_returns_zero() {
        assert_eq!(haze_extinction_per_m(0.0), Vec3::ZERO);
        assert_eq!(haze_extinction_per_m(-1.0), Vec3::ZERO);
    }
}
