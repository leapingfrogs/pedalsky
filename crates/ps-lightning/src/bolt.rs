//! Fractal lightning bolt geometry. Given a start (cloud origin) and
//! end (ground attach) point, generate a polyline trunk with mid-point
//! displacement and a small number of secondary forks branching off
//! the trunk.
//!
//! Output is a flat list of `BoltSegment { a, b, thickness }` entries
//! the renderer can issue as billboarded quads. Trunk segments carry
//! more thickness (and the renderer multiplies trunk emission by the
//! tuning's `bolt_peak_emission`); fork segments are thinner and
//! dimmer.

use glam::Vec3;

use crate::rng::Pcg32;

/// Number of midpoint-displacement subdivisions for the trunk.
/// 5 → 32 segments per bolt (2⁵ = 32). The displacement scale halves
/// at each level, so the perturbation stays bounded.
const TRUNK_SUBDIVISIONS: u32 = 5;

/// Initial perpendicular displacement scale at the outermost level,
/// expressed as a fraction of the trunk length. 0.10 = ±10% lateral.
const ROOT_DISPLACEMENT_FRAC: f32 = 0.10;

/// Number of secondary forks per bolt.
const FORK_COUNT: u32 = 3;

/// Thickness (m) of the trunk's billboard quad.
const TRUNK_THICKNESS_M: f32 = 6.0;

/// Thickness (m) of fork quads.
const FORK_THICKNESS_M: f32 = 3.0;

/// One billboarded segment of a bolt.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoltSegment {
    /// Segment start (world).
    pub a: Vec3,
    /// Segment end (world).
    pub b: Vec3,
    /// Quad thickness in metres.
    pub thickness: f32,
    /// Per-segment emission scalar in `[0, 1]`. Trunk = 1.0, forks
    /// taper toward 0 along their length.
    pub emission_scale: f32,
}

/// A complete bolt: trunk + forks, flattened.
#[derive(Clone, Debug, Default)]
pub struct Bolt {
    /// All segments in draw order. Trunk first, then forks. The
    /// renderer doesn't distinguish between them at render time.
    pub segments: Vec<BoltSegment>,
}

/// Generate a bolt from `origin` (cloud base) to `attach` (ground).
pub fn generate_bolt(origin: Vec3, attach: Vec3, rng: &mut Pcg32) -> Bolt {
    let mut trunk = vec![origin, attach];
    let total_len = (attach - origin).length().max(1.0);

    // Midpoint displacement: at each level, insert a new vertex at
    // every existing midpoint, displaced perpendicular to the local
    // segment direction by a Gaussian step that halves at each
    // subdivision.
    let mut amplitude = total_len * ROOT_DISPLACEMENT_FRAC;
    for _ in 0..TRUNK_SUBDIVISIONS {
        let mut next = Vec::with_capacity(trunk.len() * 2 - 1);
        for i in 0..trunk.len() - 1 {
            let a = trunk[i];
            let b = trunk[i + 1];
            let mid = (a + b) * 0.5;
            let dir = (b - a).normalize_or_zero();
            let perp = rng.orthogonal_unit(dir);
            let disp = perp * amplitude * rng.normal();
            next.push(a);
            next.push(mid + disp);
        }
        next.push(*trunk.last().unwrap());
        trunk = next;
        amplitude *= 0.5;
    }

    let mut segments = Vec::with_capacity(trunk.len() - 1 + FORK_COUNT as usize * 6);
    for i in 0..trunk.len() - 1 {
        segments.push(BoltSegment {
            a: trunk[i],
            b: trunk[i + 1],
            thickness: TRUNK_THICKNESS_M,
            emission_scale: 1.0,
        });
    }

    // Forks branch off random trunk vertices, deviate by ~30°, and
    // run for ~20–40% of the trunk length. Each fork gets its own
    // shorter midpoint subdivision.
    for _ in 0..FORK_COUNT {
        let attach_idx = (rng.next_f32() * (trunk.len() - 2) as f32) as usize + 1;
        let fork_start = trunk[attach_idx];
        let trunk_dir = (attach - origin).normalize_or_zero();
        let perp = rng.orthogonal_unit(trunk_dir);
        // Bias the fork dominantly downward + sideways.
        let down = Vec3::new(0.0, -1.0, 0.0);
        let fork_dir =
            (down * 0.6 + trunk_dir * 0.2 + perp * 0.4 * rng.range_f32(-1.0, 1.0))
                .normalize_or_zero();
        let fork_len = total_len * rng.range_f32(0.20, 0.40);
        let fork_end = fork_start + fork_dir * fork_len;
        // Quick 2-level subdivision for forks.
        let mut fork_pts = vec![fork_start, fork_end];
        let mut fork_amp = fork_len * ROOT_DISPLACEMENT_FRAC;
        for _ in 0..3 {
            let mut next = Vec::with_capacity(fork_pts.len() * 2 - 1);
            for i in 0..fork_pts.len() - 1 {
                let a = fork_pts[i];
                let b = fork_pts[i + 1];
                let mid = (a + b) * 0.5;
                let dir = (b - a).normalize_or_zero();
                let perp = rng.orthogonal_unit(dir);
                let disp = perp * fork_amp * rng.normal();
                next.push(a);
                next.push(mid + disp);
            }
            next.push(*fork_pts.last().unwrap());
            fork_pts = next;
            fork_amp *= 0.5;
        }
        let fork_seg_count = (fork_pts.len() - 1) as f32;
        for (i, win) in fork_pts.windows(2).enumerate() {
            // Forks taper toward zero emission at the tip.
            let t = i as f32 / fork_seg_count;
            let emission_scale = 0.55 * (1.0 - t);
            segments.push(BoltSegment {
                a: win[0],
                b: win[1],
                thickness: FORK_THICKNESS_M * (1.0 - 0.5 * t),
                emission_scale,
            });
        }
    }

    Bolt { segments }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_bolt_has_trunk_and_forks() {
        let mut rng = Pcg32::new(42);
        let bolt = generate_bolt(
            Vec3::new(0.0, 1500.0, 0.0),
            Vec3::new(2000.0, 0.0, 1000.0),
            &mut rng,
        );
        // Trunk: 2^TRUNK_SUBDIVISIONS = 32 segments minimum.
        let trunk_seg_count = 1 << TRUNK_SUBDIVISIONS;
        assert!(bolt.segments.len() > trunk_seg_count);
        // First segment starts at origin.
        let first = bolt.segments.first().unwrap();
        assert!((first.a - Vec3::new(0.0, 1500.0, 0.0)).length() < 1.0e-3);
        // All segments have positive thickness.
        for s in &bolt.segments {
            assert!(s.thickness > 0.0, "non-positive thickness");
        }
    }

    #[test]
    fn generate_bolt_is_deterministic_for_seed() {
        let mut rng_a = Pcg32::new(9);
        let mut rng_b = Pcg32::new(9);
        let a = generate_bolt(Vec3::ZERO, Vec3::new(1000.0, -1500.0, 500.0), &mut rng_a);
        let b = generate_bolt(Vec3::ZERO, Vec3::new(1000.0, -1500.0, 500.0), &mut rng_b);
        assert_eq!(a.segments.len(), b.segments.len());
        for (sa, sb) in a.segments.iter().zip(b.segments.iter()) {
            assert!((sa.a - sb.a).length() < 1.0e-5);
            assert!((sa.b - sb.b).length() < 1.0e-5);
        }
    }
}
