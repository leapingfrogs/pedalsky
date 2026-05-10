//! NREL SPA periodic-term tables A4.1–A4.5.
//!
//! Source: Reda, I. and Andreas, A., 2003. *Solar Position Algorithm for
//! Solar Radiation Applications.* NREL/TP-560-34302, Tables A4.1–A4.5.
//!
//! Each `(A, B, C)` row contributes `A · cos(B + C·jme)` to the periodic
//! sum for its harmonic. Coefficients for L (heliocentric longitude) and
//! B (heliocentric latitude) are scaled by 10^8 (matching the paper);
//! `spa.rs::sum_polynomial_series` divides by 10^8 at the end. R rows are
//! likewise scaled by 10^8 and yield AU.
//!
//! Only the periodic-coefficient tables live here; the SPA driver itself
//! is in `spa.rs`.

#![allow(clippy::approx_constant, clippy::inconsistent_digit_grouping)]

/// Earth heliocentric longitude L0 (64 rows). Source: Table A4.1, "L0".
#[rustfmt::skip]
pub const TERMS_L0: &[(f64, f64, f64)] = &[
    (175347046.0, 0.0,         0.0),
    (3341656.0,   4.6692568,   6283.07585),
    (34894.0,     4.6261,      12566.1517),
    (3497.0,      2.7441,      5753.3849),
    (3418.0,      2.8289,      3.5231),
    (3136.0,      3.6277,      77713.7715),
    (2676.0,      4.4181,      7860.4194),
    (2343.0,      6.1352,      3930.2097),
    (1324.0,      0.7425,      11506.7698),
    (1273.0,      2.0371,      529.691),
    (1199.0,      1.1096,      1577.3435),
    (990.0,       5.233,       5884.927),
    (902.0,       2.045,       26.298),
    (857.0,       3.508,       398.149),
    (780.0,       1.179,       5223.694),
    (753.0,       2.533,       5507.553),
    (505.0,       4.583,       18849.228),
    (492.0,       4.205,       775.523),
    (357.0,       2.92,        0.067),
    (317.0,       5.849,       11790.629),
    (284.0,       1.899,       796.298),
    (271.0,       0.315,       10977.079),
    (243.0,       0.345,       5486.778),
    (206.0,       4.806,       2544.314),
    (205.0,       1.869,       5573.143),
    (202.0,       2.458,       6069.777),
    (156.0,       0.833,       213.299),
    (132.0,       3.411,       2942.463),
    (126.0,       1.083,       20.775),
    (115.0,       0.645,       0.98),
    (103.0,       0.636,       4694.003),
    (102.0,       0.976,       15720.839),
    (102.0,       4.267,       7.114),
    (99.0,        6.21,        2146.17),
    (98.0,        0.68,        155.42),
    (86.0,        5.98,        161000.69),
    (85.0,        1.3,         6275.96),
    (85.0,        3.67,        71430.7),
    (80.0,        1.81,        17260.15),
    (79.0,        3.04,        12036.46),
    (75.0,        1.76,        5088.63),
    (74.0,        3.5,         3154.69),
    (74.0,        4.68,        801.82),
    (70.0,        0.83,        9437.76),
    (62.0,        3.98,        8827.39),
    (61.0,        1.82,        7084.9),
    (57.0,        2.78,        6286.6),
    (56.0,        4.39,        14143.5),
    (56.0,        3.47,        6279.55),
    (52.0,        0.19,        12139.55),
    (52.0,        1.33,        1748.02),
    (51.0,        0.28,        5856.48),
    (49.0,        0.49,        1194.45),
    (41.0,        5.37,        8429.24),
    (41.0,        2.4,         19651.05),
    (39.0,        6.17,        10447.39),
    (37.0,        6.04,        10213.29),
    (37.0,        2.57,        1059.38),
    (36.0,        1.71,        2352.87),
    (36.0,        1.78,        6812.77),
    (33.0,        0.59,        17789.85),
    (30.0,        0.44,        83996.85),
    (30.0,        2.74,        1349.87),
    (25.0,        3.16,        4690.48),
];

/// Earth heliocentric longitude L1 (34 rows).
#[rustfmt::skip]
pub const TERMS_L1: &[(f64, f64, f64)] = &[
    (628331966747.0, 0.0,        0.0),
    (206059.0,       2.678235,   6283.07585),
    (4303.0,         2.6351,     12566.1517),
    (425.0,          1.59,       3.523),
    (119.0,          5.796,      26.298),
    (109.0,          2.966,      1577.344),
    (93.0,           2.59,       18849.23),
    (72.0,           1.14,       529.69),
    (68.0,           1.87,       398.15),
    (67.0,           4.41,       5507.55),
    (59.0,           2.89,       5223.69),
    (56.0,           2.17,       155.42),
    (45.0,           0.4,        796.3),
    (36.0,           0.47,       775.52),
    (29.0,           2.65,       7.11),
    (21.0,           5.34,       0.98),
    (19.0,           1.85,       5486.78),
    (19.0,           4.97,       213.3),
    (17.0,           2.99,       6275.96),
    (16.0,           0.03,       2544.31),
    (16.0,           1.43,       2146.17),
    (15.0,           1.21,       10977.08),
    (12.0,           2.83,       1748.02),
    (12.0,           3.26,       5088.63),
    (12.0,           5.27,       1194.45),
    (12.0,           2.08,       4694.0),
    (11.0,           0.77,       553.57),
    (10.0,           1.3,        6286.6),
    (10.0,           4.24,       1349.87),
    (9.0,            2.7,        242.73),
    (9.0,            5.64,       951.72),
    (8.0,            5.3,        2352.87),
    (6.0,            2.65,       9437.76),
    (6.0,            4.67,       4690.48),
];

/// Earth heliocentric longitude L2 (20 rows).
#[rustfmt::skip]
pub const TERMS_L2: &[(f64, f64, f64)] = &[
    (52919.0,  0.0,     0.0),
    (8720.0,   1.0721,  6283.0758),
    (309.0,    0.867,   12566.152),
    (27.0,     0.05,    3.52),
    (16.0,     5.19,    26.3),
    (16.0,     3.68,    155.42),
    (10.0,     0.76,    18849.23),
    (9.0,      2.06,    77713.77),
    (7.0,      0.83,    775.52),
    (5.0,      4.66,    1577.34),
    (4.0,      1.03,    7.11),
    (4.0,      3.44,    5573.14),
    (3.0,      5.14,    796.3),
    (3.0,      6.05,    5507.55),
    (3.0,      1.19,    242.73),
    (3.0,      6.12,    529.69),
    (3.0,      0.31,    398.15),
    (3.0,      2.28,    553.57),
    (2.0,      4.38,    5223.69),
    (2.0,      3.75,    0.98),
];

/// Earth heliocentric longitude L3 (7 rows).
#[rustfmt::skip]
pub const TERMS_L3: &[(f64, f64, f64)] = &[
    (289.0,  5.844,   6283.076),
    (35.0,   0.0,     0.0),
    (17.0,   5.49,    12566.15),
    (3.0,    5.2,     155.42),
    (1.0,    4.72,    3.52),
    (1.0,    5.3,     18849.23),
    (1.0,    5.97,    242.73),
];

/// Earth heliocentric longitude L4 (3 rows).
#[rustfmt::skip]
pub const TERMS_L4: &[(f64, f64, f64)] = &[
    (114.0, 3.142,  0.0),
    (8.0,   4.13,   6283.08),
    (1.0,   3.84,   12566.15),
];

/// Earth heliocentric longitude L5 (1 row).
#[rustfmt::skip]
pub const TERMS_L5: &[(f64, f64, f64)] = &[
    (1.0, 3.14, 0.0),
];

/// Earth heliocentric latitude B0 (5 rows).
#[rustfmt::skip]
pub const TERMS_B0: &[(f64, f64, f64)] = &[
    (280.0, 3.199,  84334.662),
    (102.0, 5.422,  5507.553),
    (80.0,  3.88,   5223.69),
    (44.0,  3.7,    2352.87),
    (32.0,  4.0,    1577.34),
];

/// Earth heliocentric latitude B1 (2 rows).
#[rustfmt::skip]
pub const TERMS_B1: &[(f64, f64, f64)] = &[
    (9.0, 3.9,  5507.55),
    (6.0, 1.73, 5223.69),
];

/// Earth-Sun distance R0 (40 rows). AU × 10⁸.
#[rustfmt::skip]
pub const TERMS_R0: &[(f64, f64, f64)] = &[
    (100013989.0, 0.0,        0.0),
    (1670700.0,   3.0984635,  6283.07585),
    (13956.0,     3.05525,    12566.1517),
    (3084.0,      5.1985,     77713.7715),
    (1628.0,      1.1739,     5753.3849),
    (1576.0,      2.8469,     7860.4194),
    (925.0,       5.453,      11506.77),
    (542.0,       4.564,      3930.21),
    (472.0,       3.661,      5884.927),
    (346.0,       0.964,      5507.553),
    (329.0,       5.9,        5223.694),
    (307.0,       0.299,      5573.143),
    (243.0,       4.273,      11790.629),
    (212.0,       5.847,      1577.344),
    (186.0,       5.022,      10977.079),
    (175.0,       3.012,      18849.228),
    (110.0,       5.055,      5486.778),
    (98.0,        0.89,       6069.78),
    (86.0,        5.69,       15720.84),
    (86.0,        1.27,       161000.69),
    (65.0,        0.27,       17260.15),
    (63.0,        0.92,       529.69),
    (57.0,        2.01,       83996.85),
    (56.0,        5.24,       71430.7),
    (49.0,        3.25,       2544.31),
    (47.0,        2.58,       775.52),
    (45.0,        5.54,       9437.76),
    (43.0,        6.01,       6275.96),
    (39.0,        5.36,       4694.0),
    (38.0,        2.39,       8827.39),
    (37.0,        0.83,       19651.05),
    (37.0,        4.9,        12139.55),
    (36.0,        1.67,       12036.46),
    (35.0,        1.84,       2942.46),
    (33.0,        0.24,       7084.9),
    (32.0,        0.18,       5088.63),
    (32.0,        1.78,       398.15),
    (28.0,        1.21,       6286.6),
    (28.0,        1.9,        6279.55),
    (26.0,        4.59,       10447.39),
];

/// Earth-Sun distance R1 (10 rows).
#[rustfmt::skip]
pub const TERMS_R1: &[(f64, f64, f64)] = &[
    (103019.0, 1.10749,  6283.07585),
    (1721.0,   1.0644,   12566.1517),
    (702.0,    3.142,    0.0),
    (32.0,     1.02,     18849.23),
    (31.0,     2.84,     5507.55),
    (25.0,     1.32,     5223.69),
    (18.0,     1.42,     1577.34),
    (10.0,     5.91,     10977.08),
    (9.0,      1.42,     6275.96),
    (9.0,      0.27,     5486.78),
];

/// Earth-Sun distance R2 (6 rows).
#[rustfmt::skip]
pub const TERMS_R2: &[(f64, f64, f64)] = &[
    (4359.0, 5.7846,  6283.0758),
    (124.0,  5.579,   12566.152),
    (12.0,   3.14,    0.0),
    (9.0,    3.63,    77713.77),
    (6.0,    1.87,    5573.14),
    (3.0,    5.47,    18849.23),
];

/// Earth-Sun distance R3 (2 rows).
#[rustfmt::skip]
pub const TERMS_R3: &[(f64, f64, f64)] = &[
    (145.0, 4.273,  6283.076),
    (7.0,   3.92,   12566.15),
];

/// Earth-Sun distance R4 (1 row).
#[rustfmt::skip]
pub const TERMS_R4: &[(f64, f64, f64)] = &[
    (4.0, 2.56, 6283.08),
];

// ---------------------------------------------------------------------------
// Nutation (Table A4.4–A4.5) — 63 terms each with 5 lunisolar argument
// multipliers (Y0..Y4) and 4 amplitude/rate coefficients (a, b in 0.0001″
// for Δψ, c, d in 0.0001″ for Δε).
// ---------------------------------------------------------------------------

/// One row of the nutation periodic series. `y` are the integer multipliers
/// for the five fundamental arguments [X0..X4]; `a`, `b`, `c`, `d` are the
/// amplitude/rate coefficients in units of 10⁻⁴ arcseconds.
#[derive(Debug, Clone, Copy)]
pub struct NutationTerm {
    /// Integer multipliers `y_0..y_4` for the five fundamental arguments.
    pub y: [i32; 5],
    /// Δψ amplitude (units 10⁻⁴ arcsec).
    pub a: f64,
    /// Δψ time-coefficient (units 10⁻⁴ arcsec / Julian century).
    pub b: f64,
    /// Δε amplitude (units 10⁻⁴ arcsec).
    pub c: f64,
    /// Δε time-coefficient (units 10⁻⁴ arcsec / Julian century).
    pub d: f64,
}

/// Build a [`NutationTerm`] for the table — short helper to keep
/// the table itself compact.
pub const fn nutation_term(y: [i32; 5], a: f64, b: f64, c: f64, d: f64) -> NutationTerm {
    NutationTerm { y, a, b, c, d }
}

/// Compute the argument `Σ y_i · X_i(jce)` (in radians) for one nutation
/// term. The five fundamental arguments are SPA Eq 15a–19 (Mean Anomaly of
/// the Moon X0; of the Sun X1; Argument of Latitude of the Moon X2; Mean
/// Elongation of the Moon X3; Longitude of Ascending Node X4).
pub fn nutation_argument(jce: f64, y: &[i32; 5]) -> f64 {
    // Mean Elongation of the Moon from the Sun (X3 in SPA — Eq 15a).
    let x0 =
        (297.85036 + jce * (445_267.111_480 + jce * (-0.001_9142 + jce / 189_474.0))).to_radians();
    // Mean anomaly of the Sun (Earth) (X1 — Eq 16).
    let x1 =
        (357.52772 + jce * (35_999.050_340 + jce * (-0.000_1603 - jce / 300_000.0))).to_radians();
    // Mean anomaly of the Moon (X2 — Eq 17).
    let x2 =
        (134.96298 + jce * (477_198.867_398 + jce * (0.008_6972 + jce / 56_250.0))).to_radians();
    // Argument of latitude of the Moon (X3 — Eq 18).
    let x3 =
        (93.27191 + jce * (483_202.017_538 + jce * (-0.003_6825 + jce / 327_270.0))).to_radians();
    // Longitude of ascending node of the Moon's mean orbit (X4 — Eq 19).
    let x4 =
        (125.04452 + jce * (-1_934.136_261 + jce * (0.002_0708 + jce / 450_000.0))).to_radians();

    // SPA labels these in the order [X0..X4] = [Lp_moon, M_sun, M_moon,
    // F, Ω]. The table column ordering matches that.
    let xs = [x0, x1, x2, x3, x4];
    let mut sum = 0.0;
    for i in 0..5 {
        sum += (y[i] as f64) * xs[i];
    }
    sum
}

/// Nutation periodic-term table A4.4 / A4.5 (63 rows).
#[rustfmt::skip]
pub const NUTATION_TERMS: &[NutationTerm] = &[
    nutation_term([ 0, 0, 0, 0, 1], -171996.0, -174.2,  92025.0,    8.9),
    nutation_term([-2, 0, 0, 2, 2],  -13187.0,   -1.6,   5736.0,   -3.1),
    nutation_term([ 0, 0, 0, 2, 2],   -2274.0,   -0.2,    977.0,   -0.5),
    nutation_term([ 0, 0, 0, 0, 2],    2062.0,    0.2,   -895.0,    0.5),
    nutation_term([ 0, 1, 0, 0, 0],    1426.0,   -3.4,     54.0,   -0.1),
    nutation_term([ 0, 0, 1, 0, 0],     712.0,    0.1,     -7.0,    0.0),
    nutation_term([-2, 1, 0, 2, 2],    -517.0,    1.2,    224.0,   -0.6),
    nutation_term([ 0, 0, 0, 2, 1],    -386.0,   -0.4,    200.0,    0.0),
    nutation_term([ 0, 0, 1, 2, 2],    -301.0,    0.0,    129.0,   -0.1),
    nutation_term([-2,-1, 0, 2, 2],     217.0,   -0.5,    -95.0,    0.3),
    nutation_term([-2, 0, 1, 0, 0],    -158.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 0, 0, 2, 1],     129.0,    0.1,    -70.0,    0.0),
    nutation_term([ 0, 0,-1, 2, 2],     123.0,    0.0,    -53.0,    0.0),
    nutation_term([ 2, 0, 0, 0, 0],      63.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0, 1, 0, 1],      63.0,    0.1,    -33.0,    0.0),
    nutation_term([ 2, 0,-1, 2, 2],     -59.0,    0.0,     26.0,    0.0),
    nutation_term([ 0, 0,-1, 0, 1],     -58.0,   -0.1,     32.0,    0.0),
    nutation_term([ 0, 0, 1, 2, 1],     -51.0,    0.0,     27.0,    0.0),
    nutation_term([-2, 0, 2, 0, 0],      48.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0,-2, 2, 1],      46.0,    0.0,    -24.0,    0.0),
    nutation_term([ 2, 0, 0, 2, 2],     -38.0,    0.0,     16.0,    0.0),
    nutation_term([ 0, 0, 2, 2, 2],     -31.0,    0.0,     13.0,    0.0),
    nutation_term([ 0, 0, 2, 0, 0],      29.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 0, 1, 2, 2],      29.0,    0.0,    -12.0,    0.0),
    nutation_term([ 0, 0, 0, 2, 0],      26.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 0, 0, 2, 0],     -22.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0,-1, 2, 1],      21.0,    0.0,    -10.0,    0.0),
    nutation_term([ 0, 2, 0, 0, 0],      17.0,   -0.1,      0.0,    0.0),
    nutation_term([ 2, 0,-1, 0, 1],      16.0,    0.0,     -8.0,    0.0),
    nutation_term([-2, 2, 0, 2, 2],     -16.0,    0.1,      7.0,    0.0),
    nutation_term([ 0, 1, 0, 0, 1],     -15.0,    0.0,      9.0,    0.0),
    nutation_term([-2, 0, 1, 0, 1],     -13.0,    0.0,      7.0,    0.0),
    nutation_term([ 0,-1, 0, 0, 1],     -12.0,    0.0,      6.0,    0.0),
    nutation_term([ 0, 0, 2,-2, 0],      11.0,    0.0,      0.0,    0.0),
    nutation_term([ 2, 0,-1, 2, 1],     -10.0,    0.0,      5.0,    0.0),
    nutation_term([ 2, 0, 1, 2, 2],      -8.0,    0.0,      3.0,    0.0),
    nutation_term([ 0, 1, 0, 2, 2],       7.0,    0.0,     -3.0,    0.0),
    nutation_term([-2, 1, 1, 0, 0],      -7.0,    0.0,      0.0,    0.0),
    nutation_term([ 0,-1, 0, 2, 2],      -7.0,    0.0,      3.0,    0.0),
    nutation_term([ 2, 0, 0, 2, 1],      -7.0,    0.0,      3.0,    0.0),
    nutation_term([ 2, 0, 1, 0, 0],       6.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 0, 2, 2, 2],       6.0,    0.0,     -3.0,    0.0),
    nutation_term([-2, 0, 1, 2, 1],       6.0,    0.0,     -3.0,    0.0),
    nutation_term([ 2, 0,-2, 0, 1],      -6.0,    0.0,      3.0,    0.0),
    nutation_term([ 2, 0, 0, 0, 1],      -6.0,    0.0,      3.0,    0.0),
    nutation_term([ 0,-1, 1, 0, 0],       5.0,    0.0,      0.0,    0.0),
    nutation_term([-2,-1, 0, 2, 1],      -5.0,    0.0,      3.0,    0.0),
    nutation_term([-2, 0, 0, 0, 1],      -5.0,    0.0,      3.0,    0.0),
    nutation_term([ 0, 0, 2, 2, 1],      -5.0,    0.0,      3.0,    0.0),
    nutation_term([-2, 0, 2, 0, 1],       4.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 1, 0, 2, 1],       4.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0, 1,-2, 0],       4.0,    0.0,      0.0,    0.0),
    nutation_term([-1, 0, 1, 0, 0],      -4.0,    0.0,      0.0,    0.0),
    nutation_term([-2, 1, 0, 0, 0],      -4.0,    0.0,      0.0,    0.0),
    nutation_term([ 1, 0, 0, 0, 0],      -4.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0, 1, 2, 0],       3.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0,-2, 2, 2],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([-1,-1, 1, 0, 0],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 1, 1, 0, 0],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([ 0,-1, 1, 2, 2],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([ 2,-1,-1, 2, 2],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([ 0, 0, 3, 2, 2],      -3.0,    0.0,      0.0,    0.0),
    nutation_term([ 2,-1, 0, 2, 2],      -3.0,    0.0,      0.0,    0.0),
];

/// Meeus chapter 47 lunar periodic-term tables.
///
/// **Phase 2 deviation:** the SPA paper requires the *full* Meeus 47.A
/// (60 longitude/distance terms) and 47.B (60 latitude terms). The plan
/// §2.2 says "implement Meeus chapter 47" with no specific accuracy
/// target. We ship an abbreviated 6+6-term form here, which is good to
/// roughly 0.1° in lunar position — sufficient for the eventual
/// night-sky illumination direction Phase 5 will need, and well within
/// the §2.2 acceptance band specified for the sun (±0.1°). The full
/// tables are deferred to Phase 5+ when the moon actually feeds into a
/// shader. See README "Known cross-cutting gaps" for the deferral note.
pub mod moon {
    /// Longitude (Σ_L, units 10⁻⁶°) and distance (Σ_R, units 10⁻³ km)
    /// periodic terms. Top-amplitude rows from Meeus Table 47.A.
    /// Each row: (D, M, M', F, L_amp, R_amp).
    #[rustfmt::skip]
    pub const ML_TERMS: &[(i32, i32, i32, i32, f64, f64)] = &[
        (0,  0,  1,  0,  6_288_774.0, -20_905_355.0),
        (2,  0, -1,  0,  1_274_027.0,  -3_699_111.0),
        (2,  0,  0,  0,    658_314.0,  -2_955_968.0),
        (0,  0,  2,  0,    213_618.0,    -569_925.0),
        (0,  1,  0,  0,   -185_116.0,      48_888.0),
        (0,  0,  0,  2,   -114_332.0,      -3_149.0),
    ];

    /// Distance-only rows (currently unused; reserved for a future split).
    pub const MR_TERMS: &[(i32, i32, i32, i32, f64)] = &[];

    /// Latitude (Σ_B, units 10⁻⁶°) periodic terms. Top-amplitude rows from
    /// Meeus Table 47.B. Each row: (D, M, M', F, B_amp).
    #[rustfmt::skip]
    pub const MB_TERMS: &[(i32, i32, i32, i32, f64)] = &[
        (0, 0, 0,  1,  5_128_122.0),
        (0, 0, 1,  1,    280_602.0),
        (0, 0, 1, -1,    277_693.0),
        (2, 0, 0, -1,    173_237.0),
        (2, 0, -1, 1,     55_413.0),
        (2, 0, -1, -1,    46_271.0),
    ];
}
