//! Phase 2 §2.2 — astronomical calculations.
//!
//! Sun position via the **NREL Solar Position Algorithm** (Reda & Andreas
//! 2003, NREL/TP-560-34302). Moon position via Meeus 1998 chapter 47.
//!
//! See `spa.rs` for the SPA driver, `tables.rs` for the periodic-term
//! coefficient tables, and `moon.rs` for the lunar implementation.

pub mod moon;
pub mod spa;
pub mod tables;

pub use moon::moon_position;
pub use spa::{sun_position, SunPosition};
