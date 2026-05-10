//! Library face of `ps-app`: exposes the headless test harness used by
//! `tests/integration.rs`, plus shared helpers used by both `main.rs` and
//! the harness.
//!
//! The binary entry point lives in `src/main.rs`.

#![deny(missing_docs)]

pub mod main_helpers;
pub mod test_harness;
