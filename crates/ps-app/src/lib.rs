//! Library face of `ps-app`: exposes the headless test harness used by
//! `tests/integration.rs`, plus shared helpers used by both `main.rs` and
//! the harness.
//!
//! The binary entry point lives in `src/main.rs`.

#![deny(missing_docs)]

pub mod headless_dump;
pub mod main_helpers;
pub mod probe;
pub mod render_cli;
pub mod test_harness;

/// Internal: convert `[time]` from the engine config into a UTC `DateTime`.
/// Exposed publicly so [`headless_dump::run`] reuses the same conversion
/// the windowed binary uses.
pub fn config_initial_utc(config: &ps_core::Config) -> chrono::DateTime<chrono::Utc> {
    use chrono::{Duration, TimeZone, Utc};
    let local_naive = Utc
        .with_ymd_and_hms(
            config.time.year,
            config.time.month,
            config.time.day,
            config.time.hour,
            config.time.minute,
            config.time.second,
        )
        .single()
        .unwrap_or_else(|| {
            tracing::warn!("[time] invalid civil instant — falling back to J2000.0");
            Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap()
        });
    let offset_seconds = (config.time.timezone_offset_hours * 3_600.0).round() as i64;
    local_naive - Duration::seconds(offset_seconds)
}
