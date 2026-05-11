//! PedalSky weather-data synthesis (Phase 3).
//!
//! Turns a parsed scene + world-state into a `WeatherState` (defined in
//! `ps-core::weather`): a bundle of Pod scalars plus GPU-resident
//! textures and storage buffers.
//!
//! The Pod struct *types* (`AtmosphereParams`, `CloudLayerGpu`,
//! `SurfaceParams`, `WeatherState`) live in `ps-core::weather` so that
//! `PrepareContext` can borrow `&WeatherState` without `ps-core` taking
//! a dependency on this crate. This crate provides the synthesis
//! pipeline that *fills* those structs.
//!
//! Module map:
//! - [`koschmieder`] — visibility → Mie haze coefficient.
//! - [`cloud_layers`] — per-layer envelope synthesis from `Scene` layers.
//! - [`weather_map`] — 128×128 RGBA16Float coverage / cloud-base / precip texture.
//! - [`cloud_type_grid`] — Phase 12.1 per-pixel cloud-type R8Uint texture.
//! - [`wind_field`] — 32×32×16 RGBA16Float (u, v, w, turbulence) profile.
//! - [`density_mask`] — 2D top-down density mask for precipitation occlusion.
//! - [`ndf`] — vertical density profile (mirror of Phase 6 §6.4 WGSL).
//! - [`state`] — the [`synthesise`] entry point.

#![deny(missing_docs)]

pub mod cloud_layers;
pub mod cloud_type_grid;
pub mod density_mask;
pub mod koschmieder;
pub mod ndf;
pub mod state;
pub mod weather_map;
pub mod wind_field;

pub use cloud_layers::synthesise_cloud_layers;
pub use koschmieder::haze_extinction_per_m;
pub use state::{synthesise, SynthesisError};
