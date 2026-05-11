//! PedalSky post-processing: tone-mapping, exposure.
//!
//! Phase 0 ships an ACES Filmic (Narkowicz fit) and a passthrough/clamp
//! variant. The shader source lives at `shaders/postprocess/tonemap.wgsl`
//! and is included at compile time.

#![deny(missing_docs)]

pub mod auto_exposure;
pub mod tonemap;

pub use auto_exposure::AutoExposure;
pub use tonemap::{Tonemap, TonemapMode};
