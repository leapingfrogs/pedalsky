//! PedalSky post-processing: tone-mapping, exposure.
//!
//! Phase 0 ships an ACES Filmic (Narkowicz fit) and a passthrough variant.

#![deny(missing_docs)]

/// Tone mapping pass.
///
/// Placeholder module — Phase 0 GPU plumbing for the tone-mapper lands here
/// (TODO: Phase 0 finish — fullscreen-triangle pipeline reading the HDR
/// target and writing the swapchain).
pub mod tonemap {}
