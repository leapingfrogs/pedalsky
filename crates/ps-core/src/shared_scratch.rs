//! Host-provided full-res HDR snapshot scratch, shared across the
//! post-process subsystems (bloom, godrays) so each need not allocate
//! its own resident full-res copy.

/// A full-res HDR snapshot texture owned by the host and lent to a
/// post-process subsystem via its `set_shared_hdr_scratch` setter.
///
/// Bloom and godrays each need to sample the current HDR target while
/// also writing into it (bright-pass / radial-blur). They resolve the
/// read+write hazard by first copying HDR into a scratch texture and
/// sampling that. Left to themselves each allocates its own resident
/// full-res `Rgba16Float` copy (~66 MB at 4K). Because the two run
/// sequentially in the frame — godrays composites its shafts into HDR,
/// then bloom snapshots the result — they can time-share ONE physical
/// scratch: each copies HDR into it at its own pipeline point, just
/// before sampling. The host owns the texture, keeps its lifecycle tied
/// to the framebuffer size (reallocating only on resize so the
/// subsystems' size-keyed bind-group caches stay valid), and lends it
/// each frame.
///
/// The handles are `Arc`-backed `wgpu` resources; cloning is cheap.
#[derive(Clone)]
pub struct SharedHdrScratch {
    /// Copy destination and sample source. Must be
    /// `COPY_DST | TEXTURE_BINDING`, a single layer, sized `size`, in
    /// [`crate::framebuffer::HdrFramebufferImpl::COLOR_FORMAT`].
    pub texture: wgpu::Texture,
    /// A default view of `texture`, bound as the sample source.
    pub view: wgpu::TextureView,
    /// `(width, height)` of `texture`. A subsystem uses the shared
    /// scratch only when this matches its current framebuffer size;
    /// otherwise it falls back to a self-allocated copy for that frame.
    pub size: (u32, u32),
}
