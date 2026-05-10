//! HDR framebuffer (Rgba16Float colour + Depth32Float reverse-Z depth).
//!
//! See plan §0.3.

use crate::gpu::GpuContext;

/// HDR framebuffer: a colour and a reverse-Z depth target sized to the swapchain
/// (or to a configured size for headless renders).
///
/// The colour format is fixed at `Rgba16Float`; the depth format is fixed at
/// `Depth32Float` with reverse-Z (clear value 0.0, near-plane = 1.0).
pub struct HdrFramebufferImpl {
    /// `Rgba16Float` HDR colour target.
    pub color: wgpu::Texture,
    /// Default colour view.
    pub color_view: wgpu::TextureView,
    /// `Depth32Float` reverse-Z depth target.
    pub depth: wgpu::Texture,
    /// Default depth view.
    pub depth_view: wgpu::TextureView,
    /// Pixel size (width, height).
    pub size: (u32, u32),
}

impl HdrFramebufferImpl {
    /// HDR colour format used everywhere internal to the engine.
    pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// Reverse-Z depth format.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Allocate fresh colour + depth targets at `size`.
    pub fn new(gpu: &GpuContext, (w, h): (u32, u32)) -> Self {
        let color = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr-color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr-depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            color,
            color_view,
            depth,
            depth_view,
            size: (w, h),
        }
    }

    /// Resize: recreate both targets. Cheap to call (test harness only resizes
    /// on window events; subsystems must not cache views across resize).
    pub fn resize(&mut self, gpu: &GpuContext, size: (u32, u32)) {
        *self = Self::new(gpu, size);
    }
}
