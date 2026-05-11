//! Phase 9.2 — debug auto-exposure.
//!
//! Computes the average log-luminance of the HDR target via a single-
//! workgroup compute pass, derives an EV100 that centres mid-grey, and
//! returns it for the host to feed into [`crate::Tonemap::render`].
//!
//! Usage per frame:
//! 1. Call [`AutoExposure::dispatch`] inside the encoder, after all HDR
//!    writes are done (i.e. just before the tone-map pass).
//! 2. After `queue.submit(...)` for the frame, call
//!    [`AutoExposure::read_back_ev100`] to drain the staging buffer.
//!    This blocks on `device.poll(..wait_indefinitely..)`.
//! 3. Cache the returned EV100 and feed it into the next frame's
//!    `Tonemap::render(...)` (or this same frame's, accepting the lag).
//!
//! Gated by `[debug] auto_exposure = true`. Off by default.

use bytemuck::{Pod, Zeroable};
use ps_core::HdrFramebuffer;

const SHADER_SRC: &str = include_str!("../../../shaders/postprocess/auto_exposure.comp.wgsl");

/// Mirror of the WGSL `AeOutput` struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
struct AeOutputCpu {
    log_lum_sum: f32,
    pixel_count: f32,
}

/// Auto-exposure compute pass + staging.
pub struct AutoExposure {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl AutoExposure {
    /// Build the compute pipeline + buffers.
    pub fn new(device: &wgpu::Device, hdr: &HdrFramebuffer) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("auto_exposure.comp"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auto-exposure-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("auto-exposure-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("auto-exposure-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-exposure-output"),
            size: std::mem::size_of::<AeOutputCpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-exposure-staging"),
            size: std::mem::size_of::<AeOutputCpu>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = build_bind_group(device, &bind_group_layout, hdr, &output_buf);

        Self {
            pipeline,
            bind_group_layout,
            output_buf,
            staging_buf,
            bind_group,
        }
    }

    /// Rebuild the bind group after a framebuffer resize.
    pub fn rebuild_bindings(&mut self, device: &wgpu::Device, hdr: &HdrFramebuffer) {
        self.bind_group =
            build_bind_group(device, &self.bind_group_layout, hdr, &self.output_buf);
    }

    /// Dispatch the auto-exposure compute pass and copy the output to
    /// the staging buffer ready for read-back. Call after all HDR writes
    /// for the frame.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("auto-exposure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &self.staging_buf,
            0,
            std::mem::size_of::<AeOutputCpu>() as u64,
        );
    }

    /// Read the latest auto-exposure result and convert to EV100.
    /// Blocks until the staging buffer is mappable. Returns `None` if
    /// the read-back fails for any reason.
    ///
    /// Mid-grey calibration: we centre such that the geometric mean of
    /// the scene's luminance lands at 0.18 cd/m² in the tone-mapper's
    /// post-exposure space. Tonemap formula: `exposure = 1/(1.2 * 2^EV100)`.
    /// So `avg_lum_post = avg_lum_pre * exposure` should equal 0.18,
    /// giving `EV100 = log2(avg_lum_pre / 0.216)`.
    pub fn read_back_ev100(&self, device: &wgpu::Device) -> Option<f32> {
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok()?;
        let recv = rx.recv().ok()?;
        recv.ok()?;
        let bytes = slice.get_mapped_range().to_vec();
        self.staging_buf.unmap();
        if bytes.len() < std::mem::size_of::<AeOutputCpu>() {
            return None;
        }
        let out: AeOutputCpu = *bytemuck::from_bytes(&bytes[..std::mem::size_of::<AeOutputCpu>()]);
        if out.pixel_count <= 0.0 {
            return None;
        }
        let avg_log_lum = out.log_lum_sum / out.pixel_count;
        let avg_lum = avg_log_lum.exp2().max(1e-6);
        // 0.18 grey × 1.2 (tone-map's denominator constant) = 0.216.
        let ev100 = (avg_lum / 0.216).log2();
        Some(ev100)
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    hdr: &HdrFramebuffer,
    output_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("auto-exposure-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr.color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    })
}
