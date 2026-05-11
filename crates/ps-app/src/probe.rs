//! Phase 10.A4 — probe-pixel transmittance readback.
//!
//! One-thread compute pass that reconstructs the world-space view ray
//! at a probe pixel and samples the atmosphere transmittance LUT.
//! Result lives in a 16-byte storage buffer; host reads back per frame
//! and pushes into `UiHandle.debug.probe_transmittance`.
//!
//! No-op when atmosphere is disabled (no LUT bind group available).

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use ps_core::{
    atmosphere_lut_bind_group_layout, frame_bind_group_layout, world_bind_group_layout,
    GpuContext,
};

const SHADER_BAKED: &str = include_str!("../../../shaders/debug/probe_transmittance.comp.wgsl");
const SHADER_REL: &str = "debug/probe_transmittance.comp.wgsl";

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct ProbeUniformsGpu {
    pixel: [u32; 2],
    _pad: [u32; 2],
}

/// Probe compute pass + 16-byte readback.
pub struct ProbeReadback {
    pipeline: wgpu::ComputePipeline,
    bg_layout: wgpu::BindGroupLayout,
    uniforms_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
}

impl ProbeReadback {
    /// Build pipeline + buffers.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let live_src = ps_core::shaders::load_shader(SHADER_REL, SHADER_BAKED);
        let composed = ps_core::shaders::compose(&[
            ps_core::shaders::COMMON_UNIFORMS_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_WGSL,
            ps_core::shaders::COMMON_ATMOSPHERE_LUT_SAMPLING_WGSL,
            &live_src,
        ]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe-transmittance.comp"),
            source: wgpu::ShaderSource::Wgsl(composed.into()),
        });
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);
        let lut_layout = atmosphere_lut_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe-pl"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&world_layout),
                Some(&bg_layout),
                Some(&lut_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-uniforms"),
            size: std::mem::size_of::<ProbeUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-output"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-staging"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bg_layout,
            uniforms_buf,
            output_buf,
            staging_buf,
        }
    }

    /// Dispatch the probe compute pass.
    ///
    /// `pixel` is in framebuffer (top-left origin) coordinates.
    /// `frame_bg`, `world_bg`, `luts_bg` are the canonical engine bind
    /// groups. Returns silently if no luts are bound.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        pixel: (u32, u32),
        frame_bg: &wgpu::BindGroup,
        world_bg: &wgpu::BindGroup,
        luts_bg: &wgpu::BindGroup,
    ) {
        queue.write_buffer(
            &self.uniforms_buf,
            0,
            bytemuck::bytes_of(&ProbeUniformsGpu {
                pixel: [pixel.0, pixel.1],
                _pad: [0; 2],
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe-bg"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe-transmittance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, frame_bg, &[]);
            pass.set_bind_group(1, world_bg, &[]);
            pass.set_bind_group(2, &bg, &[]);
            pass.set_bind_group(3, luts_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.staging_buf, 0, 16);
    }

    /// Read back the previous frame's probe value. Blocks on
    /// `device.poll(wait_indefinitely)`.
    pub fn read(&self, gpu: &GpuContext) -> Result<[f32; 3]> {
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        let _ = rx.recv();
        let bytes = slice.get_mapped_range().to_vec();
        self.staging_buf.unmap();
        if bytes.len() < 16 {
            return Ok([0.0; 3]);
        }
        let r = f32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4]));
        let g = f32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4]));
        let b = f32::from_le_bytes(bytes[8..12].try_into().unwrap_or([0; 4]));
        Ok([r, g, b])
    }
}
