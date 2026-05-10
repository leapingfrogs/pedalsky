//! Canonical bind-group conventions (plan §4.2).
//!
//! - **Group 0** — `FrameUniforms` (per-frame matrices, sun, time, ev100).
//! - **Group 1** — `WorldUniforms` (planet + atmosphere constants).
//! - **Group 2** — subsystem-specific resources (no canonical layout).
//! - **Group 3** — atmosphere LUTs (Phase 5; `None` until then).
//!
//! This module exposes the *layouts* (consumed by every subsystem
//! pipeline) and the [`FrameWorldBindings`] helper that owns the two
//! uniform buffers + their bind groups in the host (ps-app).

use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::frame_uniforms::{FrameUniforms, FrameUniformsGpu};
use crate::weather::WorldUniformsGpu;

/// Build the canonical `BindGroupLayout` for **group 0** (`FrameUniforms`).
///
/// Visibility is `VERTEX_FRAGMENT | COMPUTE` so all stages can reach it.
pub fn frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ps-core::group0-frame"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<FrameUniformsGpu>() as u64
                ),
            },
            count: None,
        }],
    })
}

/// Build the canonical `BindGroupLayout` for **group 1** (`WorldUniforms`).
pub fn world_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ps-core::group1-world"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<WorldUniformsGpu>() as u64
                ),
            },
            count: None,
        }],
    })
}

/// Owns the per-frame uniform buffers + bind groups (groups 0 and 1) and
/// the layouts that subsystems pin against.
///
/// Constructed once at startup; mutated each frame via
/// [`FrameWorldBindings::write`].
pub struct FrameWorldBindings {
    /// Uniform buffer holding `FrameUniformsGpu`.
    pub frame_buffer: wgpu::Buffer,
    /// Bind group attaching `frame_buffer` at binding 0 of group 0.
    pub frame_bind_group: wgpu::BindGroup,
    /// Layout for group 0.
    pub frame_layout: wgpu::BindGroupLayout,

    /// Uniform buffer holding `WorldUniformsGpu`.
    pub world_buffer: wgpu::Buffer,
    /// Bind group attaching `world_buffer` at binding 0 of group 1.
    pub world_bind_group: wgpu::BindGroup,
    /// Layout for group 1.
    pub world_layout: wgpu::BindGroupLayout,
}

impl FrameWorldBindings {
    /// Allocate buffers + bind groups. Initial contents are zero-filled
    /// for `frame` and the [`WorldUniformsGpu::default`] for `world`.
    pub fn new(device: &wgpu::Device) -> Self {
        let frame_layout = frame_bind_group_layout(device);
        let world_layout = world_bind_group_layout(device);

        let frame_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ps-core::frame-uniforms"),
            contents: bytes_of(&FrameUniformsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let world_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ps-core::world-uniforms"),
            contents: bytes_of(&WorldUniformsGpu::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ps-core::group0-frame-bg"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });
        let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ps-core::group1-world-bg"),
            layout: &world_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: world_buffer.as_entire_binding(),
            }],
        });

        Self {
            frame_buffer,
            frame_bind_group,
            frame_layout,
            world_buffer,
            world_bind_group,
            world_layout,
        }
    }

    /// Upload fresh `frame` + `world` payloads.
    pub fn write(&self, queue: &wgpu::Queue, frame: &FrameUniforms, world: &WorldUniformsGpu) {
        queue.write_buffer(
            &self.frame_buffer,
            0,
            bytes_of(&FrameUniformsGpu::from_cpu(frame)),
        );
        queue.write_buffer(&self.world_buffer, 0, bytes_of(world));
    }
}
