//! GPU runtime for the erosion pipeline.
//!
//! Owns the wgpu device + queue + cached compute pipelines. The
//! `run` function allocates per-tile textures, uploads the initial
//! heightmap, dispatches the interleaved hydraulic/thermal loop +
//! fractal + normal map, and reads the result back to CPU.

use std::sync::Arc;

use bytemuck::bytes_of;

use super::fractal::FractalBindings;
use super::hydraulic::HydraulicBindings;
use super::normal_map::NormalMapBindings;
use super::params::{
    ErosionParams, FractalUniformGpu, HydraulicUniformGpu, NormalMapUniformGpu,
    ThermalUniformGpu,
};
use super::thermal::ThermalBindings;
use crate::progress::{TerrainProgressSink, TerrainStage};
use crate::tile::HeightmapTile;
use crate::TerrainError;

/// Cached GPU resources for the erosion pipeline. Cheap to clone via
/// `Arc`; share one across all `ErosionAugment` instances if you want
/// to avoid rebuilding pipelines.
pub(super) struct ErosionRuntime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub hydraulic: HydraulicBindings,
    pub thermal: ThermalBindings,
    pub fractal: FractalBindings,
    pub normal_map: NormalMapBindings,
}

impl ErosionRuntime {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let hydraulic = HydraulicBindings::new(&device);
        let thermal = ThermalBindings::new(&device);
        let fractal = FractalBindings::new(&device);
        let normal_map = NormalMapBindings::new(&device);
        Self { device, queue, hydraulic, thermal, fractal, normal_map }
    }
}

/// Per-run texture allocations. Lifetime: one `run` call.
struct Textures {
    terrain: wgpu::Texture,
    water_a: wgpu::Texture,
    water_b: wgpu::Texture,
    sediment_a: wgpu::Texture,
    sediment_b: wgpu::Texture,
    flux_a: wgpu::Texture,
    flux_b: wgpu::Texture,
    velocity: wgpu::Texture,
    thermal_out_a: wgpu::Texture,
    thermal_out_b: wgpu::Texture,
    normal_map: wgpu::Texture,
}

impl Textures {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mk = |label: &'static str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        Self {
            terrain: mk("erosion-terrain", wgpu::TextureFormat::R32Float),
            water_a: mk("erosion-water-a", wgpu::TextureFormat::R32Float),
            water_b: mk("erosion-water-b", wgpu::TextureFormat::R32Float),
            sediment_a: mk("erosion-sediment-a", wgpu::TextureFormat::R32Float),
            sediment_b: mk("erosion-sediment-b", wgpu::TextureFormat::R32Float),
            flux_a: mk("erosion-flux-a", wgpu::TextureFormat::Rgba32Float),
            flux_b: mk("erosion-flux-b", wgpu::TextureFormat::Rgba32Float),
            velocity: mk("erosion-velocity", wgpu::TextureFormat::Rg32Float),
            thermal_out_a: mk("erosion-thermal-out-a", wgpu::TextureFormat::Rgba32Float),
            thermal_out_b: mk("erosion-thermal-out-b", wgpu::TextureFormat::Rgba32Float),
            normal_map: mk("erosion-normal-map", wgpu::TextureFormat::Rgba8Unorm),
        }
    }
}

fn view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

pub(super) fn run(
    rt: &ErosionRuntime,
    tile: HeightmapTile,
    params: &ErosionParams,
    progress: &dyn TerrainProgressSink,
) -> Result<HeightmapTile, TerrainError> {
    let w = tile.width;
    let h = tile.height;
    let device = &*rt.device;
    let queue = &*rt.queue;

    let textures = Textures::new(device, w, h);

    // Upload initial heightmap into the R32Float terrain texture.
    upload_r32f(queue, &textures.terrain, &tile.heights_m, w, h);
    // Zero water + sediment + flux + velocity. wgpu auto-zeroes on
    // create + first write but we initialise via small writes to
    // ensure the texture is in a defined state on all backends.
    let zeros = vec![0.0_f32; (w * h) as usize];
    upload_r32f(queue, &textures.water_a, &zeros, w, h);
    upload_r32f(queue, &textures.water_b, &zeros, w, h);
    upload_r32f(queue, &textures.sediment_a, &zeros, w, h);
    upload_r32f(queue, &textures.sediment_b, &zeros, w, h);

    // ---- Hydraulic + thermal interleaved loop ----------------------
    let cell_size = params.target_resolution_m.max(0.01);
    let hu = HydraulicUniformGpu::from_params(params, cell_size);
    queue.write_buffer(&rt.hydraulic.uniforms, 0, bytes_of(&hu));
    let tu = ThermalUniformGpu::from_params(params, cell_size);
    queue.write_buffer(&rt.thermal.uniforms, 0, bytes_of(&tu));

    let terrain_view = view(&textures.terrain);
    let water_a_view = view(&textures.water_a);
    let water_b_view = view(&textures.water_b);
    let sed_a_view = view(&textures.sediment_a);
    let sed_b_view = view(&textures.sediment_b);
    let flux_a_view = view(&textures.flux_a);
    let flux_b_view = view(&textures.flux_b);
    let velocity_view = view(&textures.velocity);

    // Two hydraulic bind groups. Water + flux ping-pong each iter
    // because pass 1 reads `flux_in` (binding 4) and writes
    // `flux_out` (binding 7) — pass 2 then needs `flux_out` as its
    // primary input. Sediment, by contrast, does NOT swap: pass 3
    // writes an intermediate to `sediment_out` (binding 8 = sed_b),
    // and pass 4 reads that intermediate and writes the final
    // advected result back to `sediment_in` (binding 3 = sed_a).
    // Next iteration's pass 3 needs to read the same sed_a, so the
    // sediment bindings must stay constant across bg_a / bg_b.
    let bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion-hydraulic-bg-a"),
        layout: &rt.hydraulic.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: rt.hydraulic.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&water_a_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sed_a_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&flux_a_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&velocity_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&water_b_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&flux_b_view) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&sed_b_view) },
        ],
    });
    let bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion-hydraulic-bg-b"),
        layout: &rt.hydraulic.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: rt.hydraulic.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&water_b_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sed_a_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&flux_b_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&velocity_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&water_a_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&flux_a_view) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&sed_b_view) },
        ],
    });

    let thermal_out_a_view = view(&textures.thermal_out_a);
    let thermal_out_b_view = view(&textures.thermal_out_b);
    let thermal_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion-thermal-bg"),
        layout: &rt.thermal.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: rt.thermal.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&thermal_out_a_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&thermal_out_b_view) },
        ],
    });

    progress.stage(TerrainStage::HydraulicErosion, 0, params.iterations);

    let wg_x = w.div_ceil(16);
    let wg_y = h.div_ceil(16);

    for iter in 0..params.iterations {
        let pingpong_a = (iter % 2) == 0;
        let bind = if pingpong_a { &bg_a } else { &bg_b };

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("erosion-hydraulic-iter"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("erosion-hydraulic-pass"),
                timestamp_writes: None,
            });
            // Pass 1
            pass.set_pipeline(&rt.hydraulic.pipeline_add_water_and_flux);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            // Pass 2
            pass.set_pipeline(&rt.hydraulic.pipeline_update_water_and_velocity);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            // Pass 3
            pass.set_pipeline(&rt.hydraulic.pipeline_erosion_deposition);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            // Pass 4
            pass.set_pipeline(&rt.hydraulic.pipeline_advect_sediment);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(Some(encoder.finish()));

        // Interleaved thermal pass.
        let mut do_thermal = params.hydraulic_iterations_between_thermal > 0
            && (iter + 1) % params.hydraulic_iterations_between_thermal == 0;
        if iter + 1 == params.iterations {
            // Always include a final thermal cycle at the end so cliffs
            // from the last hydraulic step get cleaned up.
            do_thermal = true;
        }
        if do_thermal && params.thermal_erosion_rate > 0.0 {
            for _ in 0..params.thermal_iterations_per_cycle.max(1) {
                let mut th_enc = device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("erosion-thermal") },
                );
                {
                    let mut tp = th_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("erosion-thermal-pass"),
                        timestamp_writes: None,
                    });
                    tp.set_pipeline(&rt.thermal.pipeline_outflow);
                    tp.set_bind_group(0, &thermal_bg, &[]);
                    tp.dispatch_workgroups(wg_x, wg_y, 1);
                    tp.set_pipeline(&rt.thermal.pipeline_apply);
                    tp.set_bind_group(0, &thermal_bg, &[]);
                    tp.dispatch_workgroups(wg_x, wg_y, 1);
                }
                queue.submit(Some(th_enc.finish()));
            }
        }

        progress.stage(TerrainStage::HydraulicErosion, iter + 1, params.iterations);
    }
    progress.stage(TerrainStage::ThermalErosion, 1, 1);

    // ---- Stage 1.4 — fractal detail --------------------------------
    if params.fractal_amplitude_m > 0.0 {
        progress.stage(TerrainStage::FractalDetail, 0, 1);
        let fu = FractalUniformGpu::from_params(params, cell_size);
        queue.write_buffer(&rt.fractal.uniforms, 0, bytes_of(&fu));
        let frac_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("erosion-fractal-bg"),
            layout: &rt.fractal.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: rt.fractal.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("erosion-fractal"),
        });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("erosion-fractal-pass"),
                timestamp_writes: None,
            });
            p.set_pipeline(&rt.fractal.pipeline);
            p.set_bind_group(0, &frac_bg, &[]);
            p.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(Some(enc.finish()));
        progress.stage(TerrainStage::FractalDetail, 1, 1);
    }

    // ---- Stage 1.5 — normal map ------------------------------------
    progress.stage(TerrainStage::NormalMap, 0, 1);
    let nu = NormalMapUniformGpu::from_cell_size(cell_size);
    queue.write_buffer(&rt.normal_map.uniforms, 0, bytes_of(&nu));
    let nmap_view = view(&textures.normal_map);
    let nmap_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erosion-normal-map-bg"),
        layout: &rt.normal_map.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: rt.normal_map.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&nmap_view) },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("erosion-normal-map"),
    });
    {
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("erosion-normal-map-pass"),
            timestamp_writes: None,
        });
        p.set_pipeline(&rt.normal_map.pipeline);
        p.set_bind_group(0, &nmap_bg, &[]);
        p.dispatch_workgroups(wg_x, wg_y, 1);
    }
    queue.submit(Some(enc.finish()));
    progress.stage(TerrainStage::NormalMap, 1, 1);

    // ---- Read back the heightmap to CPU ----------------------------
    let heights = readback_r32f(device, queue, &textures.terrain, w, h)?;

    Ok(HeightmapTile {
        heights_m: heights,
        width: w,
        height: h,
        extent_deg: tile.extent_deg,
        source: tile.source,
        gsd_m_centre: cell_size,
    })
}

fn upload_r32f(queue: &wgpu::Queue, tex: &wgpu::Texture, data: &[f32], w: u32, h: u32) {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

/// Read an `R32Float` texture back to CPU. Allocates a staging buffer
/// + maps it synchronously via `pollster`. Blocks the calling thread.
fn readback_r32f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    w: u32,
    h: u32,
) -> Result<Vec<f32>, TerrainError> {
    // wgpu requires bytes_per_row to be a multiple of 256.
    let unaligned_bpr = w * 4;
    let padded_bpr = (unaligned_bpr + 255) & !255;
    let buffer_size = (padded_bpr * h) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("erosion-readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("erosion-readback-encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| {
        TerrainError::AugmentInvalid(format!("device poll failed: {e}"))
    })?;
    rx.recv()
        .map_err(|e| TerrainError::AugmentInvalid(format!("map_async channel closed: {e}")))?
        .map_err(|e| TerrainError::AugmentInvalid(format!("readback map failed: {e}")))?;

    let view_bytes = slice.get_mapped_range();
    let mut out = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        let row_off = (row * padded_bpr) as usize;
        let row_bytes = &view_bytes[row_off..row_off + (w * 4) as usize];
        let row_floats: &[f32] = bytemuck::cast_slice(row_bytes);
        out.extend_from_slice(row_floats);
    }
    drop(view_bytes);
    staging.unmap();
    Ok(out)
}
