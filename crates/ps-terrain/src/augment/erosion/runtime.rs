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
        let mk_with = |label: &'static str,
                       format: wgpu::TextureFormat,
                       usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let default_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let mk = |label: &'static str, format: wgpu::TextureFormat| {
            mk_with(label, format, default_usage)
        };
        Self {
            terrain: mk("erosion-terrain", wgpu::TextureFormat::R32Float),
            water_a: mk("erosion-water-a", wgpu::TextureFormat::R32Float),
            water_b: mk("erosion-water-b", wgpu::TextureFormat::R32Float),
            sediment_a: mk("erosion-sediment-a", wgpu::TextureFormat::R32Float),
            sediment_b: mk("erosion-sediment-b", wgpu::TextureFormat::R32Float),
            flux_a: mk("erosion-flux-a", wgpu::TextureFormat::Rgba32Float),
            flux_b: mk("erosion-flux-b", wgpu::TextureFormat::Rgba32Float),
            // Velocity needs TEXTURE_BINDING in addition to
            // STORAGE_BINDING because we bind a sampled read view at
            // binding 9 (see hydraulic.wgsl header — Metal can't do
            // read_write on rg32float).
            velocity: mk_with(
                "erosion-velocity",
                wgpu::TextureFormat::Rg32Float,
                default_usage | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
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
    //
    // Cell size is the *actual* metric pixel pitch of the upsampled
    // tile. The bicubic upsample preserves the geographic extent, so
    // `tile.gsd_m_centre` is the real metres-per-pixel — anywhere
    // between 30 m (raw GLO-30) and 1 m (full Section 1.1 upsample).
    // Using `params.target_resolution_m` here would be a bug: when
    // `max_working_dim` caps the upsample (e.g. a 1° tile capped to
    // 1024 px gives ~108 m/px at the equator), the shader would
    // think the cell is 1 m while the heightmap is actually 108 m
    // per pixel — fluxes would dramatically under-count water, the
    // flux-scaling step couldn't bound outflow, velocities would
    // blow up, and erosion would carve absurd peaks + valleys.
    //
    // Mei's `dt = 0.02` is calibrated for cell_size ~= 1 m. At
    // larger cell sizes we need a proportionally smaller dt to
    // respect the CFL condition `dt < cell / sqrt(g * h_max)`.
    // Scale dt by 1/cell_size to keep flux per iter bounded.
    let cell_size = tile.gsd_m_centre.max(0.01);
    // The Mei flux term is `dt * pipe_cross_section * gravity * dh /
    // pipe_length`. The spec calls for `pipe_length = cell_size` (it
    // explicitly says "normally equal to target_resolution_m"); we
    // enforce that here so a user who never touches the slider can't
    // hit the case where the pipe length is calibrated for one cell
    // size and the heightmap is at another.
    //
    // The remaining CFL constraint is `dt < cell_size / sqrt(g * h_max)`.
    // The default dt = 0.02 is calibrated for cell_size = 1 m and
    // reasonable max water depths (~1 m). At larger cell sizes we
    // gain headroom (so smaller scaling is just conservative); at
    // smaller cell sizes we need a proportionally smaller dt. Scale
    // dt by `min(1, cell_size)` so a 1 m cell uses the full 0.02
    // while a 0.5 m cell uses 0.01.
    let cfl_scale = cell_size.min(1.0);
    let scaled_params = ErosionParams {
        dt: params.dt * cfl_scale,
        pipe_length: cell_size,
        ..*params
    };
    tracing::info!(
        target: "ps_terrain::erosion",
        cell_size, target_resolution_m = params.target_resolution_m,
        dt_in = params.dt, dt_scaled = scaled_params.dt,
        "erosion: dt rescaled for actual cell size"
    );
    let hu = HydraulicUniformGpu::from_params(&scaled_params, cell_size);
    queue.write_buffer(&rt.hydraulic.uniforms, 0, bytes_of(&hu));
    let tu = ThermalUniformGpu::from_params(&scaled_params, cell_size);
    queue.write_buffer(&rt.thermal.uniforms, 0, bytes_of(&tu));

    let terrain_view = view(&textures.terrain);
    let water_a_view = view(&textures.water_a);
    let water_b_view = view(&textures.water_b);
    let sed_a_view = view(&textures.sediment_a);
    let sed_b_view = view(&textures.sediment_b);
    let flux_a_view = view(&textures.flux_a);
    let flux_b_view = view(&textures.flux_b);
    let velocity_view = view(&textures.velocity);
    let velocity_read_view = view(&textures.velocity);

    // Water + flux ping-pong each iter because pass 1 reads `flux_in`
    // (binding 4) and writes `flux_out` (binding 7) — pass 2 then needs
    // `flux_out` as its primary input. Sediment, by contrast, does NOT
    // swap: pass 3 writes an intermediate to `sediment_out` (binding 8
    // = sed_b), and pass 4 reads that intermediate and writes the final
    // advected result back to `sediment_in` (binding 3 = sed_a). Next
    // iteration's pass 3 needs to read the same sed_a, so the sediment
    // bindings must stay constant across bg_a / bg_b.
    //
    // Velocity is also split across the iteration: pass 2 writes it via
    // a write-only storage view at binding 5, and passes 3+4 read it
    // via a sampled view at binding 9 (Metal has no read_write tier for
    // rg32float — see shader header). The two views can't coexist in a
    // single bind group because wgpu validates per-dispatch using *all*
    // bindings of the active bind group regardless of which the shader
    // actually accesses: STORAGE_WRITE_ONLY is exclusive of RESOURCE on
    // the same texture, so the merge fails. We work around this with
    // separate "write" and "read" bind groups per ping-pong state and a
    // 1×1 dummy texture in whichever slot is unused for that phase.
    let velocity_dummy = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("erosion-velocity-dummy"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let velocity_dummy_view = view(&velocity_dummy);

    let mk_bg = |label: &'static str,
                 water_in: &wgpu::TextureView,
                 flux_in: &wgpu::TextureView,
                 water_out: &wgpu::TextureView,
                 flux_out: &wgpu::TextureView,
                 vel_write: &wgpu::TextureView,
                 vel_read: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &rt.hydraulic.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: rt.hydraulic.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&terrain_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(water_in) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&sed_a_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(flux_in) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(vel_write) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(water_out) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(flux_out) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&sed_b_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(vel_read) },
            ],
        })
    };
    // Write phase (passes 1+2): velocity at binding 5, dummy at binding 9.
    // Read phase (passes 3+4): dummy at binding 5, velocity at binding 9.
    let bg_a_write = mk_bg("erosion-hydraulic-bg-a-write",
        &water_a_view, &flux_a_view, &water_b_view, &flux_b_view,
        &velocity_view, &velocity_dummy_view);
    let bg_a_read = mk_bg("erosion-hydraulic-bg-a-read",
        &water_a_view, &flux_a_view, &water_b_view, &flux_b_view,
        &velocity_dummy_view, &velocity_read_view);
    let bg_b_write = mk_bg("erosion-hydraulic-bg-b-write",
        &water_b_view, &flux_b_view, &water_a_view, &flux_a_view,
        &velocity_view, &velocity_dummy_view);
    let bg_b_read = mk_bg("erosion-hydraulic-bg-b-read",
        &water_b_view, &flux_b_view, &water_a_view, &flux_a_view,
        &velocity_dummy_view, &velocity_read_view);

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
        let (bind_write, bind_read) = if pingpong_a {
            (&bg_a_write, &bg_a_read)
        } else {
            (&bg_b_write, &bg_b_read)
        };

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("erosion-hydraulic-iter"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("erosion-hydraulic-pass"),
                timestamp_writes: None,
            });
            // Passes 1+2 use the write-velocity bind group.
            pass.set_pipeline(&rt.hydraulic.pipeline_add_water_and_flux);
            pass.set_bind_group(0, bind_write, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            pass.set_pipeline(&rt.hydraulic.pipeline_update_water_and_velocity);
            pass.set_bind_group(0, bind_write, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            // Passes 3+4 swap to the read-velocity bind group.
            pass.set_pipeline(&rt.hydraulic.pipeline_erosion_deposition);
            pass.set_bind_group(0, bind_read, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            pass.set_pipeline(&rt.hydraulic.pipeline_advect_sediment);
            pass.set_bind_group(0, bind_read, &[]);
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
        // Erosion redistributes heights but doesn't change the grid
        // pitch — preserve the input's metric spacing.
        gsd_m_centre: tile.gsd_m_centre,
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
