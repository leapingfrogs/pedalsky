//! Phase 5 LUT diagnostic — drives a single frame headless, reads back
//! a sample texel from each LUT, and prints diagnostic values. Used
//! to debug the "near-zero output" problem identified by the
//! Hillaire-reference gap analysis.
//!
//! With the temporary debug instrumentation in multiscatter.comp.wgsl,
//! the multi-scatter LUT stores `(l_2.g, f_ms.g, l_full.g, 1)` per texel.
//! Reading the centre texel tells us:
//!   - l_2 (the numerator of the geometric-series closed form)
//!   - f_ms (the denominator factor, expected to be in 0.3-0.6 range)
//!   - l_full (final per-unit-illuminance multi-scatter contribution)
//!
//! Reference values per the agent's analysis:
//!   l_2.g ≈ 0.005, f_ms.g ≈ 0.4, l_full.g ≈ 0.008 per-unit-illuminance.

use std::sync::OnceLock;

use half::f16;
use ps_app::test_harness::{HeadlessApp, TestSetup};
use ps_core::{Config, GpuContext};

fn gpu() -> Option<&'static GpuContext> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| match ps_core::gpu::init_headless() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping diag — no GPU adapter: {e}");
            None
        }
    })
    .as_ref()
}

#[test]
fn dump_multiscatter_centre_texel() {
    let Some(gpu) = gpu() else { return };

    let mut config = Config::default();
    config.render.subsystems.atmosphere = true;
    config.render.subsystems.backdrop = false;
    config.render.subsystems.tint = false;
    config.render.subsystems.ground = false;
    config.render.subsystems.clouds = false;
    config.render.subsystems.precipitation = false;
    config.render.subsystems.wet_surface = false;

    let setup = TestSetup::new(gpu, &config, (64, 64));
    let mut app = HeadlessApp::new(gpu, &config, setup).expect("HeadlessApp::new");
    let _pixels = app.render_one_frame(gpu);

    // The multi-scatter LUT is 32×32 Rgba16Float. We read the (16, 16) texel.
    // To do that we need a handle to the LUT texture — pull it from the
    // HeadlessApp's atmosphere_luts.
    let luts = app
        .atmosphere_luts_for_diag()
        .expect("atmosphere enabled — LUTs should be Some");

    // Sample three multi-scatter LUT points:
    //  (16, 0)  — sun on horizon, ground level (u=0.5 → sun_cos=0)
    //  (31, 0)  — sun overhead, ground level (u=~1 → sun_cos=+1)
    //  (31, 31) — sun overhead, top of atmosphere
    let ms_horizon_ground = readback_2d_rgba16f_single_pixel(gpu, &luts.multiscatter, 16, 0);
    let ms_overhead_ground = readback_2d_rgba16f_single_pixel(gpu, &luts.multiscatter, 31, 0);
    let ms_overhead_top = readback_2d_rgba16f_single_pixel(gpu, &luts.multiscatter, 31, 31);
    eprintln!(
        "multi-scatter LUT samples:\n\
         (16, 0) sun-horizon ground: {ms_horizon_ground:?}\n\
         (31, 0) sun-overhead ground: {ms_overhead_ground:?}\n\
         (31,31) sun-overhead top:   {ms_overhead_top:?}"
    );
    let pixel = ms_overhead_ground;

    // Sample several sky-view LUT texels (192×108, RGB cd/m²·sr).
    // With Hillaire reference parametrisation: v=0 zenith, v≈0.5 horizon,
    // v=1 nadir.  Camera at 1.7 m altitude → horizon is just past
    // v=0.5.  Pick rows that straddle the sky.
    let sv_zenith = readback_2d_rgba16f_single_pixel(gpu, &luts.skyview, 96, 4);
    let sv_mid = readback_2d_rgba16f_single_pixel(gpu, &luts.skyview, 96, 30);
    let sv_horizon_at_sun = readback_2d_rgba16f_single_pixel(gpu, &luts.skyview, 0, 53);
    eprintln!(
        "sky-view zenith (96,4): {sv_zenith:?}\n\
         sky-view mid (96,30):    {sv_mid:?}\n\
         sky-view horizon-at-sun (0,53): {sv_horizon_at_sun:?}"
    );

    // Sample AP LUT — 32×32×32, but readback helper is 2D.  The test
    // camera defaults to (0, 1.7 m, 0) looking horizontally, so view
    // rays at the centre NDC tilt very slightly downward and hit the
    // ground after only a few hundred metres.  Far froxels along that
    // ray legitimately read zero (the ray no longer exists).  Read a
    // near-camera froxel to confirm the bake produces sensible values
    // for rays that haven't yet hit ground.
    let ap_near = readback_3d_rgba16f_single_pixel(gpu, &luts.aerial_perspective, 16, 16, 2);
    eprintln!("AP near (16,16,2): {ap_near:?}");

    // Reasonableness (very loose — we're just confirming non-zero in all
    // three RGB channels).  Hillaire reference per-unit-illuminance is
    // O(1e−3..1e−1); the bake currently scales by sun_illuminance so
    // values can be much larger.
    let total = pixel[0] + pixel[1] + pixel[2];
    assert!(
        total > 1e-4,
        "multi-scatter centre should be non-zero (got rgb sum {})",
        total
    );
    // Sky-view should be non-zero somewhere in the upper hemisphere.
    let sv_total = sv_mid[0] + sv_mid[1] + sv_mid[2] + sv_zenith[0] + sv_zenith[1] + sv_zenith[2];
    assert!(
        sv_total > 1e-2,
        "sky-view upper hemisphere should be non-zero (got total {})",
        sv_total
    );
}

fn readback_3d_rgba16f_single_pixel(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    x: u32,
    y: u32,
    z: u32,
) -> [f32; 4] {
    let size = texture.size();
    let w = size.width;
    let h = size.height;
    let d = size.depth_or_array_layers;
    let bytes_per_pixel = 8u32;
    let unpadded = w * bytes_per_pixel;
    let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phase5_diag-3d-staging"),
        size: (aligned * h * d) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("phase5_diag-3d-copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map recv").expect("map");
    let bytes = slice.get_mapped_range().to_vec();

    let slice_start = (z * aligned * h) as usize;
    let row_start = slice_start + (y * aligned) as usize;
    let off = row_start + (x * bytes_per_pixel) as usize;
    [
        f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32(),
        f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32(),
        f16::from_le_bytes([bytes[off + 4], bytes[off + 5]]).to_f32(),
        f16::from_le_bytes([bytes[off + 6], bytes[off + 7]]).to_f32(),
    ]
}

/// Read back a single texel at `(x, y)` from a 2D Rgba16Float texture.
/// Returns `[r, g, b, a]` as f32.
fn readback_2d_rgba16f_single_pixel(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    x: u32,
    y: u32,
) -> [f32; 4] {
    let size = texture.size();
    let w = size.width;
    let h = size.height;
    let bytes_per_pixel = 8u32;
    let unpadded = w * bytes_per_pixel;
    let aligned = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phase5_diag-staging"),
        size: (aligned * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("phase5_diag-copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map recv").expect("map");
    let bytes = slice.get_mapped_range().to_vec();
    drop(staging);

    let row_start = (y * aligned) as usize;
    let off = row_start + (x * bytes_per_pixel) as usize;
    [
        f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32(),
        f16::from_le_bytes([bytes[off + 2], bytes[off + 3]]).to_f32(),
        f16::from_le_bytes([bytes[off + 4], bytes[off + 5]]).to_f32(),
        f16::from_le_bytes([bytes[off + 6], bytes[off + 7]]).to_f32(),
    ]
}
