//! Phase 3 weather-state types — Pod structs + the `WeatherState` bundle.
//!
//! The synthesis pipeline that *populates* these lives in `ps-synthesis`;
//! the type definitions live here so `PrepareContext` can borrow
//! `&WeatherState` without `ps-core` depending on `ps-synthesis`. See
//! plan §3.2.
//!
//! [`AtmosphereParams`] doubles as the Phase 4 §4.1 `WorldUniforms`
//! payload — the WGSL declaration in `shaders/common/uniforms.wgsl`
//! mirrors this struct under the name `WorldUniforms`. The
//! [`WorldUniformsGpu`] type alias makes the bind-group-1 intent
//! explicit at use sites.

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

/// Hillaire-2020 atmosphere parameters (plan §5.1).
///
/// Internal unit convention: metres throughout. Defaults match the Earth
/// values in the plan; UI sliders later edit individual fields.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct AtmosphereParams {
    /// Planet radius in metres.
    pub planet_radius_m: f32,
    /// Top-of-atmosphere altitude in metres.
    pub atmosphere_top_m: f32,
    /// Rayleigh scale height in metres (Earth ≈ 8000).
    pub rayleigh_scale_height_m: f32,
    /// Mie scale height in metres (Earth ≈ 1200).
    pub mie_scale_height_m: f32,

    /// Rayleigh scattering coefficients per metre, w unused.
    pub rayleigh_scattering: Vec4,
    /// Mie scattering coefficients per metre, w unused.
    pub mie_scattering: Vec4,
    /// Mie absorption coefficients per metre, w unused.
    pub mie_absorption: Vec4,

    /// Henyey-Greenstein g for Mie phase function.
    pub mie_g: f32,
    /// _pad to keep mie_g + ozone scalars on a 16B boundary.
    pub _pad_after_mie_g: [f32; 3],

    /// Ozone absorption per metre, w unused.
    pub ozone_absorption: Vec4,
    /// Ozone layer centre altitude in metres (Earth ≈ 25 000).
    pub ozone_center_m: f32,
    /// Ozone layer thickness in metres (Earth ≈ 30 000).
    pub ozone_thickness_m: f32,
    /// _pad.
    pub _pad_after_ozone: [f32; 2],

    /// Linear-sRGB ground albedo, w unused.
    pub ground_albedo: Vec4,
    /// Mie haze extinction per metre (Koschmieder), w unused.
    pub haze_extinction_per_m: Vec4,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self {
            planet_radius_m: 6_360_000.0,
            atmosphere_top_m: 6_460_000.0,
            rayleigh_scale_height_m: 8_000.0,
            mie_scale_height_m: 1_200.0,

            rayleigh_scattering: Vec4::new(5.802e-6, 13.558e-6, 33.100e-6, 0.0),
            mie_scattering: Vec4::new(3.996e-6, 3.996e-6, 3.996e-6, 0.0),
            mie_absorption: Vec4::new(4.4e-6, 4.4e-6, 4.4e-6, 0.0),

            mie_g: 0.8,
            _pad_after_mie_g: [0.0; 3],

            ozone_absorption: Vec4::new(0.650e-6, 1.881e-6, 0.085e-6, 0.0),
            ozone_center_m: 25_000.0,
            ozone_thickness_m: 30_000.0,
            _pad_after_ozone: [0.0; 2],

            ground_albedo: Vec4::new(0.18, 0.18, 0.18, 0.0),
            haze_extinction_per_m: Vec4::new(0.0, 0.0, 0.0, 0.0),
        }
    }
}

/// Bind-group-1 uniform payload (Plan §4.1 / §4.2). Currently identical
/// to [`AtmosphereParams`]; once Phase 5 lands its dependent
/// transmittance / multi-scatter LUTs the binding may grow.
pub type WorldUniformsGpu = AtmosphereParams;

/// Per-cloud-layer parameters consumed by Phase 6 (plan §3.2 + §6).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct CloudLayerGpu {
    /// Layer base AGL in metres.
    pub base_m: f32,
    /// Layer top AGL in metres.
    pub top_m: f32,
    /// Coverage in [0, 1] before per-pixel weather-map modulation.
    pub coverage: f32,
    /// Optical density multiplier.
    pub density_scale: f32,
    /// Cloud type widened from the `CloudType` `repr(u8)`.
    pub cloud_type: u32,
    /// Bias on shape-noise frequency (negative = blockier).
    pub shape_bias: f32,
    /// Bias on detail-noise frequency.
    pub detail_bias: f32,
    /// Anvil bias (only meaningful for Cumulonimbus).
    pub anvil_bias: f32,
}

impl Default for CloudLayerGpu {
    fn default() -> Self {
        Self {
            base_m: 0.0,
            top_m: 0.0,
            coverage: 0.0,
            density_scale: 1.0,
            cloud_type: 0,
            shape_bias: 0.0,
            detail_bias: 0.0,
            anvil_bias: 0.0,
        }
    }
}

/// Surface scalars consumed by ground / wet-surface / precipitation
/// shaders. Plan §3.2 SurfaceParams.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, PartialEq)]
pub struct SurfaceParams {
    /// Meteorological visibility in metres.
    pub visibility_m: f32,
    /// Surface temperature in °C.
    pub temperature_c: f32,
    /// Dew-point in °C.
    pub dewpoint_c: f32,
    /// Mean sea-level pressure in hPa.
    pub pressure_hpa: f32,
    /// Wind direction in degrees (meteorological — direction wind comes from).
    pub wind_dir_deg: f32,
    /// Wind speed in m/s at 10 m AGL.
    pub wind_speed_mps: f32,
    /// Continuous wetness scalar in [0, 1].
    pub ground_wetness: f32,
    /// Fraction of ground covered by puddles in [0, 1].
    pub puddle_coverage: f32,
    /// Snow depth in metres.
    pub snow_depth_m: f32,
    /// Wetness threshold for puddles, default 0.6.
    pub puddle_start: f32,
    /// Precipitation intensity in mm/h. Mirrors the source value the
    /// Phase 8 precip subsystem reads from `Scene.precipitation`. Lives
    /// here so the Phase 7 ground shader can also consult it (for ripple
    /// strength). `0` when there is no precipitation.
    pub precip_intensity_mm_per_h: f32,
    /// Precipitation kind. `0` = none, `1` = rain, `2` = snow, `3` = sleet
    /// (matches `ps_core::PrecipKind` ordering). Stored as `f32` so it
    /// lives inside the same uniform without alignment surprises.
    pub precip_kind: f32,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self {
            visibility_m: 30_000.0,
            temperature_c: 17.0,
            dewpoint_c: 7.0,
            pressure_hpa: 1018.0,
            wind_dir_deg: 240.0,
            wind_speed_mps: 5.0,
            ground_wetness: 0.0,
            puddle_coverage: 0.0,
            snow_depth_m: 0.0,
            puddle_start: 0.6,
            precip_intensity_mm_per_h: 0.0,
            precip_kind: 0.0,
        }
    }
}

/// GPU-resident textures synthesised by `ps-synthesis::synthesise`.
#[derive(Debug)]
pub struct WeatherTextures {
    /// 128×128 RGBA16Float weather map (R=coverage, G=reserved, B=base offset, A=precip).
    pub weather_map: wgpu::Texture,
    /// Default view onto `weather_map`.
    pub weather_map_view: wgpu::TextureView,
    /// 32×32×16 RGBA16Float wind field (channels: u, v, w, turbulence).
    pub wind_field: wgpu::Texture,
    /// Default view onto `wind_field`.
    pub wind_field_view: wgpu::TextureView,
    /// 2D R8Unorm top-down density mask matching weather-map extent.
    pub top_down_density_mask: wgpu::Texture,
    /// Default view onto `top_down_density_mask`.
    pub top_down_density_mask_view: wgpu::TextureView,
    /// Phase 12.1 — 128×128 R8Uint per-pixel cloud-type override
    /// grid. Pixel value 0..7 selects a cloud type (matching
    /// [`crate::CloudType`]); value 255 is the sentinel "use the
    /// per-layer cloud_type instead".
    pub cloud_type_grid: wgpu::Texture,
    /// Default view onto `cloud_type_grid`.
    pub cloud_type_grid_view: wgpu::TextureView,
}

/// Synthesised weather state passed through `PrepareContext` to every
/// rendering subsystem. Plan §3.2.
///
/// The struct definition lives in `ps-core` so `PrepareContext` can
/// borrow `&WeatherState`; the synthesis pipeline that *fills* it lives
/// in `ps-synthesis` (which depends on `ps-core`).
#[derive(Debug)]
pub struct WeatherState {
    /// Hillaire-2020 atmosphere parameters.
    pub atmosphere: AtmosphereParams,
    /// Per-layer cloud envelopes (synthesised for the current scene).
    pub cloud_layers: Vec<CloudLayerGpu>,
    /// Surface scalars.
    pub surface: SurfaceParams,
    /// Mie haze extinction per metre, RGB.
    pub haze_extinction_per_m: Vec3,
    /// World-space sun direction (refreshed from `WorldState` each frame).
    pub sun_direction: Vec3,
    /// Sun illuminance proxy (RGB lux); Phase 5 widens this once the
    /// transmittance LUT is available.
    pub sun_illuminance: Vec3,
    /// GPU-resident textures.
    pub textures: WeatherTextures,
    /// Storage buffer carrying `cloud_layers` for the GPU.
    pub cloud_layers_buffer: wgpu::Buffer,
    /// Number of valid entries in `cloud_layers_buffer`.
    pub cloud_layer_count: u32,
    /// Phase 12.3 — Poisson trigger rate for the lightning subsystem,
    /// taken verbatim from `Scene.lightning.strikes_per_min_per_km2`.
    /// Lives here so `LightningSubsystem::prepare` can read it via
    /// `ctx.weather` without needing a separate scene reference.
    pub scene_strikes_per_min_per_km2: f32,
}

impl WeatherState {
    /// Construct a minimal `WeatherState` with 1×1×1 zeroed textures
    /// and an empty cloud-layer buffer. Used by `ps-core` integration
    /// tests that need a borrow-able `&WeatherState` without dragging
    /// in `ps-synthesis`.
    ///
    /// This is **not** suitable for rendering: the synthesised pipeline
    /// shapes the textures to specific dimensions (128×128 weather map,
    /// 32×32×16 wind field). Real users build a `WeatherState` via
    /// `ps_synthesis::synthesise`.
    pub fn stub_for_tests(gpu: &crate::GpuContext) -> Self {
        let make_2d = |label: &'static str, format: wgpu::TextureFormat| {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let make_3d = || {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("stub-wind"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (weather_map, weather_map_view) = make_2d("stub-wm", wgpu::TextureFormat::Rgba16Float);
        // Upload coverage = 1.0 so cloud subsystem tests against this stub
        // see a fully-covered weather map; the cloud layer's own coverage
        // scalar (0.4 by default) still does the gating.
        {
            let coverage_pixel = [
                half::f16::from_f32(1.0),
                half::f16::from_f32(0.0),
                half::f16::from_f32(0.0),
                half::f16::from_f32(0.0),
            ];
            let bytes: &[u8] = bytemuck::cast_slice(&coverage_pixel);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &weather_map,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        let (wind_field, wind_field_view) = make_3d();
        let (top_down_density_mask, top_down_density_mask_view) =
            make_2d("stub-mask", wgpu::TextureFormat::R8Unorm);
        // Upload mask = 0 ("clear sky") so ground-shading tests that
        // use this stub see the standard sky-view-LUT ambient, not
        // the Phase 12.6 cloud-modulated overcast diffuse. Tests that
        // need the precip cloud-occlusion gate (mask = 1) override
        // this via test_harness::HeadlessApp's cloud_mask_override.
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &top_down_density_mask,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        // Phase 12.1 — cloud type grid (R8Uint). Stub filled with 255
        // (sentinel meaning "use per-layer type"); tests that need a
        // specific override should upload their own.
        let (cloud_type_grid, cloud_type_grid_view) =
            make_2d("stub-cloud-type-grid", wgpu::TextureFormat::R8Uint);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cloud_type_grid,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let cloud_layers_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stub-cloud-layers"),
            size: std::mem::size_of::<CloudLayerGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            atmosphere: AtmosphereParams::default(),
            cloud_layers: Vec::new(),
            surface: SurfaceParams::default(),
            haze_extinction_per_m: Vec3::ZERO,
            sun_direction: Vec3::Y,
            sun_illuminance: Vec3::splat(127_500.0),
            textures: WeatherTextures {
                weather_map,
                weather_map_view,
                wind_field,
                wind_field_view,
                top_down_density_mask,
                top_down_density_mask_view,
                cloud_type_grid,
                cloud_type_grid_view,
            },
            cloud_layers_buffer,
            cloud_layer_count: 0,
            scene_strikes_per_min_per_km2: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<AtmosphereParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<CloudLayerGpu>() % 16, 0);
        assert_eq!(std::mem::size_of::<SurfaceParams>() % 16, 0);
    }

    #[test]
    fn struct_sizes_pinned() {
        // Phase 4 will lock these against WGSL via naga; pin sizes now so
        // accidental layout changes are visible in code review.
        assert_eq!(std::mem::size_of::<AtmosphereParams>(), 144);
        assert_eq!(std::mem::size_of::<CloudLayerGpu>(), 32);
        assert_eq!(std::mem::size_of::<SurfaceParams>(), 48);
    }
}
