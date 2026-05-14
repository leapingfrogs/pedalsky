//! `synthesise` — Phase 3 entry point.
//!
//! Builds a [`ps_core::WeatherState`] from a parsed scene, the engine
//! config, the current `WorldState`, and a `GpuContext`. Allocates GPU
//! textures + storage buffer and uploads CPU-synthesised data via
//! `queue.write_*`.

use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use ps_core::{
    AtmosphereParams, CloudLayerGpu, Config, GpuContext, Scene, SurfaceParams, WeatherState,
    WeatherTextures, WorldState,
};

/// Monotonic counter shared across the process; bumped once per
/// `synthesise` invocation. Stamped into `WeatherState::revision` so
/// downstream subsystems can cache bind groups keyed on it (cheap u64
/// comparison per frame vs rebuilding a multi-entry wgpu bind group).
///
/// Starts at 1 so the default-initialised `WeatherState` (revision 0)
/// always compares unequal to a real synthesised state.
static SYNTHESIS_REVISION: AtomicU64 = AtomicU64::new(1);

use crate::cloud_layers::{check_non_overlap, synthesise_cloud_layers};
use crate::cloud_type_grid::CloudTypeGrid;
use crate::density_mask::TopDownMask;
use crate::haze_extinction_per_m;
use crate::weather_map::WeatherMap;
use crate::wind_field::WindField;

/// Errors raised by [`synthesise`].
#[derive(Debug, Error)]
pub enum SynthesisError {
    /// Two cloud layers overlap in altitude (plan §3.2.2).
    #[error(
        "cloud layers {a} and {b} overlap in altitude; v1 requires vertically disjoint layers"
    )]
    OverlappingCloudLayers {
        /// Index of first overlapping layer.
        a: usize,
        /// Index of second overlapping layer.
        b: usize,
    },
}

/// Build a [`WeatherState`] from `scene` + `config` + `world`, allocating
/// GPU resources via `gpu`.
pub fn synthesise(
    scene: &Scene,
    config: &Config,
    world: &WorldState,
    gpu: &GpuContext,
) -> Result<WeatherState, SynthesisError> {
    // §3.2.2 cloud-layer envelope synthesis + non-overlap.
    let cloud_layers = synthesise_cloud_layers(&scene.clouds.layers);
    if let Err((a, b, _, _)) = check_non_overlap(&cloud_layers) {
        return Err(SynthesisError::OverlappingCloudLayers { a, b });
    }

    // §3.2.1 visibility → Mie haze.
    let haze = haze_extinction_per_m(scene.surface.visibility_m);

    // §3.2 atmosphere params: physical defaults + config-driven overrides.
    let atmosphere = AtmosphereParams {
        planet_radius_m: config.world.ground_radius_m,
        atmosphere_top_m: config.world.ground_radius_m + config.world.atmosphere_top_m,
        ground_albedo: glam::Vec4::new(
            config.world.ground_albedo[0],
            config.world.ground_albedo[1],
            config.world.ground_albedo[2],
            0.0,
        ),
        haze_extinction_per_m: glam::Vec4::new(haze.x, haze.y, haze.z, 0.0),
        ..AtmosphereParams::default()
    };

    // §3.2 surface scalars.
    let surface = SurfaceParams {
        visibility_m: scene.surface.visibility_m,
        temperature_c: scene.surface.temperature_c,
        dewpoint_c: scene.surface.dewpoint_c,
        pressure_hpa: scene.surface.pressure_hpa,
        wind_dir_deg: scene.surface.wind_dir_deg,
        wind_speed_mps: scene.surface.wind_speed_mps,
        ground_wetness: scene.surface.wetness.ground_wetness,
        puddle_coverage: scene.surface.wetness.puddle_coverage,
        snow_depth_m: scene.surface.wetness.snow_depth_m,
        puddle_start: scene.surface.wetness.puddle_start,
        precip_intensity_mm_per_h: scene.precipitation.intensity_mm_per_h,
        precip_kind: match scene.precipitation.kind {
            ps_core::PrecipKind::None => 0.0,
            ps_core::PrecipKind::Rain => 1.0,
            ps_core::PrecipKind::Snow => 2.0,
            ps_core::PrecipKind::Sleet => 3.0,
        },
        material: scene.surface.material.as_u32() as f32,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };

    // Phase 12.1 — load the gridded coverage once. None when the
    // scene has no `coverage_grid` block or when the file failed to
    // load. Used by both the weather map (R-channel spatial gate)
    // and the top-down density mask (per-pixel column integral).
    let loaded_grid = crate::coverage_grid::load(scene);

    // §3.2.3 weather map.
    let weather_map = WeatherMap::synthesise(scene, &scene.surface);
    let (wm_tex, wm_view) = weather_map.upload(&gpu.device, &gpu.queue);

    // §3.2.4 wind field.
    let wind = WindField::synthesise(scene);
    let (wf_tex, wf_view) = wind.upload(&gpu.device, &gpu.queue);

    // §3.2.5 top-down density mask. Phase 12.1 / followup #76 — when
    // a coverage grid is loaded the per-pixel mask reflects it; the
    // ground shader's overcast-diffuse modulation (Phase 12.6) then
    // brightens up the gaps between cloud lozenges as expected.
    let mask = TopDownMask::synthesise(&cloud_layers, loaded_grid.as_ref());
    let (m_tex, m_view) = mask.upload(&gpu.device, &gpu.queue);

    // Phase 12.1 — per-pixel cloud-type override grid. Sentinel-
    // filled (255) when the scene has no `coverage_grid.type_path`.
    let type_grid = CloudTypeGrid::synthesise(scene);
    let (ct_tex, ct_view) = type_grid.upload(&gpu.device, &gpu.queue);

    // Cloud-layers storage buffer — at least one slot so the shader's
    // array<...> binding is non-empty.
    let cloud_layer_count = cloud_layers.len() as u32;
    let buffer_data: Vec<CloudLayerGpu> = if cloud_layers.is_empty() {
        vec![CloudLayerGpu::default()]
    } else {
        cloud_layers.clone()
    };
    let cloud_layers_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloud-layers"),
        size: std::mem::size_of_val::<[CloudLayerGpu]>(buffer_data.as_slice()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&cloud_layers_buffer, 0, bytemuck::cast_slice(&buffer_data));

    Ok(WeatherState {
        atmosphere,
        cloud_layers,
        surface,
        haze_extinction_per_m: haze,
        sun_direction: world.sun_direction_world,
        sun_illuminance: glam::Vec3::splat(world.toa_illuminance_lux),
        textures: WeatherTextures {
            weather_map: wm_tex,
            weather_map_view: wm_view,
            wind_field: wf_tex,
            wind_field_view: wf_view,
            overcast_field: m_tex,
            overcast_field_view: m_view,
            cloud_type_grid: ct_tex,
            cloud_type_grid_view: ct_view,
        },
        cloud_layers_buffer,
        cloud_layer_count,
        scene_strikes_per_min_per_km2: scene.lightning.strikes_per_min_per_km2,
        scene_aurora_kp: scene.aurora.kp_index,
        scene_aurora_intensity_override: scene.aurora.intensity_override,
        scene_aurora_colour_bias: {
            let bias = scene.aurora.predominant_colour.bias();
            [bias[0], bias[1], bias[2], 0.0]
        },
        scene_water: scene.water.clone(),
        // Synthesis assumes clouds are rendered; the host downgrades
        // this in `WeatherState` for the executor when
        // `[render.subsystems].clouds = false`. Audit §3.2.
        cloud_render_active: true,
        // Fresh synthesis re-populated the overcast field; the host
        // will re-zero it (and stamp this) if clouds-disabled this
        // frame. Audit §H3.
        overcast_zeroed_at_revision: None,
        revision: SYNTHESIS_REVISION.fetch_add(1, Ordering::Relaxed),
    })
}
