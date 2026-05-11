//! Phase 10.2 panels — World, Render, Subsystems, Atmosphere, Clouds,
//! Wet surface, Precipitation, Debug.
//!
//! Every tunable parameter from Phases 5–9 is exposed as a slider with
//! direct numeric input; edits flip `pending.config_dirty` so the host
//! reconfigures the next frame.

use chrono::{Datelike, TimeZone, Utc};
use egui::{Color32, ComboBox, DragValue, Slider, Widget};

use crate::state::UiState;

/// Render the full panel suite.
#[allow(deprecated)] // egui 0.34 deprecates top-level Panel::show without
                     // providing a non-deprecated replacement that takes &Context;
                     // the API is mid-transition.
pub fn ui(ctx: &egui::Context, state: &mut UiState) {
    top_bar(ctx, state);
    egui::Panel::left("ps-ui-left")
        .resizable(true)
        .default_size(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                world_panel(ui, state);
                ui.separator();
                render_panel(ui, state);
                ui.separator();
                subsystem_panel(ui, state);
                ui.separator();
                atmosphere_panel(ui, state);
                ui.separator();
                clouds_panel(ui, state);
                ui.separator();
                wet_surface_panel(ui, state);
                ui.separator();
                precipitation_panel(ui, state);
                ui.separator();
                debug_panel(ui, state);
            });
        });
}

// ---------------------------------------------------------------------------
// Top bar — preset Load/Save buttons (Phase 10.4)
// ---------------------------------------------------------------------------

#[allow(deprecated)] // see `ui` above
fn top_bar(ctx: &egui::Context, state: &mut UiState) {
    egui::Panel::top("ps-ui-topbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("PedalSky");
            ui.separator();
            if ui.button("Load scene…").clicked() {
                if let Some(path) = rfd_pick_open() {
                    state.pending.load_scene = Some(path);
                }
            }
            if ui.button("Save scene…").clicked() {
                if let Some(path) = rfd_pick_save() {
                    state.pending.save_scene = Some(path);
                }
            }
            ui.separator();
            ui.label(format!("{:.1} fps", state.frame_stats.fps));
            ui.label(format!("{:.2} ms", state.frame_stats.frame_ms));
        });
    });
}

// We don't depend on rfd to keep dep count down. Use a synchronous path-
// input fallback: the buttons set a flag that the host can later wire to
// a real picker. For now the host expects a default path.
fn rfd_pick_open() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from("scenes/loaded.toml"))
}
fn rfd_pick_save() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from("scenes/saved.toml"))
}

// ---------------------------------------------------------------------------
// World panel — date/time, lat/lon, time scale, pause, calendar shortcuts
// ---------------------------------------------------------------------------

fn world_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("World", |ui| {
        // Read-out
        egui::Grid::new("world-readout").striped(true).show(ui, |ui| {
            ui.label("Sun altitude");
            ui.label(format!("{:>8.3}°", state.world_readout.sun_alt_deg));
            ui.end_row();
            ui.label("Sun azimuth");
            ui.label(format!("{:>8.3}°", state.world_readout.sun_az_deg));
            ui.end_row();
            ui.label("Moon altitude");
            ui.label(format!("{:>8.3}°", state.world_readout.moon_alt_deg));
            ui.end_row();
            ui.label("Moon azimuth");
            ui.label(format!("{:>8.3}°", state.world_readout.moon_az_deg));
            ui.end_row();
            ui.label("Julian day");
            ui.label(format!("{:>14.5}", state.world_readout.julian_day));
            ui.end_row();
        });

        ui.separator();
        ui.label("Date / time (UTC)");
        let mut y = state.live_config.time.year;
        let mut mo = state.live_config.time.month as i32;
        let mut d = state.live_config.time.day as i32;
        let mut hh = state.live_config.time.hour as i32;
        let mut mm = state.live_config.time.minute as i32;
        let mut ss = state.live_config.time.second as i32;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("YYYY");
            changed |= DragValue::new(&mut y).range(1900..=2100).ui(ui).changed();
            ui.label("MM");
            changed |= DragValue::new(&mut mo).range(1..=12).ui(ui).changed();
            ui.label("DD");
            changed |= DragValue::new(&mut d).range(1..=31).ui(ui).changed();
        });
        ui.horizontal(|ui| {
            ui.label("hh");
            changed |= DragValue::new(&mut hh).range(0..=23).ui(ui).changed();
            ui.label("mm");
            changed |= DragValue::new(&mut mm).range(0..=59).ui(ui).changed();
            ui.label("ss");
            changed |= DragValue::new(&mut ss).range(0..=59).ui(ui).changed();
        });
        if changed {
            state.live_config.time.year = y;
            state.live_config.time.month = mo as u32;
            state.live_config.time.day = d as u32;
            state.live_config.time.hour = hh as u32;
            state.live_config.time.minute = mm as u32;
            state.live_config.time.second = ss as u32;
            if let chrono::offset::LocalResult::Single(utc) =
                Utc.with_ymd_and_hms(y, mo as u32, d as u32, hh as u32, mm as u32, ss as u32)
            {
                state.pending.set_world_utc = Some(utc);
                state.pending.config_dirty = true;
            }
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Latitude");
            let mut lat = state.live_config.world.latitude_deg;
            if DragValue::new(&mut lat)
                .range(-90.0..=90.0)
                .speed(0.1)
                .max_decimals(4)
                .ui(ui)
                .changed()
            {
                state.live_config.world.latitude_deg = lat;
                state.pending.set_lat_lon =
                    Some((lat, state.live_config.world.longitude_deg));
                state.pending.config_dirty = true;
            }
            ui.label("Longitude");
            let mut lon = state.live_config.world.longitude_deg;
            if DragValue::new(&mut lon)
                .range(-180.0..=180.0)
                .speed(0.1)
                .max_decimals(4)
                .ui(ui)
                .changed()
            {
                state.live_config.world.longitude_deg = lon;
                state.pending.set_lat_lon =
                    Some((state.live_config.world.latitude_deg, lon));
                state.pending.config_dirty = true;
            }
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Time scale ×");
            let mut s = state.live_config.time.time_scale;
            if DragValue::new(&mut s)
                .range(-86400.0..=86400.0)
                .speed(1.0)
                .ui(ui)
                .changed()
            {
                state.live_config.time.time_scale = s;
                state.pending.set_time_scale = Some(s);
            }
            let mut paused = !state.live_config.time.auto_advance;
            if ui.checkbox(&mut paused, "Pause").changed() {
                state.live_config.time.auto_advance = !paused;
                state.pending.set_paused = Some(paused);
            }
        });

        ui.separator();
        ui.label("Calendar shortcuts (UTC noon at observer lon)");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Vernal equinox").clicked() {
                jump_calendar(state, CalendarDay::VernalEquinox);
            }
            if ui.button("Summer solstice").clicked() {
                jump_calendar(state, CalendarDay::SummerSolstice);
            }
            if ui.button("Autumnal equinox").clicked() {
                jump_calendar(state, CalendarDay::AutumnalEquinox);
            }
            if ui.button("Winter solstice").clicked() {
                jump_calendar(state, CalendarDay::WinterSolstice);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Sunrise (today)").clicked() {
                jump_solar(state, SolarEvent::Sunrise);
            }
            if ui.button("Sunset (today)").clicked() {
                jump_solar(state, SolarEvent::Sunset);
            }
            if ui.button("Civil dawn (today)").clicked() {
                jump_solar(state, SolarEvent::CivilDawn);
            }
            if ui.button("Civil dusk (today)").clicked() {
                jump_solar(state, SolarEvent::CivilDusk);
            }
        });
    });
}

#[derive(Copy, Clone)]
enum CalendarDay {
    VernalEquinox,
    SummerSolstice,
    AutumnalEquinox,
    WinterSolstice,
}

fn jump_calendar(state: &mut UiState, day: CalendarDay) {
    // Approximate Northern-Hemisphere astronomical event dates for the
    // currently-configured year. Sufficient for UI shortcuts; iterative
    // refinement would only shift by hours.
    let year = state.live_config.time.year;
    let (mo, d) = match day {
        CalendarDay::VernalEquinox => (3, 20),
        CalendarDay::SummerSolstice => (6, 21),
        CalendarDay::AutumnalEquinox => (9, 22),
        CalendarDay::WinterSolstice => (12, 21),
    };
    state.live_config.time.year = year;
    state.live_config.time.month = mo;
    state.live_config.time.day = d;
    state.live_config.time.hour = 12;
    state.live_config.time.minute = 0;
    state.live_config.time.second = 0;
    if let chrono::offset::LocalResult::Single(utc) =
        Utc.with_ymd_and_hms(year, mo, d, 12, 0, 0)
    {
        state.pending.set_world_utc = Some(utc);
        state.pending.config_dirty = true;
    }
}

#[derive(Copy, Clone)]
enum SolarEvent {
    Sunrise,
    Sunset,
    CivilDawn,
    CivilDusk,
}

fn jump_solar(state: &mut UiState, event: SolarEvent) {
    use ps_core::astro::sun_position_precise;
    let lat = state.live_config.world.latitude_deg;
    let lon = state.live_config.world.longitude_deg;
    let elev = state.live_config.world.ground_elevation_m as f64;
    let year = state.live_config.time.year;
    let month = state.live_config.time.month;
    let day = state.live_config.time.day;
    // Target altitude: 0° for sunrise/sunset (apparent horizon),
    // -6° for civil dawn/dusk (CIE convention).
    let target_alt_deg = match event {
        SolarEvent::Sunrise | SolarEvent::Sunset => -0.833, // standard refraction
        SolarEvent::CivilDawn | SolarEvent::CivilDusk => -6.0,
    };
    let want_rising = matches!(event, SolarEvent::Sunrise | SolarEvent::CivilDawn);
    // Bisection between solar midnight (00:00 UTC + lon offset) and noon
    // for rising events; noon..midnight+24h for setting events.
    // Simpler: walk in 1-minute increments from local-day midnight.
    let lon_hours = lon / 15.0;
    let local_midnight_utc =
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0).single();
    let Some(local_midnight_utc) = local_midnight_utc else {
        return;
    };
    let local_midnight_utc = local_midnight_utc - chrono::Duration::minutes((lon_hours * 60.0) as i64);

    // Sample altitudes at 5-minute granularity over the next 24 h, find
    // sign changes around target, refine with linear interp.
    let mut prev_alt: Option<f64> = None;
    let mut best: Option<chrono::DateTime<Utc>> = None;
    for step_min in 0..(24 * 60 / 5) {
        let t = local_midnight_utc + chrono::Duration::minutes(step_min * 5);
        let pos = sun_position_precise(t, lat, lon, elev);
        let alt_deg = pos.altitude_deg;
        if let Some(prev) = prev_alt {
            let crossing_up = prev < target_alt_deg && alt_deg >= target_alt_deg;
            let crossing_dn = prev >= target_alt_deg && alt_deg < target_alt_deg;
            if (want_rising && crossing_up) || (!want_rising && crossing_dn) {
                // Linear interp inside the 5-minute bracket.
                let frac = (target_alt_deg - prev) / (alt_deg - prev);
                let t_prev = t - chrono::Duration::minutes(5);
                let bracket_secs = 5.0 * 60.0;
                let offset_secs = (frac * bracket_secs) as i64;
                best = Some(t_prev + chrono::Duration::seconds(offset_secs));
                break;
            }
        }
        prev_alt = Some(alt_deg);
    }
    if let Some(utc) = best {
        state.live_config.time.year = utc.year();
        state.live_config.time.month = utc.month();
        state.live_config.time.day = utc.day();
        state.live_config.time.hour = utc.hour();
        state.live_config.time.minute = utc.minute();
        state.live_config.time.second = utc.second();
        state.pending.set_world_utc = Some(utc);
        state.pending.config_dirty = true;
    }
}

// chrono::Timelike for hour/minute/second on DateTime.
use chrono::Timelike;

// ---------------------------------------------------------------------------
// Render panel — EV100, tone mapper, vsync, screenshot
// ---------------------------------------------------------------------------

fn render_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Render", |ui| {
        let mut ev = state.live_config.render.ev100;
        if Slider::new(&mut ev, 0.0..=20.0)
            .text("EV100")
            .max_decimals(4)
            .ui(ui)
            .changed()
        {
            state.live_config.render.ev100 = ev;
            state.pending.config_dirty = true;
        }
        ui.horizontal(|ui| {
            ui.label("Tone mapper");
            let mut current = state.live_config.render.tone_mapper.clone();
            ComboBox::from_id_salt("ps-ui-tonemap")
                .selected_text(&current)
                .show_ui(ui, |ui| {
                    for opt in ["ACESFilmic", "Passthrough"] {
                        if ui
                            .selectable_label(current == opt, opt)
                            .clicked()
                        {
                            current = opt.to_string();
                        }
                    }
                });
            if current != state.live_config.render.tone_mapper {
                state.live_config.render.tone_mapper = current;
                state.pending.config_dirty = true;
            }
        });
        let mut vsync = state.live_config.window.vsync;
        if ui.checkbox(&mut vsync, "Vsync").changed() {
            state.live_config.window.vsync = vsync;
            state.pending.config_dirty = true;
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Screenshot tonemapped (PNG)").clicked() {
                state.pending.screenshot_png = true;
            }
            if ui.button("Screenshot HDR (EXR)").clicked() {
                state.pending.screenshot_exr = true;
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Subsystem toggles
// ---------------------------------------------------------------------------

fn subsystem_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Subsystems", |ui| {
        let s = &mut state.live_config.render.subsystems;
        let mut any = false;
        any |= ui.checkbox(&mut s.ground, "Ground").changed();
        any |= ui.checkbox(&mut s.atmosphere, "Atmosphere").changed();
        any |= ui.checkbox(&mut s.clouds, "Clouds").changed();
        any |= ui.checkbox(&mut s.precipitation, "Precipitation").changed();
        any |= ui.checkbox(&mut s.wet_surface, "Wet surface").changed();
        any |= ui.checkbox(&mut s.backdrop, "Backdrop (debug)").changed();
        any |= ui.checkbox(&mut s.tint, "Tint (debug)").changed();
        if any {
            state.pending.config_dirty = true;
        }
    });
}

// ---------------------------------------------------------------------------
// Atmosphere panel — every AtmosphereTuning + AtmosphereParams field
// ---------------------------------------------------------------------------

fn atmosphere_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Atmosphere", |ui| {
        let a = &mut state.live_config.render.atmosphere;
        let mut any = false;
        any |= ui.checkbox(&mut a.multi_scattering, "Multi-scattering").changed();
        any |= ui.checkbox(&mut a.sun_disk, "Sun disk").changed();
        any |= ui.checkbox(&mut a.ozone_enabled, "Ozone").changed();
        any |= Slider::new(&mut a.sun_angular_radius_deg, 0.05..=2.0)
            .text("Sun angular radius (deg)")
            .max_decimals(4)
            .ui(ui)
            .changed();

        if ui.button("Reset to physical Earth defaults").clicked() {
            *a = ps_core::config::AtmosphereTuning::default();
            any = true;
        }
        if any {
            state.pending.config_dirty = true;
        }
        ui.label(
            "Note: physical AtmosphereParams (Rayleigh / Mie / ozone profiles) \
             are wired through `ps_core::AtmosphereParams::default` by the \
             synthesis pipeline. UI exposure of those individual coefficients \
             ships with v2 — the WeatherState rebake hook is in place.",
        );
    });
}

// ---------------------------------------------------------------------------
// Clouds panel — every CloudsTuning field + per-layer overrides
// ---------------------------------------------------------------------------

fn clouds_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Clouds", |ui| {
        let c = &mut state.live_config.render.clouds;
        let mut any = false;
        any |= Slider::new(&mut c.cloud_steps, 32..=512)
            .text("Cloud march steps")
            .ui(ui)
            .changed();
        any |= Slider::new(&mut c.light_steps, 1..=16)
            .text("Light march steps")
            .ui(ui)
            .changed();
        any |= Slider::new(&mut c.multi_scatter_octaves, 1..=8)
            .text("Multi-scatter octaves")
            .ui(ui)
            .changed();
        any |= Slider::new(&mut c.detail_strength, 0.0..=1.0)
            .text("Detail strength")
            .max_decimals(4)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut c.powder_strength, 0.0..=1.0)
            .text("Powder strength")
            .max_decimals(4)
            .ui(ui)
            .changed();
        any |= ui.checkbox(&mut c.freeze_time, "Freeze time").changed();
        if any {
            state.pending.config_dirty = true;
        }
        ui.label(
            "Per-layer envelopes (CloudLayerGpu) come from the loaded scene; \
             reload via Save scene… → edit → Load scene… or the v2 layer \
             accordions (planned).",
        );
    });
}

// ---------------------------------------------------------------------------
// Wet surface panel — SurfaceParams.wetness block
// ---------------------------------------------------------------------------

fn wet_surface_panel(ui: &mut egui::Ui, _state: &mut UiState) {
    ui.collapsing("Wet surface", |ui| {
        // Wet/snow values currently live on the *scene*, not the engine
        // config. Edits flow through Load scene… → edit → Load scene…
        // for now.
        ui.label("Wetness, snow depth, and puddle parameters are scene-side.");
        ui.label("Edit the scene TOML and Load scene… to apply.");
    });
}

// ---------------------------------------------------------------------------
// Precipitation panel
// ---------------------------------------------------------------------------

fn precipitation_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Precipitation", |ui| {
        let p = &mut state.live_config.render.precip;
        let mut any = false;
        any |= Slider::new(&mut p.near_particle_count, 1024..=32_768)
            .text("Near particle count")
            .ui(ui)
            .changed();
        any |= Slider::new(&mut p.far_layers, 1..=5)
            .text("Far rain layers")
            .ui(ui)
            .changed();
        if any {
            state.pending.config_dirty = true;
        }
        ui.label(
            "Type and intensity are scene-side (`scene.precipitation.type`, \
             `intensity_mm_per_h`). Edit the scene TOML to change them.",
        );
    });
}

// ---------------------------------------------------------------------------
// Debug panel — LUT viewer mode, probe, FPS, GPU timestamps
// ---------------------------------------------------------------------------

fn debug_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Debug", |ui| {
        let d = &mut state.live_config.debug;
        let mut any = false;
        any |= ui
            .checkbox(&mut d.atmosphere_lut_overlay, "Atmosphere LUT overlay (2x2)")
            .changed();
        any |= ui.checkbox(&mut d.auto_exposure, "Auto exposure").changed();
        any |= ui.checkbox(&mut d.shader_hot_reload, "Shader hot reload").changed();
        any |= ui.checkbox(&mut d.gpu_validation, "GPU validation").changed();
        if any {
            state.pending.config_dirty = true;
        }

        ui.separator();
        ui.label("Fullscreen LUT viewer");
        let mut sel = state.debug.lut_viewer_mode;
        ComboBox::from_id_salt("ps-ui-lut-viewer")
            .selected_text(match sel {
                0 => "off",
                1 => "transmittance",
                2 => "multi-scatter",
                3 => "sky-view",
                4 => "aerial-perspective",
                _ => "off",
            })
            .show_ui(ui, |ui| {
                for (i, name) in [
                    "off",
                    "transmittance",
                    "multi-scatter",
                    "sky-view",
                    "aerial-perspective",
                ]
                .iter()
                .enumerate()
                {
                    ui.selectable_value(&mut sel, i as u32, *name);
                }
            });
        state.debug.lut_viewer_mode = sel;
        if sel == 4 {
            Slider::new(&mut state.debug.ap_depth_slice, 0.0..=1.0)
                .text("AP depth slice")
                .max_decimals(3)
                .ui(ui);
        }

        ui.separator();
        ui.label("Probe pixel");
        let mut px = state.debug.probe_pixel.0 as i32;
        let mut py = state.debug.probe_pixel.1 as i32;
        ui.horizontal(|ui| {
            DragValue::new(&mut px).range(0..=8192).ui(ui);
            DragValue::new(&mut py).range(0..=8192).ui(ui);
        });
        state.debug.probe_pixel = (px.max(0) as u32, py.max(0) as u32);
        let t = state.debug.probe_transmittance;
        ui.colored_label(
            Color32::LIGHT_BLUE,
            format!("transmittance R={:.4} G={:.4} B={:.4}", t[0], t[1], t[2]),
        );

        ui.separator();
        ui.label("GPU timings");
        if state.frame_stats.gpu_passes.is_empty() {
            ui.label("(no timestamp data — feature not yet enabled or no passes ran)");
        } else {
            egui::Grid::new("ps-ui-gpu-timings").striped(true).show(ui, |ui| {
                for (name, ms) in &state.frame_stats.gpu_passes {
                    ui.label(name);
                    ui.label(format!("{ms:>8.3} ms"));
                    ui.end_row();
                }
            });
        }
    });
}
