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
    // Phase 13.6 — compass rose overlay (top-right corner). Drawn
    // before the left panel so its layer order is below the sliders
    // when they overlap on narrow windows.
    compass_overlay(ctx, state);
    egui::Panel::left("ps-ui-left")
        .resizable(true)
        .default_size(360.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                world_panel(ui, state);
                ui.separator();
                camera_panel(ui, state);
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
                water_panel(ui, state);
                ui.separator();
                precipitation_panel(ui, state);
                ui.separator();
                godrays_panel(ui, state);
                ui.separator();
                bloom_panel(ui, state);
                ui.separator();
                lightning_panel(ui, state);
                ui.separator();
                aurora_panel(ui, state);
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
            // Phase 14 — Fetch real weather. Pulls Open-Meteo +
            // (optionally) the nearest METAR and synthesises a
            // `Scene` for the configured lat/lon at the current
            // world clock time. While in flight the button shows
            // "Fetching…" and is disabled; the host fetches on a
            // background thread and feeds the result back through
            // `pending.live_scene` so the resulting scene swap
            // takes the same path a user-loaded scene would.
            let in_flight = state.weather_fetch.in_flight;
            ui.add_enabled_ui(!in_flight, |ui| {
                let label = if in_flight {
                    "Fetching…"
                } else {
                    "Fetch real weather"
                };
                if ui
                    .button(label)
                    .on_hover_text(
                        "Pull live forecast + observed weather for the \
                         configured lat/lon (Dunblane default) and \
                         replace the active scene with a synthesised \
                         multi-layer cloud + surface representation.",
                    )
                    .clicked()
                {
                    let now = state
                        .pending
                        .set_world_utc
                        .unwrap_or_else(chrono::Utc::now);
                    state.pending.fetch_real_weather =
                        Some(crate::state::WeatherFetchRequest {
                            lat: state.live_config.world.latitude_deg,
                            lon: state.live_config.world.longitude_deg,
                            time: now,
                            enrich_with_metar: true,
                        });
                }
            });
            // Show last fetch result inline (success → green
            // summary; failure → red error). Stays visible until
            // the next click.
            if let Some(err) = &state.weather_fetch.last_error {
                ui.colored_label(egui::Color32::from_rgb(220, 60, 60), err);
            } else if let Some(summary) = &state.weather_fetch.last_summary {
                ui.colored_label(egui::Color32::from_rgb(80, 180, 80), summary);
            }
            ui.separator();
            ui.label(format!("{:.1} fps", state.frame_stats.fps));
            ui.label(format!("{:.2} ms", state.frame_stats.frame_ms));
        });
    });
}

// Native file dialogs via `rfd` (plan §10.4). Filtered to .toml so
// only valid scene files appear. Default directory is the workspace
// `tests/scenes/` since that's where the reference scenes live;
// `scenes/` (the live-app directory) is reachable from there in one
// click via the dialog's parent navigation.
fn rfd_pick_open() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .set_title("Load scene")
        .add_filter("Scene TOML", &["toml"])
        .set_directory(default_scenes_dir())
        .pick_file()
}

fn rfd_pick_save() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .set_title("Save scene")
        .add_filter("Scene TOML", &["toml"])
        .set_directory(default_scenes_dir())
        .set_file_name("scene.toml")
        .save_file()
}

/// Locate a sensible default directory for the file picker. Tries
/// the workspace's `tests/scenes/` first (the reference scenes
/// live there); falls back to the current working directory.
fn default_scenes_dir() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let candidate = cwd.join("tests").join("scenes");
    if candidate.is_dir() {
        candidate
    } else {
        cwd
    }
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
// Camera panel — fov / near / speed (plan §0.4)
// ---------------------------------------------------------------------------

fn camera_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Camera", |ui| {
        let Some(current) = state.latest_camera else {
            ui.label("(awaiting first frame to mirror camera state)");
            return;
        };
        let mut cam = current;
        let mut changed = false;

        let mut fov_deg = cam.fov_y_rad.to_degrees();
        if Slider::new(&mut fov_deg, 20.0..=120.0)
            .text("FOV (vertical, °)")
            .max_decimals(2)
            .ui(ui)
            .changed()
        {
            cam.fov_y_rad = fov_deg.to_radians();
            changed = true;
        }

        if Slider::new(&mut cam.near_m, 0.01..=10.0)
            .text("Near plane (m)")
            .logarithmic(true)
            .max_decimals(4)
            .ui(ui)
            .changed()
        {
            changed = true;
        }

        if Slider::new(&mut cam.speed_mps, 0.1..=200.0)
            .text("Speed (m/s)")
            .logarithmic(true)
            .max_decimals(2)
            .ui(ui)
            .changed()
        {
            changed = true;
        }

        if changed {
            state.latest_camera = Some(cam);
            state.pending.live_camera = Some(cam);
        }
    });
}

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
        // Phase 13.8 — `ground`, `clouds`, `windsock`, and `water` all
        // bind the atmosphere LUT bundle. With atmosphere disabled
        // their pass closures no-op cleanly, but it's surprising for
        // the user to toggle them on and see no result. Grey out the
        // affected boxes and explain the dependency in a tooltip.
        let atmosphere_on = s.atmosphere;
        any |= ui
            .add_enabled_ui(atmosphere_on, |ui| {
                ui.checkbox(&mut s.ground, "Ground")
                    .on_disabled_hover_text(
                        "Ground PBR samples the atmosphere transmittance / \
                         sky-view / aerial-perspective LUTs. Enable Atmosphere first.",
                    )
                    .changed()
            })
            .inner;
        any |= ui.checkbox(&mut s.atmosphere, "Atmosphere").changed();
        any |= ui
            .add_enabled_ui(atmosphere_on, |ui| {
                ui.checkbox(&mut s.clouds, "Clouds")
                    .on_disabled_hover_text(
                        "Cloud march reads the atmosphere LUTs for sky-ambient \
                         and AP composite. Enable Atmosphere first.",
                    )
                    .changed()
            })
            .inner;
        any |= ui.checkbox(&mut s.precipitation, "Precipitation").changed();
        any |= ui.checkbox(&mut s.wet_surface, "Wet surface").changed();
        any |= ui.checkbox(&mut s.godrays, "Godrays").changed();
        any |= ui.checkbox(&mut s.lightning, "Lightning").changed();
        any |= ui.checkbox(&mut s.aurora, "Aurora").changed();
        any |= ui.checkbox(&mut s.bloom, "Bloom").changed();
        any |= ui
            .add_enabled_ui(atmosphere_on, |ui| {
                ui.checkbox(&mut s.windsock, "Windsock")
                    .on_disabled_hover_text(
                        "Windsock samples the AP LUT so it fades into haze. \
                         Enable Atmosphere first.",
                    )
                    .changed()
            })
            .inner;
        any |= ui
            .add_enabled_ui(atmosphere_on, |ui| {
                ui.checkbox(&mut s.water, "Water")
                    .on_disabled_hover_text(
                        "Water reflection samples the sky-view LUT and AP LUT. \
                         Enable Atmosphere first.",
                    )
                    .changed()
            })
            .inner;
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
        // Tuning toggles (config-side).
        let a = &mut state.live_config.render.atmosphere;
        let mut tuning_changed = false;
        tuning_changed |= ui.checkbox(&mut a.multi_scattering, "Multi-scattering").changed();
        tuning_changed |= ui.checkbox(&mut a.sun_disk, "Sun disk").changed();
        tuning_changed |= ui.checkbox(&mut a.ozone_enabled, "Ozone").changed();
        tuning_changed |= Slider::new(&mut a.sun_angular_radius_deg, 0.05..=2.0)
            .text("Sun angular radius (deg)")
            .max_decimals(4)
            .ui(ui)
            .changed();
        if tuning_changed {
            state.pending.config_dirty = true;
        }

        ui.separator();
        ui.label("Physical parameters (AtmosphereParams)");
        // Coefficient sliders (WeatherState-side).
        let Some(current) = state.latest_atmosphere else {
            ui.label("(awaiting first frame to mirror WeatherState)");
            return;
        };
        let mut atmo = current;
        let defaults = ps_core::AtmosphereParams::default();
        let mut any = false;
        // Planet geometry.
        any |= f32_with_reset(ui, &mut atmo.planet_radius_m, defaults.planet_radius_m,
                              "planet_radius_m", 1.0e6..=1.0e8, 1000.0);
        any |= f32_with_reset(ui, &mut atmo.atmosphere_top_m, defaults.atmosphere_top_m,
                              "atmosphere_top_m", 1.0e4..=1.0e8, 1000.0);
        any |= f32_with_reset(ui, &mut atmo.rayleigh_scale_height_m,
                              defaults.rayleigh_scale_height_m,
                              "rayleigh_scale_height_m", 100.0..=20000.0, 100.0);
        any |= f32_with_reset(ui, &mut atmo.mie_scale_height_m, defaults.mie_scale_height_m,
                              "mie_scale_height_m", 100.0..=10000.0, 100.0);
        // Rayleigh scattering (per-channel).
        ui.label("Rayleigh scattering (per metre)");
        any |= vec3_with_reset(ui, &mut atmo.rayleigh_scattering,
                                defaults.rayleigh_scattering, "rayleigh", 1e-7);
        // Mie scattering / absorption.
        ui.label("Mie scattering (per metre)");
        any |= vec3_with_reset(ui, &mut atmo.mie_scattering,
                                defaults.mie_scattering, "mie_s", 1e-7);
        ui.label("Mie absorption (per metre)");
        any |= vec3_with_reset(ui, &mut atmo.mie_absorption,
                                defaults.mie_absorption, "mie_a", 1e-7);
        any |= f32_with_reset(ui, &mut atmo.mie_g, defaults.mie_g,
                              "mie_g (HG anisotropy)", 0.0..=0.99, 0.01);
        // Ozone.
        ui.label("Ozone absorption (per metre)");
        any |= vec3_with_reset(ui, &mut atmo.ozone_absorption,
                                defaults.ozone_absorption, "ozone", 1e-8);
        any |= f32_with_reset(ui, &mut atmo.ozone_center_m, defaults.ozone_center_m,
                              "ozone_center_m", 0.0..=80000.0, 100.0);
        any |= f32_with_reset(ui, &mut atmo.ozone_thickness_m, defaults.ozone_thickness_m,
                              "ozone_thickness_m", 100.0..=80000.0, 100.0);
        // Ground albedo (for atmosphere bounce).
        ui.label("Ground albedo (atmosphere bounce)");
        any |= vec3_with_reset(ui, &mut atmo.ground_albedo,
                                defaults.ground_albedo, "albedo", 0.01);

        if ui.button("Reset all to Earth defaults").clicked() {
            atmo = defaults;
            any = true;
        }

        if any {
            state.latest_atmosphere = Some(atmo);
            state.pending.live_atmosphere = Some(atmo);
        }
    });
}

/// f32 slider with a per-field reset (↺) button. Returns true if
/// either the slider value or a reset click changed `value`.
fn f32_with_reset(
    ui: &mut egui::Ui,
    value: &mut f32,
    default: f32,
    label: &str,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        let mut v = *value;
        if DragValue::new(&mut v)
            .range(*range.start()..=*range.end())
            .speed(speed)
            .max_decimals(6)
            .ui(ui)
            .changed()
        {
            *value = v;
            changed = true;
        }
        ui.label(label);
        if ui
            .small_button("↺")
            .on_hover_text("Reset to Earth default")
            .clicked()
            && *value != default
        {
            *value = default;
            changed = true;
        }
    });
    changed
}

/// vec3 (xyz) drag-input row with a per-field reset button.
fn vec3_with_reset(
    ui: &mut egui::Ui,
    value: &mut glam::Vec4,
    default: glam::Vec4,
    label: &str,
    speed: f32,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for (i, suffix) in ["r", "g", "b"].iter().enumerate() {
            ui.label(format!("{label}.{suffix}"));
            let mut v = match i {
                0 => value.x,
                1 => value.y,
                _ => value.z,
            };
            if DragValue::new(&mut v).speed(speed).max_decimals(8).ui(ui).changed() {
                match i {
                    0 => value.x = v,
                    1 => value.y = v,
                    _ => value.z = v,
                }
                changed = true;
            }
        }
        if ui
            .small_button("↺")
            .on_hover_text("Reset to Earth default")
            .clicked()
            && *value != default
        {
            *value = default;
            changed = true;
        }
    });
    changed
}

fn drag_f32(
    ui: &mut egui::Ui,
    value: &mut f32,
    label: &str,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
) -> bool {
    ui.horizontal(|ui| {
        let mut v = *value;
        let r = DragValue::new(&mut v)
            .range(*range.start()..=*range.end())
            .speed(speed)
            .max_decimals(6)
            .ui(ui);
        ui.label(label);
        if r.changed() {
            *value = v;
            true
        } else {
            false
        }
    })
    .inner
}


// ---------------------------------------------------------------------------
// Clouds panel — every CloudsTuning field + per-layer accordions
// ---------------------------------------------------------------------------

fn clouds_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Clouds", |ui| {
        // Engine-side march tuning.
        let c = &mut state.live_config.render.clouds;
        let mut tuning_changed = false;
        tuning_changed |= Slider::new(&mut c.cloud_steps, 32..=512)
            .text("Cloud march steps")
            .ui(ui)
            .changed();
        tuning_changed |= Slider::new(&mut c.light_steps, 1..=16)
            .text("Light march steps")
            .ui(ui)
            .changed();
        tuning_changed |= Slider::new(&mut c.multi_scatter_octaves, 1..=8)
            .text("Multi-scatter octaves")
            .ui(ui)
            .changed();
        tuning_changed |= Slider::new(&mut c.detail_strength, 0.0..=1.0)
            .text("Detail strength")
            .max_decimals(4)
            .ui(ui)
            .changed();
        tuning_changed |= Slider::new(&mut c.powder_strength, 0.0..=1.0)
            .text("Powder strength")
            .max_decimals(4)
            .ui(ui)
            .changed();

        // Approximate Mie (Jendersie & d'Eon 2023) — single droplet
        // diameter knob replaces the previous HG bias triple.
        // Default 1.0 ⇒ use the per-cloud-type diameter from
        // synthesis unchanged (water clouds ≈ 16–20 µm; ice ≈ 50 µm).
        tuning_changed |= Slider::new(&mut c.droplet_diameter_bias, 0.25..=2.0)
            .text("Droplet diameter bias")
            .max_decimals(3)
            .ui(ui)
            .on_hover_text(
                "Multiplies the per-layer droplet effective diameter \
                 fed to the Approximate Mie phase function. Lower → \
                 smaller droplets (broader, hazier forward scatter); \
                 higher → larger droplets (sharper silver lining). \
                 Shader clamps the resulting diameter to the paper's \
                 5–50 µm fit range.",
            )
            .changed();

        tuning_changed |= ui.checkbox(&mut c.freeze_time, "Freeze time").changed();
        // Phase 13.9 — optional temporal rotation of the spatial blue
        // noise. Auto-gated by `freeze_time` engine-side, but show the
        // disabled state here so the user understands why the toggle
        // isn't taking effect while paused.
        ui.add_enabled_ui(!c.freeze_time, |ui| {
            tuning_changed |= ui
                .checkbox(&mut c.temporal_jitter, "Temporal jitter (TAA prep)")
                .on_hover_text(
                    "Rotate the cloud-march blue-noise lookup with a \
                     16-frame cycle. Off by default — without a TAA \
                     pass downstream this looks like obvious per-frame \
                     noise. Automatically disabled while paused so \
                     screenshots stay deterministic.",
                )
                .changed();
        });
        if tuning_changed {
            state.pending.config_dirty = true;
        }

        ui.separator();
        ui.label("Layers (Scene.clouds.layers)");
        let Some(scene) = state.latest_scene.as_ref() else {
            ui.label("(awaiting first frame to mirror Scene)");
            return;
        };
        let mut new_scene = scene.clone();
        let mut layers_changed = false;
        for (i, layer) in new_scene.clouds.layers.iter_mut().enumerate() {
            ui.collapsing(format!("Layer {i}: {:?}", layer.cloud_type), |ui| {
                layers_changed |= drag_f32(ui, &mut layer.base_m, "base_m",
                                            0.0..=18000.0, 10.0);
                layers_changed |= drag_f32(ui, &mut layer.top_m, "top_m",
                                            0.0..=18000.0, 10.0);
                layers_changed |= Slider::new(&mut layer.coverage, 0.0..=1.0)
                    .text("coverage")
                    .max_decimals(4)
                    .ui(ui)
                    .changed();
                // density_scale is Option<f32>: None ⇒ use the
                // per-cloud-type default from ps-core. Show the
                // effective value (current Some, else default) and
                // write Some(v) on change so the user's edit becomes
                // explicit.
                let density_default =
                    ps_core::default_density_scale(layer.cloud_type);
                let mut density_val = layer.density_scale.unwrap_or(density_default);
                let density_resp = Slider::new(&mut density_val, 0.0..=4.0)
                    .text("density_scale")
                    .max_decimals(4)
                    .ui(ui)
                    .on_hover_text(
                        "Per-layer optical density multiplier. Defaults \
                         track the per-cloud-type optical-depth ranges; \
                         move the slider to override.",
                    );
                if density_resp.changed() {
                    layer.density_scale = Some(density_val);
                    layers_changed = true;
                }
                ui.horizontal(|ui| {
                    ui.label("type");
                    ComboBox::from_id_salt(format!("ps-ui-cloud-kind-{i}"))
                        .selected_text(format!("{:?}", layer.cloud_type))
                        .show_ui(ui, |ui| {
                            use ps_core::CloudType;
                            for k in [
                                CloudType::Cumulus,
                                CloudType::Stratus,
                                CloudType::Stratocumulus,
                                CloudType::Altocumulus,
                                CloudType::Altostratus,
                                CloudType::Cirrus,
                                CloudType::Cirrostratus,
                                CloudType::Cumulonimbus,
                            ] {
                                if ui
                                    .selectable_label(
                                        layer.cloud_type == k,
                                        format!("{k:?}"),
                                    )
                                    .clicked()
                                    && layer.cloud_type != k
                                {
                                    layer.cloud_type = k;
                                    layers_changed = true;
                                }
                            }
                        });
                });
                let shape_resp = Slider::new(&mut layer.shape_octave_bias, -1.0..=1.0)
                    .text("shape_octave_bias")
                    .max_decimals(4)
                    .ui(ui)
                    .on_hover_text(
                        "Bias on the base-shape low-frequency Worley \
                         FBM weighting. Subtle — redistributes the bulk \
                         density envelope. Most effect lives near 0.",
                    );
                layers_changed |= shape_resp.changed();
                let detail_resp = Slider::new(&mut layer.detail_octave_bias, -1.0..=1.0)
                    .text("detail_octave_bias")
                    .max_decimals(4)
                    .ui(ui)
                    .on_hover_text(
                        "Bias on the boundary erosion strength. The \
                         Schneider remap is non-monotonic: positive \
                         bias culls more low-density samples while \
                         concentrating survivors, so the cloud may \
                         read as either thinner or denser depending \
                         on the layer's coverage + density_scale. \
                         Recommended range ~±0.1.",
                    );
                layers_changed |= detail_resp.changed();
                // Phase 13 follow-up — anvil bias is meaningful only
                // for Cumulonimbus. Default of None means "use the
                // per-type default" (Cumulonimbus = 1.0). Show the
                // slider only for Cumulonimbus so the panel stays
                // concise.
                if layer.cloud_type == ps_core::CloudType::Cumulonimbus {
                    let mut anvil_val = layer.anvil_bias.unwrap_or(1.0);
                    let anvil_resp = Slider::new(&mut anvil_val, 0.0..=2.0)
                        .text("anvil_bias")
                        .max_decimals(3)
                        .ui(ui)
                        .on_hover_text(
                            "Strength of the cumulonimbus anvil top in \
                             the NDF. 0 = no anvil, 1 = v1 default, \
                             2 = double the anvil mass.",
                        );
                    if anvil_resp.changed() {
                        layer.anvil_bias = Some(anvil_val);
                        layers_changed = true;
                    }
                }
            });
        }
        if layers_changed {
            state.latest_scene = Some(new_scene.clone());
            state.pending.live_scene = Some(new_scene);
        }
    });
}

// ---------------------------------------------------------------------------
// Wet surface panel — SurfaceParams.wetness block
// ---------------------------------------------------------------------------

fn wet_surface_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Wet surface", |ui| {
        let Some(scene) = state.latest_scene.as_ref() else {
            ui.label("(awaiting first frame to mirror Scene)");
            return;
        };
        let mut new_scene = scene.clone();
        let mut any = false;

        // Phase 13.4 — surface material picker. Lives in this
        // panel because the dry / wet PBR paths and the puddle
        // appearance all depend on the chosen palette/roughness/F0.
        ui.horizontal(|ui| {
            ui.label("Material");
            for (label, value) in [
                ("Grass", ps_core::SurfaceMaterial::Grass),
                ("BareSoil", ps_core::SurfaceMaterial::BareSoil),
                ("Tarmac", ps_core::SurfaceMaterial::Tarmac),
                ("Sand", ps_core::SurfaceMaterial::Sand),
                ("WaterEdge", ps_core::SurfaceMaterial::WaterEdge),
            ] {
                let selected = new_scene.surface.material == value;
                if ui.selectable_label(selected, label).clicked()
                    && new_scene.surface.material != value
                {
                    new_scene.surface.material = value;
                    any = true;
                }
            }
        });

        let w = &mut new_scene.surface.wetness;
        any |= Slider::new(&mut w.ground_wetness, 0.0..=1.0)
            .text("ground_wetness")
            .max_decimals(4)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut w.puddle_coverage, 0.0..=1.0)
            .text("puddle_coverage")
            .max_decimals(4)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut w.puddle_start, 0.0..=1.0)
            .text("puddle_start")
            .max_decimals(4)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut w.snow_depth_m, 0.0..=1.0)
            .text("snow_depth_m")
            .max_decimals(4)
            .ui(ui)
            .changed();
        if any {
            state.latest_scene = Some(new_scene.clone());
            state.pending.live_scene = Some(new_scene);
        }
    });
}

// ---------------------------------------------------------------------------
// Precipitation panel
// ---------------------------------------------------------------------------

fn precipitation_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Precipitation", |ui| {
        // Engine-side pool tuning (config).
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

        ui.separator();
        // Scene-side type + intensity.
        let Some(scene) = state.latest_scene.as_ref() else {
            ui.label("(awaiting first frame to mirror Scene)");
            return;
        };
        let mut new_scene = scene.clone();
        let mut scene_changed = false;
        ui.horizontal(|ui| {
            ui.label("type");
            ComboBox::from_id_salt("ps-ui-precip-kind")
                .selected_text(format!("{:?}", new_scene.precipitation.kind))
                .show_ui(ui, |ui| {
                    use ps_core::PrecipKind;
                    for k in [
                        PrecipKind::None,
                        PrecipKind::Rain,
                        PrecipKind::Snow,
                        PrecipKind::Sleet,
                    ] {
                        if ui
                            .selectable_label(
                                new_scene.precipitation.kind == k,
                                format!("{k:?}"),
                            )
                            .clicked()
                            && new_scene.precipitation.kind != k
                        {
                            new_scene.precipitation.kind = k;
                            scene_changed = true;
                        }
                    }
                });
        });
        scene_changed |= Slider::new(&mut new_scene.precipitation.intensity_mm_per_h, 0.0..=50.0)
            .text("intensity_mm_per_h")
            .max_decimals(4)
            .ui(ui)
            .changed();
        if scene_changed {
            state.latest_scene = Some(new_scene.clone());
            state.pending.live_scene = Some(new_scene);
        }
    });
}

// ---------------------------------------------------------------------------
// Godrays panel — Phase 12.4 screen-space crepuscular rays.
// ---------------------------------------------------------------------------

fn godrays_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Godrays", |ui| {
        // The on/off lives in the Subsystems panel
        // (`[render.subsystems].godrays`); this panel only tunes
        // the four parameters once enabled.
        let g = &mut state.live_config.render.godrays;
        let mut changed = false;

        changed |= Slider::new(&mut g.samples, 8..=256)
            .text("Samples (per pixel)")
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut g.decay, 0.80..=0.999)
            .text("Decay (per sample)")
            .max_decimals(4)
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut g.intensity, 0.0..=2.0)
            .text("Intensity")
            .max_decimals(3)
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut g.bright_threshold, 0.0..=20_000.0)
            .text("Bright threshold (cd/m²)")
            .max_decimals(0)
            .logarithmic(true)
            .ui(ui)
            .changed();

        if changed {
            state.pending.config_dirty = true;
        }

        ui.label(
            "Crepuscular rays appear only when the sun is in front of \
             the camera. Look toward the sun (e.g. yaw 180° at noon \
             at default Dunblane lat) to see the effect.",
        );
    });
}

// ---------------------------------------------------------------------------
// Bloom panel (Phase 13.3) — bright-pass threshold + intensity
// ---------------------------------------------------------------------------

fn bloom_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Bloom", |ui| {
        let b = &mut state.live_config.render.bloom;
        let mut any = false;
        any |= Slider::new(&mut b.threshold_ev100, -2.0..=8.0)
            .text("Threshold (EV stops above ev100)")
            .max_decimals(2)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut b.knee_ev, 0.0..=2.0)
            .text("Knee width (EV)")
            .max_decimals(2)
            .ui(ui)
            .changed();
        any |= Slider::new(&mut b.intensity, 0.0..=4.0)
            .text("Intensity")
            .max_decimals(3)
            .ui(ui)
            .changed();
        if any {
            state.pending.config_dirty = true;
        }
        ui.label(
            "Bloom highlights: extracts pixels brighter than \
             2^(ev100 + threshold), runs a 3-level Gaussian \
             pyramid, additively blends back into HDR. The sun \
             disk (~10⁹ cd/m²) is the dominant contributor.",
        );
    });
}

// ---------------------------------------------------------------------------
// Lightning panel (Phase 12.3) — strike RNG seed + intensity tunables
// ---------------------------------------------------------------------------

fn lightning_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Lightning", |ui| {
        // The on/off lives in the Subsystems panel
        // (`[render.subsystems].lightning`); this panel only tunes
        // the engine-side render parameters once enabled.
        let l = &mut state.live_config.render.lightning;
        let mut changed = false;

        // Seed is u64 — the egui Slider is generic but stepping a
        // u64 through a slider isn't useful. Expose it as a numeric
        // input so the user can paste a value for repeatable bolts.
        ui.horizontal(|ui| {
            ui.label("Seed");
            let mut seed_str = format!("{}", l.seed);
            if ui.text_edit_singleline(&mut seed_str).lost_focus() {
                if let Ok(v) = seed_str.parse::<u64>() {
                    if v != l.seed {
                        l.seed = v;
                        changed = true;
                    }
                }
            }
        });

        changed |= Slider::new(&mut l.peak_cloud_illuminance, 0.0..=200_000.0)
            .text("Peak cloud illuminance (cd/m²·sr)")
            .max_decimals(0)
            .logarithmic(true)
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut l.bolt_peak_emission, 0.0..=2.0e9)
            .text("Bolt peak emission (cd/m²·sr)")
            .max_decimals(0)
            .logarithmic(true)
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut l.illumination_radius_m, 0.0..=20_000.0)
            .text("Illumination radius (m)")
            .max_decimals(0)
            .ui(ui)
            .changed();
        changed |= Slider::new(&mut l.max_active_strikes, 1..=32)
            .text("Max active strikes")
            .ui(ui)
            .changed();

        if changed {
            state.pending.config_dirty = true;
        }

        ui.label(
            "Bolts trigger from the scene's strikes_per_min_per_km2 \
             via a Poisson process. Set the rate in the scene file \
             (e.g. tests/scenes/diag_lightning.toml).",
        );
    });
}

// ---------------------------------------------------------------------------
// Aurora panel (Phase 12.5) — scene kp/colour + engine tunables
// ---------------------------------------------------------------------------

fn aurora_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Aurora", |ui| {
        // Engine-side render tunables come from
        // `[render.aurora]`; scene inputs (kp_index,
        // intensity_override, predominant_colour) live on
        // `Scene.aurora` and feed the synthesis pipeline. The
        // pattern matches clouds_panel: clone latest_scene, mutate,
        // publish via pending.live_scene.
        let mut any_engine = false;

        ui.label("Scene inputs (apply on synthesis):");
        if let Some(scene_ref) = state.latest_scene.as_ref() {
            let mut new_scene = scene_ref.clone();
            let mut scene_changed = false;
            scene_changed |= Slider::new(&mut new_scene.aurora.kp_index, 0.0..=9.0)
                .text("kp_index")
                .max_decimals(1)
                .ui(ui)
                .changed();
            scene_changed |= Slider::new(
                &mut new_scene.aurora.intensity_override,
                -1.0..=1.0,
            )
            .text("intensity_override (-1 = derive from kp)")
            .max_decimals(2)
            .ui(ui)
            .changed();

            ui.horizontal(|ui| {
                ui.label("predominant_colour");
                for option in ["green", "red", "purple", "mixed"] {
                    let selected = new_scene.aurora.predominant_colour == option;
                    if ui.selectable_label(selected, option).clicked() {
                        new_scene.aurora.predominant_colour = option.into();
                        scene_changed = true;
                    }
                }
            });

            if scene_changed {
                state.latest_scene = Some(new_scene.clone());
                state.pending.live_scene = Some(new_scene);
            }
        } else {
            ui.label("(awaiting first frame to mirror Scene)");
        }

        ui.separator();
        ui.label("Engine tunables (live):");
        let a = &mut state.live_config.render.aurora;
        any_engine |= Slider::new(&mut a.march_steps, 4..=32)
            .text("March steps")
            .ui(ui)
            .changed();
        any_engine |= Slider::new(&mut a.peak_emission, 100.0..=500_000.0)
            .text("Peak emission (cd/m²·sr)")
            .max_decimals(0)
            .logarithmic(true)
            .ui(ui)
            .changed();
        any_engine |= Slider::new(&mut a.motion_hz, 0.0..=1.0)
            .text("Motion speed (Hz)")
            .max_decimals(3)
            .ui(ui)
            .changed();
        any_engine |= Slider::new(&mut a.min_latitude_abs_deg, 0.0..=90.0)
            .text("Latitude gate min (°)")
            .max_decimals(1)
            .ui(ui)
            .changed();
        any_engine |= Slider::new(&mut a.peak_latitude_abs_deg, 0.0..=90.0)
            .text("Latitude gate peak (°)")
            .max_decimals(1)
            .ui(ui)
            .changed();
        any_engine |= Slider::new(&mut a.fade_latitude_abs_deg, 0.0..=90.0)
            .text("Latitude gate fade (°)")
            .max_decimals(1)
            .ui(ui)
            .changed();

        if any_engine {
            state.pending.config_dirty = true;
        }

        ui.label(
            "Auroras are gated by absolute latitude and kp_index. \
             At Dunblane (56°N) the gate is partially open; at lat \
             65° kp 5 you should see clear green curtains looking \
             north at midnight.",
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
        if any {
            state.pending.config_dirty = true;
        }
        // shader_hot_reload + gpu_validation are startup-only by design
        // (the watcher is built once at start, validation is set on
        // device init). Show them disabled with a tooltip so the value
        // is visible but the user can't be misled into thinking edits
        // take effect at runtime.
        let r = d.shader_hot_reload;
        let g = d.gpu_validation;
        let mut r_dummy = r;
        let mut g_dummy = g;
        ui.add_enabled(false, egui::Checkbox::new(&mut r_dummy, "Shader hot reload"))
            .on_disabled_hover_text("Startup-only — edit pedalsky.toml and restart.");
        ui.add_enabled(false, egui::Checkbox::new(&mut g_dummy, "GPU validation"))
            .on_disabled_hover_text("Startup-only — edit pedalsky.toml and restart.");

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
        // Optical depth = -ln(T). Clamp T to a small floor so the log
        // stays finite when transmittance is zero.
        let od = [
            -(t[0].max(1e-12)).ln(),
            -(t[1].max(1e-12)).ln(),
            -(t[2].max(1e-12)).ln(),
        ];
        ui.colored_label(
            Color32::LIGHT_BLUE,
            format!("optical depth (total) R={:.4} G={:.4} B={:.4}", od[0], od[1], od[2]),
        );
        // Phase 13.10 — per-component OD breakdown along the same
        // view ray. Sum should equal the total above (within
        // integration error). Useful for validating which absorber
        // is dominating the haze at the probe pixel — Rayleigh
        // dominates blue, Mie dominates white/grey, ozone bites
        // green at high sun-zenith angles.
        let r = state.debug.probe_od_rayleigh;
        let m = state.debug.probe_od_mie;
        let o = state.debug.probe_od_ozone;
        ui.colored_label(
            Color32::LIGHT_BLUE,
            format!("  Rayleigh R={:.4} G={:.4} B={:.4}", r[0], r[1], r[2]),
        );
        ui.colored_label(
            Color32::LIGHT_BLUE,
            format!("  Mie      R={:.4} G={:.4} B={:.4}", m[0], m[1], m[2]),
        );
        ui.colored_label(
            Color32::LIGHT_BLUE,
            format!("  ozone    R={:.4} G={:.4} B={:.4}", o[0], o[1], o[2]),
        );

        ui.separator();
        ui.label("GPU timings");
        if state.frame_stats.gpu_passes.is_empty() {
            ui.label("(awaiting first frame's resolved timestamps — \
                     feature requires TIMESTAMP_QUERY_INSIDE_ENCODERS)");
        } else {
            egui::Grid::new("ps-ui-gpu-timings").striped(true).show(ui, |ui| {
                let mut total_ms = 0.0;
                for (name, ms) in &state.frame_stats.gpu_passes {
                    ui.label(name);
                    ui.label(format!("{ms:>8.3} ms"));
                    ui.end_row();
                    total_ms += ms;
                }
                ui.label("(total)");
                ui.label(format!("{total_ms:>8.3} ms"));
                ui.end_row();
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Phase 13.5 — water plane editor.
//
// Toggles the optional `[scene.water]` block on/off and exposes its
// bounds, altitude, and roughness range. When off, the scene-side
// field is `None` and the water subsystem renders nothing.
// ---------------------------------------------------------------------------

fn water_panel(ui: &mut egui::Ui, state: &mut UiState) {
    ui.collapsing("Water", |ui| {
        let Some(scene) = state.latest_scene.as_ref() else {
            ui.label("(awaiting first frame to mirror Scene)");
            return;
        };
        let mut new_scene = scene.clone();
        let mut any = false;

        let mut enabled = new_scene.water.is_some();
        if ui.checkbox(&mut enabled, "Water plane present").changed() {
            new_scene.water = if enabled {
                Some(ps_core::Water::default())
            } else {
                None
            };
            any = true;
        }
        if let Some(w) = new_scene.water.as_mut() {
            any |= drag_f32(ui, &mut w.xmin, "xmin (m)", -1000.0..=1000.0, 1.0);
            any |= drag_f32(ui, &mut w.xmax, "xmax (m)", -1000.0..=1000.0, 1.0);
            any |= drag_f32(ui, &mut w.zmin, "zmin (m)", -1000.0..=1000.0, 1.0);
            any |= drag_f32(ui, &mut w.zmax, "zmax (m)", -1000.0..=1000.0, 1.0);
            any |= drag_f32(ui, &mut w.altitude_m, "altitude (m)", -100.0..=100.0, 0.1);
            any |= Slider::new(&mut w.roughness_min, 0.001..=0.2)
                .text("roughness_min")
                .max_decimals(4)
                .ui(ui)
                .changed();
            any |= Slider::new(&mut w.roughness_max, 0.001..=0.5)
                .text("roughness_max")
                .max_decimals(4)
                .ui(ui)
                .changed();
        } else {
            ui.label("(no water plane on this scene — tick to add)");
        }
        if any {
            state.latest_scene = Some(new_scene.clone());
            state.pending.live_scene = Some(new_scene);
        }
    });
}

// ---------------------------------------------------------------------------
// Phase 13.6 — compass-rose overlay
//
// Floats in the top-right corner. Shows:
//   - cardinal N/E/S/W marks on the outer ring
//   - a yellow wedge for the camera's current heading (where the
//     viewer is looking on the horizontal plane)
//   - a sun and moon disc at their (azimuth, altitude) — the
//     altitude controls the radius (zenith = centre, horizon = ring)
//   - a small wind barb showing the direction the wind is blowing
//     FROM, with the barb length scaled by speed
//
// Pure egui paint into a single `Area`; consumes no UiState (it only
// reads from `world_readout`).
// ---------------------------------------------------------------------------

fn compass_overlay(ctx: &egui::Context, state: &UiState) {
    let r = &state.world_readout;
    egui::Area::new(egui::Id::new("ps-ui-compass"))
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 12.0))
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            // Use a fixed canvas. egui's allocate_painter sizes the
            // overlay to this rect, so no auto-shrink.
            let canvas_size = egui::vec2(150.0, 150.0);
            let (response, painter) =
                ui.allocate_painter(canvas_size, egui::Sense::hover());
            let rect = response.rect;
            let centre = rect.center();
            let radius = (rect.width().min(rect.height()) * 0.5) - 6.0;

            // Background disc (semi-transparent so the rendered scene
            // shows through).
            painter.circle_filled(
                centre,
                radius,
                Color32::from_black_alpha(140),
            );
            painter.circle_stroke(
                centre,
                radius,
                egui::Stroke::new(1.0, Color32::from_white_alpha(180)),
            );

            // Cardinal points.
            let cardinals = [(0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W")];
            for (az_deg, label) in cardinals {
                let p = polar_point(centre, radius - 4.0, az_deg);
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(13.0),
                    if label == "N" {
                        Color32::from_rgb(255, 180, 120)
                    } else {
                        Color32::WHITE
                    },
                );
            }

            // Camera heading — yellow wedge from centre toward the
            // direction the camera is facing. Yaw=0 means looking
            // down −Z (= north), positive yaw rotates the camera
            // counter-clockwise when viewed from above (right-handed
            // about +Y); on a compass that *clockwise* azimuth runs
            // the opposite way, so we negate.
            let yaw_az_deg = (-r.camera_yaw_deg) as f32;
            let cam_tip = polar_point(centre, radius - 8.0, yaw_az_deg);
            painter.line_segment(
                [centre, cam_tip],
                egui::Stroke::new(2.5, Color32::from_rgb(240, 220, 60)),
            );
            // Small arrowhead.
            for side in [-1.0_f32, 1.0] {
                let p = polar_point(
                    centre,
                    radius - 18.0,
                    yaw_az_deg + 8.0 * side,
                );
                painter.line_segment(
                    [cam_tip, p],
                    egui::Stroke::new(2.0, Color32::from_rgb(240, 220, 60)),
                );
            }

            // Sun marker. Above-horizon: filled gold disc; below
            // horizon: hollow circle, faded.
            paint_celestial(
                &painter,
                centre,
                radius,
                r.sun_az_deg as f32,
                r.sun_alt_deg as f32,
                Color32::from_rgb(255, 210, 100),
            );
            // Moon marker — paler.
            paint_celestial(
                &painter,
                centre,
                radius,
                r.moon_az_deg as f32,
                r.moon_alt_deg as f32,
                Color32::from_rgb(200, 200, 220),
            );

            // Wind barb. Show as a line from the rim *inward*: the
            // wind comes from the indicated direction. Length scales
            // with speed (0 mps → invisible nub; 20 mps → most of the
            // radius). Calm wind (≈0) draws nothing.
            if r.wind_speed_mps > 0.5 {
                let len = (r.wind_speed_mps / 20.0).clamp(0.15, 0.95) * (radius - 10.0);
                let from_rim = polar_point(centre, radius - 4.0, r.wind_dir_deg);
                let toward_centre = polar_point(centre, radius - 4.0 - len, r.wind_dir_deg);
                let blue = Color32::from_rgb(120, 180, 240);
                painter.line_segment(
                    [from_rim, toward_centre],
                    egui::Stroke::new(2.0, blue),
                );
                // Barb tick at the upwind end.
                let perp_az = r.wind_dir_deg + 90.0;
                let tip_a = polar_offset(toward_centre, 6.0, perp_az);
                let tip_b = polar_offset(toward_centre, -6.0, perp_az);
                painter.line_segment([toward_centre, tip_a], egui::Stroke::new(1.5, blue));
                painter.line_segment([toward_centre, tip_b], egui::Stroke::new(1.5, blue));
            }
        });
}

/// Project `(azimuth_deg, distance)` (azimuth measured clockwise from
/// north, screen coordinates with `+y` down) onto the screen plane
/// centred at `centre`. North → straight up (so we flip the sign of
/// the cosine term to account for egui's +y-down convention).
fn polar_point(centre: egui::Pos2, distance: f32, az_deg: f32) -> egui::Pos2 {
    polar_offset(centre, distance, az_deg)
}

fn polar_offset(centre: egui::Pos2, distance: f32, az_deg: f32) -> egui::Pos2 {
    let theta = az_deg.to_radians();
    let dx = distance * theta.sin();
    let dy = -distance * theta.cos();
    centre + egui::vec2(dx, dy)
}

/// Paint sun / moon marker at a given (azimuth, altitude). Above
/// the horizon → coloured disc; below → faded outline.
fn paint_celestial(
    painter: &egui::Painter,
    centre: egui::Pos2,
    radius: f32,
    az_deg: f32,
    alt_deg: f32,
    colour: Color32,
) {
    // Radius scales with cosine of altitude: zenith → centre, horizon
    // → rim. (1 - alt/90) is the equivalent linear mapping and is
    // closer to what users expect from a "planispheric" overlay.
    let alt_norm = (alt_deg / 90.0).clamp(-0.5, 1.0);
    let r_screen = (1.0 - alt_norm.max(0.0)) * (radius - 8.0);
    let p = polar_point(centre, r_screen, az_deg);
    if alt_deg >= 0.0 {
        painter.circle_filled(p, 4.0, colour);
    } else {
        painter.circle_stroke(
            p,
            4.0,
            egui::Stroke::new(1.2, colour.linear_multiply(0.4)),
        );
    }
}
