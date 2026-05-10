//! Phase 1 Group A tests for `Config` parsing and validation.

use std::path::Path;

use ps_core::{Config, ConfigError};

fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn pedalsky_toml_loads_unchanged() {
    let root = workspace_root();
    let path = root.join("pedalsky.toml");
    let cfg = Config::load(&path).expect("pedalsky.toml at workspace root should parse");
    cfg.validate().expect("pedalsky.toml should validate");
    assert_eq!(cfg.window.title, "PedalSky");
    // Demo subsystems are present in the schema and default to backdrop=true / tint=false.
    assert!(cfg.render.subsystems.backdrop, "backdrop default should be true");
    assert!(!cfg.render.subsystems.tint, "tint default should be false");
}

#[test]
fn partial_config_uses_defaults() {
    let toml = r#"
[window]
width = 800
"#;
    let cfg = Config::parse(toml).expect("partial config should parse");
    assert_eq!(cfg.window.width, 800);
    // Height untouched → default.
    assert_eq!(cfg.window.height, 1080);
    // Other top-level blocks are also defaulted.
    assert_eq!(cfg.world.latitude_deg, 56.1922);
    assert!(cfg.render.subsystems.backdrop);
}

#[test]
fn unknown_field_is_rejected() {
    let toml = r#"
[window]
potato = 42
"#;
    let err = Config::parse(toml).expect_err("unknown field should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("potato"),
        "error should mention the offending field: {msg}"
    );
}

#[test]
fn latitude_out_of_range_is_rejected() {
    let toml = r#"
[world]
latitude_deg = 91.0
"#;
    let cfg = Config::parse(toml).expect("syntactically valid");
    let err = cfg.validate().expect_err("validate must reject out-of-range latitude");
    let msg = format!("{err}");
    assert!(matches!(err, ConfigError::Invalid(_)));
    assert!(msg.contains("latitude_deg"), "{msg}");
}

#[test]
fn empty_file_uses_defaults() {
    let cfg = Config::parse("").expect("empty config should parse via defaults");
    cfg.validate().expect("default config validates");
    assert_eq!(cfg.window.width, 1920);
}

#[test]
fn invalid_reprojection_is_rejected() {
    let toml = r#"
[render.clouds]
reprojection = "checker_2x2"
"#;
    let cfg = Config::parse(toml).expect("syntactically valid");
    let err = cfg.validate().expect_err("only \"off\" is supported in v1");
    assert!(format!("{err}").contains("reprojection"));
}

#[test]
fn paths_resolve_relative_to_workspace() {
    // Confirms the default scene path matches the layout under [paths].weather.
    let cfg = Config::default();
    assert_eq!(
        cfg.paths.weather,
        Path::new("scenes/broken_cumulus_afternoon.toml")
    );
}
