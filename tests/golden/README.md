Phase 11 — golden-image regression
==================================

The nine scenes below are the reference scene library (the Phase 11
eight plus the hail-mode fixture). The golden PNGs in this directory
are the regression baseline; SSIM ≥ 0.99 is enforced by
`cargo test -p ps-app --test golden`.

How to regenerate after a deliberate visual change:

    cargo run -p ps-app --bin ps-bless

How to run the regression check:

    cargo test -p ps-app --test golden

Per-scene status
----------------

All nine scenes render visible, plausible imagery for the
meteorological situation described:

    clear_summer_noon          deep blue sunny sky over sunlit ground
    broken_cumulus_afternoon   visible cumulus puffs over sunlit ground
    overcast_drizzle           visible stratus deck
    thunderstorm               storm cloud + sunlit ground
    hail_storm                 storm cloud + opaque white hail splats
    high_cirrus_sunset         cirrus wisps + warm horizon
    winter_overcast_snow       overcast cloud + snowy ground
    twilight_civil             pre-dawn red horizon
    mountain_wave_clouds       altocumulus pattern + sunlit ground

Known followups (not Phase-11 blockers)
---------------------------------------

  #55 — UI bug: opening Atmosphere panel changes rendering
  #56 — Atmosphere bug: sun disk position offset from sky brightening
