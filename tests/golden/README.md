Phase 11 — golden-image regression status
==========================================

The eight scenes below are the Phase 11 reference scene library. The
golden PNGs in this directory are the regression baseline; SSIM ≥ 0.99
is enforced by `cargo test -p ps-app --test golden`.

The regression *infrastructure* (headless render, scene loading, SSIM
comparison, ps-bless re-blessing) is the Phase 11 deliverable and is
fully working. The *visual quality* of the goldens themselves is
limited by several pre-Phase-11 rendering bugs surfaced during the
acceptance pass — these have been catalogued as followups (see below)
rather than fixed in Phase 11.

How to regenerate after a deliberate visual change:

    cargo run -p ps-app --bin ps-bless

How to run the regression check:

    cargo test -p ps-app --test golden

Per-scene status
----------------

Two scenes show visible cloud structure that's broadly plausible:

    broken_cumulus_afternoon   cumulus puffs visible against blue sky
    overcast_drizzle           visible cloud deck

Two scenes are clean-sky baselines (no clouds by design) and look
correct apart from the patchy-dark-ground rendering issue (#62 / #63):

    clear_summer_noon          deep blue sky over dark ground
    twilight_civil             pre-dawn red horizon

Four scenes round-trip through the regression harness at SSIM = 1.0
but have known visual defects from upstream rendering bugs. They are
blessed as placeholders so the harness covers the full library;
re-blessing will be needed once the linked followup tasks land:

    thunderstorm               cloud not visible at altitude (#61)
    high_cirrus_sunset         cirrus deck not visible (#61)
    winter_overcast_snow       extremely dark; low-sun-angle (#62)
    mountain_wave_clouds       lenticulars not visible (#61)

Known followups (defer until Phase 12+)
---------------------------------------

  #56 — sun disk position offset from the brightest sky patch
  #57 — Schneider coverage remap visible band is narrow (~0.6-0.75)
  #60 — pitch-dependent shadow band in atmosphere/cloud lighting
  #61 — cloud march fails for high-altitude / high-vertical-extent layers
  #62 — low-sun-angle ground rendering is excessively dark
  #63 — ground appears over-dark / patchy even in midday scenes

When any of these followups lands, the affected goldens will need
re-blessing — the harness will catch the change as an SSIM drop.
