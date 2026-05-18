# PedalBack terrain enhancement and meshing pipeline

## Document purpose

This document specifies two replacements for the current placeholder identity transformation in the PedalBack terrain pipeline:

1. **Section 1** — DEM enhancement: take a Copernicus 30m DEM tile and produce a higher-resolution, geomorphologically plausible heightmap using physical erosion simulation plus fractal detail.
2. **Section 2** — Mesh decimation: turn the enhanced heightmap into a triangle mesh with the minimum vertex count needed to preserve visual fidelity, especially in areas of high detail produced by Section 1.

Both stages should be implementable into the existing tile baking pipeline. Section 1 runs on the GPU as wgpu compute passes; Section 2 runs on the CPU via the `delatin` crate. Output of Section 1 is the input to Section 2.

The intended caller is offline, per-route baking driven by the GPX track corridor. Enhanced tiles and decimated meshes are cached to disk keyed by tile coordinates and processing parameters.

---

## Section 1: DEM enhancement pipeline

### Overview

The enhancement pipeline takes a Copernicus 30m DEM tile (or higher-resolution source if available) and produces a hydrologically and geomorphologically consistent heightmap at a configurable target resolution (typically 1m–2m). The pipeline runs in five stages, all in the heightmap/texture domain before any meshing.

```
Copernicus tile ──> [1.1 Upsample] ──> [1.2 Hydraulic erosion] ──>
                    [1.3 Thermal erosion] ──> [1.4 Fractal detail] ──>
                    [1.5 Normal map] ──> Enhanced tile output
```

Stages 1.2 and 1.3 may be interleaved (run hydraulic for N iterations, then thermal for M iterations, repeat). This produces more natural results than running each to completion separately.

### Stage 1.1 — Upsample

Bicubic (Catmull-Rom) upsampling from the source resolution to the working resolution. This does not add real detail; it only provides a denser grid for the erosion stages to operate on.

If the source already has detail richer than your target working resolution (e.g. 1m UK Environment Agency LIDAR), skip this stage and downsample if needed.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_resolution_m` | `f32` | 30.0 | Spatial resolution of the input DEM in metres per cell. Copernicus DEM GLO-30 is 30m at the equator (varies with latitude); set this from the GeoTIFF metadata. |
| `target_resolution_m` | `f32` | 1.0 | Resolution of the enhanced heightmap in metres per cell. 1m matches typical Environment Agency LIDAR; 2m halves memory and runtime with marginal visual cost for cycling-distance viewing. |

### Stage 1.2 — Hydraulic erosion (Mei, Decaudin, Hu 2007)

The "virtual pipes" shallow-water model. Reference: Mei, X., Decaudin, P., Hu, B.-G. "Fast Hydraulic Erosion Simulation and Visualization on GPU", *Pacific Graphics 2007*. Open implementations to consult: `bshishov/UnityTerrainErosionGPU`, `huw-man/Interactive-Erosion-Simulator-on-GPU`, `LanLou123/Webgl-Erosion`.

#### State textures

All textures are 2D, `R32Float` storage textures, dimensions = enhanced heightmap dimensions. Use ping-pong pairs (`_a` / `_b`) for the textures that are read and written within the same dispatch chain.

| Texture | Format | Purpose |
|---|---|---|
| `terrain_height` (b) | `R32Float` ping-pong | Bedrock + sediment elevation in metres |
| `water_height` (d) | `R32Float` ping-pong | Water column depth in metres |
| `suspended_sediment` (s) | `R32Float` ping-pong | Sediment dissolved in the water column, in metres-equivalent |
| `flux` (f_L, f_R, f_T, f_B) | `Rgba32Float` ping-pong | Outflow rates through the four virtual pipes to each neighbour, m³/s |
| `velocity` (u, v) | `Rg32Float` | Depth-averaged flow velocity in m/s, derived from flux |

#### Per-iteration compute passes

Each iteration runs four dispatches, each one a grid-stride compute over the heightmap. Workgroup size 8×8 or 16×16 is typical for wgpu.

**Pass 1 — Add water, compute flux**

For each cell `(x, y)`:

1. Add `rainfall_rate * dt` to `water_height`.
2. For each of the four neighbours `n ∈ {L, R, T, B}`:
   - `dh = (terrain + water)[cell] - (terrain + water)[n]`
   - `new_flux[n] = max(0, flux[n] + dt * pipe_cross_section * gravity * dh / pipe_length)`
3. Scale all four fluxes by `K = min(1, water_height * cell_area / (sum_of_fluxes * dt))` to prevent draining more water than is present (this is the "scaling factor" in Mei §3.2.1).
4. Write new flux to the output flux texture.

**Pass 2 — Update water level and velocity**

For each cell `(x, y)`:

1. `inflow = flux_R[x-1, y] + flux_L[x+1, y] + flux_B[x, y-1] + flux_T[x, y+1]`
2. `outflow = flux_L[x, y] + flux_R[x, y] + flux_T[x, y] + flux_B[x, y]`
3. `delta_water = dt * (inflow - outflow) / cell_area`
4. `water_height_new = water_height + delta_water`
5. Compute velocity:
   - `u = (flux_R[x-1, y] - flux_L[x, y] + flux_R[x, y] - flux_L[x+1, y]) / (2 * avg_water_depth * cell_size)`
   - `v = similar in the y direction`

**Pass 3 — Erosion / deposition**

For each cell `(x, y)`:

1. Compute local terrain slope from neighbour heights. Use `sin(angle)` form, not gradient magnitude — the literature has been inconsistent here, but `sin(angle)` matches Mei's formulation and is bounded in [0, 1].
2. Sediment capacity `C = sediment_capacity_constant * sin(local_tilt) * |velocity| * water_depth_scaling`
   - Water depth scaling: `lerp(0.1, 1.0, smoothstep(0.0, shallow_water_threshold, water_depth))` — prevents puddles in flat areas from picking up unbounded sediment.
3. If `C > suspended_sediment`: dissolve — `amount = dissolution_rate * (C - suspended_sediment) * dt`; subtract from `terrain_height`, add to `suspended_sediment`.
4. Else: deposit — `amount = deposition_rate * (suspended_sediment - C) * dt`; add to `terrain_height`, subtract from `suspended_sediment`.

**Pass 4 — Sediment transport (semi-Lagrangian advection)**

For each cell `(x, y)`:

1. Backtrack: `prev_pos = (x, y) - velocity * dt / cell_size`
2. Bilinear sample of suspended sediment at `prev_pos`
3. Write to the output suspended_sediment texture
4. Optional: evaporate water — `water_height *= (1 - evaporation_rate * dt)`

#### Parameters (hydraulic)

These map closely to Mei et al.'s notation. Defaults are tuned for the 1m target resolution; if you change `target_resolution_m` you will need to retune (in particular `dt`, `sediment_capacity_constant`, and `rainfall_rate`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `iterations` | `u32` | 200 | Number of full erosion steps. More iterations carve deeper valleys and produce more dendritic drainage networks. 50 is a quick preview; 200 is the standard quality target; 500+ approaches steady state and is rarely worth the cost for visual purposes. |
| `dt` | `f32` | 0.02 | Time step in seconds. Smaller is more stable but slower. The CFL condition for the flux pass means `dt` must satisfy `dt < cell_size / (gravity * max_water_depth)^0.5`; the default is safe for typical inputs. Halve it if you see oscillations or NaN propagation. |
| `rainfall_rate` | `f32` | 0.012 | Metres of rain added to each cell per second of simulated time. Controls how much water is in the system. Higher values produce wider, deeper rivers and more erosion overall but also wash out finer detail. |
| `evaporation_rate` | `f32` | 0.015 | Fraction of water column lost per second. Balances rainfall to maintain a steady-state water budget. Higher evaporation concentrates flow into established channels (more dendritic look); lower evaporation pools water in basins (more lakes and braided streams). |
| `pipe_cross_section` | `f32` | 1.0 | Virtual pipe area in m². Scales the rate at which height differences translate into flux. Higher values make water respond faster to terrain (sharper channels); lower values produce more diffuse flow. |
| `pipe_length` | `f32` | 1.0 | Virtual pipe length, normally equal to `target_resolution_m`. Don't change unless you understand the implications for `dt` stability. |
| `gravity` | `f32` | 9.81 | Standard gravity. Available as a parameter mainly because some references use 10.0 for simplicity; leave at 9.81. |
| `sediment_capacity_constant` | `f32` | 0.5 | How much sediment moving water can hold per unit (velocity × slope × depth). The single most impactful parameter for visual character. Higher values produce more dramatic V-shaped river-cut valleys; lower values give gentler, more diffuse erosion. Range 0.1–2.0; values above 2.0 tend to over-carve. |
| `dissolution_rate` | `f32` | 0.5 | How quickly sediment is picked up from the bed when capacity exceeds current load. Higher = faster carving in active channels but also more chaotic surfaces if rate exceeds simulation stability. |
| `deposition_rate` | `f32` | 1.0 | How quickly sediment is dropped from the water column when capacity falls below current load. Higher = sharper sediment fans where flow slows; lower = sediment spreads more uniformly. |
| `min_slope` | `f32` | 0.01 | Minimum effective slope for the capacity formula, to avoid zero-capacity-in-flat-areas pathologies. The default is small enough not to affect realistic terrain but prevents NaN cascades. |
| `shallow_water_threshold` | `f32` | 0.05 | Water depth in metres below which the capacity formula is attenuated. Prevents thin sheets of water from carving rapidly. Tuned so shallow puddles deposit sediment rather than dig. |

### Stage 1.3 — Thermal erosion (Olsen 2004, Št'ava et al. 2008)

A separate compute pass that simulates the angle of repose. Where terrain slopes exceed a critical "talus angle", material slides from higher to lower neighbours. This cleans up overhangs and cliffs left by hydraulic erosion and produces realistic scree slopes.

Reference: Olsen, J. "Realtime Procedural Terrain Generation" (2004). Refinements in Št'ava et al. "Interactive Terrain Modeling Using Hydraulic Erosion", *SCA 2008*.

#### Algorithm (per pass)

For each cell `(x, y)`:

1. For each of the eight neighbours (Moore neighbourhood):
   - Compute `dh = terrain[x, y] - terrain[neighbour]`
   - Compute distance to neighbour (`cell_size` for orthogonal, `cell_size * sqrt(2)` for diagonal)
   - If `dh / distance > tan(talus_angle)` then this neighbour is "below the repose threshold"
2. Sum `dh - tan(talus_angle) * distance` over all neighbours below threshold — call this `excess_total`
3. Material to move: `move_amount = min(0.5 * max_excess, thermal_erosion_rate * excess_total)`
4. For each below-threshold neighbour, transfer `move_amount * (its_excess / excess_total)` from this cell to that neighbour

Because adjacent cells may simultaneously try to push material to each other, use two passes with a write-buffer to avoid races. Run thermal erosion every 5–10 hydraulic iterations rather than at every step.

#### Parameters (thermal)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `talus_angle_degrees` | `f32` | 35.0 | The angle of repose. Slopes steeper than this will slough material to the lower side. 30–35° corresponds to loose dirt and dry sand; 40–45° to coarser, more cohesive material (gravel, weathered rock); >50° only for solid rock. For cycling-realistic terrain in upland UK or similar, 35° works well. |
| `thermal_erosion_rate` | `f32` | 0.3 | Fraction of "excess" material transferred per thermal pass. 0.5 is the maximum stable value (any more risks oscillation). Lower values give more gradual relaxation; higher values are more aggressive. |
| `thermal_iterations_per_cycle` | `u32` | 1 | Thermal passes between hydraulic iteration cycles. 1 is usually enough; increase to 3–5 if you see persistent cliffs that should have eroded. |
| `hydraulic_iterations_between_thermal` | `u32` | 10 | How often to run thermal erosion in the interleaved schedule. 10 hydraulic steps followed by 1 thermal step is a good default. |

### Stage 1.4 — Fractal detail injection

After erosion has produced the large- and medium-scale features (valleys, ridges, drainage), add high-frequency procedural detail at sub-metre scales. This is the cheap stuff — it doesn't have geomorphological meaning but it breaks up the smoothness of the bilinearly-interpolated source data and gives surfaces the visual character of natural rock and soil.

#### Algorithm

For each cell `(x, y)` at world position `(wx, wy)`:

1. Sample a heterogeneous multifractal: an fBm where each octave's amplitude is scaled by the previous accumulated value.
2. Scale the result by a `slope_mask` derived from local gradient magnitude — flat areas (lakebeds, plateaus, valley floors) get little or no fractal noise; steep slopes get the full amplitude.
3. Optionally gate further by `elevation_mask` — high-altitude ridges get sharper, more ridged noise; valley walls get smoother fBm.
4. Add the result to `terrain_height`.

Reference for the noise construction: Musgrave, F. K. "Fractal Models of Mountainous Terrain" in *Texturing & Modeling: A Procedural Approach* (Ebert, Musgrave, Peachey, Perlin, Worley), 3rd edition, Chapter 16.

Hash-based gradient noise (Perlin or Simplex) is the standard choice. There is a `noise` crate on crates.io but for compute-shader use you typically inline the noise function in WGSL. The standard `permute` / `fade` Perlin implementation in WGSL is widely available.

#### Parameters (fractal)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fractal_amplitude_m` | `f32` | 0.4 | Maximum fractal detail amplitude in metres, before slope and elevation masking. 0.4m gives perceptible surface texture without distorting actual landforms. Set to 0 to disable fractal detail entirely. |
| `fractal_base_frequency` | `f32` | 0.5 | Frequency of the lowest noise octave in cycles per metre. 0.5 puts the largest features at ~2m wavelength, which is appropriate for sub-resolution detail on a 1m heightmap. |
| `fractal_octaves` | `u32` | 5 | Number of frequency-doubled octaves summed. 5 covers two and a half orders of magnitude in frequency; more octaves add finer detail at diminishing visual return and increasing cost. |
| `fractal_lacunarity` | `f32` | 2.0 | Frequency multiplier per octave. Standard fractal value. |
| `fractal_persistence` | `f32` | 0.5 | Amplitude multiplier per octave. Below 0.5 = smoother surface; above 0.5 = rougher. |
| `fractal_ridged` | `bool` | `false` | If true, transforms each octave with `1 - |noise|` to produce sharp ridge-like features. Use for high-altitude rocky areas; leave false for general surface detail. |
| `slope_mask_strength` | `f32` | 1.0 | How strongly the slope mask attenuates detail in flat areas. 1.0 = flat areas get zero fractal noise; 0.0 = uniform application regardless of slope. |
| `slope_mask_threshold_degrees` | `f32` | 5.0 | Slope angle below which detail is fully attenuated, in degrees. |

### Stage 1.5 — Normal map generation

A trivial single compute pass: for each cell, compute the surface normal from neighbour heights using a Sobel operator or central differences, and write to an `Rgb10A2Unorm` or `Rgba8Unorm` texture (packed XYZ in [-1, +1] mapped to [0, 1]).

This is consumed by the lighting pass at render time. Normals computed at full enhanced resolution will give better lighting than normals interpolated from a decimated mesh, so do this *before* meshing and pass the normal map to the renderer as a separate texture.

### Implementation notes for wgpu

#### Tile boundary handling

The erosion simulation does not respect tile boundaries. If you erode tiles independently, neighbouring tiles will have visibly different drainage patterns at their shared edge. Three options:

1. **Padding (recommended for v0.1)**: Process tiles with a buffer of overlap. For a 1024×1024 working tile, fetch the 1152×1152 region from the source DEM (64-cell padding per side), erode the whole region, then crop the central 1024×1024 for output. Padding must be larger than the longest erosion influence distance; 64 cells works for the default iteration count.
2. **Stitched processing**: Process a 3×3 grid of tiles together, output only the central tile. More memory but smoother boundaries.
3. **Catchment-based processing**: Process a hydrologic catchment as a unit, regardless of tile boundaries. Best results, most complex orchestration.

For PedalBack v0.1, padding is the right call.

#### Caching strategy

Cache key for an enhanced tile: `(tile_x, tile_y, source_dem_version, params_hash)` where `params_hash` is a hash of all Section 1 parameters. This means if you change parameters, caches invalidate cleanly.

Storage format: `.bin` files containing the raw `f32` heightmap, plus a sidecar JSON with metadata (resolution, bounding box, parameters used).

#### Performance budget

On reasonable GPU hardware (e.g. RTX 3060), 200 iterations of hydraulic erosion on a 1024×1024 tile takes ~1–2 seconds. Adding thermal interleaving and fractal injection adds maybe 30%. A 1 km × 1 km tile at 1m resolution thus bakes in 2–3 seconds — well within "tolerable offline build step" territory. A typical 50 km route corridor at 2 km width = 100 tiles ≈ 5 minutes wallclock.

#### Required wgpu features

None beyond the baseline. Compute shaders, storage textures, and bind groups are all standard. No mesh shaders, no ray tracing.

---

## Section 2: Mesh decimation

### Overview

Given the enhanced heightmap from Section 1, produce a triangle mesh whose vertex density follows the heightmap's detail rather than uniformly sampling it. Areas with high detail (river-cut valleys, ridges, eroded surfaces) get dense triangles; flat areas (plateaus, valley floors, lakes) get sparse triangles.

The algorithm is greedy Garland-Heckbert refinement: start with the four corners triangulated, iteratively insert the point of maximum vertical error until the error threshold is met. The Rust crate `delatin` implements this directly.

Reference: Garland, M. and Heckbert, P. S. "Fast Polygonal Approximation of Terrains and Height Fields", Carnegie Mellon technical report CMU-CS-95-181, 1995. Implementation lineage: Garland & Heckbert (C++) → Fogleman's `hmm` (modern C++) → Agafonkin's `delatin` (JavaScript) → `delatin` (Rust port on crates.io).

### Recommended Rust crate

**`delatin`** on crates.io. Status: 0.2.1 as of the latest release, ISC licensed, ~640 SLoC, no significant transitive dependencies.

```toml
[dependencies]
delatin = "0.2"
```

Minimal usage:

```rust
use delatin::{triangulate, Error};

// `heights` is the enhanced heightmap as a flat row-major f32 array
let max_error = Error(0.25);  // target max vertical error in heightmap units (metres)
let (points, triangles) = triangulate(&heights, width, height, max_error)?;
// points: Vec<(usize, usize)>  -- (col, row) grid indices into the heightmap
// triangles: Vec<(usize, usize, usize)>  -- triplets of indices into `points`
```

The crate returns points as grid indices, so to convert to world coordinates you multiply by `target_resolution_m` and look up the height from the source heightmap.

### Multi-LOD pipeline

For each tile, generate four LOD levels by running `delatin` with four different `max_error` thresholds:

| LOD | `max_error` (metres) | Approx. triangle count for 1km² mountain terrain | Used at distance |
|---|---|---|---|
| LOD 0 | 0.05 | ~50k–100k | < 50m (rider's immediate foreground) |
| LOD 1 | 0.25 | ~10k–20k | 50–200m (foreground) |
| LOD 2 | 1.0 | ~2k–5k | 200m–1km (midground) |
| LOD 3 | 5.0 | ~300–800 | > 1km (background) |

Counts depend heavily on terrain type — open moorland decimates much more aggressively than complex eroded ridges. The defaults above are typical of upland UK terrain (Scottish Highlands, Lake District, Snowdonia).

### Skirts for tile borders

Adjacent tiles at different LODs will have different edge vertex sets, producing visible cracks. The cheap solution is "skirts": at each tile boundary, drop a strip of vertices vertically downward by some fixed amount (e.g. 10m) so the geometry overlaps with the neighbour and any crack is hidden behind the skirt.

To add skirts post-decimation: identify all vertices on the tile boundary, duplicate them, displace duplicates downward by `skirt_depth_m`, and add triangles forming a vertical strip between the original boundary edge and the displaced duplicates.

For PedalBack v0.1, skirts are simpler than full stitching and look fine at typical cycling viewing distances.

### Multi-mesh structure for runtime

Per-tile output:

```rust
pub struct TileMesh {
    pub lods: [Lod; 4],
    pub tile_x: i32,
    pub tile_y: i32,
    pub bounds_metres: (f32, f32, f32, f32),  // (min_x, min_y, max_x, max_y)
}

pub struct Lod {
    pub vertices: Vec<[f32; 3]>,       // world-space positions
    pub indices: Vec<u32>,             // triangle indices
    pub max_error_m: f32,              // error threshold used
}
```

At render time, distance-based LOD selection chooses one Lod per tile. Switch LODs at the boundary of each LOD distance bracket; transition pops are hidden by skirts and by the fact that LOD switches happen at fairly long range.

Normals are not stored in the mesh — sample them from the high-resolution normal map (Section 1.5) in the fragment shader at full source resolution. This is the key trick: the mesh provides geometry only, while the normal map provides the visual fidelity of lighting.

### Performance

Single-threaded `delatin` runs at roughly 30k–50k triangles per second for the greedy insertion phase. A 1024×1024 tile decimated to 50k triangles thus takes ~1–2 seconds per LOD. Four LODs per tile = ~5–8 seconds per tile decimation. This runs in parallel with — or after — Section 1 baking, so it fits comfortably within the offline budget. Use `rayon` to parallelise over tiles if you have many.

### Parameters (decimation)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lod_max_errors_m` | `[f32; 4]` | `[0.05, 0.25, 1.0, 5.0]` | Maximum allowed vertical error per LOD in metres. Lower values produce denser meshes that hug the heightmap more tightly; higher values produce sparser meshes that flatten subtle features. For a cycling sim the LOD 0 value matters most — it determines how faithfully the ride surface tracks the actual terrain. 5cm is appropriate for road/path realism; relax to 10–15cm for off-road tolerance. |
| `lod_distances_m` | `[f32; 3]` | `[50.0, 200.0, 1000.0]` | Distances at which LOD switches occur. The mesh switches from LOD `i` to LOD `i+1` at distance `lod_distances_m[i]`. Tune up or down depending on display field-of-view and how often the rider stops to look around; longer ranges mean more visible terrain but more triangles drawn. |
| `skirt_depth_m` | `f32` | 10.0 | Vertical depth of the skirt added at each tile boundary, in metres. Must exceed the maximum LOD error to fully hide cracks. 10m comfortably covers all four LODs and is invisible because skirts are below ground. |
| `max_triangles_per_lod` | `Option<u32>` | `None` | Optional hard upper bound on triangles per LOD. Useful as a memory safety net for pathologically detailed tiles. When set, `delatin` stops refining when the count is reached, even if the error threshold has not been met. |
| `parallelism` | `usize` | `num_cpus` | Number of tiles to decimate in parallel via rayon. The Garland-Heckbert algorithm is single-threaded internally, so parallelism is across tiles only. |

### Integration with the existing pipeline

The existing identity placeholder transformation should be replaced as follows:

1. Where the current code returns the raw upsampled tile, insert the Section 1 compute pipeline. Input: Copernicus tile + Section 1 parameters. Output: `(heightmap_buffer, normal_map_texture)`.
2. Where the current code produces a regular-grid mesh, replace with the Section 2 decimation. Input: `heightmap_buffer` + Section 2 parameters. Output: `TileMesh` with four LODs.
3. Cache both the enhanced heightmap (for re-decimation with different LOD parameters without re-erosion) and the per-LOD meshes (for direct upload to wgpu vertex/index buffers).
4. The runtime renderer changes: per-frame LOD selection per visible tile, with skirts always drawn.

### Parameter summary — what most users will want to tune

Most of the parameters above can stay at defaults. The handful that matter most for tuning visual character:

- `iterations` (hydraulic) — overall erosion intensity
- `sediment_capacity_constant` — how dramatically rivers carve their valleys
- `rainfall_rate` ÷ `evaporation_rate` — wetness of the terrain, channel concentration
- `talus_angle_degrees` — overall steepness ceiling
- `fractal_amplitude_m` — high-frequency surface roughness
- `lod_max_errors_m[0]` — fidelity of the ridable surface
- `lod_distances_m` — visible draw distance vs cost trade-off

A "make it look better" exploration would adjust these in roughly this order: start with default everything, increase `iterations` to 500 for one tile to see what extra cost buys you, then try `sediment_capacity_constant` at 0.3 and 1.0 to see how dramatic you want valleys, then tune `talus_angle_degrees` to the visual style you want (35° = scree, 45° = blockier), then dial in `fractal_amplitude_m` for surface character.

---

## References

- Mei, X., Decaudin, P., Hu, B.-G. "Fast Hydraulic Erosion Simulation and Visualization on GPU." *Pacific Graphics 2007*. https://hal.inria.fr/inria-00402079
- Št'ava, O., Beneš, B., Brisbin, M., Křivánek, J. "Interactive Terrain Modeling Using Hydraulic Erosion." *SCA 2008*.
- Olsen, J. "Realtime Procedural Terrain Generation." Master's thesis, University of Aarhus, 2004.
- Garland, M., Heckbert, P. S. "Fast Polygonal Approximation of Terrains and Height Fields." Carnegie Mellon CMU-CS-95-181, 1995. https://www.cs.cmu.edu/~garland/scape/scape.pdf
- Musgrave, F. K. "Fractal Models of Mountainous Terrain" in Ebert, D., Musgrave, F. K., Peachey, D., Perlin, K., Worley, S. *Texturing & Modeling: A Procedural Approach*, 3rd edition, Morgan Kaufmann, 2002, Chapter 16.
- Galin, E., Guérin, E., Peytavie, A., Cordonnier, G., Cani, M.-P., Beneš, B., Gain, J. "A Review of Digital Terrain Modeling." *Computer Graphics Forum* 38(2), 2019. https://hal.inria.fr/hal-02097510
- Schott, H., Galin, E., Guérin, E., Peytavie, A., Paris, A. "Terrain Amplification using Multi-Scale Erosion." *ACM TOG* 43(4), SIGGRAPH 2024 — for future enhancement directions beyond v0.1.

Rust implementation references:

- `delatin` crate: https://crates.io/crates/delatin / https://docs.rs/delatin
- Fogleman, M. `hmm` (reference C++ implementation): https://github.com/fogleman/hmm
- Agafonkin, V. `delatin` (JavaScript original): https://github.com/mapbox/delatin
- HERE Maps `tin-terrain` (C++, documents the algorithm well): https://github.com/heremaps/tin-terrain
