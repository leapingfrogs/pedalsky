//! `HeightmapTile` and the geographic/request types that flow with it.

use std::path::PathBuf;

use crate::simplify::SimplifyTarget;

/// A rectangular heightmap covering a geographic extent.
///
/// `heights_m.len() == width * height` is a load-bearing invariant —
/// the mesh builder emits exactly one vertex per pixel, so any augment
/// stage that changes `width` or `height` MUST resize `heights_m` to
/// match. [`crate::validate_tile`] enforces this between stages.
///
/// Storage is row-major, north-up: `heights_m[row * width + col]` where
/// `row = 0` corresponds to the northernmost row (`extent_deg.north`)
/// and `col = 0` is the westernmost column (`extent_deg.west`).
pub struct HeightmapTile {
    /// Heights in metres above the WGS84 ellipsoid (Copernicus DSM
    /// reference).
    pub heights_m: Vec<f32>,
    /// Number of pixels east-west.
    pub width: u32,
    /// Number of pixels north-south.
    pub height: u32,
    /// Geographic extent (degrees).
    pub extent_deg: GeoExtent,
    /// Source label for tracing (e.g. `"copernicus-glo30"`,
    /// `"copernicus-glo30+fbm4x"`).
    pub source: &'static str,
    /// Per-pixel ground sample distance in metres at tile centre. Used
    /// to convert pixel deltas to metric XZ deltas in mesh build.
    pub gsd_m_centre: f32,
}

/// Geographic extent of a tile in degrees.
#[derive(Debug, Copy, Clone)]
pub struct GeoExtent {
    /// Westernmost longitude (degrees, signed; -180..180).
    pub west: f64,
    /// Easternmost longitude.
    pub east: f64,
    /// Southernmost latitude (degrees, signed; -90..90).
    pub south: f64,
    /// Northernmost latitude.
    pub north: f64,
}

impl GeoExtent {
    /// Centre latitude/longitude of the extent in degrees.
    pub fn centre(self) -> (f64, f64) {
        ((self.south + self.north) * 0.5, (self.west + self.east) * 0.5)
    }
}

/// A request for terrain at a specific lat/lon.
///
/// Mirrors `ps_weather_feed::FetchOptions` in shape so the call sites
/// look similar.
#[derive(Debug, Clone)]
pub struct TileRequest {
    /// Observer latitude (degrees north).
    pub lat: f64,
    /// Observer longitude (degrees east).
    pub lon: f64,
    /// Desired half-extent in metres around the observer. The mesh is
    /// cropped to a square of `(2 * radius_m)` per side after augment.
    pub radius_m: f32,
    /// Cache directory. Created on demand by the source.
    pub cache_dir: PathBuf,
    /// Optional simplify target. `None` ⇒ `Ratio(1.0)` (identity).
    pub simplify_target: Option<SimplifyTarget>,
}

impl TileRequest {
    /// Construct a request with the default 30 km observer radius.
    pub fn around(lat: f64, lon: f64, cache_dir: PathBuf) -> Self {
        Self {
            lat,
            lon,
            radius_m: 30_000.0,
            cache_dir,
            simplify_target: None,
        }
    }
}

/// Crop a heightmap to a square `2*radius_m` extent centred on the
/// observer, returning a new tile whose pixel grid still satisfies
/// `heights_m.len() == width * height`.
///
/// This trims rows/columns of the source `HeightmapTile` rather than
/// resampling. A future LOD-ring implementation would tile multiple
/// resolutions; v1 ships a single uniform crop.
///
/// If the requested radius is larger than the tile already covers,
/// the tile is returned unmodified.
pub(crate) fn crop_to_radius(
    tile: HeightmapTile,
    req: &TileRequest,
) -> HeightmapTile {
    // The Copernicus DSM is in geographic coordinates, so a metre-radius
    // crop has to map back to pixel deltas via the GSD.
    let gsd = tile.gsd_m_centre.max(1.0);
    let half_pixels = (req.radius_m / gsd).round() as i64;

    // Pixel centre of the observer inside this tile. We assume the
    // tile already covers the observer (the source picks the right
    // 1° tile based on lat/lon).
    let (lat_c, lon_c) = tile.extent_deg.centre();
    let lat_span = tile.extent_deg.north - tile.extent_deg.south;
    let lon_span = tile.extent_deg.east - tile.extent_deg.west;

    let row_f = ((tile.extent_deg.north - req.lat) / lat_span) * (tile.height as f64 - 1.0);
    let col_f = ((req.lon - tile.extent_deg.west) / lon_span) * (tile.width as f64 - 1.0);
    let row_c = row_f.round() as i64;
    let col_c = col_f.round() as i64;

    // Compute the crop window, clamped to the tile.
    let r0 = (row_c - half_pixels).max(0) as u32;
    let r1 = ((row_c + half_pixels) as u32).min(tile.height - 1);
    let c0 = (col_c - half_pixels).max(0) as u32;
    let c1 = ((col_c + half_pixels) as u32).min(tile.width - 1);

    if r0 == 0 && c0 == 0 && r1 == tile.height - 1 && c1 == tile.width - 1 {
        // Crop window covers (or exceeds) the tile — no-op.
        let _ = (lat_c, lon_c);
        return tile;
    }

    let new_w = c1 - c0 + 1;
    let new_h = r1 - r0 + 1;
    let mut new_heights = Vec::with_capacity((new_w as usize) * (new_h as usize));
    let src_w = tile.width as usize;
    for r in r0..=r1 {
        let row_start = (r as usize) * src_w;
        for c in c0..=c1 {
            new_heights.push(tile.heights_m[row_start + c as usize]);
        }
    }

    // Update the geographic extent to match the new corners.
    let north = tile.extent_deg.north - (r0 as f64 / (tile.height as f64 - 1.0)) * lat_span;
    let south = tile.extent_deg.north - (r1 as f64 / (tile.height as f64 - 1.0)) * lat_span;
    let west = tile.extent_deg.west + (c0 as f64 / (tile.width as f64 - 1.0)) * lon_span;
    let east = tile.extent_deg.west + (c1 as f64 / (tile.width as f64 - 1.0)) * lon_span;

    HeightmapTile {
        heights_m: new_heights,
        width: new_w,
        height: new_h,
        extent_deg: GeoExtent { west, east, south, north },
        source: tile.source,
        gsd_m_centre: tile.gsd_m_centre,
    }
}
