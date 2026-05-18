//! `RgbTile` + the geographic-extent types + Web-Mercator tile math.
//!
//! The Web-Mercator math is the same XYZ slippy-map formula used by
//! every web mapping library; the formulas live here in one place so
//! source impls don't duplicate them.

use std::path::PathBuf;

/// A stitched RGBA8 raster covering a rectangular geographic extent.
///
/// `pixels_rgba.len() == width * height * 4` is a load-bearing
/// invariant — the GPU upload assumes tight RGBA8 packing.
pub struct RgbTile {
    /// Pixels in row-major RGBA8 order. `[r, g, b, a, r, g, b, a, ...]`.
    pub pixels_rgba: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Geographic extent (degrees) covered by the raster. The
    /// north-west corner is at pixel (0, 0); the south-east at
    /// (width-1, height-1).
    pub extent_deg: GeoExtent,
    /// Source label for tracing (`"eox-s2cloudless-2024"`, etc.).
    pub source: &'static str,
}

/// Geographic extent of a tile in degrees.
#[derive(Debug, Copy, Clone)]
pub struct GeoExtent {
    /// Westernmost longitude (degrees, signed; -180..180).
    pub west: f64,
    /// Easternmost longitude.
    pub east: f64,
    /// Southernmost latitude (degrees, signed; -85..85 for Web Mercator).
    pub south: f64,
    /// Northernmost latitude.
    pub north: f64,
}

impl GeoExtent {
    /// Centre lat/lon.
    pub fn centre(self) -> (f64, f64) {
        ((self.south + self.north) * 0.5, (self.west + self.east) * 0.5)
    }
}

/// Resolution preset for satellite imagery fetches. Picks the target
/// stitched-image pixel count, which the source impl translates into
/// a WMTS zoom level.
///
/// EOX s2cloudless is derived from Sentinel-2's ~10 m native ground
/// resolution. Beyond [`ImageryResolution::Max`] additional zoom adds
/// HTTP traffic without revealing real new detail (the server just
/// upsamples the same source pixels).
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum ImageryResolution {
    /// ~2048 px across the requested extent. ~30 m/pixel at mid
    /// latitudes — a handful of WMTS tiles, fast fetch. Default.
    #[default]
    Standard,
    /// ~4096 px across. ~15 m/pixel. ~4× the tile count of
    /// `Standard`; still under 100 tiles for a 30 km radius.
    High,
    /// ~8192 px across. ~7 m/pixel — at or below Sentinel-2's
    /// native resolution. ~16× `Standard`'s tile count; slow first
    /// fetch but the on-disk cache makes re-fetches instant.
    Max,
}

impl ImageryResolution {
    /// Target stitched-image side length in pixels. The source impl
    /// picks the zoom that gets close to this number.
    pub fn target_pixels(self) -> u32 {
        match self {
            Self::Standard => 2048,
            Self::High => 4096,
            Self::Max => 8192,
        }
    }

    /// Short label for tracing + UI display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::High => "High",
            Self::Max => "Max",
        }
    }
}

/// One imagery fetch.
#[derive(Debug, Clone)]
pub struct ImageryRequest {
    /// Observer latitude (degrees north).
    pub lat: f64,
    /// Observer longitude (degrees east).
    pub lon: f64,
    /// Half-extent in metres around the observer. Mirrors
    /// `ps_terrain::TileRequest::radius_m`.
    pub radius_m: f32,
    /// Cache directory. Created on demand by the source.
    pub cache_dir: PathBuf,
    /// Resolution preset. Defaults to [`ImageryResolution::Standard`].
    pub resolution: ImageryResolution,
    /// Hard upper bound on either dimension of the returned `RgbTile`,
    /// in pixels. The source clamps tile selection so the stitched
    /// image fits inside this and box-filters down if it doesn't.
    ///
    /// **Why this exists:** wgpu device texture-size limits (typically
    /// 8192 or 16384 px) are a hard wall — exceeding them in
    /// `Device::create_texture` triggers a validation panic. The
    /// caller passes its GPU `limits.max_texture_dimension_2d` so the
    /// pipeline can stay inside the device's safe envelope. Default
    /// is 8192 which is the universal wgpu downlevel minimum.
    pub max_texture_dim: u32,
}

impl ImageryRequest {
    /// Convenience: 30 km half-extent matches the default terrain
    /// fetch; resolution defaults to `Standard`; texture-cap defaults
    /// to 8192.
    pub fn around(lat: f64, lon: f64, cache_dir: PathBuf) -> Self {
        Self {
            lat,
            lon,
            radius_m: 30_000.0,
            cache_dir,
            resolution: ImageryResolution::default(),
            max_texture_dim: 8192,
        }
    }
}

// -- Web Mercator (EPSG:3857 / "g" / GoogleMapsCompatible) ----------------
//
// Standard slippy-map formulas. lon/lat in degrees, zoom 0..=21.
// Returns floating-point tile coordinates so the caller can floor
// for the north-west tile and ceil for the south-east tile of a
// requested bbox.

/// Project `(lon_deg, lat_deg)` to fractional Web-Mercator tile XY at
/// the given zoom. The integer part is the tile index; the fractional
/// part is the position within that tile.
pub fn lonlat_to_tile_xy(lon_deg: f64, lat_deg: f64, zoom: u32) -> (f64, f64) {
    let n = (1u64 << zoom) as f64;
    let lat_rad = lat_deg.to_radians();
    let x = (lon_deg + 180.0) / 360.0 * n;
    let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    (x, y)
}

/// Inverse — the NW corner (in lon/lat degrees) of integer tile
/// `(x, y)` at zoom.
pub fn tile_xy_to_lonlat_nw(x: u32, y: u32, zoom: u32) -> (f64, f64) {
    let n = (1u64 << zoom) as f64;
    let lon_deg = x as f64 / n * 360.0 - 180.0;
    let lat_rad =
        (std::f64::consts::PI * (1.0 - 2.0 * y as f64 / n)).sinh().atan();
    (lon_deg, lat_rad.to_degrees())
}

/// Pick a zoom level such that the visible diameter `2 * radius_m`
/// at the given latitude maps to roughly `target_pixels` pixels.
/// Higher zoom = more pixels per metre = sharper detail.
///
/// Web-Mercator ground-sample distance at the equator at zoom `z`
/// is `156543.03 / 2^z` metres/pixel; at latitude φ this is scaled
/// by `cos(φ)`.
pub fn pick_zoom_for_radius(lat_deg: f64, radius_m: f32, target_pixels: u32) -> u32 {
    let cos_lat = lat_deg.to_radians().cos().abs().max(0.05);
    // Solve target_pixels = (2 * radius) / gsd  ⇒  gsd = 2*radius / target_pixels
    let target_gsd = (2.0 * radius_m as f64) / target_pixels as f64;
    let z = (156_543.03_f64 * cos_lat / target_gsd).log2().ceil() as i32;
    z.clamp(0, 18) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dunblane_tile_at_z12() {
        // From the verified EOX URL: lat 56.19°N, lon -3.96°W at z=12
        // should land in tile (2002, 1271).
        let (xf, yf) = lonlat_to_tile_xy(-3.96, 56.19, 12);
        assert_eq!(xf.floor() as u32, 2002);
        assert_eq!(yf.floor() as u32, 1271);
    }

    #[test]
    fn round_trip_nw_corner() {
        // NW corner of tile (2002, 1271, z=12) should match the
        // lon/lat that floors into that tile.
        let (lon, lat) = tile_xy_to_lonlat_nw(2002, 1271, 12);
        let (xf, yf) = lonlat_to_tile_xy(lon, lat, 12);
        assert!((xf - 2002.0).abs() < 1e-6);
        assert!((yf - 1271.0).abs() < 1e-6);
    }

    #[test]
    fn zoom_picks_reasonable_level() {
        // 30 km radius at lat ~56°, want ~2000 px → expect z somewhere
        // in 11..13.
        let z = pick_zoom_for_radius(56.19, 30_000.0, 2000);
        assert!((11..=13).contains(&z), "got z={z}");
    }

    #[test]
    fn zoom_clamps_at_extremes() {
        // Trying to fit thousands of pixels across a 1 m radius would
        // demand a far-too-large zoom; the formula clamps at 18.
        let z = pick_zoom_for_radius(0.0, 1.0, 4096);
        assert_eq!(z, 18);
    }

    #[test]
    fn resolution_preset_pixel_counts_monotonic() {
        // Each preset must request strictly more pixels than the last
        // — that's the whole point of the dropdown.
        assert!(
            ImageryResolution::Standard.target_pixels()
                < ImageryResolution::High.target_pixels()
        );
        assert!(
            ImageryResolution::High.target_pixels()
                < ImageryResolution::Max.target_pixels()
        );
    }

    #[test]
    fn resolution_preset_picks_higher_zoom() {
        // At a fixed location + radius, picking a higher preset
        // should map to a zoom >= the lower preset's. Roughly each
        // doubling of pixel count = +1 zoom level.
        let std_z = pick_zoom_for_radius(
            56.19, 30_000.0, ImageryResolution::Standard.target_pixels(),
        );
        let high_z = pick_zoom_for_radius(
            56.19, 30_000.0, ImageryResolution::High.target_pixels(),
        );
        let max_z = pick_zoom_for_radius(
            56.19, 30_000.0, ImageryResolution::Max.target_pixels(),
        );
        assert!(high_z > std_z, "High zoom {high_z} should exceed Standard {std_z}");
        assert!(max_z > high_z, "Max zoom {max_z} should exceed High {high_z}");
    }
}
