//! Copernicus GLO-30 source.
//!
//! Fetches 1° × 1° Float32 GeoTIFF tiles from the public AWS Open
//! Data bucket and decodes them into [`HeightmapTile`]s.
//!
//! Bucket: `https://copernicus-dem-30m.s3.amazonaws.com/` — public,
//! no auth.
//!
//! Tile naming: `Copernicus_DSM_COG_10_<NS><lat>_00_<EW><lon>_00_DEM/
//! Copernicus_DSM_COG_10_<NS><lat>_00_<EW><lon>_00_DEM.tif`
//! where `<NS>` is `N` for `lat >= 0` else `S` (with `|lat|`), and
//! `<EW>` is `E` for `lon >= 0` else `W` (with `|lon|`). Latitude is
//! zero-padded to 2 digits; longitude to 3.
//!
//! Format: Cloud-Optimized GeoTIFF, Float32 samples, DEFLATE
//! compression with `Predictor = 3` (floating-point predictor). The
//! pure-Rust `tiff` crate decodes this without GDAL/C dependencies.

use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context};
use tiff::decoder::{Decoder, DecodingResult};

use super::HeightmapSource;
use crate::cache::BinaryCache;
use crate::tile::{GeoExtent, HeightmapTile, TileRequest};
use crate::TerrainError;

/// Default TTL for cached DEM tiles. They effectively never change so
/// 30 days is a long-enough cushion against stale-bucket weirdness.
pub const DEFAULT_TTL: Duration = Duration::from_secs(30 * 24 * 3600);

/// Cache directory name under `req.cache_dir`.
const CACHE_SUBSOURCE: &str = "copernicus-glo30";

/// HTTP user-agent. The S3 bucket doesn't require one but it shows up
/// in S3 access logs and helps us be a polite citizen.
const UA: &str = concat!("ps-terrain/", env!("CARGO_PKG_VERSION"));

/// GLO-30 fetcher.
pub struct CopernicusGlo30 {
    cache_root: PathBuf,
}

impl CopernicusGlo30 {
    /// Build a fetcher rooted at `cache_root`. The actual cache
    /// directory used is `cache_root` itself (matches the
    /// per-`TileRequest.cache_dir` discipline of the weather-feed
    /// crate, but the source ignores the per-request `cache_dir`
    /// override — set the cache root once at pipeline construction).
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Build the public S3 URL for a 1° tile corner.
    fn url(lat_floor: i32, lon_floor: i32) -> String {
        let (ns, lat) = if lat_floor >= 0 { ('N', lat_floor) } else { ('S', -lat_floor) };
        let (ew, lon) = if lon_floor >= 0 { ('E', lon_floor) } else { ('W', -lon_floor) };
        let stem = format!("Copernicus_DSM_COG_10_{ns}{lat:02}_00_{ew}{lon:03}_00_DEM");
        format!("https://copernicus-dem-30m.s3.amazonaws.com/{stem}/{stem}.tif")
    }
}

impl HeightmapSource for CopernicusGlo30 {
    fn fetch(&self, req: &TileRequest) -> Result<HeightmapTile, TerrainError> {
        // Copernicus 1° tiles are named by their south-west corner for
        // northern/eastern hemispheres but actually cover (corner..corner+1).
        // The integer floor of the observer lat/lon picks the tile.
        let lat_floor = req.lat.floor() as i32;
        let lon_floor = req.lon.floor() as i32;

        let cache = BinaryCache::new(&self.cache_root);
        let body = match cache
            .read_fresh(CACHE_SUBSOURCE, lat_floor, lon_floor, DEFAULT_TTL)
            .map_err(|e| TerrainError::Fetch(anyhow!(e)))?
        {
            Some(b) => {
                tracing::info!(
                    target: "ps_terrain",
                    lat_floor, lon_floor,
                    "GLO-30 cache hit"
                );
                b
            }
            None => {
                let url = Self::url(lat_floor, lon_floor);
                tracing::info!(target: "ps_terrain", %url, "GLO-30 fetching");
                let bytes = download(&url).map_err(TerrainError::Fetch)?;
                cache
                    .write(CACHE_SUBSOURCE, lat_floor, lon_floor, &bytes)
                    .map_err(|e| TerrainError::Fetch(anyhow!(e)))?;
                bytes
            }
        };

        decode_tiff(&body, lat_floor, lon_floor)
            .map_err(TerrainError::Decode)
    }
}

fn download(url: &str) -> Result<Vec<u8>, anyhow::Error> {
    let resp = ureq::get(url)
        .set("User-Agent", UA)
        .call()
        .with_context(|| format!("GET {url}"))?;
    if resp.status() != 200 {
        return Err(anyhow!("GET {} returned HTTP {}", url, resp.status()));
    }
    // Cap the read at 64 MiB so a misbehaving response can't OOM us.
    // Copernicus GLO-30 tiles are typically 1–20 MiB.
    let reader = resp.into_reader();
    let mut limited = reader.take(64 * 1024 * 1024);
    let mut buf = Vec::new();
    limited
        .read_to_end(&mut buf)
        .with_context(|| format!("read body from {url}"))?;
    Ok(buf)
}

fn decode_tiff(
    body: &[u8],
    lat_floor: i32,
    lon_floor: i32,
) -> Result<HeightmapTile, anyhow::Error> {
    let cursor = Cursor::new(body);
    let mut decoder = Decoder::new(cursor)
        .context("open TIFF decoder")?;

    let (width, height) = decoder.dimensions().context("read TIFF dimensions")?;
    if width < 2 || height < 2 {
        return Err(anyhow!("TIFF too small: {}x{}", width, height));
    }

    let heights_m = match decoder.read_image().context("decode TIFF image")? {
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        _ => return Err(anyhow!("unsupported TIFF sample type")),
    };

    let expected = (width as usize) * (height as usize);
    if heights_m.len() != expected {
        return Err(anyhow!(
            "TIFF image size mismatch: got {} samples, expected {} ({}x{})",
            heights_m.len(),
            expected,
            width,
            height
        ));
    }

    // 1° tile, north up: the south-west corner is at (lat_floor, lon_floor).
    let extent_deg = GeoExtent {
        west: lon_floor as f64,
        east: (lon_floor + 1) as f64,
        south: lat_floor as f64,
        north: (lat_floor + 1) as f64,
    };

    // GSD: roughly 30 m EW at the equator, shrinks as cos(lat) toward
    // the poles for the EW direction; NS is roughly constant ~30 m.
    // The tile is in geographic projection so EW and NS pixel counts
    // are identical, but the *metric* spacing differs. For the v1 mesh
    // builder we treat the cell as a square of `gsd_m_centre` size and
    // accept the EW distortion away from the equator — fine for the
    // visual sense-of-place we want; not fine for survey-grade work.
    let centre_lat_rad = ((lat_floor as f64) + 0.5).to_radians();
    let metres_per_deg_lat = 111_320.0_f64;
    let gsd_ns_m = (metres_per_deg_lat / height as f64) as f32;
    let gsd_ew_m = (metres_per_deg_lat * centre_lat_rad.cos() / width as f64) as f32;
    // Average isn't perfect but it's a single scalar; the augment
    // stage is free to refine if it wants accurate XYZ.
    let gsd_m_centre = (gsd_ns_m + gsd_ew_m) * 0.5;

    Ok(HeightmapTile {
        heights_m,
        width,
        height,
        extent_deg,
        source: "copernicus-glo30",
        gsd_m_centre,
    })
}

/// Helper used by tests in this crate's `tests/` dir: decode a raw
/// TIFF body without going through HTTP.
#[doc(hidden)]
pub fn decode_for_tests(
    body: &[u8],
    lat_floor: i32,
    lon_floor: i32,
) -> Result<HeightmapTile, anyhow::Error> {
    decode_tiff(body, lat_floor, lon_floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_dunblane() {
        // Dunblane is 56.19°N, 3.96°W → tile N56_W004.
        assert_eq!(
            CopernicusGlo30::url(56, -4),
            "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N56_00_W004_00_DEM/Copernicus_DSM_COG_10_N56_00_W004_00_DEM.tif"
        );
    }

    #[test]
    fn url_southern_hemisphere() {
        // Sydney is ~-33.87°S, 151.2°E → floor = (-34, 151) → tile S34_E151.
        assert_eq!(
            CopernicusGlo30::url(-34, 151),
            "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S34_00_E151_00_DEM/Copernicus_DSM_COG_10_S34_00_E151_00_DEM.tif"
        );
    }
}
