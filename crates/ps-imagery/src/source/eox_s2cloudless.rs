//! EOX s2cloudless source — Sentinel-2 cloud-free yearly composite.
//!
//! Public WMTS endpoint at <https://tiles.maps.eox.at>, no auth.
//! URL template (REST, verified 2026-05-18):
//!
//! ```text
//! https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg
//! ```
//!
//! - `s2cloudless-2024_3857` — yearly composite (latest available),
//!   Web Mercator variant.
//! - `g` — `GoogleMapsCompatible` tile-matrix-set (EPSG:3857), 256×256 JPEG tiles.
//! - `{z}/{y}/{x}` — standard slippy-map XYZ coordinates.
//!
//! Licence: CC BY 4.0 — see [`crate::ATTRIBUTION`].
//!
//! Fair-use guidance from EOX: don't hammer the service. We rate-limit
//! to one request at a time (sequential per fetch), and the cache
//! means a re-fetch at the same location is instant.

use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context};

use super::ImagerySource;
use crate::cache::BinaryCache;
use crate::tile::{
    lonlat_to_tile_xy, pick_zoom_for_radius, tile_xy_to_lonlat_nw, GeoExtent, ImageryRequest,
    RgbTile,
};
use crate::{ImageryError, ImageryProgressSink};

/// Default TTL — the s2cloudless composite is a yearly product so
/// month-long caching is fine.
pub const DEFAULT_TTL: Duration = Duration::from_secs(30 * 24 * 3600);

/// Layer identifier on the EOX WMTS endpoint. Bump the year when EOX
/// publishes a newer composite.
const LAYER: &str = "s2cloudless-2024_3857";

/// Cache subsource label so the disk cache keeps separate per year.
const CACHE_SUBSOURCE: &str = "eox-s2cloudless-2024";

/// Source-label string stamped onto every `RgbTile` we produce.
const SOURCE_LABEL: &str = "eox-s2cloudless-2024";

/// Tile size — fixed at 256 by the EOX tile-matrix-set.
const TILE_PX: u32 = 256;

/// HTTP user-agent.
const UA: &str = concat!("ps-imagery/", env!("CARGO_PKG_VERSION"));

/// EOX s2cloudless fetcher.
pub struct EoxS2Cloudless {
    cache_root: PathBuf,
}

impl EoxS2Cloudless {
    /// Build a fetcher rooted at `cache_root`.
    pub fn new(cache_root: PathBuf) -> Self {
        Self { cache_root }
    }

    /// Build the public REST URL for a `(z, x, y)` tile.
    fn url(zoom: u32, x: u32, y: u32) -> String {
        format!("https://tiles.maps.eox.at/wmts/1.0.0/{LAYER}/default/g/{zoom}/{y}/{x}.jpg")
    }
}

impl ImagerySource for EoxS2Cloudless {
    fn fetch_with_progress(
        &self,
        req: &ImageryRequest,
        progress: &dyn ImageryProgressSink,
    ) -> Result<RgbTile, ImageryError> {
        // 1. Pick a zoom level matched to the request radius +
        // resolution preset, then cap so the stitched grid fits the
        // device's texture-size limit.
        //
        // The Max preset at low/mid latitudes can pick z=14, which
        // covers a 30 km radius with ~45×45 tiles = 11520×11520 px.
        // wgpu adapters typically expose `max_texture_dimension_2d`
        // = 8192, so without a cap the GPU upload validation panics.
        // We step zoom down one level at a time until the projected
        // stitched dimension is within the cap; this loses some
        // sharpness at the highest preset on tiles where the cap is
        // tight, but never exceeds the GPU limit.
        let mut zoom = pick_zoom_for_radius(req.lat, req.radius_m, req.resolution.target_pixels());
        let cap = req.max_texture_dim.max(TILE_PX);
        let projected_dim = |z: u32| -> u32 {
            let (n_x, n_y) = tile_grid_dims(req.lat, req.lon, req.radius_m, z);
            n_x.max(n_y) * TILE_PX
        };
        while zoom > 0 && projected_dim(zoom) > cap {
            zoom -= 1;
        }

        // 2. Compute the tile grid covering the request bbox. The
        // Mercator y-axis grows southward, so the NW corner has the
        // smaller y.
        let radius_lat_deg = req.radius_m as f64 / 111_320.0;
        let cos_lat = req.lat.to_radians().cos().abs().max(0.05);
        let radius_lon_deg = (req.radius_m as f64 / (111_320.0 * cos_lat)).min(180.0);

        let nw_lon = req.lon - radius_lon_deg;
        let nw_lat = (req.lat + radius_lat_deg).min(85.0511);
        let se_lon = req.lon + radius_lon_deg;
        let se_lat = (req.lat - radius_lat_deg).max(-85.0511);

        let (xf0, yf0) = lonlat_to_tile_xy(nw_lon, nw_lat, zoom);
        let (xf1, yf1) = lonlat_to_tile_xy(se_lon, se_lat, zoom);
        let x0 = xf0.floor() as u32;
        let y0 = yf0.floor() as u32;
        let x1 = xf1.ceil() as u32;
        let y1 = yf1.ceil() as u32;
        let n_tiles_x = x1.saturating_sub(x0).max(1);
        let n_tiles_y = y1.saturating_sub(y0).max(1);

        tracing::info!(
            target: "ps_imagery",
            zoom, x0, y0, n_tiles_x, n_tiles_y,
            radius_m = req.radius_m,
            resolution = req.resolution.label(),
            tile_count = n_tiles_x * n_tiles_y,
            stitched_dim = n_tiles_x.max(n_tiles_y) * TILE_PX,
            tex_cap = cap,
            "EOX: fetching tile grid"
        );

        let cache = BinaryCache::new(&self.cache_root);

        // 3. Fetch each tile (cache-first), decode JPEG, write into
        // the stitched buffer.
        let stitched_w = n_tiles_x * TILE_PX;
        let stitched_h = n_tiles_y * TILE_PX;
        let mut pixels_rgba = vec![0u8; (stitched_w * stitched_h * 4) as usize];

        let total_tiles = n_tiles_x * n_tiles_y;
        // Publish the total up-front so the UI can render an empty
        // bar immediately (otherwise the bar would only appear after
        // the first tile completes, which can be 1–2 seconds on a
        // cold cache).
        progress.update(0, total_tiles);
        let mut done_tiles = 0u32;

        for ty in 0..n_tiles_y {
            for tx in 0..n_tiles_x {
                let tile_x = x0 + tx;
                let tile_y = y0 + ty;
                let bytes = match cache
                    .read_fresh(CACHE_SUBSOURCE, zoom, tile_x, tile_y, DEFAULT_TTL)
                    .map_err(|e| ImageryError::Fetch(anyhow!(e)))?
                {
                    Some(b) => b,
                    None => {
                        let url = Self::url(zoom, tile_x, tile_y);
                        let bytes = download(&url).map_err(ImageryError::Fetch)?;
                        cache
                            .write(CACHE_SUBSOURCE, zoom, tile_x, tile_y, &bytes)
                            .map_err(|e| ImageryError::Fetch(anyhow!(e)))?;
                        bytes
                    }
                };

                // Decode the JPEG and blit into the stitched buffer.
                let img = image::load_from_memory_with_format(
                    &bytes,
                    image::ImageFormat::Jpeg,
                )
                .with_context(|| format!("decode tile z{zoom}/x{tile_x}/y{tile_y}"))
                .map_err(ImageryError::Decode)?;
                let rgba = img.to_rgba8();
                if rgba.width() != TILE_PX || rgba.height() != TILE_PX {
                    return Err(ImageryError::Decode(anyhow!(
                        "unexpected tile size {}x{}, expected {}x{}",
                        rgba.width(),
                        rgba.height(),
                        TILE_PX,
                        TILE_PX
                    )));
                }

                let dx = tx * TILE_PX;
                let dy = ty * TILE_PX;
                for row in 0..TILE_PX {
                    let src_off = (row * TILE_PX * 4) as usize;
                    let dst_off = (((dy + row) * stitched_w + dx) * 4) as usize;
                    let src_row = &rgba.as_raw()[src_off..src_off + (TILE_PX as usize) * 4];
                    pixels_rgba[dst_off..dst_off + (TILE_PX as usize) * 4]
                        .copy_from_slice(src_row);
                }

                done_tiles += 1;
                progress.update(done_tiles, total_tiles);
            }
        }

        // 4. Compute the geographic extent of the stitched raster
        // (NW corner of tile (x0, y0); SE corner of tile (x1, y1) is
        // the NW corner of tile (x1+1, y1+1)).
        let (nw_lon_actual, nw_lat_actual) = tile_xy_to_lonlat_nw(x0, y0, zoom);
        let (se_lon_actual, se_lat_actual) =
            tile_xy_to_lonlat_nw(x0 + n_tiles_x, y0 + n_tiles_y, zoom);

        // 5. Belt-and-braces: if the zoom cap above somehow missed
        // (e.g. an EOX tile size other than 256 in a future variant),
        // downsample 2× with a box filter until the result fits the
        // device limit. Step doubles each pass so this terminates in
        // at most log2(stitched_w / cap) iterations.
        let (pixels_rgba, width, height) =
            downsample_to_fit(pixels_rgba, stitched_w, stitched_h, cap);

        Ok(RgbTile {
            pixels_rgba,
            width,
            height,
            extent_deg: GeoExtent {
                west: nw_lon_actual,
                east: se_lon_actual,
                north: nw_lat_actual,
                south: se_lat_actual,
            },
            source: SOURCE_LABEL,
        })
    }
}

/// Return the `(n_tiles_x, n_tiles_y)` covering the bbox at the given
/// zoom. Hoisted out of `fetch` so the pre-flight zoom cap can use
/// the same computation without duplicating the bbox math.
fn tile_grid_dims(lat: f64, lon: f64, radius_m: f32, zoom: u32) -> (u32, u32) {
    let radius_lat_deg = radius_m as f64 / 111_320.0;
    let cos_lat = lat.to_radians().cos().abs().max(0.05);
    let radius_lon_deg = (radius_m as f64 / (111_320.0 * cos_lat)).min(180.0);
    let nw_lon = lon - radius_lon_deg;
    let nw_lat = (lat + radius_lat_deg).min(85.0511);
    let se_lon = lon + radius_lon_deg;
    let se_lat = (lat - radius_lat_deg).max(-85.0511);
    let (xf0, yf0) = lonlat_to_tile_xy(nw_lon, nw_lat, zoom);
    let (xf1, yf1) = lonlat_to_tile_xy(se_lon, se_lat, zoom);
    let x0 = xf0.floor() as u32;
    let y0 = yf0.floor() as u32;
    let x1 = xf1.ceil() as u32;
    let y1 = yf1.ceil() as u32;
    (x1.saturating_sub(x0).max(1), y1.saturating_sub(y0).max(1))
}

/// 2× box-filter downsample until both dimensions fit `cap`. RGBA8 in,
/// RGBA8 out. Returns the buffer + final `(width, height)`.
fn downsample_to_fit(
    mut pixels: Vec<u8>,
    mut w: u32,
    mut h: u32,
    cap: u32,
) -> (Vec<u8>, u32, u32) {
    while w > cap || h > cap {
        let new_w = (w / 2).max(1);
        let new_h = (h / 2).max(1);
        let mut next = vec![0u8; (new_w * new_h * 4) as usize];
        for ny in 0..new_h {
            for nx in 0..new_w {
                // Average the 2×2 source block.
                let sx0 = nx * 2;
                let sy0 = ny * 2;
                let sx1 = (sx0 + 1).min(w - 1);
                let sy1 = (sy0 + 1).min(h - 1);
                let mut acc = [0u32; 4];
                for &(sx, sy) in &[(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)] {
                    let off = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += pixels[off + c] as u32;
                    }
                }
                let dst_off = ((ny * new_w + nx) * 4) as usize;
                for c in 0..4 {
                    next[dst_off + c] = (acc[c] / 4) as u8;
                }
            }
        }
        pixels = next;
        w = new_w;
        h = new_h;
        tracing::info!(
            target: "ps_imagery",
            new_w = w, new_h = h,
            "EOX: downsampled to fit texture cap"
        );
    }
    (pixels, w, h)
}

fn download(url: &str) -> Result<Vec<u8>, anyhow::Error> {
    let resp = ureq::get(url)
        .set("User-Agent", UA)
        .call()
        .with_context(|| format!("GET {url}"))?;
    if resp.status() != 200 {
        return Err(anyhow!("GET {} returned HTTP {}", url, resp.status()));
    }
    // 4 MiB cap — JPEG tiles from EOX are typically 10–60 KiB.
    let reader = resp.into_reader();
    let mut limited = reader.take(4 * 1024 * 1024);
    let mut buf = Vec::new();
    limited
        .read_to_end(&mut buf)
        .with_context(|| format!("read body from {url}"))?;
    Ok(buf)
}

/// Test hook — decode + blit a single tile body without HTTP.
#[doc(hidden)]
pub fn decode_one_tile_for_tests(body: &[u8]) -> Result<image::RgbaImage, anyhow::Error> {
    let cursor = Cursor::new(body);
    let img = image::load(std::io::BufReader::new(cursor), image::ImageFormat::Jpeg)
        .context("decode test tile")?;
    Ok(img.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_dunblane_z12() {
        // From the verified EOX response (200 OK, image/jpeg, 14495 B):
        assert_eq!(
            EoxS2Cloudless::url(12, 2002, 1271),
            "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/12/1271/2002.jpg"
        );
    }

    #[test]
    fn downsample_to_fit_halves_until_within_cap() {
        // 4×4 RGBA, all white. Cap = 1 → must downsample twice.
        let pixels = vec![255u8; 4 * 4 * 4];
        let (out, w, h) = downsample_to_fit(pixels, 4, 4, 1);
        assert_eq!((w, h), (1, 1));
        // Average of all-white is still white.
        assert_eq!(out, vec![255, 255, 255, 255]);
    }

    #[test]
    fn downsample_to_fit_is_noop_when_already_within_cap() {
        let pixels = vec![128u8; 2 * 2 * 4];
        let (out, w, h) = downsample_to_fit(pixels.clone(), 2, 2, 8192);
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, pixels);
    }

    /// Regression — the Max-preset crash: a 30 km radius around
    /// Dunblane (56.19°N) at the Max preset's target_pixels (8192)
    /// initially picks z=14, which would produce a ~11520×11520 px
    /// stitched image. After the pre-flight zoom cap, the projected
    /// stitched dimension must be ≤ the device limit (8192).
    #[test]
    fn max_preset_zoom_capped_to_device_limit() {
        let cap = 8192u32;
        let lat = 56.19_f64;
        let lon = -3.96_f64;
        let radius_m = 30_000.0_f32;
        let mut zoom = pick_zoom_for_radius(
            lat,
            radius_m,
            crate::tile::ImageryResolution::Max.target_pixels(),
        );
        // Replicate the cap loop in fetch().
        let projected_dim = |z: u32| {
            let (n_x, n_y) = tile_grid_dims(lat, lon, radius_m, z);
            n_x.max(n_y) * TILE_PX
        };
        while zoom > 0 && projected_dim(zoom) > cap {
            zoom -= 1;
        }
        assert!(
            projected_dim(zoom) <= cap,
            "after cap: zoom={zoom}, projected_dim={}, cap={cap}",
            projected_dim(zoom)
        );
    }
}
