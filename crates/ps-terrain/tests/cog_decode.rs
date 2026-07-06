//! End-to-end offline test: synthesize a tiny Float32 TIFF in memory,
//! decode it via the Copernicus source's public test hook, then build
//! a mesh and assert the load-bearing invariants.
//!
//! No network. No bundled binary fixture (we generate one in-process
//! via the `tiff` crate's encoder), so the test fully describes the
//! data it exercises.

use std::io::Cursor;
use std::path::PathBuf;

use ps_terrain::source::copernicus_glo30::decode_for_tests;
use ps_terrain::{build_grid_mesh, TileRequest};

use tiff::encoder::{colortype, TiffEncoder};

/// Encode an 8×8 Float32 grayscale TIFF whose heights count up by row.
fn synth_tiff() -> Vec<u8> {
    let w: u32 = 8;
    let h: u32 = 8;
    let mut pixels: Vec<f32> = Vec::with_capacity((w * h) as usize);
    for r in 0..h {
        for c in 0..w {
            // Heights climb from south to north so we can spot row order
            // mistakes: row 0 (north edge in our convention) = 0, row 7 = 70.
            pixels.push((r as f32) * 10.0 + (c as f32));
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut enc = TiffEncoder::new(cursor).expect("TiffEncoder::new");
        let img = enc
            .new_image::<colortype::Gray32Float>(w, h)
            .expect("new_image");
        img.write_data(&pixels).expect("write_data");
    }
    buf
}

#[test]
fn end_to_end_tiff_to_mesh() {
    let body = synth_tiff();
    // Fictitious tile location chosen so cropping is a no-op.
    let tile = decode_for_tests(&body, 56, -4).expect("decode_for_tests");
    assert_eq!(tile.width, 8);
    assert_eq!(tile.height, 8);
    assert_eq!(tile.heights_m.len(), 64);
    assert_eq!(tile.source, "copernicus-glo30");

    // Row 0 = north edge in our convention, row 7 = south edge.
    // We wrote row r => heights r*10..r*10+7. The decoder is row-major
    // and the GeoTIFF convention is north-up, so heights at index 0
    // should still be 0.0 (row 0, col 0).
    assert_eq!(tile.heights_m[0], 0.0);
    assert_eq!(tile.heights_m[8], 10.0); // row 1, col 0
    assert_eq!(tile.heights_m[63], 77.0); // row 7, col 7

    let req = TileRequest {
        lat: 56.5,
        lon: -3.5,
        radius_m: 1_000_000.0, // larger than the tile so crop is a no-op
        cache_dir: PathBuf::from("."),
        simplify_target: None,
    };
    let mesh = build_grid_mesh(&tile, &req);

    // Load-bearing invariant: one vertex per pixel.
    assert_eq!(mesh.positions.len(), 64);
    // Load-bearing invariant: (W-1)*(H-1)*6 indices.
    assert_eq!(mesh.indices.len(), (7 * 7 * 6) as usize);

    // All normals unit-length.
    for v in &mesh.positions {
        let [x, y, z] = v.normal;
        let len = (x * x + y * y + z * z).sqrt();
        assert!((len - 1.0).abs() < 1e-3, "normal not unit: {len}");
    }

    // Heights monotonically increase south (positive Z, larger row).
    // Compare row 0 col 0 vs row 7 col 0 in mesh order.
    let nw_y = mesh.positions[0].position[1];
    let sw_y = mesh.positions[7 * 8].position[1];
    assert!(sw_y > nw_y, "south should be higher (synthetic gradient)");
}

#[test]
fn full_pipeline_via_passthrough_stages() {
    use ps_terrain::{HeightmapPipeline, PassthroughAugment, PassthroughSimplify};

    // Build a pipeline whose source is a stub returning our synthetic tile.
    struct StubSource(Vec<u8>);
    impl ps_terrain::HeightmapSource for StubSource {
        fn fetch(
            &self,
            _req: &TileRequest,
        ) -> Result<ps_terrain::HeightmapTile, ps_terrain::TerrainError> {
            decode_for_tests(&self.0, 56, -4).map_err(ps_terrain::TerrainError::Decode)
        }
    }

    let pipeline = HeightmapPipeline {
        source: Box::new(StubSource(synth_tiff())),
        augment: Box::new(PassthroughAugment),
        simplify: Box::new(PassthroughSimplify),
    };

    let req = TileRequest {
        lat: 56.5,
        lon: -3.5,
        radius_m: 1_000_000.0,
        cache_dir: PathBuf::from("."),
        simplify_target: None,
    };
    let mesh = pipeline.run(&req).expect("pipeline.run");

    assert_eq!(mesh.positions.len(), 64);
    assert_eq!(mesh.indices.len(), (7 * 7 * 6) as usize);
}
