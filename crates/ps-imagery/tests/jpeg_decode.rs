//! Offline test: synthesize an in-memory JPEG, decode it via the
//! EOX source's test hook, verify RGBA8 output. No network.

use ps_imagery::source::eox_s2cloudless::decode_one_tile_for_tests;

fn synth_jpeg_tile() -> Vec<u8> {
    // 256x256 image, vertical gradient from black (top) to red
    // (bottom). JPEG-encoded — exercises the same decoder path the
    // real EOX response goes through.
    let w = 256u32;
    let h = 256u32;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for r in 0..h {
        let red = (r * 255 / (h - 1)) as u8;
        for _ in 0..w {
            rgb.push(red);
            rgb.push(0);
            rgb.push(0);
        }
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
    encoder
        .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
        .expect("encode jpeg");
    buf
}

#[test]
fn decode_synthetic_jpeg_to_rgba8() {
    let body = synth_jpeg_tile();
    let img = decode_one_tile_for_tests(&body).expect("decode");
    assert_eq!(img.width(), 256);
    assert_eq!(img.height(), 256);

    // Top row should be near-black; bottom row near-red. JPEG is
    // lossy so we allow a wide tolerance.
    let top = img.get_pixel(128, 0);
    let bot = img.get_pixel(128, 255);
    assert!(top.0[0] < 20, "top row R = {} should be near 0", top.0[0]);
    assert!(bot.0[0] > 200, "bottom row R = {} should be near 255", bot.0[0]);
    // Alpha channel from RGB→RGBA conversion should be 255.
    assert_eq!(top.0[3], 255);
    assert_eq!(bot.0[3], 255);
}
