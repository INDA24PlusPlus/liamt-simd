use criterion::Criterion;
use image::*;
use std::simd::*;

// Invert color conversion
pub fn convert(img: DynamicImage, benchmark: bool) -> DynamicImage {
    let img_buf = convert_img_to_vec8(img.clone());
    let (r, g, b) = convert_to_channels(img_buf.clone());

    if benchmark {
        let mut criterion = Criterion::default()
            .sample_size(1000)
            .warm_up_time(std::time::Duration::from_secs(5))
            .measurement_time(std::time::Duration::from_secs(10))
            .significance_level(0.1);
        let mut group = criterion.benchmark_group("inverts");

        group.bench_function("invert no simd", |be| {
            be.iter(|| {
                invert(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("invert simd 8", |be| {
            be.iter(|| {
                invert_simd_8(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("invert simd 16", |be| {
            be.iter(|| {
                invert_simd_16(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("invert simd 32", |be| {
            be.iter(|| {
                invert_simd_32(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("invert simd 64", |be| {
            be.iter(|| {
                invert_simd_64(r.clone(), g.clone(), b.clone());
            })
        });

        // Make sure all functions return the same result
        let inverted_img_buf = invert(r.clone(), g.clone(), b.clone());
        let inverted_img_buf_simd_8 = invert_simd_8(r.clone(), g.clone(), b.clone());
        let inverted_img_buf_simd_16 = invert_simd_16(r.clone(), g.clone(), b.clone());
        let inverted_img_buf_simd_32 = invert_simd_32(r.clone(), g.clone(), b.clone());
        let inverted_img_buf_simd_64 = invert_simd_64(r.clone(), g.clone(), b.clone());
        assert_eq!(inverted_img_buf, inverted_img_buf_simd_8);
        assert_eq!(inverted_img_buf, inverted_img_buf_simd_16);
        assert_eq!(inverted_img_buf, inverted_img_buf_simd_32);
        assert_eq!(inverted_img_buf, inverted_img_buf_simd_64);

        group.finish();
    }

    println!("Inverting image...");
    let (inv_r, inv_g, inv_b) = invert(r, g, b);
    let (width, height) = img.dimensions();

    // Return inverted image
    convert_vec8_to_img(convert_from_channels(inv_r, inv_g, inv_b), width, height)
}

// Convert image to a vec of RGB values
fn convert_img_to_vec8(img: DynamicImage) -> Vec<[u8; 3]> {
    let mut res = Vec::new();

    let buf = img.to_rgb8();
    buf.enumerate_pixels().for_each(|(_x, _y, pixel)| {
        res.push([pixel[0], pixel[1], pixel[2]]);
    });

    res
}

// Convert back the RGB vec to an image
fn convert_vec8_to_img(buf: Vec<[u8; 3]>, width: u32, height: u32) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(width, height);

    for (i, pixel) in buf.iter().enumerate() {
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        img.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
    }

    img
}

// Convert vec of R,G,B u8 to 3 separate vecs
fn convert_to_channels(buf: Vec<[u8; 3]>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r = Vec::new();
    let mut g = Vec::new();
    let mut b = Vec::new();

    for pixel in buf {
        r.push(pixel[0]);
        g.push(pixel[1]);
        b.push(pixel[2]);
    }

    (r, g, b)
}

// From three vecs of R,G,B u8 to a vec of the RGB values
fn convert_from_channels(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> Vec<[u8; 3]> {
    let mut res = Vec::new();

    for i in 0..r.len() {
        res.push([r[i], g[i], b[i]]);
    }

    res
}

fn invert(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let r_res = r.iter().map(|&x| 255 - x).collect();
    let g_res = g.iter().map(|&x| 255 - x).collect();
    let b_res = b.iter().map(|&x| 255 - x).collect();

    (r_res, g_res, b_res)
}

fn invert_simd_8(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r_res = Vec::new();
    let mut g_res = Vec::new();
    let mut b_res = Vec::new();

    for i in (0..r.len()).step_by(8) {
        if i + 8 > r.len() {
            break;
        }

        // Create vectors for each channel, each vector hold 8 elemts
        // Then subtract the vector from 255 to invert the colors
        let r_vector = u8x8::splat(255) - u8x8::from_slice(&r[i..i + 8]);
        let g_vector = u8x8::splat(255) - u8x8::from_slice(&g[i..i + 8]);
        let b_vector = u8x8::splat(255) - u8x8::from_slice(&b[i..i + 8]);

        // Add the result to the final vec
        r_res.extend_from_slice(&r_vector.to_array());
        g_res.extend_from_slice(&g_vector.to_array());
        b_res.extend_from_slice(&b_vector.to_array());
    }

    // Handle the elements that are not multiple of 8, they get individually inverted
    let remaining = r.len() % 8;
    for i in r.len() - remaining..r.len() {
        r_res.push(255 - r[i]);
        g_res.push(255 - g[i]);
        b_res.push(255 - b[i]);
    }

    (r_res, g_res, b_res)
}

fn invert_simd_16(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r_res = Vec::new();
    let mut g_res = Vec::new();
    let mut b_res = Vec::new();

    for i in (0..r.len()).step_by(16) {
        if i + 16 > r.len() {
            break;
        }

        let r_vector = u8x16::splat(255) - u8x16::from_slice(&r[i..i + 16]);
        let g_vector = u8x16::splat(255) - u8x16::from_slice(&g[i..i + 16]);
        let b_vector = u8x16::splat(255) - u8x16::from_slice(&b[i..i + 16]);

        r_res.extend_from_slice(&r_vector.to_array());
        g_res.extend_from_slice(&g_vector.to_array());
        b_res.extend_from_slice(&b_vector.to_array());
    }

    let remaining = r.len() % 16;
    for i in r.len() - remaining..r.len() {
        r_res.push(255 - r[i]);
        g_res.push(255 - g[i]);
        b_res.push(255 - b[i]);
    }

    (r_res, g_res, b_res)
}

fn invert_simd_32(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r_res = Vec::new();
    let mut g_res = Vec::new();
    let mut b_res = Vec::new();

    for i in (0..r.len()).step_by(32) {
        if i + 32 > r.len() {
            break;
        }

        let r_vector = u8x32::splat(255) - u8x32::from_slice(&r[i..i + 32]);
        let g_vector = u8x32::splat(255) - u8x32::from_slice(&g[i..i + 32]);
        let b_vector = u8x32::splat(255) - u8x32::from_slice(&b[i..i + 32]);

        r_res.extend_from_slice(&r_vector.to_array());
        g_res.extend_from_slice(&g_vector.to_array());
        b_res.extend_from_slice(&b_vector.to_array());
    }

    let remaining = r.len() % 32;
    for i in r.len() - remaining..r.len() {
        r_res.push(255 - r[i]);
        g_res.push(255 - g[i]);
        b_res.push(255 - b[i]);
    }

    (r_res, g_res, b_res)
}

fn invert_simd_64(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r_res = Vec::new();
    let mut g_res = Vec::new();
    let mut b_res = Vec::new();

    for i in (0..r.len()).step_by(64) {
        if i + 64 > r.len() {
            break;
        }

        let r_vector = u8x64::splat(255) - u8x64::from_slice(&r[i..i + 64]);
        let g_vector = u8x64::splat(255) - u8x64::from_slice(&g[i..i + 64]);
        let b_vector = u8x64::splat(255) - u8x64::from_slice(&b[i..i + 64]);

        r_res.extend_from_slice(&r_vector.to_array());
        g_res.extend_from_slice(&g_vector.to_array());
        b_res.extend_from_slice(&b_vector.to_array());
    }

    let remaining = r.len() % 64;
    for i in r.len() - remaining..r.len() {
        r_res.push(255 - r[i]);
        g_res.push(255 - g[i]);
        b_res.push(255 - b[i]);
    }

    (r_res, g_res, b_res)
}
