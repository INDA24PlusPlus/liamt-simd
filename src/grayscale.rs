use criterion::Criterion;
use image::*;
use std::simd::num::*;
use std::simd::*;

// Grayscale conversion
pub fn convert(img: DynamicImage, benchmark: bool) -> DynamicImage {
    let img_buf = convert_img_to_vec8(img.clone());
    let (r, g, b) = convert_to_channels(img_buf.clone());

    if benchmark {
        let mut criterion = Criterion::default();
        let mut group = criterion.benchmark_group("grayscales");

        group.bench_function("grayscale no simd", |be| {
            be.iter(|| {
                grayscale(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("grayscale simd 8", |be| {
            be.iter(|| {
                grayscale_simd_8(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("grayscale simd 16", |be| {
            be.iter(|| {
                grayscale_simd_16(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("grayscale simd 32", |be: &mut criterion::Bencher<'_>| {
            be.iter(|| {
                grayscale_simd_32(r.clone(), g.clone(), b.clone());
            })
        });

        group.bench_function("grayscale simd 64", |be| {
            be.iter(|| {
                grayscale_simd_64(r.clone(), g.clone(), b.clone());
            })
        });

        // Make sure all functions return the same result
        let gray_img_buf = grayscale(r.clone(), g.clone(), b.clone());
        let gray_img_buf_simd_8 = grayscale_simd_8(r.clone(), g.clone(), b.clone());
        let gray_img_buf_simd_16 = grayscale_simd_16(r.clone(), g.clone(), b.clone());
        let gray_img_buf_simd_32 = grayscale_simd_32(r.clone(), g.clone(), b.clone());
        let gray_img_buf_simd_64 = grayscale_simd_64(r.clone(), g.clone(), b.clone());
        assert_eq!(gray_img_buf, gray_img_buf_simd_8);
        assert_eq!(gray_img_buf, gray_img_buf_simd_16);
        assert_eq!(gray_img_buf, gray_img_buf_simd_32);
        assert_eq!(gray_img_buf, gray_img_buf_simd_64);

        group.finish();
    }

    println!("Converting image to grayscale...");
    let (r_gray, g_gray, b_gray) = grayscale(r.clone(), g.clone(), b.clone());
    let (width, height) = img.dimensions();

    // Return grayscale image
    convert_vec8_to_img(convert_from_channels(r_gray, g_gray, b_gray), width, height)
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

fn grayscale(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut res = Vec::new();

    for i in 0..r.len() {
        // I'm not using floats here because the SIMD implementation will use integers as well
        let gray = (((r[i] as u32) * 299 + (g[i] as u32) * 587 + (b[i] as u32) * 114) / 1000) as u8;
        res.push(gray);
    }

    (res.clone(), res.clone(), res.clone())
}

// First SIMD implementation, uses u32x8 (vector with 8 u32 values)
fn grayscale_simd_8(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut res = Vec::new();

    for i in (0..r.len()).step_by(8) {
        if i + 8 > r.len() {
            break;
        }

        // Create a vector with 8 u32 values from the RGB values
        let r_vector = u8x8::from_slice(&r[i..i + 8]).cast::<u32>();
        let g_vector = u8x8::from_slice(&g[i..i + 8]).cast::<u32>();
        let b_vector = u8x8::from_slice(&b[i..i + 8]).cast::<u32>();

        // Multiply each element of the vector with the given weight
        // Then divide by 1000 (Because its * 0.299 and not * 299)
        let gray = ((r_vector * u32x8::splat(299)
            + g_vector * u32x8::splat(587)
            + b_vector * u32x8::splat(114))
            / u32x8::splat(1000))
        .cast::<u8>();

        res.extend_from_slice(&gray.to_array());
    }

    // And if the RGB values are not a multiple of 8
    // Then we do individual conversions for each
    let remainder = r.len() % 8;
    for i in r.len() - remainder..r.len() {
        let gray = (((r[i] as u32) * 299 + (g[i] as u32) * 587 + (b[i] as u32) * 114) / 1000) as u8;
        res.push(gray);
    }

    (res.clone(), res.clone(), res.clone())
}

// Same as above but with u32x16
fn grayscale_simd_16(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut res = Vec::new();

    for i in (0..r.len()).step_by(16) {
        if i + 16 > r.len() {
            break;
        }

        let r_vector = u8x16::from_slice(&r[i..i + 16]).cast::<u32>();
        let g_vector = u8x16::from_slice(&g[i..i + 16]).cast::<u32>();
        let b_vector = u8x16::from_slice(&b[i..i + 16]).cast::<u32>();

        let gray = ((r_vector * u32x16::splat(299)
            + g_vector * u32x16::splat(587)
            + b_vector * u32x16::splat(114))
            / u32x16::splat(1000))
        .cast::<u8>();

        res.extend_from_slice(&gray.to_array());
    }

    let remainder = r.len() % 16;
    for i in r.len() - remainder..r.len() {
        let gray = (((r[i] as u32) * 299 + (g[i] as u32) * 587 + (b[i] as u32) * 114) / 1000) as u8;
        res.push(gray);
    }

    (res.clone(), res.clone(), res.clone())
}

fn grayscale_simd_32(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut res = Vec::new();

    for i in (0..r.len()).step_by(32) {
        if i + 32 > r.len() {
            break;
        }

        let r_vector = u8x32::from_slice(&r[i..i + 32]).cast::<u32>();
        let g_vector = u8x32::from_slice(&g[i..i + 32]).cast::<u32>();
        let b_vector = u8x32::from_slice(&b[i..i + 32]).cast::<u32>();

        let gray = ((r_vector * u32x32::splat(299)
            + g_vector * u32x32::splat(587)
            + b_vector * u32x32::splat(114))
            / u32x32::splat(1000))
        .cast::<u8>();

        res.extend_from_slice(&gray.to_array());
    }

    let remainder = r.len() % 32;
    for i in r.len() - remainder..r.len() {
        let gray = (((r[i] as u32) * 299 + (g[i] as u32) * 587 + (b[i] as u32) * 114) / 1000) as u8;
        res.push(gray);
    }

    (res.clone(), res.clone(), res.clone())
}

fn grayscale_simd_64(r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut res = Vec::new();

    for i in (0..r.len()).step_by(64) {
        if i + 64 > r.len() {
            break;
        }

        let r_vector = u8x64::from_slice(&r[i..i + 64]).cast::<u32>();
        let g_vector = u8x64::from_slice(&g[i..i + 64]).cast::<u32>();
        let b_vector = u8x64::from_slice(&b[i..i + 64]).cast::<u32>();

        let gray = ((r_vector * u32x64::splat(299)
            + g_vector * u32x64::splat(587)
            + b_vector * u32x64::splat(114))
            / u32x64::splat(1000))
        .cast::<u8>();

        res.extend_from_slice(&gray.to_array());
    }

    let remainder = r.len() % 64;
    for i in r.len() - remainder..r.len() {
        let gray = (((r[i] as u32) * 299 + (g[i] as u32) * 587 + (b[i] as u32) * 114) / 1000) as u8;
        res.push(gray);
    }

    (res.clone(), res.clone(), res.clone())
}
