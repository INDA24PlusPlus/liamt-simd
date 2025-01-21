use criterion::{black_box, Criterion};
use image::*;
use std::simd::num::*;
use std::simd::*;

// Grayscale conversion
pub fn convert(img: DynamicImage, benchmark: bool) -> DynamicImage {
    let img_buf = convert_img_to_vec8(img.clone());

    if benchmark {
        let mut criterion = Criterion::default();
        let mut group = criterion.benchmark_group("grayscales");

        group.bench_function("grayscale no simd", |b| {
            b.iter(|| {
                grayscale(black_box(img_buf.clone()));
            })
        });

        group.bench_function("grayscale simd 8", |b| {
            b.iter(|| {
                grayscale_simd_8(black_box(img_buf.clone()));
            })
        });

        group.bench_function("grayscale simd 16", |b| {
            b.iter(|| {
                grayscale_simd_16(black_box(img_buf.clone()));
            })
        });

        group.bench_function("grayscale simd 32", |b| {
            b.iter(|| {
                grayscale_simd_32(black_box(img_buf.clone()));
            })
        });

        group.bench_function("grayscale simd 64", |b| {
            b.iter(|| {
                grayscale_simd_64(black_box(img_buf.clone()));
            })
        });

        // Make sure all functions return the same result
        let gray_img_buf = grayscale(img_buf.clone());
        let gray_img_buf_simd_8 = grayscale_simd_8(img_buf.clone());
        let gray_img_buf_simd_16 = grayscale_simd_16(img_buf.clone());
        let gray_img_buf_simd_32 = grayscale_simd_32(img_buf.clone());
        let gray_img_buf_simd_64 = grayscale_simd_64(img_buf.clone());
        assert_eq!(gray_img_buf, gray_img_buf_simd_8);
        assert_eq!(gray_img_buf, gray_img_buf_simd_16);
        assert_eq!(gray_img_buf, gray_img_buf_simd_32);
        assert_eq!(gray_img_buf, gray_img_buf_simd_64);

        group.finish();
    }

    println!("Converting image to grayscale...");
    let gray_img_buf = grayscale(img_buf);
    let (width, height) = img.dimensions();

    // Return grayscale image
    convert_vec8_to_img(gray_img_buf, width, height)
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

fn grayscale(pixels: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    let mut result = Vec::new();
    for pixel in pixels {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        // Take each pixel and use formula to get the grayscale value given RGB values
        let gray = (r * 0.299 + g * 0.587 + b * 0.114) as u8;

        result.push([gray, gray, gray]);
    }

    result
}

// First SIMD implementation, uses f32x8 (vector with 8 f32 values)
fn grayscale_simd_8(pixels: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    let mut result = Vec::new();

    // Take chunks of 8 pixels and batch-convert them
    for chunk in pixels.chunks_exact(8) {
        // This is very ugly, but also the fastest (performance) way to do it
        let r = f32x8::from_array([
            chunk[0][0] as f32,
            chunk[1][0] as f32,
            chunk[2][0] as f32,
            chunk[3][0] as f32,
            chunk[4][0] as f32,
            chunk[5][0] as f32,
            chunk[6][0] as f32,
            chunk[7][0] as f32,
        ]);

        let g = f32x8::from_array([
            chunk[0][1] as f32,
            chunk[1][1] as f32,
            chunk[2][1] as f32,
            chunk[3][1] as f32,
            chunk[4][1] as f32,
            chunk[5][1] as f32,
            chunk[6][1] as f32,
            chunk[7][1] as f32,
        ]);

        let b = f32x8::from_array([
            chunk[0][2] as f32,
            chunk[1][2] as f32,
            chunk[2][2] as f32,
            chunk[3][2] as f32,
            chunk[4][2] as f32,
            chunk[5][2] as f32,
            chunk[6][2] as f32,
            chunk[7][2] as f32,
        ]);

        // Multiply each R,G,B with their given weight (0.299 for red, 0.587 for green, 0.114 for blue) and sum them (dot product)
        let gray = (r * f32x8::splat(0.299) + g * f32x8::splat(0.587) + b * f32x8::splat(0.114))
            .cast::<u8>();
        let gray_arr: [u8; 8] = gray.into();

        for c in gray_arr {
            result.push([c, c, c]);
        }
    }

    // If image are not a multiple of 8 pixels, process the remaining pixels one by one
    for &[r, g, b] in pixels.chunks_exact(8).remainder() {
        let gray = ((r as f32) * 0.299 + (g as f32) * 0.587 + (b as f32) * 0.114) as u8;
        result.push([gray, gray, gray]);
    }

    result
}

// Uses f32x16 (vector with 16 f32 values)
// Same implentation as above
fn grayscale_simd_16(pixels: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    let mut result = Vec::new();

    for chunk in pixels.chunks_exact(16) {
        let r = f32x16::from_array([
            chunk[0][0] as f32,
            chunk[1][0] as f32,
            chunk[2][0] as f32,
            chunk[3][0] as f32,
            chunk[4][0] as f32,
            chunk[5][0] as f32,
            chunk[6][0] as f32,
            chunk[7][0] as f32,
            chunk[8][0] as f32,
            chunk[9][0] as f32,
            chunk[10][0] as f32,
            chunk[11][0] as f32,
            chunk[12][0] as f32,
            chunk[13][0] as f32,
            chunk[14][0] as f32,
            chunk[15][0] as f32,
        ]);

        let g = f32x16::from_array([
            chunk[0][1] as f32,
            chunk[1][1] as f32,
            chunk[2][1] as f32,
            chunk[3][1] as f32,
            chunk[4][1] as f32,
            chunk[5][1] as f32,
            chunk[6][1] as f32,
            chunk[7][1] as f32,
            chunk[8][1] as f32,
            chunk[9][1] as f32,
            chunk[10][1] as f32,
            chunk[11][1] as f32,
            chunk[12][1] as f32,
            chunk[13][1] as f32,
            chunk[14][1] as f32,
            chunk[15][1] as f32,
        ]);

        let b = f32x16::from_array([
            chunk[0][2] as f32,
            chunk[1][2] as f32,
            chunk[2][2] as f32,
            chunk[3][2] as f32,
            chunk[4][2] as f32,
            chunk[5][2] as f32,
            chunk[6][2] as f32,
            chunk[7][2] as f32,
            chunk[8][2] as f32,
            chunk[9][2] as f32,
            chunk[10][2] as f32,
            chunk[11][2] as f32,
            chunk[12][2] as f32,
            chunk[13][2] as f32,
            chunk[14][2] as f32,
            chunk[15][2] as f32,
        ]);

        let gray = (r * f32x16::splat(0.299) + g * f32x16::splat(0.587) + b * f32x16::splat(0.114))
            .cast::<u8>();
        let gray_arr: [u8; 16] = gray.into();

        for c in gray_arr {
            result.push([c, c, c]);
        }
    }

    for &[r, g, b] in pixels.chunks_exact(16).remainder() {
        let gray = ((r as f32) * 0.299 + (g as f32) * 0.587 + (b as f32) * 0.114) as u8;
        result.push([gray, gray, gray]);
    }

    result
}

// Uses f32x32 (vector with 32 f32 values)
fn grayscale_simd_32(pixels: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    let mut result = Vec::new();

    for chunk in pixels.chunks_exact(32) {
        let r = f32x32::from_array([
            chunk[0][0] as f32,
            chunk[1][0] as f32,
            chunk[2][0] as f32,
            chunk[3][0] as f32,
            chunk[4][0] as f32,
            chunk[5][0] as f32,
            chunk[6][0] as f32,
            chunk[7][0] as f32,
            chunk[8][0] as f32,
            chunk[9][0] as f32,
            chunk[10][0] as f32,
            chunk[11][0] as f32,
            chunk[12][0] as f32,
            chunk[13][0] as f32,
            chunk[14][0] as f32,
            chunk[15][0] as f32,
            chunk[16][0] as f32,
            chunk[17][0] as f32,
            chunk[18][0] as f32,
            chunk[19][0] as f32,
            chunk[20][0] as f32,
            chunk[21][0] as f32,
            chunk[22][0] as f32,
            chunk[23][0] as f32,
            chunk[24][0] as f32,
            chunk[25][0] as f32,
            chunk[26][0] as f32,
            chunk[27][0] as f32,
            chunk[28][0] as f32,
            chunk[29][0] as f32,
            chunk[30][0] as f32,
            chunk[31][0] as f32,
        ]);

        let g = f32x32::from_array([
            chunk[0][1] as f32,
            chunk[1][1] as f32,
            chunk[2][1] as f32,
            chunk[3][1] as f32,
            chunk[4][1] as f32,
            chunk[5][1] as f32,
            chunk[6][1] as f32,
            chunk[7][1] as f32,
            chunk[8][1] as f32,
            chunk[9][1] as f32,
            chunk[10][1] as f32,
            chunk[11][1] as f32,
            chunk[12][1] as f32,
            chunk[13][1] as f32,
            chunk[14][1] as f32,
            chunk[15][1] as f32,
            chunk[16][1] as f32,
            chunk[17][1] as f32,
            chunk[18][1] as f32,
            chunk[19][1] as f32,
            chunk[20][1] as f32,
            chunk[21][1] as f32,
            chunk[22][1] as f32,
            chunk[23][1] as f32,
            chunk[24][1] as f32,
            chunk[25][1] as f32,
            chunk[26][1] as f32,
            chunk[27][1] as f32,
            chunk[28][1] as f32,
            chunk[29][1] as f32,
            chunk[30][1] as f32,
            chunk[31][1] as f32,
        ]);

        let b = f32x32::from_array([
            chunk[0][2] as f32,
            chunk[1][2] as f32,
            chunk[2][2] as f32,
            chunk[3][2] as f32,
            chunk[4][2] as f32,
            chunk[5][2] as f32,
            chunk[6][2] as f32,
            chunk[7][2] as f32,
            chunk[8][2] as f32,
            chunk[9][2] as f32,
            chunk[10][2] as f32,
            chunk[11][2] as f32,
            chunk[12][2] as f32,
            chunk[13][2] as f32,
            chunk[14][2] as f32,
            chunk[15][2] as f32,
            chunk[16][2] as f32,
            chunk[17][2] as f32,
            chunk[18][2] as f32,
            chunk[19][2] as f32,
            chunk[20][2] as f32,
            chunk[21][2] as f32,
            chunk[22][2] as f32,
            chunk[23][2] as f32,
            chunk[24][2] as f32,
            chunk[25][2] as f32,
            chunk[26][2] as f32,
            chunk[27][2] as f32,
            chunk[28][2] as f32,
            chunk[29][2] as f32,
            chunk[30][2] as f32,
            chunk[31][2] as f32,
        ]);

        let gray = (r * f32x32::splat(0.299) + g * f32x32::splat(0.587) + b * f32x32::splat(0.114))
            .cast::<u8>();
        let gray_arr: [u8; 32] = gray.into();

        for c in gray_arr {
            result.push([c, c, c]);
        }
    }

    for &[r, g, b] in pixels.chunks_exact(32).remainder() {
        let gray = ((r as f32) * 0.299 + (g as f32) * 0.587 + (b as f32) * 0.114) as u8;
        result.push([gray, gray, gray]);
    }

    result
}

// Uses f32x64 (vector with 64 f32 values)
fn grayscale_simd_64(pixels: Vec<[u8; 3]>) -> Vec<[u8; 3]> {
    let mut result = Vec::new();

    for chunk in pixels.chunks_exact(64) {
        let r = f32x64::from_array([
            chunk[0][0] as f32,
            chunk[1][0] as f32,
            chunk[2][0] as f32,
            chunk[3][0] as f32,
            chunk[4][0] as f32,
            chunk[5][0] as f32,
            chunk[6][0] as f32,
            chunk[7][0] as f32,
            chunk[8][0] as f32,
            chunk[9][0] as f32,
            chunk[10][0] as f32,
            chunk[11][0] as f32,
            chunk[12][0] as f32,
            chunk[13][0] as f32,
            chunk[14][0] as f32,
            chunk[15][0] as f32,
            chunk[16][0] as f32,
            chunk[17][0] as f32,
            chunk[18][0] as f32,
            chunk[19][0] as f32,
            chunk[20][0] as f32,
            chunk[21][0] as f32,
            chunk[22][0] as f32,
            chunk[23][0] as f32,
            chunk[24][0] as f32,
            chunk[25][0] as f32,
            chunk[26][0] as f32,
            chunk[27][0] as f32,
            chunk[28][0] as f32,
            chunk[29][0] as f32,
            chunk[30][0] as f32,
            chunk[31][0] as f32,
            chunk[32][0] as f32,
            chunk[33][0] as f32,
            chunk[34][0] as f32,
            chunk[35][0] as f32,
            chunk[36][0] as f32,
            chunk[37][0] as f32,
            chunk[38][0] as f32,
            chunk[39][0] as f32,
            chunk[40][0] as f32,
            chunk[41][0] as f32,
            chunk[42][0] as f32,
            chunk[43][0] as f32,
            chunk[44][0] as f32,
            chunk[45][0] as f32,
            chunk[46][0] as f32,
            chunk[47][0] as f32,
            chunk[48][0] as f32,
            chunk[49][0] as f32,
            chunk[50][0] as f32,
            chunk[51][0] as f32,
            chunk[52][0] as f32,
            chunk[53][0] as f32,
            chunk[54][0] as f32,
            chunk[55][0] as f32,
            chunk[56][0] as f32,
            chunk[57][0] as f32,
            chunk[58][0] as f32,
            chunk[59][0] as f32,
            chunk[60][0] as f32,
            chunk[61][0] as f32,
            chunk[62][0] as f32,
            chunk[63][0] as f32,
        ]);

        let g = f32x64::from_array([
            chunk[0][1] as f32,
            chunk[1][1] as f32,
            chunk[2][1] as f32,
            chunk[3][1] as f32,
            chunk[4][1] as f32,
            chunk[5][1] as f32,
            chunk[6][1] as f32,
            chunk[7][1] as f32,
            chunk[8][1] as f32,
            chunk[9][1] as f32,
            chunk[10][1] as f32,
            chunk[11][1] as f32,
            chunk[12][1] as f32,
            chunk[13][1] as f32,
            chunk[14][1] as f32,
            chunk[15][1] as f32,
            chunk[16][1] as f32,
            chunk[17][1] as f32,
            chunk[18][1] as f32,
            chunk[19][1] as f32,
            chunk[20][1] as f32,
            chunk[21][1] as f32,
            chunk[22][1] as f32,
            chunk[23][1] as f32,
            chunk[24][1] as f32,
            chunk[25][1] as f32,
            chunk[26][1] as f32,
            chunk[27][1] as f32,
            chunk[28][1] as f32,
            chunk[29][1] as f32,
            chunk[30][1] as f32,
            chunk[31][1] as f32,
            chunk[32][1] as f32,
            chunk[33][1] as f32,
            chunk[34][1] as f32,
            chunk[35][1] as f32,
            chunk[36][1] as f32,
            chunk[37][1] as f32,
            chunk[38][1] as f32,
            chunk[39][1] as f32,
            chunk[40][1] as f32,
            chunk[41][1] as f32,
            chunk[42][1] as f32,
            chunk[43][1] as f32,
            chunk[44][1] as f32,
            chunk[45][1] as f32,
            chunk[46][1] as f32,
            chunk[47][1] as f32,
            chunk[48][1] as f32,
            chunk[49][1] as f32,
            chunk[50][1] as f32,
            chunk[51][1] as f32,
            chunk[52][1] as f32,
            chunk[53][1] as f32,
            chunk[54][1] as f32,
            chunk[55][1] as f32,
            chunk[56][1] as f32,
            chunk[57][1] as f32,
            chunk[58][1] as f32,
            chunk[59][1] as f32,
            chunk[60][1] as f32,
            chunk[61][1] as f32,
            chunk[62][1] as f32,
            chunk[63][1] as f32,
        ]);

        let b = f32x64::from_array([
            chunk[0][2] as f32,
            chunk[1][2] as f32,
            chunk[2][2] as f32,
            chunk[3][2] as f32,
            chunk[4][2] as f32,
            chunk[5][2] as f32,
            chunk[6][2] as f32,
            chunk[7][2] as f32,
            chunk[8][2] as f32,
            chunk[9][2] as f32,
            chunk[10][2] as f32,
            chunk[11][2] as f32,
            chunk[12][2] as f32,
            chunk[13][2] as f32,
            chunk[14][2] as f32,
            chunk[15][2] as f32,
            chunk[16][2] as f32,
            chunk[17][2] as f32,
            chunk[18][2] as f32,
            chunk[19][2] as f32,
            chunk[20][2] as f32,
            chunk[21][2] as f32,
            chunk[22][2] as f32,
            chunk[23][2] as f32,
            chunk[24][2] as f32,
            chunk[25][2] as f32,
            chunk[26][2] as f32,
            chunk[27][2] as f32,
            chunk[28][2] as f32,
            chunk[29][2] as f32,
            chunk[30][2] as f32,
            chunk[31][2] as f32,
            chunk[32][2] as f32,
            chunk[33][2] as f32,
            chunk[34][2] as f32,
            chunk[35][2] as f32,
            chunk[36][2] as f32,
            chunk[37][2] as f32,
            chunk[38][2] as f32,
            chunk[39][2] as f32,
            chunk[40][2] as f32,
            chunk[41][2] as f32,
            chunk[42][2] as f32,
            chunk[43][2] as f32,
            chunk[44][2] as f32,
            chunk[45][2] as f32,
            chunk[46][2] as f32,
            chunk[47][2] as f32,
            chunk[48][2] as f32,
            chunk[49][2] as f32,
            chunk[50][2] as f32,
            chunk[51][2] as f32,
            chunk[52][2] as f32,
            chunk[53][2] as f32,
            chunk[54][2] as f32,
            chunk[55][2] as f32,
            chunk[56][2] as f32,
            chunk[57][2] as f32,
            chunk[58][2] as f32,
            chunk[59][2] as f32,
            chunk[60][2] as f32,
            chunk[61][2] as f32,
            chunk[62][2] as f32,
            chunk[63][2] as f32,
        ]);

        let gray = (r * f32x64::splat(0.299) + g * f32x64::splat(0.587) + b * f32x64::splat(0.114))
            .cast::<u8>();
        let gray_arr: [u8; 64] = gray.into();

        for c in gray_arr {
            result.push([c, c, c]);
        }
    }

    for &[r, g, b] in pixels.chunks_exact(64).remainder() {
        let gray = ((r as f32) * 0.299 + (g as f32) * 0.587 + (b as f32) * 0.114) as u8;
        result.push([gray, gray, gray]);
    }

    result
}
