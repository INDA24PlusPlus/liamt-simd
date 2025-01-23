# SIMD Image processing
**Usage:** `cargo run <COMMAND> <IMAGE> [OPTIONS]`

**Commands:**
  - `grayscale`  Convert image to grayscale
  - `invert`     Invert the colors of image

**Image:** Path to the image file

**Options:**
  - `--benchmark`  Benchmark SIMD implementations against the non-SIMD version

###### *ps. run with `--release` flag for better performance*
## Examples
```sh
cargo run grayscale path/to/image.jpg --benchmark
cargo run invert path/to/image.jpg
```

## Benchmarks
The benchmarks seems to be inconsistent, they differ a bit between each run :(
### Grayscale
`cargo run --release grayscale img.jpg --benchmark`
| Non-SIMD | SIMD u32x8 | SIMD u32x16 | SIMD u32x32 | SIMD u32x64 |
|----------|------------|-------------|-------------|-------------|
| 2.3317 ms| 1.9064 ms  | 1.7944 ms   | 1.4068 ms   | 1.4101 ms   |

### Invert
`cargo run --release invert img.jpg --benchmark`
| Non-SIMD | SIMD u8x8 | SIMD u8x16 | SIMD u8x32 | SIMD u8x64 |
|----------|-----------|------------|------------|------------|
| 1.2015 ms| 1.3193 ms | 1.1249 ms  | 822.05 µs  | 812.96 µs  |