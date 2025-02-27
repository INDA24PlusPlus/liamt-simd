#![feature(portable_simd)]
mod grayscale;
mod invert;

use clap::{Parser as ClapParser, Subcommand};
use std::path::PathBuf;

#[derive(ClapParser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert image to grayscale
    Grayscale {
        #[arg(short, long)]
        benchmark: bool,
        file: PathBuf,
    },
    /// Invert the colors of image
    Invert {
        #[arg(short, long)]
        benchmark: bool,
        file: PathBuf,
    },
}

fn main() {
    let args = Cli::parse();

    match &args.command {
        Commands::Grayscale { benchmark, file } => {
            let img = image::open(file).unwrap();
            let new_img = grayscale::convert(img, *benchmark);
            new_img
                .save("grey_".to_string() + file.to_str().unwrap())
                .unwrap();
        }
        Commands::Invert { benchmark, file } => {
            let img = image::open(file).unwrap();
            let new_img = invert::convert(img, *benchmark);
            new_img
                .save("invert_".to_string() + file.to_str().unwrap())
                .unwrap();
        }
    }
}
