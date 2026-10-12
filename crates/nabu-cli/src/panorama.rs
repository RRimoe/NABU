//! `nabu panorama`: stitching straight from the terminal.
//!
//! The one command that does not need the editor: the stitcher is a plain Rust
//! library, so a folder of photos becomes a panorama on a server with no
//! display at all.

use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use nabu_stitch::{stitch, Options, Projection, Stage, StitchError};
use std::io::{self, Write};
use std::path::Path;

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// Decodes a photo the right way up: phones store most pictures sideways and
/// say so in the EXIF orientation, which the stitcher must not ignore.
pub fn load(path: &Path) -> io::Result<RgbaImage> {
    let fail = |e: image::ImageError| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display()));
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder().map_err(fail)?;
    let orientation = decoder.orientation().map_err(fail)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(fail)?;
    image.apply_orientation(orientation);
    Ok(image.to_rgba8())
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Features => "finding features",
        Stage::Matching => "matching",
        Stage::Aligning => "aligning",
        Stage::Refining => "refining",
        Stage::Exposure => "balancing exposure",
        Stage::Seams => "finding seams",
        Stage::Blending => "blending",
    }
}

pub fn run(args: &[String]) -> io::Result<i32> {
    let mut options = Options::default();
    let mut paths = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--no-crop" => options.crop = false,
            "--projection" => {
                options.projection = match iter.next().map(String::as_str) {
                    Some("auto") => None,
                    Some("rectilinear") => Some(Projection::Rectilinear),
                    Some("cylindrical") => Some(Projection::Cylindrical),
                    Some("spherical") => Some(Projection::Spherical),
                    other => return Err(invalid(format!("unknown projection {other:?}"))),
                }
            }
            "--max-mp" => {
                let raw = iter.next().ok_or_else(|| invalid("--max-mp needs a value".into()))?;
                options.max_megapixels = raw.parse().map_err(|_| invalid(format!("--max-mp: \"{raw}\"")))?;
            }
            flag if flag.starts_with("--") => return Err(invalid(format!("unknown option {flag}"))),
            path => paths.push(path.to_string()),
        }
    }
    if paths.len() < 3 {
        return Err(invalid("usage: nabu panorama [options] <output> <photo> <photo> [...]".into()));
    }
    let output = paths.remove(0);

    let images = paths.iter().map(|p| load(Path::new(p))).collect::<io::Result<Vec<_>>>()?;
    let mut last = None;
    let result = stitch(&images, &options, &mut |stage, _| {
        if last != Some(stage) {
            eprintln!("{}…", stage_name(stage));
            let _ = io::stderr().flush();
            last = Some(stage);
        }
        true
    });

    let panorama = match result {
        Ok(panorama) => panorama,
        Err(StitchError::TooFewImages) => return Err(invalid("at least two photos are needed".into())),
        Err(StitchError::NoOverlap) => {
            return Err(io::Error::other("the photos do not overlap enough to be stitched"))
        }
        Err(err) => return Err(io::Error::other(format!("{err:?}"))),
    };

    let lower = output.to_ascii_lowercase();
    let picture = if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        // JPEG has no transparency: what is outside the panorama becomes white.
        let mut rgb = image::RgbImage::new(panorama.image.width(), panorama.image.height());
        for (out, px) in rgb.pixels_mut().zip(panorama.image.pixels()) {
            let a = px[3] as f32 / 255.0;
            for c in 0..3 {
                out[c] = (px[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            }
        }
        DynamicImage::ImageRgb8(rgb)
    } else {
        DynamicImage::ImageRgba8(panorama.image)
    };
    picture
        .save(&output)
        .map_err(|e| io::Error::other(format!("{output}: {e}")))?;

    let (h, v) = panorama.field_of_view;
    eprintln!(
        "{output}: {}x{} {:?}, {} of {} photos, alignment error {:.2} px, field of view {h:.0}°x{v:.0}°",
        picture.width(),
        picture.height(),
        panorama.projection,
        panorama.used.len(),
        paths.len(),
        panorama.rms,
    );
    for i in &panorama.dropped {
        eprintln!("left out (no overlap): {}", paths[*i]);
    }
    Ok(0)
}
