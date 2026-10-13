//! Stitches the photos given on the command line:
//! `cargo run --release --example stitch -- out.png a.jpg b.jpg ...`
//! Add `--projection rectilinear|cylindrical|spherical`, `--no-crop` or
//! `--no-refine` before the output path to override the defaults, and
//! `--in-order` when the photos are listed in shooting order.

use nabu_stitch::{stitch, Options, Projection};
use std::time::Instant;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut options = Options::default();
    while args.first().is_some_and(|a| a.starts_with("--")) {
        match args.remove(0).as_str() {
            "--no-crop" => options.crop = false,
            "--no-refine" => options.fine_alignment = false,
            "--in-order" => options.in_order = true,
            "--projection" => {
                options.projection = Some(match args.remove(0).as_str() {
                    "rectilinear" => Projection::Rectilinear,
                    "cylindrical" => Projection::Cylindrical,
                    "planar" => Projection::Planar,
                    _ => Projection::Spherical,
                })
            }
            other => panic!("unknown option {other}"),
        }
    }
    let output = args.remove(0);
    let images: Vec<_> = args
        .iter()
        .map(|p| image::open(p).unwrap_or_else(|e| panic!("{p}: {e}")).to_rgba8())
        .collect();
    let start = Instant::now();
    let mut last = None;
    let result = stitch(&images, &options, &mut |stage, value| {
        if last != Some(stage) {
            eprintln!("{:>7.2}s {stage:?}", start.elapsed().as_secs_f64());
            last = Some(stage);
        }
        let _ = value;
        true
    })
    .expect("stitching failed");
    eprintln!(
        "{:>7.2}s done: {}x{} {:?}, rms {:.3}px, fov {:.1}°x{:.1}°, focal {:.1}px k1 {:.4} k2 {:.4}, dropped {:?}",
        start.elapsed().as_secs_f64(),
        result.image.width(),
        result.image.height(),
        result.projection,
        result.rms,
        result.field_of_view.0,
        result.field_of_view.1,
        result.lens.focal,
        result.lens.k1,
        result.lens.k2,
        result.dropped
    );
    image::DynamicImage::ImageRgba8(result.image).to_rgb8().save(&output).expect("save");
}
