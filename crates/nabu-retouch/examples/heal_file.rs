//! Heals a photo from the command line, for trying the fill on real pictures:
//! `cargo run --release --example heal_file -- photo.jpg mask.png out.png`
//! (white in the mask is what gets rebuilt).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [photo, mask, out] = args.as_slice() else {
        eprintln!("usage: heal_file <photo> <mask> <out>");
        std::process::exit(2);
    };
    let photo = image::open(photo).expect("photo").to_rgba8();
    let mask = image::open(mask).expect("mask").to_luma8();
    let start = std::time::Instant::now();
    let healed = nabu_retouch::heal(&photo, &mask, &Default::default(), &|_| true).expect("heal");
    eprintln!("healed in {:.2?}", start.elapsed());
    healed.save(out).expect("save");
}
