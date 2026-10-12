<div align="center">

<img src="docs/nabu logo color.png" width="200"/>

# NABU

### An image editor that has already looked at your photo.

Everyone else added a text box.
We took ours out.

---

<img src="docs/nabu 1.png" width="900"/>

<img src="docs/nabu 2.png" width="900"/>

</div>

## The idea

Image editors have barely changed in thirty years. They keep adding tools,
panels and buttons. Then AI arrived, and most applications added one more
button.

Nabu had a text box too, and we removed it — because asking someone to write a
good prompt is asking them to do the hard part.

> **You should not have to describe your photo to an editor that can see it.**

So Nabu looks first. It measures the light, the contrast, the colour and the
focus, and reads what is in the frame. By the time the photo has finished
opening, it already has something to say about it — and everything it proposes
shows you the result on your own picture before you accept it.

---

## How it works instead

**It measures.** Exposure, contrast, saturation, clipped shadows and
highlights, colour cast, sharpness — read straight from the pixels, in
milliseconds, with no model downloaded. That is where the proposals come from,
which is why they are specific: *"the colours are close to grey and can be
brought out"*, not *"enhance"*.

**You point.** Draw roughly around something. Nabu refines the outline to the
real object, and the actions it then offers depend on what you circled: a
person can be brought forward, a plant can change colour, a stray object can
be removed. Your drawing is a constraint, not a hint — a model that misreads
the picture can be wrong about the shape of a hand, never about which half of
the photograph the hand is in.

**It proposes, twice.** The measured proposals appear instantly. If a language
model is installed, a second opinion arrives a few seconds later with a
sentence about this particular photo and a recipe to match — and it is never
waited for, because the first tier was already complete.

No prompt to phrase. No menu to hunt. Nothing to unlearn.

---

## What it does today

**Without downloading anything**

Non-destructive adjustments (tone, colour, presence, effects) · tone curves on
four channels with a live histogram · 25 looks, each previewed on your own
photo · crop with aspect presets · free transform with rotation and mirroring
· freehand, rectangular and elliptical selections · brush and eraser · selective focus — subject,
band (tilt-shift), ellipse · corrections confined to one region · layers,
blend modes, full history · save the work as a `.nabu` project and reopen it
with every layer and edit still live · export to PNG, JPEG or WebP, with
quality, size and the file weight shown first · copy and paste between photos.

**With local models, downloaded once from inside the app**

| What | Model | Licence |
|---|---|---|
| Reads the scene | D-FINE nano | Apache-2.0 |
| Cuts out what you circle | SlimSAM-77 | Apache-2.0 |
| Removes objects and rebuilds behind them | LaMa | Apache-2.0 |
| Removes backgrounds | ISNet · RMBG-1.4 | Apache/MIT · non-commercial |
| Increases resolution | Swin2SR · Real-ESRGAN | Apache-2.0 · BSD-3 |
| Measures distance, for real depth of field | Depth Anything V2 Small | Apache-2.0 |
| Restores faces | CodeFormer | non-commercial |
| Writes the second-tier advice | Qwen2.5 0.5B / 1.5B | Apache-2.0 |

Every model carries its licence, and anything that forbids commercial use is
flagged in the manager before you download it. Nabu itself is MIT, and the
default path is deliberately all-permissive: a fresh install is usable by
anyone, for anything.

---

## Local-first, and honest about it

Your images stay on your computer. Models run locally — no account, no cloud,
no subscription. The status bar tells you how many are active and what is
running them.

Nabu also tries not to overstate itself. When a capability is missing it says
so instead of doing something adjacent: no model for the scene means it will
tell you it cannot see, rather than guessing. "Blend in" matches light,
colour, grain and adds a contact shadow — and says plainly that it cannot
relight a subject that faces the wrong way, because that needs a model it does
not have yet.

---

## Extending it without touching the code

A model is described by a **JSON manifest**: which file to load, how the input
tensor is prepared, and which decoder *family* reads the output. Drop a folder
containing a `manifest.json` and its ONNX file into the models directory and
Nabu picks it up at startup. Models inside a family Nabu already knows —
DETR-style detectors, SAM-style segmenters, LaMa-style inpainters, DPT-style
depth — cost a JSON file and nothing else.

Translations work the same way: a flat `code.json` of `key: text` pairs dropped
into the languages folder is a new language, no build required. English is the
reference; anything a translation omits falls back to it.

---

## Running it

```bash
npm install
npm run tauri:llm     # desktop app, with the language engine
npm run tauri dev     # desktop app, without it
npm run dev           # interface only, in a browser (no models)
```

Building: `npm run build:llm`. Tests: `cargo test --all-targets` in
`src-tauri/`, and `npm run check:locales` for translation coverage.

---

## Roadmap

**Working now**

- [x] Open, edit non-destructively, export
- [x] Scene understanding, smart selections, cutouts
- [x] Object and background removal
- [x] Super-resolution and depth of field
- [x] Local model manager, with licences shown
- [x] Compositing between photos with light matching
- [x] Sideloadable models and translations
- [x] Project files that keep every layer and edit live, and export to
      PNG, JPEG and WebP
- [x] Brush and eraser on any pixel layer, confined to the selection
- [x] Before and after: hold \\ to see the original, Y for a draggable split
- [x] Your own looks, saved from the sliders, and copy/paste of settings
      between photos (Ctrl+Shift+C / Ctrl+Shift+V)

**Next**

- [ ] **Perspective placement** — a pasted object should sit on the surface
      you drop it on, not float. Needs surface normals (Metric3D v2, BSD-2)
      and a WebGL layer renderer, since a homography is not an affine
      transform.
- [ ] **Learned harmonisation** — PCT-Net (MPL-2.0) predicts a colour
      transform rather than pixels, which slots into the existing pipeline.
- [ ] **Tiled inpainting** — a hole covering most of the frame still exceeds
      what a 512px model can reconstruct in one pass.
- [ ] **Naming what the detector cannot** — an open-vocabulary grounder, so
      "the yellow sign" resolves outside COCO's 80 classes.

**Later**

- [ ] Text: read it (RapidOCR, Apache-2.0), translate it, remove it
- [ ] Faces as a capability (YuNet, MIT) — find them, work on them by name
- [ ] Semantic segmentation for sky and ground (PP-LiteSeg, Apache-2.0)
- [ ] RAW import, batch processing, layer masks panel

---

## Contributing

Nabu is open source because creativity grows faster together.

Ideas, issues, discussions and pull requests are always welcome — and if you
have an ONNX export of something in a family listed above, it is a JSON file
away from working.

---

## License

MIT License
