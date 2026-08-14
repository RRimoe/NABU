// src/engine/matte.ts
//
// Turning a mask into a cutout you would not notice was cut out.
//
// A mask says which pixels belong to the object. That is not the same as a
// clean cutout, for two reasons that both show up as a halo:
//
//  * the boundary pixels of a photograph are *mixtures* — half subject, half
//    whatever was behind it — so keeping them keeps a ring of the old
//    background, and dropping them eats into the subject;
//  * a hard edge on a soft photograph reads as a sticker.
//
// So the edge is pulled in by a pixel, softened by a pixel, and then the
// colours in the transition band are replaced with colours from inside the
// object. That last step is the one that actually removes the halo: the
// transparency was never the problem, the borrowed colour was.

/** How the edge of a cutout is treated. */
export interface MatteOptions {
  /** Pixels pulled in from the mask edge, dropping the last ring of background. */
  erode: number;
  /** Softness of the final edge, in pixels. */
  feather: number;
  /** Passes of colour decontamination along the edge. */
  defringe: number;
}

/**
 * Defaults for a cutout taken from a photograph.
 *
 * Deliberately small. A wide erosion hides a bad mask by eating the subject,
 * which trades a visible halo for an invisible amputation — and the second is
 * worse, because nobody can see what is missing.
 */
export const PHOTO_MATTE: MatteOptions = { erode: 1, feather: 1.2, defringe: 3 };

/** Alpha below this counts as outside the object; above it, as inside. */
const OUTSIDE = 0.02;
const INSIDE = 0.97;

/** Alpha a neighbour needs before its colour is trusted as the object's own. */
const CLEAN = 0.88;

/** Reads a greyscale mask into a 0..1 coverage field. */
export function readCoverage(
  mask: CanvasImageSource,
  width: number,
  height: number
): Float32Array | null {
  const canvas = window.document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(mask, 0, 0, width, height);

  let pixels: Uint8ClampedArray;
  try {
    pixels = ctx.getImageData(0, 0, width, height).data;
  } catch {
    return null;
  }

  const coverage = new Float32Array(width * height);
  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    coverage[p] = pixels[i + 1] / 255;
  }
  return coverage;
}

/**
 * Shrinks the covered region by `radius` pixels.
 *
 * A separable minimum filter: exact for a square, and linear in the pixels
 * rather than quadratic in the radius, which matters on a twelve-megapixel
 * layer.
 */
export function erodeCoverage(
  coverage: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  if (radius <= 0) return coverage;
  const r = Math.round(radius);

  const horizontal = new Float32Array(coverage.length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let lowest = 1;
      for (let dx = -r; dx <= r; dx += 1) {
        const sx = x + dx;
        // Past the border the object is treated as absent, so a subject
        // touching the frame is trimmed there like anywhere else.
        const value = sx < 0 || sx >= width ? 0 : coverage[row + sx];
        if (value < lowest) lowest = value;
        if (lowest === 0) break;
      }
      horizontal[row + x] = lowest;
    }
  }

  const out = new Float32Array(coverage.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let lowest = 1;
      for (let dy = -r; dy <= r; dy += 1) {
        const sy = y + dy;
        const value = sy < 0 || sy >= height ? 0 : horizontal[sy * width + x];
        if (value < lowest) lowest = value;
        if (lowest === 0) break;
      }
      out[y * width + x] = lowest;
    }
  }
  return out;
}

/** Fast separable box blur using an O(1) sliding window accumulator. */
export function blurCoverage(
  coverage: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  const r = Math.round(radius);
  if (r <= 0 || width <= 0 || height <= 0) return coverage;
  const span = r * 2 + 1;
  const invSpan = 1 / span;

  const horizontal = new Float32Array(coverage.length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    let acc = 0;
    for (let dx = -r; dx <= r; dx += 1) {
      const sx = Math.min(width - 1, Math.max(0, dx));
      acc += coverage[row + sx];
    }
    for (let x = 0; x < width; x += 1) {
      horizontal[row + x] = acc * invSpan;
      const leftX = Math.min(width - 1, Math.max(0, x - r));
      const rightX = Math.min(width - 1, Math.max(0, x + r + 1));
      acc += coverage[row + rightX] - coverage[row + leftX];
    }
  }

  const out = new Float32Array(coverage.length);
  for (let x = 0; x < width; x += 1) {
    let acc = 0;
    for (let dy = -r; dy <= r; dy += 1) {
      const sy = Math.min(height - 1, Math.max(0, dy));
      acc += horizontal[sy * width + x];
    }
    for (let y = 0; y < height; y += 1) {
      out[y * width + x] = acc * invSpan;
      const topY = Math.min(height - 1, Math.max(0, y - r));
      const bottomY = Math.min(height - 1, Math.max(0, y + r + 1));
      acc += horizontal[bottomY * width + x] - horizontal[topY * width + x];
    }
  }
  return out;
}

/**
 * Replaces the colours in the transition band with colours from inside the object.
 *
 * Employs adaptive multi-radius search and distance-weighted progressive diffusion:
 * solid inner pixels propagate color outwards into partial-alpha regions, eliminating
 * background color fringe even along wide, soft, or complex edges.
 */
export function decontaminate(
  data: Uint8ClampedArray,
  coverage: Float32Array,
  width: number,
  height: number,
  passes: number
): void {
  if (passes <= 0) return;

  for (let pass = 0; pass < passes; pass += 1) {
    // The pass reads a snapshot of the current pixel buffer.
    const snapshot = new Uint8ClampedArray(data);
    // Trust threshold relaxes gradually across passes.
    const trust = Math.max(0.2, CLEAN - pass * 0.18);
    // Search radius expands for outer transition rings if needed.
    const maxRadius = pass > 1 ? 2 : 1;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const p = y * width + x;
        const alpha = coverage[p];
        if (alpha <= OUTSIDE || alpha >= INSIDE) continue;

        let weight = 0;
        let r = 0;
        let g = 0;
        let b = 0;

        // Multi-radius adaptive search
        for (let rad = 1; rad <= maxRadius; rad += 1) {
          for (let dy = -rad; dy <= rad; dy += 1) {
            const sy = y + dy;
            if (sy < 0 || sy >= height) continue;
            for (let dx = -rad; dx <= rad; dx += 1) {
              if (dx === 0 && dy === 0) continue;
              const sx = x + dx;
              if (sx < 0 || sx >= width) continue;

              const q = sy * width + sx;
              if (coverage[q] < trust) continue;

              // Distance-weighted contribution
              const distSq = dx * dx + dy * dy;
              const spatialWeight = 1.0 / distSq;
              const w = coverage[q] * spatialWeight;
              const i = q * 4;
              weight += w;
              r += snapshot[i] * w;
              g += snapshot[i + 1] * w;
              b += snapshot[i + 2] * w;
            }
          }
          if (weight > 0) break; // Found confident color at this radius
        }

        if (weight <= 0) continue;
        const i = p * 4;
        data[i] = Math.round(r / weight);
        data[i + 1] = Math.round(g / weight);
        data[i + 2] = Math.round(b / weight);
      }
    }
  }
}

/**
 * Cuts `image` out along `mask`, cleanly.
 *
 * Returns a canvas the size of the source with everything outside the object
 * transparent; `trimToContent` reduces it to the part that matters.
 */
export function cutOut(
  image: CanvasImageSource,
  mask: CanvasImageSource,
  width: number,
  height: number,
  options: MatteOptions = PHOTO_MATTE
): HTMLCanvasElement | null {
  const raw = readCoverage(mask, width, height);
  if (!raw) return null;

  const coverage = blurCoverage(
    erodeCoverage(raw, width, height, options.erode),
    width,
    height,
    options.feather
  );

  const canvas = window.document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(image, 0, 0, width, height);

  let pixels: ImageData;
  try {
    pixels = ctx.getImageData(0, 0, width, height);
  } catch {
    return null;
  }

  // Colour first, transparency second: decontamination reads the neighbours'
  // colours, and those have to still be the photograph's own.
  decontaminate(pixels.data, coverage, width, height, options.defringe);

  const data = pixels.data;
  for (let p = 0, i = 3; p < coverage.length; p += 1, i += 4) {
    // The mask multiplies whatever transparency the layer already had, so
    // cutting something out of a cutout does not resurrect erased pixels.
    data[i] = data[i] * coverage[p];
  }

  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

/**
 * Cleans up a cutout that already carries its own transparency.
 *
 * The segmenter hands back the object on a transparent background, and that
 * result has the same halo problem as any other mask — it just wears it as
 * alpha instead of as a separate picture. Same treatment, coverage read from
 * the image itself.
 */
export function refineCutout(
  cutout: CanvasImageSource,
  width: number,
  height: number,
  options: MatteOptions = PHOTO_MATTE
): HTMLCanvasElement | null {
  const canvas = window.document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(cutout, 0, 0, width, height);

  let pixels: ImageData;
  try {
    pixels = ctx.getImageData(0, 0, width, height);
  } catch {
    return null;
  }

  const raw = new Float32Array(width * height);
  for (let p = 0, i = 3; p < raw.length; p += 1, i += 4) {
    raw[p] = pixels.data[i] / 255;
  }

  const coverage = blurCoverage(
    erodeCoverage(raw, width, height, options.erode),
    width,
    height,
    options.feather
  );

  decontaminate(pixels.data, coverage, width, height, options.defringe);

  for (let p = 0, i = 3; p < coverage.length; p += 1, i += 4) {
    pixels.data[i] = coverage[p] * 255;
  }

  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

/** Bounds of everything not fully transparent, or null when nothing is left. */
export function contentBounds(
  data: Uint8ClampedArray,
  width: number,
  height: number
): { x: number; y: number; width: number; height: number } | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4 + 3] <= 2) continue;
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
  }

  if (x1 < x0 || y1 < y0) return null;
  return { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
}

/**
 * Crops a cutout down to what it actually contains.
 *
 * A cutout that keeps the full frame carries megabytes of transparency around
 * with it, and lands in the destination scaled as if it were that big.
 */
export function trimToContent(canvas: HTMLCanvasElement): HTMLCanvasElement | null {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  let pixels: ImageData;
  try {
    pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  } catch {
    return canvas;
  }

  const bounds = contentBounds(pixels.data, canvas.width, canvas.height);
  if (!bounds) return null;
  if (bounds.width === canvas.width && bounds.height === canvas.height) return canvas;

  const out = window.document.createElement('canvas');
  out.width = bounds.width;
  out.height = bounds.height;
  const outCtx = out.getContext('2d');
  if (!outCtx) return canvas;

  outCtx.drawImage(
    canvas,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    bounds.width,
    bounds.height
  );
  return out;
}
