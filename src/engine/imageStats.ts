// src/engine/imageStats.ts
//
// What the photo measures, read straight off the pixels.
//
// This exists so Nabu can propose something sensible *without* any model being
// installed: a detector tells you there is a boat, but only the pixels tell you
// the picture is two stops dark. The measurements are deliberately the ones a
// suggestion can act on — every field below maps to a slider that would fix it.

/** Everything measured, all normalised to 0..1 unless noted. */
export interface ImageStats {
  /** Average brightness. */
  meanLuma: number;
  /** Spread of brightness: low means flat, high means already contrasty. */
  contrast: number;
  /** Average distance from grey; low means a washed-out picture. */
  saturation: number;
  /** Share of pixels crushed to black. */
  clippedShadows: number;
  /** Share of pixels blown to white. */
  clippedHighlights: number;
  /** Warm/cool bias, -1 (cold) to 1 (warm), from the red-blue difference. */
  warmth: number;
  /** Rough focus estimate from neighbouring-pixel differences. */
  detail: number;
}

/** Side of the square the statistics are read from. */
const SAMPLE = 128;

/** Below this the pixel counts as crushed; above 1 - it, as blown. */
const CLIP_EDGE = 0.02;

/**
 * Measures an image.
 *
 * A downscaled copy is enough: none of these numbers change meaningfully
 * between a thumbnail and a 24-megapixel original, and reading the original
 * would stall the interface every time a photo is opened.
 */
export function measureImage(image: CanvasImageSource, width: number, height: number): ImageStats | null {
  const aspect = height > 0 ? width / height : 1;
  const sampleWidth = Math.max(1, aspect >= 1 ? SAMPLE : Math.round(SAMPLE * aspect));
  const sampleHeight = Math.max(1, aspect >= 1 ? Math.round(SAMPLE / aspect) : SAMPLE);

  const canvas = window.document.createElement('canvas');
  canvas.width = sampleWidth;
  canvas.height = sampleHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(image, 0, 0, sampleWidth, sampleHeight);

  let pixels: Uint8ClampedArray;
  try {
    pixels = ctx.getImageData(0, 0, sampleWidth, sampleHeight).data;
  } catch {
    return null;
  }

  let counted = 0;
  let sumLuma = 0;
  let sumLumaSquared = 0;
  let sumSaturation = 0;
  let sumWarmth = 0;
  let clippedShadows = 0;
  let clippedHighlights = 0;

  // Luma is kept in a parallel array so the detail pass can walk neighbours
  // without decoding the same pixels a second time.
  const luma = new Float32Array(sampleWidth * sampleHeight);

  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    const r = pixels[i] / 255;
    const g = pixels[i + 1] / 255;
    const b = pixels[i + 2] / 255;
    const value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    luma[p] = value;

    // Transparent pixels are not part of the picture: on a cutout they would
    // otherwise read as a large black region and drag every number down.
    if (pixels[i + 3] < 8) continue;

    counted += 1;
    sumLuma += value;
    sumLumaSquared += value * value;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    sumSaturation += max - min;
    sumWarmth += r - b;

    if (value <= CLIP_EDGE) clippedShadows += 1;
    else if (value >= 1 - CLIP_EDGE) clippedHighlights += 1;
  }

  if (counted === 0) return null;

  const meanLuma = sumLuma / counted;
  const variance = Math.max(0, sumLumaSquared / counted - meanLuma * meanLuma);

  // Gradient magnitude against the right and lower neighbour: a blurred photo
  // has small differences everywhere, a sharp one has large ones at the edges.
  let sumGradient = 0;
  let gradientCount = 0;
  for (let y = 0; y < sampleHeight - 1; y += 1) {
    for (let x = 0; x < sampleWidth - 1; x += 1) {
      const index = y * sampleWidth + x;
      sumGradient +=
        Math.abs(luma[index] - luma[index + 1]) +
        Math.abs(luma[index] - luma[index + sampleWidth]);
      gradientCount += 1;
    }
  }

  return {
    meanLuma,
    // The standard deviation of a normal photo sits near 0.25, so it is scaled
    // to put "ordinary contrast" around the middle of the 0..1 range.
    contrast: Math.min(1, Math.sqrt(variance) / 0.32),
    saturation: sumSaturation / counted,
    clippedShadows: clippedShadows / counted,
    clippedHighlights: clippedHighlights / counted,
    warmth: Math.max(-1, Math.min(1, (sumWarmth / counted) * 4)),
    detail: gradientCount > 0 ? Math.min(1, (sumGradient / gradientCount) * 12) : 0,
  };
}
