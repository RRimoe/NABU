// src/engine/maskAlpha.ts
//
// Turning a black-and-white mask into an alpha channel.
//
// Masks travel as greyscale pictures because that is what a segmenter and a
// depth model produce, and what a human can look at and understand. But canvas
// compositing works on alpha, not on brightness, so somewhere the one has to
// become the other. Doing it here, once, keeps every caller from reinventing it
// slightly differently.

/**
 * Returns a canvas whose alpha channel is the brightness of `mask`.
 *
 * White in equals fully opaque out, black equals fully transparent, and the
 * colour underneath is irrelevant — only coverage survives.
 */
export function alphaFromLuminance(
  mask: CanvasImageSource,
  width: number,
  height: number
): HTMLCanvasElement | null {
  const canvas = window.document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(mask, 0, 0, canvas.width, canvas.height);

  let pixels: ImageData;
  try {
    pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  } catch {
    return null;
  }

  const data = pixels.data;
  for (let i = 0; i < data.length; i += 4) {
    // The mask is grey, so any channel would do; the green one is the least
    // noisy if a JPEG ever sneaks in as a mask.
    data[i + 3] = data[i + 1];
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
  }

  ctx.putImageData(pixels, 0, 0);
  return canvas;
}

/**
 * Draws `over` on top of `under`, letting it through only where the mask is
 * bright. Both layers must already be the same size.
 */
export function compositeThroughMask(
  under: CanvasImageSource,
  over: CanvasImageSource,
  maskAlpha: CanvasImageSource,
  width: number,
  height: number
): HTMLCanvasElement | null {
  const clipped = window.document.createElement('canvas');
  clipped.width = width;
  clipped.height = height;
  const clippedCtx = clipped.getContext('2d');
  if (!clippedCtx) return null;

  clippedCtx.drawImage(over, 0, 0, width, height);
  // `destination-in` keeps the top layer only where the mask has opacity, which
  // is what the mask's brightness became.
  clippedCtx.globalCompositeOperation = 'destination-in';
  clippedCtx.drawImage(maskAlpha, 0, 0, width, height);

  const out = window.document.createElement('canvas');
  out.width = width;
  out.height = height;
  const ctx = out.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(under, 0, 0, width, height);
  ctx.drawImage(clipped, 0, 0);
  return out;
}
