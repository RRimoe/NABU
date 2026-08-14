// src/engine/thumbnail.ts
//
// Small square previews, shared by every panel that shows "what this would look
// like". Centre-cropped rather than squashed: a panorama reduced to a letterbox
// tells you nothing about a look, and a grid of different shapes is unreadable.

/**
 * Draws a centre-cropped square copy of an image.
 * Returns null only when a 2D context cannot be had at all.
 */
export function squareThumbnail(
  image: HTMLImageElement,
  size: number
): HTMLCanvasElement | null {
  const side = Math.min(image.naturalWidth, image.naturalHeight);
  if (side <= 0) return null;

  const canvas = window.document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(
    image,
    (image.naturalWidth - side) / 2,
    (image.naturalHeight - side) / 2,
    side,
    side,
    0,
    0,
    size,
    size
  );
  return canvas;
}
