// src/engine/compositing.ts
// Bitmap operations used to combine AI results with the existing content of a
// layer.

import type { LayerNode, Point2D, Selection } from '../types/document';
import { loadImage } from './RenderEngine';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Bounding box of the lasso, clamped to the usable area, plus a margin. */
export function selectionBounds(
  selection: Selection,
  maxWidth: number,
  maxHeight: number,
  padding = 8
): BoundingBox {
  const xs = selection.path.map((p) => p.x);
  const ys = selection.path.map((p) => p.y);

  const x = Math.max(0, Math.floor(Math.min(...xs) - padding));
  const y = Math.max(0, Math.floor(Math.min(...ys) - padding));
  const right = Math.min(maxWidth, Math.ceil(Math.max(...xs) + padding));
  const bottom = Math.min(maxHeight, Math.ceil(Math.max(...ys) + padding));

  return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
}

function newCanvas(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = window.document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Contesto 2D non disponibile');
  return [canvas, ctx];
}

function tracePath(ctx: CanvasRenderingContext2D, path: Point2D[], offset: Point2D): void {
  ctx.beginPath();
  ctx.moveTo(path[0].x - offset.x, path[0].y - offset.y);
  for (const point of path.slice(1)) {
    ctx.lineTo(point.x - offset.x, point.y - offset.y);
  }
  ctx.closePath();
}

/** Effective size of a raster layer. */
export function layerSize(layer: LayerNode, docWidth: number, docHeight: number) {
  return { width: layer.width ?? docWidth, height: layer.height ?? docHeight };
}

/**
 * Extracts the part of the layer that will be fed to the model.
 *
 * Cropping around the selection does more than limit the effect: the subject
 * then fills a larger share of the model's 1024x1024 input, so the resulting
 * edge is more precise than it would be on the whole image.
 */
export async function cropRegion(
  imageDataUrl: string,
  size: { width: number; height: number },
  box: BoundingBox
): Promise<string> {
  const image = await loadImage(imageDataUrl);
  const [canvas, ctx] = newCanvas(box.width, box.height);
  ctx.drawImage(
    image,
    box.x,
    box.y,
    box.width,
    box.height,
    0,
    0,
    box.width,
    box.height
  );
  // `size` reconciles any difference between file pixels and layer pixels.
  void size;
  return canvas.toDataURL('image/png');
}

/**
 * Recomposes the model output into a full-size layer.
 *
 * When a selection exists, everything outside it is cleared: the user pointed
 * at a precise area and the result must not spill beyond it.
 */
export async function composeResult(
  resultDataUrl: string,
  size: { width: number; height: number },
  box: BoundingBox,
  selection: Selection | null,
  layerOffset: Point2D
): Promise<string> {
  const image = await loadImage(resultDataUrl);
  const [canvas, ctx] = newCanvas(size.width, size.height);

  if (selection) {
    // Clip to the lasso before drawing, in layer coordinates.
    tracePath(ctx, selection.path, layerOffset);
    ctx.clip();
  }

  ctx.drawImage(image, box.x, box.y, box.width, box.height);
  return canvas.toDataURL('image/png');
}

/** Resizes an image to exact dimensions. */
export async function scaleImage(
  imageDataUrl: string,
  width: number,
  height: number
): Promise<string> {
  const image = await loadImage(imageDataUrl);
  const [canvas, ctx] = newCanvas(width, height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(image, 0, 0, width, height);
  return canvas.toDataURL('image/png');
}
