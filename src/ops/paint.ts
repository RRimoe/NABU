// src/ops/paint.ts
//
// Brush and eraser: a stroke drawn on the canvas, baked into the active layer's
// pixels as one history step.
//
// The stroke arrives in document coordinates, exactly as the pointer produced
// it. Rather than converting every point, the layer's canvas is given the
// inverse of the transform the renderer uses to place it, so the stroke (and
// the selection that confines it) can be drawn in document space and land on
// the right pixels at any position, scale, rotation or mirroring.

import { BACKGROUND_LAYER_ID, LayerNode, NabuDocument, Point2D } from '../types/document';
import { layerBoxSize, loadImage } from '../engine/RenderEngine';

export type PaintMode = 'paint' | 'erase';

export interface PaintOptions {
  mode: PaintMode;
  /** Diameter in document pixels. */
  size: number;
  /** 0..1, applied to the stroke as a whole. */
  opacity: number;
  color: string;
  /** When present, paint lands only inside it, the way a selection is expected to work. */
  clip?: Point2D[] | null;
}

/**
 * Where a layer pixel ends up in the document: the same chain of operations
 * paintLayers applies, written as one matrix.
 */
export function layerToDocument(layer: LayerNode, doc: NabuDocument): DOMMatrix {
  const scaleX = layer.transform.scaleX || 1;
  const scaleY = layer.transform.scaleY || 1;
  const naturalWidth = layer.width ?? doc.width;
  const naturalHeight = layer.height ?? doc.height;
  const box = layerBoxSize(layer, doc);

  let m = new DOMMatrix();
  if (layer.transform.rotation) {
    m = m
      .translate(layer.transform.x + box.width / 2, layer.transform.y + box.height / 2)
      .rotate((layer.transform.rotation * 180) / Math.PI)
      .translate(-box.width / 2, -box.height / 2);
  } else {
    m = m.translate(layer.transform.x, layer.transform.y);
  }
  m = m.scale(scaleX, scaleY);
  if (scaleX < 0 || scaleY < 0) {
    m = m.translate(scaleX < 0 ? -naturalWidth : 0, scaleY < 0 ? -naturalHeight : 0);
  }
  return m;
}

/** Layers the brush can work on: pixels, visible, not locked. */
export function canPaint(layer: LayerNode | undefined): layer is LayerNode {
  return Boolean(
    layer &&
      (layer.type === 'raster' || layer.type === 'ai_generated') &&
      layer.visible &&
      !layer.locked
  );
}

function traceStroke(ctx: CanvasRenderingContext2D, stroke: Point2D[], size: number) {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = size;
  ctx.beginPath();
  ctx.moveTo(stroke[0].x, stroke[0].y);
  for (const point of stroke.slice(1)) ctx.lineTo(point.x, point.y);
  // A single tap would produce no stroke at all: nudge it a hair.
  if (stroke.length === 1) ctx.lineTo(stroke[0].x + 0.01, stroke[0].y);
}

/**
 * Applies one stroke to the layer and returns its new pixels as a PNG data URL,
 * or null when there is nothing to paint on.
 *
 * The stroke is drawn opaque on its own sheet first and only then laid down at
 * the chosen opacity: drawn directly, every place the path crosses itself would
 * come out darker, which reads as a defect rather than as a brush.
 */
export async function paintStroke(
  layer: LayerNode,
  doc: NabuDocument,
  stroke: Point2D[],
  options: PaintOptions
): Promise<string | null> {
  if (stroke.length === 0) return null;

  const width = Math.round(layer.width ?? doc.width);
  const height = Math.round(layer.height ?? doc.height);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  if (layer.imageDataUrl) {
    ctx.drawImage(await loadImage(layer.imageDataUrl), 0, 0, width, height);
  } else if (layer.id === BACKGROUND_LAYER_ID && doc.background === 'white') {
    // The white sheet is drawn by the renderer only while the layer is empty;
    // once it has pixels, the sheet has to be part of them or it would vanish.
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
  } else if (options.mode === 'erase') {
    return null;
  }

  const sheet = document.createElement('canvas');
  sheet.width = width;
  sheet.height = height;
  const sheetCtx = sheet.getContext('2d');
  if (!sheetCtx) return null;

  sheetCtx.setTransform(layerToDocument(layer, doc).inverse());
  if (options.clip && options.clip.length >= 3) {
    sheetCtx.beginPath();
    sheetCtx.moveTo(options.clip[0].x, options.clip[0].y);
    for (const point of options.clip.slice(1)) sheetCtx.lineTo(point.x, point.y);
    sheetCtx.closePath();
    sheetCtx.clip();
  }
  traceStroke(sheetCtx, stroke, options.size);
  sheetCtx.strokeStyle = options.mode === 'erase' ? '#000' : options.color;
  sheetCtx.stroke();

  ctx.save();
  ctx.globalAlpha = Math.min(1, Math.max(0, options.opacity));
  ctx.globalCompositeOperation = options.mode === 'erase' ? 'destination-out' : 'source-over';
  ctx.drawImage(sheet, 0, 0);
  ctx.restore();

  return canvas.toDataURL('image/png');
}
