// src/ops/mask.ts
//
// Turning "this bit here" into a mask.
//
// Two sources, one result: the outline the user drew, or the shape a segmenter
// found inside it. Shared because three different features now need the same
// answer — selective focus, local corrections and recolouring — and a mask that
// differed slightly between them would show up as a visible seam the moment two
// were used together.

import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from '../store/documentStore';
import { useModelStore } from '../store/modelStore';
import { layerSize } from '../engine/compositing';
import { loadImage } from '../engine/RenderEngine';
import type { Point2D, Selection } from '../types/document';

/**
 * Softness applied to the edge of a hand-drawn mask, as a fraction of the
 * shorter side. A hard edge is what makes a blurred background look cut out.
 */
const DRAWN_FEATHER = 0.012;

/** Softness applied to a segmented edge: it is already accurate, so barely any. */
const SEGMENTED_FEATHER = 0.004;

export function maskCanvas(
  width: number,
  height: number
): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = window.document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D context unavailable');
  return [canvas, ctx];
}

/** Paints a filled polygon in white on black, softened at the edge. */
export function maskFromPath(
  path: Point2D[],
  offset: Point2D,
  width: number,
  height: number
): string {
  const [canvas, ctx] = maskCanvas(width, height);

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const feather = Math.max(1, Math.min(width, height) * DRAWN_FEATHER);
  ctx.filter = `blur(${feather.toFixed(1)}px)`;
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.moveTo(path[0].x - offset.x, path[0].y - offset.y);
  for (const point of path.slice(1)) ctx.lineTo(point.x - offset.x, point.y - offset.y);
  ctx.closePath();
  ctx.fill();
  ctx.filter = 'none';

  return canvas.toDataURL('image/png');
}

/**
 * Turns a cutout into a mask.
 *
 * The segmenter returns the object on transparency, so its alpha channel is
 * already the answer — it only has to be flattened to greyscale and softened
 * a hair, because a mathematically perfect edge reads as a sticker.
 */
export async function maskFromCutout(
  cutoutDataUrl: string,
  width: number,
  height: number
): Promise<string> {
  const image = await loadImage(cutoutDataUrl);

  // White wherever the cutout is opaque: draw it, then flood white through its
  // own alpha. That converts coverage into luminance without reading a pixel.
  const [silhouette, silhouetteCtx] = maskCanvas(width, height);
  silhouetteCtx.drawImage(image, 0, 0, silhouette.width, silhouette.height);
  silhouetteCtx.globalCompositeOperation = 'source-in';
  silhouetteCtx.fillStyle = '#ffffff';
  silhouetteCtx.fillRect(0, 0, silhouette.width, silhouette.height);

  const [canvas, ctx] = maskCanvas(width, height);
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const feather = Math.max(0.6, Math.min(width, height) * SEGMENTED_FEATHER);
  ctx.filter = `blur(${feather.toFixed(1)}px)`;
  ctx.drawImage(silhouette, 0, 0);
  ctx.filter = 'none';

  return canvas.toDataURL('image/png');
}

/**
 * The drawn outline in layer pixels, as the backend wants it.
 *
 * Thinned to a sane number of points: a lasso traced slowly carries hundreds,
 * and the engine rasterises it either way, so the extra precision buys nothing
 * and the payload crosses the bridge for no reason.
 */
export function outlineOf(selection: Selection, offset: Point2D): Array<[number, number]> {
  const step = Math.max(1, Math.ceil(selection.path.length / 160));
  const points = selection.path
    .filter((_, index) => index % step === 0)
    .map((point): [number, number] => [point.x - offset.x, point.y - offset.y]);

  // Three points is the least that encloses anything; below that the backend
  // falls back to the box on its own.
  return points.length >= 3 ? points : [];
}

export interface BuiltMask {
  maskDataUrl: string;
  /** True when a segmenter refined the drawn shape into the real object. */
  refined: boolean;
}

/**
 * Builds a mask from the current selection.
 *
 * With a segmentation model installed the outline is snapped to the real
 * object, because a hand-drawn circle used as a mask reads as a smudge rather
 * than as a subject. Without one the drawn shape is used as it is — every
 * feature that consumes a mask still works, just less precisely, which is the
 * whole point of not making the model mandatory.
 */
export async function buildSelectionMask(): Promise<BuiltMask | null> {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const selection = session?.selection ?? null;
  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;

  if (!doc || !layer?.imageDataUrl || !selection) return null;

  const size = layerSize(layer, doc.width, doc.height);
  const offset = { x: layer.transform.x, y: layer.transform.y };

  // A selection the user corrected by pointing at it wins over anything this
  // function would work out on its own. They looked at the answer and said
  // where it was wrong; asking the segmenter again would throw that away.
  if (session?.refineCutDataUrl) {
    return {
      maskDataUrl: await maskFromCutout(session.refineCutDataUrl, size.width, size.height),
      refined: true,
    };
  }

  const model = useModelStore.getState().readyFor('segmentation');

  if (!model) {
    return {
      maskDataUrl: maskFromPath(selection.path, offset, size.width, size.height),
      refined: false,
    };
  }

  const xs = selection.path.map((p) => p.x - offset.x);
  const ys = selection.path.map((p) => p.y - offset.y);

  const cutout = await invoke<string>('segment_object', {
    modelId: model.id,
    imageDataUrl: layer.imageDataUrl,
    x0: Math.max(0, Math.min(...xs)),
    y0: Math.max(0, Math.min(...ys)),
    x1: Math.min(size.width, Math.max(...xs)),
    y1: Math.min(size.height, Math.max(...ys)),
    // The shape itself, not only the box around it. The box is where the
    // segmenter used to be told to look, and on anything that is not roughly
    // square — a hand, a strap, a leg — the middle of the box is not even
    // inside the drawing.
    outline: outlineOf(selection, offset),
  });

  return {
    maskDataUrl: await maskFromCutout(cutout, size.width, size.height),
    refined: true,
  };
}
