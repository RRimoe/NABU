// src/ops/heal.ts
//
// Content-aware fill and the healing brush: whatever the mask covers is rebuilt
// from the rest of the photo, with no model to download.
//
// The synthesis runs in the desktop backend (crates/nabu-retouch). Only the
// part of the layer the fill can sample from travels there: the same window
// Photoshop uses, a square about four times the size of the hole around it.
// Pixels go as raw bytes, the crop followed by one mask byte per pixel, and
// come back the same way.

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { BACKGROUND_LAYER_ID, LayerNode, NabuDocument, Point2D } from '../types/document';
import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { loadImage } from '../engine/RenderEngine';
import { canPaint, layerToDocument } from './paint';
import { report } from './notify';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';

/** True inside the desktop application, where the fill runs. */
export const retouchAvailable = () =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * The sampling window for a hole with bounding box `hole`: a square of side
 * 4·√(max(w, 50)·max(h, 50)) centred on it and slid inside the layer. Mirrors
 * `sampling_window` in crates/nabu-retouch.
 */
export function samplingWindow(hole: Box, width: number, height: number): Box {
  const w = Math.max(hole.x1 - hole.x0, 50);
  const h = Math.max(hole.y1 - hole.y0, 50);
  const side = Math.round(4 * Math.sqrt(w * h));
  const span = (a: number, b: number, extent: number): [number, number] => {
    const s = Math.max(side, b - a);
    if (s >= extent) return [0, extent];
    const start = Math.min(Math.max(Math.floor((a + b - s) / 2), 0), extent - s);
    return [start, start + s];
  };
  const [x0, x1] = span(hole.x0, hole.x1, width);
  const [y0, y1] = span(hole.y0, hole.y1, height);
  return { x0, y0, x1, y1 };
}

/** Where `box` (document space) lands on the layer, rounded out and clamped. */
function toLayerBox(box: Box, toLayer: DOMMatrix, width: number, height: number): Box | null {
  const corners = [
    toLayer.transformPoint(new DOMPoint(box.x0, box.y0)),
    toLayer.transformPoint(new DOMPoint(box.x1, box.y0)),
    toLayer.transformPoint(new DOMPoint(box.x0, box.y1)),
    toLayer.transformPoint(new DOMPoint(box.x1, box.y1)),
  ];
  const out = {
    x0: Math.max(0, Math.floor(Math.min(...corners.map((p) => p.x)))),
    y0: Math.max(0, Math.floor(Math.min(...corners.map((p) => p.y)))),
    x1: Math.min(width, Math.ceil(Math.max(...corners.map((p) => p.x)))),
    y1: Math.min(height, Math.ceil(Math.max(...corners.map((p) => p.y)))),
  };
  return out.x1 > out.x0 && out.y1 > out.y0 ? out : null;
}

/** A mask shape in document space: draws itself white on a prepared context. */
export type MaskShape = {
  bounds: Box;
  draw: (ctx: CanvasRenderingContext2D) => void;
};

/** A brush stroke of diameter `size`, as a mask. */
export function strokeShape(stroke: Point2D[], size: number, clip?: Point2D[] | null): MaskShape {
  const r = size / 2 + 1;
  return {
    bounds: {
      x0: Math.min(...stroke.map((p) => p.x)) - r,
      y0: Math.min(...stroke.map((p) => p.y)) - r,
      x1: Math.max(...stroke.map((p) => p.x)) + r,
      y1: Math.max(...stroke.map((p) => p.y)) + r,
    },
    draw: (ctx) => {
      if (clip && clip.length >= 3) {
        tracePolygon(ctx, clip);
        ctx.clip();
      }
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = size;
      ctx.strokeStyle = '#fff';
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (const point of stroke.slice(1)) ctx.lineTo(point.x, point.y);
      if (stroke.length === 1) ctx.lineTo(stroke[0].x + 0.01, stroke[0].y);
      ctx.stroke();
    },
  };
}

/** A closed outline (a selection), as a mask. */
export function pathShape(path: Point2D[]): MaskShape {
  return {
    bounds: {
      x0: Math.min(...path.map((p) => p.x)) - 1,
      y0: Math.min(...path.map((p) => p.y)) - 1,
      x1: Math.max(...path.map((p) => p.x)) + 1,
      y1: Math.max(...path.map((p) => p.y)) + 1,
    },
    draw: (ctx) => {
      tracePolygon(ctx, path);
      ctx.fillStyle = '#fff';
      ctx.fill();
    },
  };
}

function tracePolygon(ctx: CanvasRenderingContext2D, path: Point2D[]) {
  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  for (const point of path.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
}

/**
 * Rebuilds what `shape` covers on `layer` and returns the layer's new pixels as
 * a PNG data URL, or null when the shape misses the layer.
 */
export async function healLayer(
  layer: LayerNode,
  doc: NabuDocument,
  shape: MaskShape,
  onProgress: (percent: number) => void = () => undefined
): Promise<string | null> {
  if (!retouchAvailable()) throw new Error(t('heal.desktopOnly'));

  const width = Math.round(layer.width ?? doc.width);
  const height = Math.round(layer.height ?? doc.height);
  const toLayer = layerToDocument(layer, doc).inverse();

  const hole = toLayerBox(shape.bounds, toLayer, width, height);
  if (!hole) return null;
  const win = samplingWindow(hole, width, height);
  const cw = win.x1 - win.x0;
  const ch = win.y1 - win.y0;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  if (layer.imageDataUrl) {
    ctx.drawImage(await loadImage(layer.imageDataUrl), 0, 0, width, height);
  } else if (layer.id === BACKGROUND_LAYER_ID && doc.background === 'white') {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
  } else {
    return null;
  }

  const sheet = document.createElement('canvas');
  sheet.width = cw;
  sheet.height = ch;
  const sheetCtx = sheet.getContext('2d', { willReadFrequently: true });
  if (!sheetCtx) return null;
  sheetCtx.setTransform(new DOMMatrix().translate(-win.x0, -win.y0).multiply(toLayer));
  shape.draw(sheetCtx);

  const pixels = ctx.getImageData(win.x0, win.y0, cw, ch).data;
  const mask = sheetCtx.getImageData(0, 0, cw, ch).data;
  const body = new Uint8Array(cw * ch * 5);
  body.set(pixels);
  let any = false;
  for (let i = 0, o = cw * ch * 4; i < cw * ch; i++) {
    const on = mask[i * 4 + 3] > 127;
    body[o + i] = on ? 255 : 0;
    any ||= on;
  }
  if (!any) return null;

  const unlisten = await listen<{ percent: number }>('retouch-progress', (event) =>
    onProgress(event.payload.percent)
  );
  let healed: ArrayBuffer;
  try {
    healed = await invoke<ArrayBuffer>('heal_region', body, {
      headers: { 'x-nabu-width': String(cw), 'x-nabu-height': String(ch) },
    });
  } finally {
    unlisten();
  }

  ctx.putImageData(new ImageData(new Uint8ClampedArray(healed), cw, ch), win.x0, win.y0);
  return canvas.toDataURL('image/png');
}

export const cancelHeal = () => invoke('cancel_heal').catch(() => undefined);

/**
 * Runs a fill on the active layer as one history step, with progress in the
 * busy indicator and a line in the activity panel. Returns whether it changed
 * anything.
 */
export async function healActive(shapeOf: () => MaskShape | null, titleKey: string): Promise<boolean> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  const layerId = doc?.activeLayerId;
  const layer = layerId ? doc.layers[layerId] : undefined;
  if (!doc || !layerId) {
    report(t('error.noImageOpen'));
    return false;
  }
  if (!canPaint(layer)) {
    report(t('brush.cannotPaint'));
    return false;
  }
  const shape = shapeOf();
  if (!shape) return false;

  const activity = useActivityStore.getState();
  const job = activity.begin(titleKey);
  state.setAIBusy(true, '', 2);
  try {
    activity.step(job, 'activity.rebuildingBackground');
    const result = await healLayer(layer, doc, shape, (percent) =>
      useDocumentStore.getState().setAIBusy(true, '', Math.max(2, percent))
    );
    if (!result) {
      activity.end(job, 'done');
      return false;
    }
    // The document may have changed hands while the fill ran.
    const now = useDocumentStore.getState();
    if (now.sessions[now.activeId]?.document.layers[layerId] === undefined) {
      activity.end(job, 'done');
      return false;
    }
    now.updateLayer(layerId, { imageDataUrl: result });
    now.dismissWelcome();
    activity.end(job, 'done');
    return true;
  } catch (err) {
    activity.end(job, 'failed', err);
    report(describeError(err));
    return false;
  } finally {
    setTimeout(() => useDocumentStore.getState().setAIBusy(false), 200);
  }
}

/** One stroke of the healing brush, in document coordinates. */
export function healStroke(stroke: Point2D[], size: number, clip?: Point2D[] | null): Promise<boolean> {
  if (stroke.length === 0) return Promise.resolve(false);
  return healActive(() => strokeShape(stroke, size, clip), 'activity.heal');
}
