// src/ops/crop.ts
//
// Cropping: the single most used operation in any photo editor, and the one
// people reach for before anything else.
//
// It is destructive by nature — the pixels outside the frame are gone — so it
// goes through the history like every other change, and the frame is always
// confirmed explicitly rather than applied as the rectangle is dragged.

import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { loadImage } from '../engine/RenderEngine';
import type { LayerNode } from '../types/document';
import { t } from '../i18n';
import { report } from './notify';

/** A frame in document coordinates. */
export interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Smallest frame the interface will accept, in pixels. */
export const MIN_CROP = 16;

/** The ratios worth a button. `null` is a free frame. */
export const CROP_RATIOS: Array<{ id: string; ratio: number | null }> = [
  { id: 'free', ratio: null },
  { id: 'original', ratio: 0 },
  { id: 'square', ratio: 1 },
  { id: 'portrait', ratio: 4 / 5 },
  { id: 'classic', ratio: 3 / 4 },
  { id: 'wide', ratio: 16 / 9 },
];

/** Clamps a frame inside the document and enforces the minimum size. */
export function clampCrop(rect: CropRect, docWidth: number, docHeight: number): CropRect {
  const width = Math.min(docWidth, Math.max(MIN_CROP, rect.width));
  const height = Math.min(docHeight, Math.max(MIN_CROP, rect.height));
  return {
    width,
    height,
    x: Math.min(docWidth - width, Math.max(0, rect.x)),
    y: Math.min(docHeight - height, Math.max(0, rect.y)),
  };
}

/** The largest frame of a given ratio that fits, centred on the document. */
export function fitRatio(ratio: number, docWidth: number, docHeight: number): CropRect {
  const wide = docWidth / docHeight > ratio;
  const width = wide ? docHeight * ratio : docWidth;
  const height = wide ? docHeight : docWidth / ratio;
  return {
    x: (docWidth - width) / 2,
    y: (docHeight - height) / 2,
    width,
    height,
  };
}

/**
 * Cuts the document down to `rect`.
 *
 * Every layer is redrawn into the new frame rather than merely offset: a layer
 * that falls entirely outside it keeps its (now empty) box, so the stack stays
 * the shape the layer panel shows.
 */
export async function cropDocument(rect: CropRect): Promise<boolean> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  if (!doc) {
    report(t('error.noImageOpen'));
    return false;
  }

  const frame = clampCrop(rect, doc.width, doc.height);
  const width = Math.max(1, Math.round(frame.width));
  const height = Math.max(1, Math.round(frame.height));
  const offsetX = Math.round(frame.x);
  const offsetY = Math.round(frame.y);

  if (width === doc.width && height === doc.height && offsetX === 0 && offsetY === 0) {
    return false;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.crop');
  activity.step(job, 'activity.cropping', { width, height });

  try {
    const updates: Record<string, Partial<LayerNode>> = {};

    for (const id of doc.rootLayerIds) {
      const layer = doc.layers[id];
      if (!layer) continue;

      if (!layer.imageDataUrl) {
        // Nothing to redraw: the empty layer simply adopts the new size.
        updates[id] = {
          width,
          height,
          transform: { ...layer.transform, x: 0, y: 0 },
        };
        continue;
      }

      const canvas = window.document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;

      const image = await loadImage(layer.imageDataUrl);
      const layerWidth = layer.width ?? doc.width;
      const layerHeight = layer.height ?? doc.height;
      const scaleX = layer.transform.scaleX || 1;
      const scaleY = layer.transform.scaleY || 1;

      // The layer is painted exactly as the renderer would, then the frame is
      // read off it: that is the only way a rotated or scaled layer crops to
      // what the user actually saw.
      ctx.save();
      ctx.translate(-offsetX, -offsetY);
      const boxWidth = Math.abs(layerWidth * scaleX);
      const boxHeight = Math.abs(layerHeight * scaleY);
      if (layer.transform.rotation) {
        ctx.translate(layer.transform.x + boxWidth / 2, layer.transform.y + boxHeight / 2);
        ctx.rotate(layer.transform.rotation);
        ctx.translate(-boxWidth / 2, -boxHeight / 2);
      } else {
        ctx.translate(layer.transform.x, layer.transform.y);
      }
      ctx.scale(scaleX, scaleY);
      if (scaleX < 0 || scaleY < 0) {
        ctx.translate(scaleX < 0 ? -layerWidth : 0, scaleY < 0 ? -layerHeight : 0);
      }
      ctx.drawImage(image, 0, 0, layerWidth, layerHeight);
      ctx.restore();

      updates[id] = {
        width,
        height,
        imageDataUrl: canvas.toDataURL('image/png'),
        // The pixels now sit in the frame, so the placement starts over.
        transform: { ...layer.transform, x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0 },
      };
    }

    state.applyResize(width, height, updates);
    state.setSelection(null);
    state.requestFitToScreen();
    activity.end(job, 'done');
    return true;
  } catch (err) {
    activity.end(job, 'failed', err);
    report(t('crop.failed'));
    return false;
  }
}
