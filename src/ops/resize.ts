// src/ops/resize.ts
//
// Resampling the document to different pixel dimensions.
//
// This is not "Enhance". Resizing resamples what is already there, so enlarging
// a photo this way makes it softer; the upscale model reconstructs detail
// instead. Two different operations that must stay clearly named apart, or
// people reach for the wrong one.

import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { scaleImage } from '../engine/compositing';
import type { LayerNode } from '../types/document';
import { t } from '../i18n';

/** Smallest and largest side the interface will accept, in pixels. */
export const MIN_SIDE = 16;
export const MAX_SIDE = 20000;

export interface ResizeOutcome {
  messageKey: string;
  params?: Record<string, string | number>;
  handled: boolean;
}

export const clampSide = (value: number): number =>
  Math.max(MIN_SIDE, Math.min(MAX_SIDE, Math.round(value)));

/**
 * Resamples the document and everything in it to `width` x `height`.
 *
 * Every layer is redrawn at the new scale and its offset moved with it, so a
 * cutout stays exactly where it sat relative to the picture.
 */
export async function resizeDocument(width: number, height: number): Promise<ResizeOutcome> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  if (!doc) return { messageKey: 'error.noImageOpen', handled: false };

  const targetWidth = clampSide(width);
  const targetHeight = clampSide(height);
  if (targetWidth === doc.width && targetHeight === doc.height) {
    return { messageKey: 'resize.unchanged', handled: false };
  }

  const scaleX = targetWidth / doc.width;
  const scaleY = targetHeight / doc.height;

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.resize');
  activity.step(job, 'activity.resampling', {
    width: targetWidth,
    height: targetHeight,
  });

  try {
    const updates: Record<string, Partial<LayerNode>> = {};

    for (const id of doc.rootLayerIds) {
      const layer = doc.layers[id];
      if (!layer) continue;

      const layerWidth = Math.max(1, Math.round((layer.width ?? doc.width) * scaleX));
      const layerHeight = Math.max(1, Math.round((layer.height ?? doc.height) * scaleY));

      updates[id] = {
        width: layerWidth,
        height: layerHeight,
        transform: {
          ...layer.transform,
          x: layer.transform.x * scaleX,
          y: layer.transform.y * scaleY,
        },
        // Layers with no pixels keep their new box and nothing else to do.
        ...(layer.imageDataUrl
          ? { imageDataUrl: await scaleImage(layer.imageDataUrl, layerWidth, layerHeight) }
          : {}),
      };
    }

    state.applyResize(targetWidth, targetHeight, updates);
    state.requestFitToScreen();
    activity.end(job, 'done');

    return {
      messageKey: 'resize.done',
      params: { width: targetWidth, height: targetHeight },
      handled: true,
    };
  } catch (err) {
    activity.end(job, 'failed', err);
    alert(t('resize.failed'));
    return { messageKey: 'resize.failed', handled: false };
  }
}

/** Resamples by a percentage of the current size. */
export async function resizeByPercent(percent: number): Promise<ResizeOutcome> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  if (!doc) return { messageKey: 'error.noImageOpen', handled: false };

  const factor = percent / 100;
  if (!Number.isFinite(factor) || factor <= 0) {
    return { messageKey: 'resize.unchanged', handled: false };
  }
  return resizeDocument(doc.width * factor, doc.height * factor);
}

/**
 * Resamples so the longest side becomes `side`, keeping the proportions.
 * This is what "make it 1920 wide" usually means.
 */
export async function resizeLongestSide(side: number): Promise<ResizeOutcome> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  if (!doc) return { messageKey: 'error.noImageOpen', handled: false };

  const factor = side / Math.max(doc.width, doc.height);
  return resizeDocument(doc.width * factor, doc.height * factor);
}
