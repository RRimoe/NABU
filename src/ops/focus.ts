// src/ops/focus.ts
//
// Building the mask that decides what stays sharp.
//
// Two sources, same result: the outline the user drew, or the shape the
// segmenter found. The drawn one needs no model at all, which matters — the
// effect has to work on a fresh install, and a rough circle around a face is
// already enough for a believable depth-of-field.

import { useDocumentStore } from '../store/documentStore';
import { useDepthStore } from '../store/depthStore';
import { useActivityStore } from '../store/activityStore';
import { layerSize, selectionBounds } from '../engine/compositing';
import { loadImage } from '../engine/RenderEngine';
import { buildSelectionMask, maskCanvas } from './mask';
import { NEUTRAL_FOCUS, type FocusEffect } from '../types/document';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';
import { report } from './notify';

/**
 * Turns on depth-of-field driven by a real depth map.
 *
 * This is the honest version of "blur the background": a mask has one boundary
 * and blurs everything past it equally, while a depth map blurs by distance —
 * the far hills more than the near hedge, and a branch in front of the subject
 * too. It is also the only mode that needs no selection at all.
 *
 * The focus plane is picked from what is actually in the picture: the selection
 * if there is one, otherwise the centre, which is where subjects live.
 */
export async function focusOnDepth(): Promise<void> {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;

  if (!doc || !layer?.imageDataUrl || !layerId) {
    report(t('error.noImageOpen'));
    return;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.focus');
  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.measuringDepth');
    const mapDataUrl = await useDepthStore.getState().analyzeActive();
    if (!mapDataUrl) {
      activity.end(job, 'failed');
      report(t('error.noDepthModel'));
      return;
    }

    activity.step(job, 'activity.buildingMask');
    const size = layerSize(layer, doc.width, doc.height);
    const offset = { x: layer.transform.x, y: layer.transform.y };
    const region = session?.selection
      ? selectionBounds(session.selection, size.width + offset.x, size.height + offset.y, 0)
      : {
          // No selection: read the middle of the frame, which is where the
          // thing being photographed almost always is.
          x: size.width * 0.35,
          y: size.height * 0.35,
          width: size.width * 0.3,
          height: size.height * 0.3,
        };

    const plane = await averageDepth(mapDataUrl, {
      x: region.x - (session?.selection ? offset.x : 0),
      y: region.y - (session?.selection ? offset.y : 0),
      width: region.width,
      height: region.height,
    });

    state.setFocus(
      layerId,
      {
        ...NEUTRAL_FOCUS,
        ...layer.focus,
        mode: 'depth',
        plane: plane ?? NEUTRAL_FOCUS.plane,
        maskDataUrl: mapDataUrl,
      },
      { history: true }
    );
    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
  } catch (err) {
    activity.end(job, 'failed', err);
    report(describeError(err));
  } finally {
    setTimeout(() => state.setAIBusy(false), 200);
  }
}

/** Mean brightness of the depth map inside a box, i.e. how far that region is. */
async function averageDepth(
  mapDataUrl: string,
  box: { x: number; y: number; width: number; height: number }
): Promise<number | null> {
  const image = await loadImage(mapDataUrl);
  const [canvas, ctx] = maskCanvas(image.naturalWidth, image.naturalHeight);
  ctx.drawImage(image, 0, 0);

  const x = Math.max(0, Math.min(canvas.width - 1, Math.round(box.x)));
  const y = Math.max(0, Math.min(canvas.height - 1, Math.round(box.y)));
  const width = Math.max(1, Math.min(canvas.width - x, Math.round(box.width)));
  const height = Math.max(1, Math.min(canvas.height - y, Math.round(box.height)));

  try {
    const { data } = ctx.getImageData(x, y, width, height);
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) sum += data[i];
    return sum / (data.length / 4) / 255;
  } catch {
    return null;
  }
}

/**
 * Puts the current selection out of focus, or everything but it.
 *
 * When a segmentation model is installed the outline is refined to the real
 * object first, because a hand-drawn circle blurred at its own edge looks like
 * a smudge rather than like depth. Without one the drawn shape is used as is,
 * which is still the whole effect, just less precise.
 */
export async function focusOnSelection(options: { invert?: boolean } = {}): Promise<void> {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const selection = session?.selection ?? null;
  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;

  if (!doc || !layer?.imageDataUrl || !layerId) {
    report(t('error.noImageOpen'));
    return;
  }
  if (!selection) {
    report(t('error.noSelection'));
    return;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.focus');
  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.buildingMask');
    const built = await buildSelectionMask();
    if (!built) {
      activity.end(job, 'failed');
      return;
    }

    const focus: FocusEffect = {
      ...NEUTRAL_FOCUS,
      ...layer.focus,
      mode: 'mask',
      invert: options.invert ?? false,
      maskDataUrl: built.maskDataUrl,
    };

    state.setFocus(layerId, focus);
    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
  } catch (err) {
    activity.end(job, 'failed', err);
    report(describeError(err));
  } finally {
    setTimeout(() => state.setAIBusy(false), 200);
  }
}
