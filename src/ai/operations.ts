// src/ai/operations.ts
//
// Execution of AI operations on the active document.
//
// The crop -> inference -> recomposition sequence lives here in one place, so
// the contextual bubble and the command line run exactly the same steps.

import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from '../store/documentStore';
import { useModelStore } from '../store/modelStore';
import { useActivityStore } from '../store/activityStore';
import { cropRegion, composeResult, layerSize } from '../engine/compositing';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';
import type { Intent } from './intents';
import type { ModelTaskKind } from '../types/models';

function activeContext() {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document ?? null;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  return { state, doc, layer, selection: session?.selection ?? null };
}

function findModel(task: ModelTaskKind) {
  return useModelStore.getState().models.find((m) => m.task === task && m.state === 'ready');
}

/** Removes the background, restricted to the selection when there is one. */
export async function removeBackground(): Promise<void> {
  const { state, doc, layer } = activeContext();
  const activity = useActivityStore.getState();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }

  const model = findModel('background-removal');
  if (!model) {
    alert(t('error.noBackgroundModel'));
    return;
  }

  const job = activity.begin('activity.removeBackground');
  const size = layerSize(layer, doc.width, doc.height);
  // Deliberately the whole layer: cropping to the selection starves the model
  // of context and it starts answering "everything is the subject".
  const box = { x: 0, y: 0, width: size.width, height: size.height };

  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.preparing');
    const input = await cropRegion(layer.imageDataUrl, size, box);

    activity.step(job, 'activity.segmenting', { model: model.name });
    const result = await invoke<string>('process_ai_task', {
      taskType: 'background-removal',
      modelId: model.id,
      imageDataUrl: input,
    });

    activity.step(job, 'activity.composing');
    const composed = await composeResult(result, size, box, null, { x: 0, y: 0 });

    const newId = state.addImageLayer(
      `${layer.name} +`,
      composed,
      size.width,
      size.height
    );

    // The result inherits the original's position; the original is hidden
    // rather than deleted, so transparency shows immediately while the source
    // pixels stay available to the refinement brush.
    state.updateLayer(newId, {
      transform: { ...layer.transform },
      sourceLayerId: doc.activeLayerId,
    });
    state.updateLayer(doc.activeLayerId, { visible: false });

    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
  } catch (err) {
    activity.end(job, 'failed', err);
    alert(describeError(err));
  } finally {
    setTimeout(() => state.setAIBusy(false), 300);
  }
}

/** Increases the resolution of the active layer. */
export async function upscale(): Promise<void> {
  const { state, doc, layer } = activeContext();
  const activity = useActivityStore.getState();

  if (!doc || !layer?.imageDataUrl) {
    alert(t('error.noImageOpen'));
    return;
  }

  const model = findModel('upscale');
  if (!model) {
    alert(t('error.noUpscaleModel'));
    return;
  }

  const job = activity.begin('activity.upscale');
  const scale = model.scaleFactor ?? 2;
  const size = layerSize(layer, doc.width, doc.height);

  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.superResolution', { scale, model: model.name });
    const result = await invoke<string>('process_ai_task', {
      taskType: 'upscale',
      modelId: model.id,
      imageDataUrl: layer.imageDataUrl,
    });

    activity.step(job, 'activity.addingLayer');
    state.addImageLayer(
      `${layer.name} ${scale}x`,
      result,
      size.width * scale,
      size.height * scale
    );

    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
  } catch (err) {
    activity.end(job, 'failed', err);
    alert(describeError(err));
  } finally {
    setTimeout(() => state.setAIBusy(false), 300);
  }
}

/**
 * Cuts out the object inside the selection.
 *
 * This is a different question from background removal. RMBG looks for "the
 * subject of the photo"; given a crop that is almost entirely one object it
 * answers "all of it", which is why a rough lasso used to come back as the
 * shape of the lasso. SAM instead is asked "which object is here", and returns
 * the outline of the car rather than the outline of the gesture.
 */
export async function extractObject(): Promise<void> {
  const { state, doc, layer, selection } = activeContext();
  const activity = useActivityStore.getState();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }
  if (!selection) {
    alert(t('error.noSelection'));
    return;
  }

  const model = findModel('segmentation');
  if (!model) {
    alert(t('error.noSegmentationModel'));
    return;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const offset = { x: layer.transform.x, y: layer.transform.y };

  // The lasso is in document coordinates; SAM wants the box in layer pixels.
  const xs = selection.path.map((p) => p.x - offset.x);
  const ys = selection.path.map((p) => p.y - offset.y);
  const x0 = Math.max(0, Math.min(...xs));
  const y0 = Math.max(0, Math.min(...ys));
  const x1 = Math.min(size.width, Math.max(...xs));
  const y1 = Math.min(size.height, Math.max(...ys));

  const job = activity.begin('activity.extractObject');
  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.segmenting', { model: model.name });

    // The whole layer is sent: SAM needs the context around the object to know
    // where it ends, which is exactly what cropping used to throw away.
    const result = await invoke<string>('segment_object', {
      modelId: model.id,
      imageDataUrl: layer.imageDataUrl,
      x0,
      y0,
      x1,
      y1,
    });

    activity.step(job, 'activity.addingLayer');
    const newId = state.addImageLayer(
      `${layer.name} ✂`,
      result,
      size.width,
      size.height
    );
    state.updateLayer(newId, {
      transform: { ...layer.transform },
      sourceLayerId: doc.activeLayerId,
    });
    state.updateLayer(doc.activeLayerId, { visible: false });

    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
  } catch (err) {
    activity.end(job, 'failed', err);
    alert(describeError(err));
  } finally {
    setTimeout(() => state.setAIBusy(false), 300);
  }
}

export async function runIntent(intent: Intent): Promise<void> {
  switch (intent.kind) {
    case 'remove-background':
      return removeBackground();
    case 'upscale':
      return upscale();
    default:
      return;
  }
}
