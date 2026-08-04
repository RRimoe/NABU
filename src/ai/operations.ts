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
import { selectionBounds, cropRegion, composeResult, layerSize } from '../engine/compositing';
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
  const { state, doc, layer, selection } = activeContext();
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
  const offset = { x: layer.transform.x, y: layer.transform.y };

  // The lasso is in document coordinates; the layer needs it in its own.
  const local = selection
    ? { path: selection.path.map((p) => ({ x: p.x - offset.x, y: p.y - offset.y })) }
    : null;

  const box = local
    ? selectionBounds(local, size.width, size.height)
    : { x: 0, y: 0, width: size.width, height: size.height };

  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, selection ? 'activity.cropSelection' : 'activity.preparing');
    const input = await cropRegion(layer.imageDataUrl, size, box);

    activity.step(job, 'activity.segmenting', { model: model.name });
    const result = await invoke<string>('process_ai_task', {
      taskType: 'background-removal',
      modelId: model.id,
      imageDataUrl: input,
    });

    activity.step(job, 'activity.composing');
    const composed = await composeResult(result, size, box, local, { x: 0, y: 0 });

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
