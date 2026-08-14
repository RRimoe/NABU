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
import { cropDocument } from '../ops/crop';
import { outlineOf } from '../ops/mask';
import { refineCutout } from '../engine/matte';
import { loadImage } from '../engine/RenderEngine';
import type { Intent } from './intents';
import type { Suggestion } from './suggestions';
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
  const { doc, layer, selection } = activeContext();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }
  if (!selection) {
    alert(t('error.noSelection'));
    return;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const offset = { x: layer.transform.x, y: layer.transform.y };

  // The lasso is in document coordinates; SAM wants the box in layer pixels.
  const xs = selection.path.map((p) => p.x - offset.x);
  const ys = selection.path.map((p) => p.y - offset.y);

  await extractObjectByBbox(
    [
      Math.max(0, Math.min(...xs)),
      Math.max(0, Math.min(...ys)),
      Math.min(size.width, Math.max(...xs)),
      Math.min(size.height, Math.max(...ys)),
    ],
    outlineOf(selection, offset)
  );
}

/**
 * Cuts out whatever sits inside `bbox`, in layer pixels.
 *
 * The box can come from a lasso the user drew or from the scene model, which is
 * why it is the shared entry point: a suggestion that says "cut out the boat"
 * and a hand-drawn circle around the boat run exactly the same code.
 */
export async function extractObjectByBbox(
  bbox: [number, number, number, number],
  /** The drawn shape, when there was one. It constrains the segmenter. */
  outline: Array<[number, number]> = []
): Promise<void> {
  const { state, doc, layer } = activeContext();
  const activity = useActivityStore.getState();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }

  const model = findModel('segmentation');
  if (!model) {
    alert(t('error.noSegmentationModel'));
    return;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const [x0, y0, x1, y1] = bbox;

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
      outline,
    });

    // The segmenter's edge is accurate but hard, and its boundary pixels still
    // carry the colour of whatever was behind the object. Refining here means
    // every cutout in Nabu gets the same treatment, whether it came from this
    // button or from a copy.
    activity.step(job, 'activity.refiningEdge');
    const refined = await refineEdge(result, size.width, size.height);

    activity.step(job, 'activity.addingLayer');
    const newId = state.addImageLayer(
      `${layer.name} ✂`,
      refined,
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

/**
 * Erases the object inside `bbox` (layer pixels) and rebuilds the background.
 *
 * The box comes from the scene model: the router resolved "the man on the
 * right" to an object id before this is called. The backend chains whatever
 * segmenter and inpainter are installed — this function neither knows nor
 * cares which models those are.
 */
export async function removeObjectByBbox(
  bbox: [number, number, number, number],
  outline: Array<[number, number]> = []
): Promise<void> {
  const { state, doc, layer } = activeContext();
  const activity = useActivityStore.getState();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const [x0, y0, x1, y1] = bbox;

  const job = activity.begin('activity.removeObject');
  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.rebuildingBackground');
    const result = await invoke<string>('remove_object', {
      imageDataUrl: layer.imageDataUrl,
      x0,
      y0,
      x1,
      y1,
      outline,
    });

    activity.step(job, 'activity.addingLayer');
    const newId = state.addImageLayer(
      `${layer.name} −`,
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

/**
 * Erases whatever is inside the current selection and rebuilds the background.
 *
 * The escape hatch for everything the detector cannot name: a sign, a stain,
 * a cable. The user points, the same segmenter → inpainter chain runs — no
 * scene entry required.
 */
export async function removeSelectedRegion(): Promise<void> {
  const { doc, layer, selection } = activeContext();

  if (!doc || !layer?.imageDataUrl || !doc.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }
  if (!selection) {
    alert(t('error.noSelection'));
    return;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const offset = { x: layer.transform.x, y: layer.transform.y };

  const xs = selection.path.map((p) => p.x - offset.x);
  const ys = selection.path.map((p) => p.y - offset.y);
  const x0 = Math.max(0, Math.min(...xs));
  const y0 = Math.max(0, Math.min(...ys));
  const x1 = Math.min(size.width, Math.max(...xs));
  const y1 = Math.min(size.height, Math.max(...ys));

  await removeObjectByBbox([x0, y0, x1, y1], outlineOf(selection, offset));
}

/**
 * Runs the matte over a cutout, falling back to the original if anything in
 * the pipeline declines. A slightly rough cutout beats no cutout.
 */
async function refineEdge(
  cutoutDataUrl: string,
  width: number,
  height: number
): Promise<string> {
  try {
    const image = await loadImage(cutoutDataUrl);
    const refined = refineCutout(image, width, height);
    return refined ? refined.toDataURL('image/png') : cutoutDataUrl;
  } catch {
    return cutoutDataUrl;
  }
}

export async function runIntent(intent: Intent): Promise<void> {
  if (intent.kind === 'tool') {
    switch (intent.call.name) {
      case 'remove_background':
        return removeBackground();
      case 'upscale':
        return upscale();
      default:
        return;
    }
  }
}

/**
 * Carries out a proposal from the suggestions panel.
 *
 * Slider suggestions are applied here too, so a card is one call whatever it
 * turns out to do, and the panel never has to know which kind it is holding.
 */
export async function runSuggestion(suggestion: Suggestion): Promise<void> {
  const { state, doc } = activeContext();
  if (!doc?.activeLayerId) {
    alert(t('error.noImageOpen'));
    return;
  }

  if (suggestion.adjustments) {
    state.applyAdjustmentDelta(doc.activeLayerId, suggestion.adjustments);
    state.dismissWelcome();
    return;
  }

  const action = suggestion.action;
  if (!action) return;

  switch (action.type) {
    case 'remove-background':
      return removeBackground();
    case 'upscale':
      return upscale();
    case 'extract-object':
      return extractObjectByBbox(action.bbox);
    case 'remove-object':
      return removeObjectByBbox(action.bbox);
    case 'crop-to': {
      const [x0, y0, x1, y1] = action.bbox;
      await cropDocument({ x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
      return;
    }
  }
}

