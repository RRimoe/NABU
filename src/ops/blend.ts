// src/ops/blend.ts
//
// "Unisci": the step between pasting something in and it belonging there.

import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { harmonizeLayer, type HarmonizeSettings } from '../engine/harmonize';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';
import { report } from './notify';

/** True when this layer has something underneath it to be matched against. */
export function canBlend(): boolean {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  const layerId = doc?.activeLayerId;
  if (!doc || !layerId) return false;

  const index = doc.rootLayerIds.indexOf(layerId);
  return (
    index >= 0 &&
    index < doc.rootLayerIds.length - 1 &&
    Boolean(doc.layers[layerId]?.imageDataUrl)
  );
}

/**
 * Relights the active layer to match the photo under it.
 *
 * One history step covers both the relit pixels and the shadow, so a single
 * undo puts everything back the way it was — a blend that took two undos to
 * reverse would be worse than no blend.
 */
export async function blendActiveLayer(settings: HarmonizeSettings): Promise<boolean> {
  const state = useDocumentStore.getState();
  const doc = state.sessions[state.activeId]?.document;
  const layerId = doc?.activeLayerId;
  const layer = layerId ? doc?.layers[layerId] : null;

  if (!doc || !layerId || !layer) {
    report(t('error.noImageOpen'));
    return false;
  }
  if (!canBlend()) {
    report(t('blend.needsBackdrop'));
    return false;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.blend');
  state.setAIBusy(true, '', 10);

  try {
    activity.step(job, 'activity.matchingLight');
    const result = await harmonizeLayer(doc, layerId, settings);

    if (!result) {
      activity.end(job, 'failed');
      report(t('blend.failed'));
      return false;
    }

    if (result.shadowDataUrl) {
      activity.step(job, 'activity.contactShadow');
      state.addLayerBelow(
        layerId,
        `${layer.name} ${t('blend.shadowSuffix')}`,
        result.shadowDataUrl,
        doc.width,
        doc.height
      );
    }

    activity.step(job, 'activity.applyingBlend');
    state.updateLayer(layerId, { imageDataUrl: result.imageDataUrl });
    // The layer keeps focus so the user can nudge it afterwards, but the
    // active layer is reset to it in case adding the shadow moved it.
    state.setActiveLayer(layerId);

    activity.end(job, 'done');
    return true;
  } catch (err) {
    activity.end(job, 'failed', err);
    report(describeError(err));
    return false;
  } finally {
    setTimeout(() => state.setAIBusy(false), 200);
  }
}

// No live preview on purpose. Harmonising is a full pass over the layer's
// pixels, so a preview that updated as the sliders moved would either be a
// different calculation from the real one — and therefore a lie — or would
// stall on every drag. Applying is one click and undoing it is one keystroke,
// which is a better deal than either.
