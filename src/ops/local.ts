// src/ops/local.ts
//
// Corrections that apply to one thing rather than to the photograph.
//
// "Brighten the subject" is a different sentence from "brighten the photo", and
// until now Nabu could only say the second one. The mask makes the difference,
// and it is the same mask everything else uses.

import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { buildSelectionMask } from './mask';
import type { Adjustments } from '../types/document';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';

/**
 * Bringing a subject forward.
 *
 * A little exposure so it separates from what is behind it, local contrast so
 * the detail reads, a touch of warmth because a lifted subject that stays cold
 * looks pasted on, and a small saturation push. Deliberately modest: the point
 * is that the eye lands on the subject first, not that the subject glows.
 */
const ENHANCE: Partial<Adjustments> = {
  exposure: 12,
  clarity: 16,
  contrast: 8,
  temperature: 5,
  vibrance: 12,
};

/** Applies a set of deltas to whatever the selection covers. */
async function applyToSelection(
  adjustments: Partial<Adjustments>,
  titleKey: string
): Promise<boolean> {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layerId = doc?.activeLayerId ?? null;

  if (!doc || !layerId) {
    alert(t('error.noImageOpen'));
    return false;
  }
  if (!session?.selection) {
    alert(t('error.noSelection'));
    return false;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin(titleKey);
  state.setAIBusy(true, '', 5);

  try {
    activity.step(job, 'activity.buildingMask');
    const built = await buildSelectionMask();
    if (!built) {
      activity.end(job, 'failed');
      return false;
    }

    activity.step(job, 'activity.applyingLocal');
    state.setLocalAdjustment(layerId, {
      maskDataUrl: built.maskDataUrl,
      adjustments,
      amount: 1,
    });
    state.setSelection(null);
    state.dismissWelcome();
    activity.end(job, 'done');
    return true;
  } catch (err) {
    activity.end(job, 'failed', err);
    alert(describeError(err));
    return false;
  } finally {
    setTimeout(() => state.setAIBusy(false), 200);
  }
}

/** Lifts the selected subject out of its background. */
export function enhanceSubject(): Promise<boolean> {
  return applyToSelection(ENHANCE, 'activity.enhanceSubject');
}

/**
 * Rotates the hue of whatever is selected.
 *
 * Restricted to hue on purpose: shifting saturation or brightness with it is
 * how a recoloured object stops matching the light in the rest of the frame.
 */
export function recolourSelection(degrees: number): Promise<boolean> {
  return applyToSelection({ hue: degrees }, 'activity.recolour');
}
