// src/ai/recipes.ts
//
// Compound edits, named.
//
// A recipe is several operations that only make sense together: "make the
// bench stand out" is a mask, a background blur, local contrast and a lift of
// the subject's exposure, and doing three of the four is worse than doing none.
//
// This is also the closed vocabulary the language model chooses from. It never
// writes an operation — it picks a name from this list, and a name that is not
// in the list is dropped. The same rule that keeps the router honest: the
// lookup guarantees the result, not the prompt.

import { useDocumentStore } from '../store/documentStore';
import { useDepthStore } from '../store/depthStore';
import { useActivityStore } from '../store/activityStore';
import { buildSelectionMask, maskFromPath } from '../ops/mask';
import { layerSize } from '../engine/compositing';
import {
  NEUTRAL_FOCUS,
  type Adjustments,
  type FocusEffect,
  type LocalAdjustment,
} from '../types/document';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';
import { report } from '../ops/notify';

export const RECIPE_IDS = [
  'highlight-subject',
  'separate-subject',
  'lift-shadows',
  'recover-highlights',
  'add-punch',
  'warm-up',
  'cool-down',
  'miniature',
] as const;

export type RecipeId = (typeof RECIPE_IDS)[number];

export function isRecipeId(value: unknown): value is RecipeId {
  return typeof value === 'string' && (RECIPE_IDS as readonly string[]).includes(value);
}

interface Recipe {
  /** Deltas applied to the whole picture. */
  global?: Partial<Adjustments>;
  /** Deltas applied only inside the subject's mask. */
  local?: Partial<Adjustments>;
  /** Selective focus. `subject` needs a mask; `band` needs nothing. */
  focus?: { kind: 'subject' | 'band'; amount: number; invert?: boolean };
  /** True when the recipe cannot run without knowing where the subject is. */
  needsSubject: boolean;
}

const RECIPES: Record<RecipeId, Recipe> = {
  // The flagship: the subject comes forward and the background steps back.
  // Neither half works alone — blurring without lifting leaves a dull subject
  // in a soft frame, lifting without blurring leaves it competing.
  'highlight-subject': {
    focus: { kind: 'subject', amount: 22 },
    local: { exposure: 11, clarity: 15, contrast: 7, vibrance: 10 },
    global: { vignette: 10 },
    needsSubject: true,
  },
  'separate-subject': {
    focus: { kind: 'subject', amount: 34 },
    needsSubject: true,
  },
  'lift-shadows': {
    global: { shadows: 30, blacks: 12 },
    needsSubject: false,
  },
  'recover-highlights': {
    global: { highlights: -32, whites: -14 },
    needsSubject: false,
  },
  'add-punch': {
    global: { contrast: 22, vibrance: 24, clarity: 14, blacks: -8 },
    needsSubject: false,
  },
  'warm-up': {
    global: { temperature: 22, vibrance: 8 },
    needsSubject: false,
  },
  'cool-down': {
    global: { temperature: -22, tint: -5 },
    needsSubject: false,
  },
  // The band plus the colour push: a tilt-shift alone reads as a mistake, and
  // it is the saturation that sells the toy.
  miniature: {
    focus: { kind: 'band', amount: 46 },
    global: { saturation: 22, contrast: 16, clarity: 10 },
    needsSubject: false,
  },
};

export function recipeNeedsSubject(id: RecipeId): boolean {
  return RECIPES[id].needsSubject;
}

/**
 * Carries out a recipe.
 *
 * `target` is a box in layer pixels — from the scene model, when the proposal
 * named an object. With no target the current selection is used, and with
 * neither, a recipe that needs a subject declines rather than guessing at one.
 *
 * Everything lands in a single history step: a compound edit that took four
 * undos to reverse would be worse than four separate buttons.
 */
export async function runRecipe(
  id: RecipeId,
  target?: [number, number, number, number]
): Promise<boolean> {
  const recipe = RECIPES[id];
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;

  if (!doc || !layerId || !layer) {
    report(t('error.noImageOpen'));
    return false;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin(`recipe.${id}`);
  state.setAIBusy(true, '', 5);

  try {
    const needsMask = recipe.needsSubject || recipe.focus?.kind === 'subject' || Boolean(recipe.local);
    let maskDataUrl: string | null = null;

    if (needsMask) {
      activity.step(job, 'activity.buildingMask');

      if (session?.selection) {
        const built = await buildSelectionMask();
        maskDataUrl = built?.maskDataUrl ?? null;
      } else if (target) {
        // A named object gives a box; the box becomes a rounded shape rather
        // than a rectangle, because a rectangular halo of sharpness around a
        // person is the most obvious tell a composite can have.
        const size = layerSize(layer, doc.width, doc.height);
        maskDataUrl = maskFromPath(
          ellipseInBox(target),
          { x: 0, y: 0 },
          size.width,
          size.height
        );
      }

      if (!maskDataUrl && recipe.needsSubject) {
        activity.end(job, 'failed');
        report(t('recipe.needsSubject'));
        return false;
      }
    }

    activity.step(job, 'activity.applyingRecipe');

    let focus: FocusEffect | undefined;
    if (recipe.focus?.kind === 'subject' && maskDataUrl) {
      focus = {
        ...NEUTRAL_FOCUS,
        ...layer.focus,
        mode: 'mask',
        amount: recipe.focus.amount,
        invert: recipe.focus.invert ?? false,
        maskDataUrl,
      };
    } else if (recipe.focus?.kind === 'band') {
      focus = {
        ...NEUTRAL_FOCUS,
        ...layer.focus,
        mode: 'linear',
        amount: recipe.focus.amount,
        size: 0.13,
        feather: 0.1,
        maskDataUrl: undefined,
      };
    }

    const local: LocalAdjustment | undefined =
      recipe.local && maskDataUrl
        ? { maskDataUrl, adjustments: recipe.local, amount: 1 }
        : undefined;

    state.applyRecipe(layerId, { delta: recipe.global, focus, local });
    state.setSelection(null);
    state.dismissWelcome();
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

/** Points of an ellipse inscribed in a box, for a soft-edged subject mask. */
function ellipseInBox([x0, y0, x1, y1]: [number, number, number, number]) {
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  // Slightly larger than the detector's box: a tight ellipse clips shoulders
  // and ears, and a mask that cuts the subject is worse than one that includes
  // a little background.
  const rx = ((x1 - x0) / 2) * 1.08;
  const ry = ((y1 - y0) / 2) * 1.08;

  return Array.from({ length: 48 }, (_, index) => {
    const angle = (index / 48) * Math.PI * 2;
    return { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry };
  });
}

/** Warms the depth map in the background: several recipes read it. */
export function primeDepth(): void {
  void useDepthStore.getState().analyzeActive();
}
