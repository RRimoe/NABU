// src/ai/semantics.ts
//
// What you selected decides what you are offered.
//
// This is the difference between a photo editor and something that understands
// photographs. A traditional editor shows the same eight tools whatever is
// under the cursor, because it has no idea what is under the cursor. Nabu does:
// the detector already named every object before the drag started, so the bar
// can offer "brighten the subject" for a person and "change the colour" for a
// plant, and leave out what makes no sense.
//
// Deliberately a lookup table and not a model. The mapping from "this is a
// person" to "these five things are worth doing to a person" is knowledge about
// photography, not a question that needs answering at run time — and a table
// answers in microseconds, in every language, identically every time.

import type { SceneModel, SceneObject } from '../types/scene';
import type { Point2D, Selection } from '../types/document';

/** The families the interface reasons about, not the detector's 80 classes. */
export type SubjectKind = 'person' | 'animal' | 'plant' | 'vehicle' | 'object';

/** Everything the contextual bar can offer. */
export type ActionId =
  | 'remove'
  | 'fill'
  | 'extract'
  | 'blur-behind'
  | 'enhance-subject'
  | 'recolour'
  | 'crop-here'
  | 'copy';

const KIND_OF: Record<string, SubjectKind> = {
  person: 'person',
  bird: 'animal',
  cat: 'animal',
  dog: 'animal',
  horse: 'animal',
  sheep: 'animal',
  cow: 'animal',
  elephant: 'animal',
  bear: 'animal',
  zebra: 'animal',
  giraffe: 'animal',
  'potted plant': 'plant',
  bicycle: 'vehicle',
  car: 'vehicle',
  motorcycle: 'vehicle',
  airplane: 'vehicle',
  bus: 'vehicle',
  train: 'vehicle',
  truck: 'vehicle',
  boat: 'vehicle',
};

/**
 * Which actions each kind of subject deserves, in the order they are shown.
 *
 * The orders differ on purpose. For a person the first thing anyone wants is to
 * make them look better; for a plant it is the colour; for a stray object in
 * the corner it is to get rid of it.
 */
const ACTIONS: Record<SubjectKind, ActionId[]> = {
  person: ['enhance-subject', 'blur-behind', 'extract', 'remove', 'fill', 'copy'],
  animal: ['enhance-subject', 'blur-behind', 'extract', 'copy', 'remove', 'fill'],
  plant: ['recolour', 'blur-behind', 'extract', 'remove', 'fill', 'copy'],
  vehicle: ['extract', 'blur-behind', 'remove', 'fill', 'copy', 'recolour'],
  object: ['remove', 'fill', 'extract', 'blur-behind', 'copy', 'crop-here'],
};

/** What is offered when nothing has been recognised under the selection. */
const UNKNOWN_ACTIONS: ActionId[] = ['remove', 'fill', 'extract', 'blur-behind', 'copy', 'crop-here'];

export function kindOf(className: string): SubjectKind {
  return KIND_OF[className.toLowerCase()] ?? 'object';
}

/**
 * The readable name of a detected class.
 *
 * The catalogue translates the classes it knows and this falls back to the raw
 * English word for the rest — a missing translation should cost a slightly
 * awkward label, never a hidden action.
 */
export function classLabelOf(raw: string, translate: (key: string) => string): string {
  const key = `class.${raw.replace(/\s+/g, '_')}`;
  const translated = translate(key);
  return translated === key ? raw : translated;
}

/** The actions to offer for a selection over `object`, or over nothing. */
export function actionsFor(object: SceneObject | null): ActionId[] {
  return object ? ACTIONS[kindOf(object.class)] : UNKNOWN_ACTIONS;
}

/** Axis-aligned bounds of a selection, in layer pixels. */
function selectionBox(selection: Selection, offset: Point2D): [number, number, number, number] {
  const xs = selection.path.map((p) => p.x - offset.x);
  const ys = selection.path.map((p) => p.y - offset.y);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function overlap(a: [number, number, number, number], b: [number, number, number, number]): number {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  if (x1 <= x0 || y1 <= y0) return 0;

  const intersection = (x1 - x0) * (y1 - y0);
  const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
  return intersection / (areaA + areaB - intersection);
}

/** Smallest agreement between a drawn shape and a box worth calling a match. */
const MATCH_THRESHOLD = 0.28;

/**
 * Which object the user circled, if any.
 *
 * Matched by overlap rather than by containment: people draw roughly, and a
 * lasso that covers two thirds of the dog is a lasso around the dog. Below the
 * threshold nothing is claimed — offering "brighten the person" over a patch of
 * grass would be worse than offering the generic list.
 */
export function objectUnderSelection(
  selection: Selection | null,
  scene: SceneModel | null,
  offset: Point2D = { x: 0, y: 0 }
): SceneObject | null {
  if (!selection || !scene || scene.objects.length === 0) return null;

  const box = selectionBox(selection, offset);
  let best: SceneObject | null = null;
  let bestScore = MATCH_THRESHOLD;

  for (const object of scene.objects) {
    const score = overlap(box, object.bbox);
    if (score > bestScore) {
      best = object;
      bestScore = score;
    }
  }

  return best;
}
