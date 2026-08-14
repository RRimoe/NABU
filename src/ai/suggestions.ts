// src/ai/suggestions.ts
//
// What Nabu proposes, and why.
//
// This replaces the typed request. Instead of asking the user to find the words
// for what is wrong with a photo, the photo is measured and the fixes it needs
// are offered by name, each one already previewable. Nothing here is a guess
// dressed up as intelligence: every proposal below comes from a number that can
// be pointed at — the picture *is* two thirds of a stop dark, the colours *are*
// close to grey, there *is* a boat on the right.
//
// No language model takes part. That is the point: a suggestion arrives in
// milliseconds and works identically in every language, which is exactly what
// the typed request could never promise.

import type { Adjustments } from '../types/document';
import type { ImageStats } from '../engine/imageStats';
import type { SceneModel, SceneObject } from '../types/scene';
import type { ModelTaskKind } from '../types/models';

/** What happens when a card is used. */
export type SuggestionAction =
  | { type: 'remove-background' }
  | { type: 'upscale' }
  | { type: 'extract-object'; bbox: [number, number, number, number] }
  | { type: 'remove-object'; bbox: [number, number, number, number] }
  | { type: 'crop-to'; bbox: [number, number, number, number] };

export interface Suggestion {
  id: string;
  titleKey: string;
  detailKey: string;
  /** Values interpolated into both strings, e.g. the name of an object. */
  params?: Record<string, string | number>;
  /**
   * Slider deltas. Present means the card can render a real preview and apply
   * instantly; absent means it runs a model instead.
   */
  adjustments?: Partial<Adjustments>;
  action?: SuggestionAction;
  /** Model the action needs. The card says so when it is not installed. */
  requires?: ModelTaskKind;
  /** Ranking. Roughly "how wrong is this", so the worst problem leads. */
  score: number;
}

/** How far from the middle is worth mentioning. */
const DARK_EDGE = 0.38;
const BRIGHT_EDGE = 0.66;
const FLAT_CONTRAST = 0.52;
const GREY_SATURATION = 0.17;
const SOFT_DETAIL = 0.3;
const COLOUR_CAST = 0.3;

/** Below this many pixels on the long side, enlarging is genuinely useful. */
const SMALL_IMAGE = 1400;

/**
 * Objects small enough that removing them is plausibly what someone wants.
 * A subject filling half the frame is the photo, not a blemish.
 */
const REMOVABLE_AREA = 0.14;

/** Extra room left around a subject when cropping to it. */
const CROP_MARGIN = 0.12;

interface Context {
  stats: ImageStats | null;
  scene: SceneModel | null;
  /** What the sliders already say, so a fix is not offered twice. */
  current: Adjustments;
  /** Size of the layer being worked on. */
  width: number;
  height: number;
}

/**
 * Ranks the proposals for one photo.
 *
 * A suggestion is dropped when the user has already moved the slider it would
 * move: having corrected the exposure by hand and still being told the photo is
 * dark is how an assistant loses credibility.
 */
export function buildSuggestions(context: Context): Suggestion[] {
  const { stats, scene, current } = context;
  const out: Suggestion[] = [];

  const untouched = (key: keyof Adjustments) => Math.abs(current[key]) < 8;

  if (stats) {
    // --- Exposure --------------------------------------------------------
    if (stats.meanLuma < DARK_EDGE && untouched('exposure')) {
      const missing = DARK_EDGE - stats.meanLuma;
      out.push({
        id: 'light',
        titleKey: 'suggest.light',
        detailKey: 'suggest.light.detail',
        adjustments: {
          exposure: Math.round(Math.min(34, missing * 130)),
          shadows: Math.round(Math.min(30, missing * 110)),
          blacks: 6,
        },
        score: 0.7 + missing,
      });
    } else if (stats.meanLuma > BRIGHT_EDGE && untouched('exposure')) {
      const excess = stats.meanLuma - BRIGHT_EDGE;
      out.push({
        id: 'tame-light',
        titleKey: 'suggest.tameLight',
        detailKey: 'suggest.tameLight.detail',
        adjustments: {
          exposure: -Math.round(Math.min(28, excess * 120)),
          highlights: -Math.round(Math.min(34, excess * 130)),
        },
        score: 0.7 + excess,
      });
    }

    // --- Clipping: a real loss of information, so it outranks taste ------
    if (stats.clippedHighlights > 0.03 && untouched('highlights')) {
      out.push({
        id: 'highlights',
        titleKey: 'suggest.highlights',
        detailKey: 'suggest.highlights.detail',
        adjustments: { highlights: -32, whites: -14 },
        score: 0.9 + stats.clippedHighlights * 3,
      });
    }
    if (stats.clippedShadows > 0.05 && untouched('shadows')) {
      out.push({
        id: 'shadows',
        titleKey: 'suggest.shadows',
        detailKey: 'suggest.shadows.detail',
        adjustments: { shadows: 30, blacks: 12 },
        score: 0.85 + stats.clippedShadows * 2,
      });
    }

    // --- Contrast and colour ---------------------------------------------
    if (stats.contrast < FLAT_CONTRAST && untouched('contrast')) {
      out.push({
        id: 'contrast',
        titleKey: 'suggest.contrast',
        detailKey: 'suggest.contrast.detail',
        adjustments: {
          contrast: Math.round(Math.min(30, (FLAT_CONTRAST - stats.contrast) * 90)),
          blacks: -6,
        },
        score: 0.55 + (FLAT_CONTRAST - stats.contrast),
      });
    }

    if (stats.saturation < GREY_SATURATION && untouched('vibrance')) {
      out.push({
        id: 'colour',
        titleKey: 'suggest.colour',
        detailKey: 'suggest.colour.detail',
        adjustments: {
          vibrance: Math.round(Math.min(38, (GREY_SATURATION - stats.saturation) * 220)),
          saturation: 6,
        },
        score: 0.6 + (GREY_SATURATION - stats.saturation) * 2,
      });
    }

    if (Math.abs(stats.warmth) > COLOUR_CAST && untouched('temperature')) {
      out.push({
        id: 'balance',
        titleKey: 'suggest.balance',
        detailKey: stats.warmth > 0 ? 'suggest.balance.warm' : 'suggest.balance.cold',
        // Corrects towards neutral by roughly half the measured cast: taking it
        // all out usually kills the light the photograph was taken in.
        adjustments: { temperature: -Math.round(stats.warmth * 26) },
        score: 0.5 + Math.abs(stats.warmth),
      });
    }

    if (stats.detail < SOFT_DETAIL && untouched('clarity')) {
      out.push({
        id: 'detail',
        titleKey: 'suggest.detail',
        detailKey: 'suggest.detail.detail',
        adjustments: {
          clarity: Math.round(Math.min(24, (SOFT_DETAIL - stats.detail) * 90)),
          sharpness: 32,
        },
        score: 0.45 + (SOFT_DETAIL - stats.detail),
      });
    }
  }

  // --- What is in the picture --------------------------------------------
  const subject = dominantObject(scene);
  if (subject) {
    out.push({
      id: `cut-${subject.id}`,
      titleKey: 'suggest.cutOut',
      detailKey: 'suggest.cutOut.detail',
      params: { subject: objectName(subject) },
      action: { type: 'extract-object', bbox: subject.bbox },
      requires: 'segmentation',
      score: 0.62,
    });

    out.push({
      id: `frame-${subject.id}`,
      titleKey: 'suggest.frameSubject',
      detailKey: 'suggest.frameSubject.detail',
      params: { subject: objectName(subject) },
      action: { type: 'crop-to', bbox: paddedBox(subject.bbox, context) },
      score: 0.4,
    });
  }

  for (const object of scene?.objects ?? []) {
    // Small things in a picture are the ones people want gone: a sign, a bin,
    // a stranger at the edge of the frame.
    if (object.areaFraction > REMOVABLE_AREA) continue;
    out.push({
      id: `clear-${object.id}`,
      titleKey: 'suggest.clear',
      detailKey: 'suggest.clear.detail',
      params: { subject: objectName(object) },
      action: { type: 'remove-object', bbox: object.bbox },
      requires: 'inpainting',
      score: 0.5 - object.areaFraction,
    });
  }

  // --- Whole-photo operations, always on offer ---------------------------
  out.push({
    id: 'cutout-background',
    titleKey: 'suggest.removeBackground',
    detailKey: 'suggest.removeBackground.detail',
    action: { type: 'remove-background' },
    requires: 'background-removal',
    score: 0.35,
  });

  const longest = Math.max(context.width, context.height);
  out.push({
    id: 'enlarge',
    titleKey: 'suggest.enlarge',
    detailKey: 'suggest.enlarge.detail',
    action: { type: 'upscale' },
    requires: 'upscale',
    score: longest < SMALL_IMAGE ? 0.66 : 0.3,
  });

  return out.sort((a, b) => b.score - a.score);
}

/** The thing the photo is about, if the scene is confident enough about one. */
function dominantObject(scene: SceneModel | null): SceneObject | null {
  if (!scene) return null;
  const candidates = scene.objects
    .filter((object) => object.prominence === 'dominant' && object.confidence > 0.45)
    .sort((a, b) => b.areaFraction - a.areaFraction);
  return candidates[0] ?? null;
}

/** Raw detector class; `classLabel` turns it into something readable. */
function objectName(object: SceneObject): string {
  return object.class;
}

/**
 * The name shown to the user.
 *
 * Detector classes are English identifiers. The catalogue translates the ones
 * it knows and this falls back to the raw class for everything else: a missing
 * word should cost a slightly awkward label, never a hidden suggestion.
 */
export function classLabel(raw: string, translate: (key: string) => string): string {
  const key = `class.${raw.replace(/\s+/g, '_')}`;
  const translated = translate(key);
  return translated === key ? raw : translated;
}

/** Grows a box to leave breathing room, clamped to the image. */
function paddedBox(
  bbox: [number, number, number, number],
  context: Context
): [number, number, number, number] {
  const [x0, y0, x1, y1] = bbox;
  const padX = (x1 - x0) * CROP_MARGIN;
  const padY = (y1 - y0) * CROP_MARGIN;
  return [
    Math.max(0, x0 - padX),
    Math.max(0, y0 - padY),
    Math.min(context.width, x1 + padX),
    Math.min(context.height, y1 + padY),
  ];
}
