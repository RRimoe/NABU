// src/engine/filters.ts
//
// Ready-made looks, in the sense a phone camera means it: one tap, one result,
// a strength slider if it went too far.
//
// A filter is not a separate rendering path — it is a set of slider positions
// added on top of whatever the user already set. That means a look costs no
// extra GPU work, survives being combined with manual corrections, and can be
// taken apart by anyone curious about what it actually does.

import {
  type Adjustments,
  type LayerFilter,
  type LayerNode,
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
} from '../types/document';

export type FilterGroup = 'signature' | 'film' | 'mono' | 'mood' | 'optics';

export interface FilterPreset {
  id: string;
  group: FilterGroup;
  /** Deviations from whatever the sliders currently say. */
  adjustments: Partial<Adjustments>;
}

export const FILTER_GROUPS: FilterGroup[] = ['signature', 'film', 'optics', 'mono', 'mood'];

/**
 * The catalogue.
 *
 * Kept deliberately short per group: twenty good looks beat a hundred that
 * differ by two points of saturation, and the grid has to stay scannable.
 */
export const FILTERS: FilterPreset[] = [
  // --- Signature: everyday corrections, the ones reached for first ---------
  {
    id: 'vivid',
    group: 'signature',
    adjustments: { contrast: 14, vibrance: 38, saturation: 6, clarity: 12 },
  },
  {
    id: 'soft',
    group: 'signature',
    adjustments: { contrast: -10, blacks: 14, clarity: -14, vibrance: 10 },
  },
  {
    id: 'crisp',
    group: 'signature',
    adjustments: { clarity: 26, sharpness: 30, contrast: 10, blacks: -6 },
  },
  {
    id: 'warm',
    group: 'signature',
    adjustments: { temperature: 26, vibrance: 12, highlights: -6 },
  },
  {
    id: 'cool',
    group: 'signature',
    adjustments: { temperature: -26, tint: -6, vibrance: 10 },
  },
  {
    id: 'golden',
    group: 'signature',
    adjustments: { temperature: 34, exposure: 6, shadows: 14, saturation: 10, vignette: 12 },
  },

  // --- Film: emulsions people can name ------------------------------------
  {
    id: 'kodachrome',
    group: 'film',
    adjustments: { contrast: 20, saturation: 18, temperature: 10, blacks: -10, grain: 18 },
  },
  {
    id: 'portra',
    group: 'film',
    adjustments: {
      temperature: 14,
      tint: 6,
      vibrance: 20,
      saturation: -6,
      highlights: -12,
      blacks: 10,
      grain: 12,
    },
  },
  {
    id: 'faded',
    group: 'film',
    adjustments: { blacks: 26, contrast: -14, saturation: -12, grain: 14 },
  },
  {
    id: 'polaroid',
    group: 'film',
    adjustments: {
      temperature: 18,
      tint: 10,
      blacks: 20,
      contrast: -8,
      saturation: -8,
      vignette: 14,
      grain: 20,
    },
  },
  {
    id: 'cinema',
    group: 'film',
    adjustments: {
      contrast: 16,
      temperature: -12,
      shadows: 10,
      highlights: -14,
      saturation: -8,
      vignette: 18,
    },
  },
  {
    id: 'retro',
    group: 'film',
    adjustments: { temperature: 20, saturation: -18, blacks: 18, grain: 26, vignette: 10 },
  },

  // --- Mono ----------------------------------------------------------------
  { id: 'mono', group: 'mono', adjustments: { saturation: -100, contrast: 12 } },
  {
    id: 'noir',
    group: 'mono',
    adjustments: {
      saturation: -100,
      contrast: 34,
      blacks: -18,
      clarity: 16,
      vignette: 22,
      grain: 12,
    },
  },
  {
    id: 'silver',
    group: 'mono',
    adjustments: {
      saturation: -100,
      contrast: 6,
      highlights: -10,
      shadows: 12,
      clarity: 8,
    },
  },
  {
    id: 'sepia',
    group: 'mono',
    adjustments: { saturation: -78, temperature: 38, tint: 8, contrast: 8, grain: 10 },
  },

  // --- Mood ----------------------------------------------------------------
  {
    id: 'dreamy',
    group: 'mood',
    adjustments: { blur: 3, clarity: -20, blacks: 18, exposure: 8, vibrance: 14 },
  },
  {
    id: 'moody',
    group: 'mood',
    adjustments: {
      exposure: -8,
      contrast: 18,
      shadows: -14,
      saturation: -10,
      vignette: 24,
      temperature: -8,
    },
  },
  {
    id: 'sunset',
    group: 'mood',
    adjustments: {
      temperature: 30,
      tint: 12,
      vibrance: 18,
      highlights: -10,
      vignette: 12,
    },
  },
  {
    id: 'punch',
    group: 'mood',
    adjustments: { contrast: 26, clarity: 20, vibrance: 26, sharpness: 22, blacks: -12 },
  },

  // --- Optics: what the glass in front of the sensor did to the light ------
  {
    id: 'glow',
    group: 'optics',
    adjustments: { bloom: 42, highlights: -10, blacks: 12, vibrance: 10 },
  },
  {
    id: 'halation',
    group: 'optics',
    adjustments: { bloom: 34, temperature: 22, blacks: 16, contrast: -6, grain: 14 },
  },
  {
    id: 'lomo',
    group: 'optics',
    adjustments: {
      aberration: 46,
      vignette: 30,
      contrast: 22,
      saturation: 16,
      grain: 18,
    },
  },
  {
    id: 'vhs',
    group: 'optics',
    adjustments: {
      aberration: 62,
      bloom: 24,
      saturation: -12,
      contrast: -10,
      blacks: 20,
      grain: 34,
    },
  },
  {
    id: 'dream',
    group: 'optics',
    adjustments: { bloom: 56, blur: 2, clarity: -18, blacks: 20, vibrance: 16 },
  },
];

const BY_ID = new Map(FILTERS.map((preset) => [preset.id, preset]));

export function filterById(id?: string): FilterPreset | undefined {
  return id ? BY_ID.get(id) : undefined;
}

const clampTo = (key: keyof Adjustments, value: number) => {
  const [min, max] = ADJUSTMENT_RANGES[key];
  return Math.min(max, Math.max(min, value));
};

/**
 * Combines the sliders with the chosen look.
 *
 * The preset is added rather than substituted, so warming a photo by hand and
 * then picking a warm look does what it says instead of throwing the manual
 * correction away. The sum is clamped to each slider's own range.
 */
export function applyFilter(base: Adjustments, filter?: LayerFilter): Adjustments {
  const preset = filterById(filter?.id);
  if (!preset || !filter) return base;

  const amount = Math.min(1, Math.max(0, filter.amount));
  if (amount === 0) return base;

  const result = { ...base };
  for (const [key, value] of Object.entries(preset.adjustments) as Array<
    [keyof Adjustments, number]
  >) {
    result[key] = clampTo(key, result[key] + value * amount);
  }
  return result;
}

/**
 * What the renderer should actually apply to a layer: its sliders plus its
 * look. Returns undefined when there is nothing to do.
 */
export function effectiveAdjustments(layer: LayerNode): Adjustments | undefined {
  if (!layer.adjustments && !filterById(layer.filter?.id)) return undefined;
  const base = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };
  return applyFilter(base, layer.filter);
}

/** Preview of a look on its own, for the thumbnails in the filter panel. */
export function previewAdjustments(
  base: Adjustments,
  presetId: string,
  amount = 1
): Adjustments {
  return applyFilter(base, { id: presetId, amount });
}
