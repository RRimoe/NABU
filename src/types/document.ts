// src/types/document.ts

/** Id of the background layer created with every new document. */
export const BACKGROUND_LAYER_ID = 'layer-root-background';

export type ThemeMode = 'dark' | 'light';

export type LayerType = 'raster' | 'vector' | 'text' | 'ai_generated' | 'adjustment' | 'group';

export type BlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'darken'
  | 'lighten'
  | 'color-dodge'
  | 'color-burn'
  | 'hard-light'
  | 'soft-light'
  | 'difference'
  | 'exclusion';

export interface Transform2D {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  rotation: number; // In radians
  skewX: number;
  skewY: number;
}

export interface LayerMask {
  id: string;
  enabled: boolean;
  inverted: boolean;
  maskDataUrl?: string; // Data URL or buffer id of the mask
}

export interface AILayerMetadata {
  prompt?: string;
  negativePrompt?: string;
  seed?: number;
  modelId?: string;
  denoiseStrength?: number;
  taskType?: 'inpainting' | 'outpainting' | 'background-removal' | 'upscale' | 'segmentation';
  generationTimeMs?: number;
}

export interface LayerNode {
  id: string;
  name: string;
  type: LayerType;
  visible: boolean;
  locked: boolean;
  opacity: number; // 0.0 - 1.0
  blendMode: BlendMode;
  transform: Transform2D;
  mask?: LayerMask;
  childrenIds?: string[]; // Child ids, for layer groups
  aiMetadata?: AILayerMetadata;
  imageDataUrl?: string; // Base64 cache used for raster rendering
  width?: number;
  height?: number;
  /**
   * Layer this one was derived from (for example a cutout relative to its
   * original). The refinement brush needs it to restore removed pixels:
   * without the original one could only erase, never bring anything back.
   */
  sourceLayerId?: string;
  /** Non-destructive adjustments; absent means none have been applied. */
  adjustments?: Adjustments;
  /** Non-destructive tone curves; absent means the identity curve. */
  curves?: Curves;
  /** Where the picture stays sharp; absent means all of it. */
  focus?: FocusEffect;
  /** Corrections confined to one region; absent means none. */
  local?: LocalAdjustment;
  /** Look chosen from the filter panel, layered on top of the sliders. */
  filter?: LayerFilter;
}

export interface NabuDocument {
  id: string;
  title: string;
  width: number;
  height: number;
  dpi: number;
  colorSpace: 'srgb' | 'display-p3' | 'adobe-rgb';
  /**
   * `white` paints an opaque sheet under the layers, `transparent` leaves it
   * empty. A document created from an imported image is transparent: after a
   * cutout the user must see transparency, not white.
   */
  background: 'transparent' | 'white';
  rootLayerIds: string[];
  layers: Record<string, LayerNode>;
  activeLayerId: string | null;
  selectedLayerIds: string[];
  createdAt: string;
  updatedAt: string;
}

/**
 * Non-destructive adjustments applied to a layer.
 *
 * Every value is a deviation from neutral, so an all-zero record means "leave
 * the pixels alone" and resetting a control is simply writing 0. They are held
 * on the layer rather than baked into its pixels: the original data is never
 * touched, and any slider can be moved again at any time.
 *
 * Ranges are -100..100 unless noted, chosen so the numbers read the same way a
 * photographer expects rather than exposing raw multipliers.
 */
export interface Adjustments {
  // Tone
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;

  // Colour
  temperature: number;
  tint: number;
  vibrance: number;
  saturation: number;
  /** Hue rotation in degrees, -180..180. */
  hue: number;

  // Presence
  clarity: number;
  sharpness: number;
  /** Blur radius in pixels, 0..20. */
  blur: number;

  // Effects
  vignette: number;
  grain: number;
  /** Highlights bleeding into their surroundings, the way film glows. */
  bloom: number;
  /** Colour fringing towards the corners, the way a cheap lens does it. */
  aberration: number;
}

export const NEUTRAL_ADJUSTMENTS: Adjustments = {
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  temperature: 0,
  tint: 0,
  vibrance: 0,
  saturation: 0,
  hue: 0,
  clarity: 0,
  sharpness: 0,
  blur: 0,
  vignette: 0,
  grain: 0,
  bloom: 0,
  aberration: 0,
};

/**
 * Range of every slider, in one place.
 *
 * The panel reads it to build its controls and the filter engine reads it to
 * clamp a preset once it is added on top of what the user already set, so a
 * range can never be defined in two places that disagree.
 */
export const ADJUSTMENT_RANGES: Record<keyof Adjustments, [number, number]> = {
  exposure: [-100, 100],
  contrast: [-100, 100],
  highlights: [-100, 100],
  shadows: [-100, 100],
  whites: [-100, 100],
  blacks: [-100, 100],
  temperature: [-100, 100],
  tint: [-100, 100],
  vibrance: [-100, 100],
  saturation: [-100, 100],
  hue: [-180, 180],
  clarity: [-100, 100],
  sharpness: [0, 100],
  blur: [0, 20],
  vignette: [-100, 100],
  grain: [0, 100],
  bloom: [0, 100],
  aberration: [0, 100],
};

/** True when nothing would change, so the processing pass can be skipped. */
export function isNeutral(adjustments?: Adjustments): boolean {
  if (!adjustments) return true;
  return (Object.keys(NEUTRAL_ADJUSTMENTS) as Array<keyof Adjustments>).every(
    (key) => adjustments[key] === 0
  );
}

// --- Tone curves -------------------------------------------------------------

/** Control point of a tone curve. Both axes run 0..1, input against output. */
export interface CurvePoint {
  x: number;
  y: number;
}

/**
 * `rgb` is the master curve applied to all three channels; the others correct
 * one channel each. They compose as master(channel(value)).
 */
export type CurveChannel = 'rgb' | 'r' | 'g' | 'b';

export const CURVE_CHANNELS: CurveChannel[] = ['rgb', 'r', 'g', 'b'];

/** The identity curve: output equals input, so the pixels are left alone. */
export const NEUTRAL_CURVE: CurvePoint[] = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
];

export type Curves = Record<CurveChannel, CurvePoint[]>;

export const NEUTRAL_CURVES: Curves = {
  rgb: NEUTRAL_CURVE,
  r: NEUTRAL_CURVE,
  g: NEUTRAL_CURVE,
  b: NEUTRAL_CURVE,
};

/** True when a single channel is still the diagonal. */
export function isCurveNeutral(points?: CurvePoint[]): boolean {
  if (!points || points.length !== 2) return !points;
  const [first, last] = points;
  return first.x === 0 && first.y === 0 && last.x === 1 && last.y === 1;
}

/** True when no channel bends, so the curve pass can be skipped entirely. */
export function areCurvesNeutral(curves?: Curves): boolean {
  if (!curves) return true;
  return CURVE_CHANNELS.every((channel) => isCurveNeutral(curves[channel]));
}

// --- Selective focus ---------------------------------------------------------

/**
 * Where the picture stays sharp.
 *
 * `mask` follows the shape of a real object — a cutout from the segmenter or the
 * outline the user drew — which is what "blur the background behind him" means.
 * `linear` and `radial` are geometric instead, and cost nothing to compute: a
 * band is the tilt-shift that turns a street into a model railway, an ellipse is
 * the soft-focus every portrait lens has been sold on since 1900.
 */
export type FocusMode = 'off' | 'mask' | 'linear' | 'radial' | 'depth';

export interface FocusEffect {
  mode: FocusMode;
  /** Blur outside the sharp zone, in pixels at the layer's own resolution. */
  amount: number;
  /** Width of the transition, as a fraction of the frame. */
  feather: number;
  /** Swaps the two sides: blur the subject and keep the background sharp. */
  invert: boolean;
  /** Centre of the sharp zone, 0..1 across and down the layer. */
  centreX: number;
  centreY: number;
  /** Half-height of the band, or radius of the ellipse, as a fraction. */
  size: number;
  /** Ellipse width relative to its height. Radial only. */
  aspect: number;
  /** Angle of the band, in degrees. Linear only. */
  angle: number;
  /**
   * The distance that stays sharp, 0 = farthest, 1 = nearest. Depth mode only.
   * This is the focus ring of a real lens: everything nearer or farther than
   * the chosen plane falls off, instead of one side of a boundary.
   */
  plane: number;
  /**
   * Greyscale PNG. In mask mode it is white where the picture stays sharp; in
   * depth mode it is the depth map itself, white near and black far.
   *
   * Stored on the layer rather than recomputed: a segmenter or a depth model
   * costs seconds, and the result has to survive an undo.
   */
  maskDataUrl?: string;
}

export const NEUTRAL_FOCUS: FocusEffect = {
  mode: 'off',
  amount: 24,
  feather: 0.12,
  invert: false,
  centreX: 0.5,
  centreY: 0.5,
  size: 0.18,
  aspect: 1.4,
  angle: 0,
  // Biased towards the near end: a photo you would want blurred behind
  // something almost always has that something in the foreground.
  plane: 0.75,
};

/** True when the focus effect would change nothing. */
export function isFocusNeutral(focus?: FocusEffect): boolean {
  if (!focus) return true;
  if (focus.mode === 'off' || focus.amount <= 0) return true;
  // Both of these read a picture rather than a shape, so without one there is
  // nothing to be out of focus against.
  return (focus.mode === 'mask' || focus.mode === 'depth') && !focus.maskDataUrl;
}

// --- Local adjustments -------------------------------------------------------

/**
 * Corrections that apply to one part of the picture only.
 *
 * This is what separates "make the photo brighter" from "bring the subject
 * forward". The mask is the same kind of greyscale picture the focus effect
 * uses — a cutout from the segmenter, a shape the user drew, a depth map — and
 * the values are deltas on top of whatever the global sliders already say.
 */
export interface LocalAdjustment {
  /** Greyscale PNG: white where the correction applies at full strength. */
  maskDataUrl: string;
  adjustments: Partial<Adjustments>;
  /** Scales the whole set, so one slider can dial the region back. */
  amount: number;
}

/** True when the region would change nothing. */
export function isLocalNeutral(local?: LocalAdjustment): boolean {
  if (!local || !local.maskDataUrl || local.amount <= 0) return true;
  return Object.values(local.adjustments).every((value) => !value);
}

// --- Filters -----------------------------------------------------------------

/**
 * A look picked from the filter panel.
 *
 * Only the name and the strength are stored: the preset itself lives in code,
 * so a photo saved today still resolves to the current definition and a filter
 * can be reworked without rewriting documents.
 */
export interface LayerFilter {
  id: string;
  /** 0..1, so a look can be dialled back instead of only on or off. */
  amount: number;
  /**
   * The look's own values, carried on the layer for looks the user saved
   * rather than ones built into Nabu. A project opened on another computer,
   * where that look was never saved, still renders exactly the same.
   */
  adjustments?: Partial<Adjustments>;
}

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Shape the drag traces.
 *
 * Everything ends up as a polygon regardless: a rectangle is four points and an
 * ellipse is a fine polygon, so the rest of the application — rendering, the
 * bounding box handed to the segmenter — never has to know which was used.
 */
export type SelectionShape = 'free' | 'rect' | 'ellipse';

/** Region drawn by the user, in document coordinates. Implicitly closed. */
export interface Selection {
  path: Point2D[];
  /** How it was drawn; absent means freehand. */
  shape?: SelectionShape;
}

/** Mode of the transparency refinement brush. */
export type RefineMode = 'restore' | 'erase';

/**
 * One click correcting a selection: this belongs to the object, or it does not.
 *
 * The clicks are kept and the mask is derived from them, never the other way
 * round. The segmenter is stateless — it is handed the whole list every time
 * and answers afresh — so the list *is* the selection. Undoing is dropping the
 * last one, and the same clicks always give the same cut.
 */
export interface RefinePoint {
  x: number;
  y: number;
  positive: boolean;
}

export type ToolType =
  | 'select'
  | 'pan'
  | 'transform'
  | 'brush'
  | 'eraser'
  | 'ai-lasso'
  | 'ai-inpaint'
  | 'crop'
  | 'zoom';

export interface ViewportTransform {
  zoom: number; // 1.0 = 100%
  panX: number;
  panY: number;
}

export interface AIModelStatus {
  id: string;
  name: string;
  downloaded: boolean;
  loading: boolean;
  ready: boolean;
  vramRequiredMb: number;
  description: string;
}
