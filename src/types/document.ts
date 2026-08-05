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
};

/** True when nothing would change, so the processing pass can be skipped. */
export function isNeutral(adjustments?: Adjustments): boolean {
  if (!adjustments) return true;
  return (Object.keys(NEUTRAL_ADJUSTMENTS) as Array<keyof Adjustments>).every(
    (key) => adjustments[key] === 0
  );
}

export interface Point2D {
  x: number;
  y: number;
}

/** Lasso drawn by the user, in document coordinates. Implicitly closed. */
export interface Selection {
  path: Point2D[];
}

/** Mode of the transparency refinement brush. */
export type RefineMode = 'restore' | 'erase';

export type ToolType =
  | 'select'
  | 'pan'
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
