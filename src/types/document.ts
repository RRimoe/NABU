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
