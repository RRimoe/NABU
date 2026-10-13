// src/engine/RenderEngine.ts
import { adjustmentKey, sharedProcessor } from './adjustments';
import { curvesKey } from './curves';
import { effectiveAdjustments } from './filters';
import { shaderLut } from './lut';
import { alphaFromLuminance, compositeThroughMask } from './maskAlpha';
import { applyMasks, maskShapeKey, masksKey, rasteriseMask } from './localMasks';
import {
  NabuDocument,
  LayerNode,
  Adjustments,
  FocusEffect,
  LocalAdjustment,
  LocalMask,
  activeMasks,
  ViewportTransform,
  BlendMode,
  ThemeMode,
  Selection,
  ADJUSTMENT_RANGES,
  areCurvesNeutral,
  isFocusNeutral,
  isLocalNeutral,
  BACKGROUND_LAYER_ID,
  NEUTRAL_ADJUSTMENTS,
} from '../types/document';

/** Colours used by the canvas, aligned with the active interface theme. */
export interface RenderTheme {
  viewportBg: string;
  checkerLight: string;
  checkerDark: string;
  paper: string;
  docBorder: string;
}

/**
 * The canvas paints outside CSS, so these have to be kept in step with the
 * variables in App.css by hand. Same blues, same faint tint in the greys.
 */
/**
 * How long settings must rest before the preview is redrawn at screen size.
 * Short enough to read as "it sharpens when I let go", long enough that a slow
 * drag does not pay for full renders between steps.
 */
const SETTLE_MS = 160;

/** Pixels a live preview works on while a slider moves, at most. */
const PREVIEW_PIXELS = 2_500_000;

/** Power-of-two reduction that brings a photo down to the preview budget. */
function proxyScale(width: number, height: number): number {
  const fraction = Math.sqrt(PREVIEW_PIXELS / Math.max(1, width * height));
  return fraction >= 1 ? 1 : 2 ** Math.floor(Math.log2(fraction));
}

export const RENDER_THEMES: Record<ThemeMode, RenderTheme> = {
  dark: {
    viewportBg: '#0d1117',
    checkerLight: '#272e3a',
    checkerDark: '#1b212a',
    paper: '#ffffff',
    docBorder: 'rgba(180, 214, 232, 0.22)',
  },
  light: {
    viewportBg: '#e4ebf2',
    checkerLight: '#ffffff',
    checkerDark: '#dfe7ef',
    paper: '#ffffff',
    docBorder: 'rgba(15, 42, 66, 0.28)',
  },
};

/**
 * Loads an image from a data URL, returning a Promise.
 * Handlers are attached before `src` so the load event is never missed.
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image could not be loaded'));
    img.src = src;
  });
}

/** Stable key for caching, matching `adjustmentKey` and `curvesKey`. */
export function focusKey(focus?: FocusEffect): string {
  if (isFocusNeutral(focus) || !focus) return '';
  return [
    focus.mode,
    focus.amount,
    focus.feather.toFixed(4),
    focus.invert ? 1 : 0,
    focus.centreX.toFixed(4),
    focus.centreY.toFixed(4),
    focus.size.toFixed(4),
    focus.aspect.toFixed(3),
    focus.angle,
    focus.plane.toFixed(4),
    focus.maskDataUrl?.length ?? 0,
  ].join(',');
}

/** Stable key for caching, matching the other `*Key` helpers. */
export function localKey(local?: LocalAdjustment): string {
  if (isLocalNeutral(local) || !local) return '';
  const values = (Object.keys(local.adjustments) as Array<keyof Adjustments>)
    .sort()
    .map((name) => `${name}:${local.adjustments[name]}`)
    .join(',');
  return `${local.amount.toFixed(3)}|${local.maskDataUrl.length}|${values}`;
}

/** Side of a transform handle, in screen pixels. */
export const TRANSFORM_HANDLE = 10;

/** Distance of the rotation grip above the box, in screen pixels. */
export const ROTATE_HANDLE_OFFSET = 26;

/** The eight scaling handles, as fractions of the box. */
export type TransformHandle =
  | 'nw'
  | 'n'
  | 'ne'
  | 'e'
  | 'se'
  | 's'
  | 'sw'
  | 'w'
  | 'rotate'
  | 'move';

export const SCALE_HANDLES: Array<{ id: TransformHandle; fx: number; fy: number }> = [
  { id: 'nw', fx: 0, fy: 0 },
  { id: 'n', fx: 0.5, fy: 0 },
  { id: 'ne', fx: 1, fy: 0 },
  { id: 'e', fx: 1, fy: 0.5 },
  { id: 'se', fx: 1, fy: 1 },
  { id: 's', fx: 0.5, fy: 1 },
  { id: 'sw', fx: 0, fy: 1 },
  { id: 'w', fx: 0, fy: 0.5 },
];

/**
 * On-screen size of a layer once its scale is taken into account.
 *
 * Always positive: a mirrored layer carries a negative scale, but it still
 * occupies a box of the same size, and a negative width would put every handle
 * on the wrong side of the picture.
 */
export function layerBoxSize(
  layer: LayerNode,
  doc: NabuDocument
): { width: number; height: number } {
  return {
    width: Math.abs((layer.width ?? doc.width) * (layer.transform.scaleX || 1)),
    height: Math.abs((layer.height ?? doc.height) * (layer.transform.scaleY || 1)),
  };
}

/**
 * Turns a document point into a position inside the layer's own unrotated box.
 *
 * Hit-testing a rotated box is the same problem as hit-testing an upright one,
 * once the point has been rotated back — doing it this way keeps a single set
 * of handle positions instead of one per angle.
 */
export function toLayerBox(
  point: { x: number; y: number },
  layer: LayerNode,
  doc: NabuDocument
): { x: number; y: number } {
  const { width, height } = layerBoxSize(layer, doc);
  const centreX = layer.transform.x + width / 2;
  const centreY = layer.transform.y + height / 2;

  const angle = -(layer.transform.rotation || 0);
  const dx = point.x - centreX;
  const dy = point.y - centreY;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  return {
    x: dx * cos - dy * sin + width / 2,
    y: dx * sin + dy * cos + height / 2,
  };
}

/** Maps the document blend mode to a Canvas 2D globalCompositeOperation. */
function mapBlendMode(blendMode: BlendMode): GlobalCompositeOperation {
  switch (blendMode) {
    case 'multiply': return 'multiply';
    case 'screen': return 'screen';
    case 'overlay': return 'overlay';
    case 'darken': return 'darken';
    case 'lighten': return 'lighten';
    case 'color-dodge': return 'color-dodge';
    case 'color-burn': return 'color-burn';
    case 'hard-light': return 'hard-light';
    case 'soft-light': return 'soft-light';
    case 'difference': return 'difference';
    case 'exclusion': return 'exclusion';
    case 'hue': return 'hue';
    case 'saturation': return 'saturation';
    case 'color': return 'color';
    case 'luminosity': return 'luminosity';
    case 'normal':
    default:
      return 'source-over';
  }
}

/**
 * Paints the layer stack in document space (origin 0,0 = top-left corner).
 * Shared between on-screen rendering and export, so the exported image is
 * identical to the one displayed.
 *
 * `resolveImage` returns null for images that are not decoded yet: the layer is
 * simply skipped for this frame.
 */
export function paintLayers(
  ctx: CanvasRenderingContext2D,
  doc: NabuDocument,
  theme: RenderTheme,
  resolveImage: (src: string) => HTMLImageElement | null,
  /** Applies the layer's adjustments; returns null to draw the raw image. */
  resolveAdjusted?: (
    layerId: string,
    image: HTMLImageElement,
    width: number,
    height: number
  ) => CanvasImageSource | null
): void {
  // rootLayerIds runs top to bottom, so it is painted in reverse.
  const bottomToTop = [...doc.rootLayerIds].reverse();

  for (const layerId of bottomToTop) {
    const layer = doc.layers[layerId];
    if (!layer || !layer.visible || layer.opacity <= 0) continue;

    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.globalCompositeOperation = mapBlendMode(layer.blendMode);

    // Rotation pivots on the centre of the layer, not on its corner: turning
    // something should spin it in place, which is also what the handles in the
    // viewport promise.
    const scaleX = layer.transform.scaleX || 1;
    const scaleY = layer.transform.scaleY || 1;
    const naturalWidth = layer.width ?? doc.width;
    const naturalHeight = layer.height ?? doc.height;
    const boxWidth = Math.abs(naturalWidth * scaleX);
    const boxHeight = Math.abs(naturalHeight * scaleY);

    if (layer.transform.rotation) {
      ctx.translate(layer.transform.x + boxWidth / 2, layer.transform.y + boxHeight / 2);
      ctx.rotate(layer.transform.rotation);
      ctx.translate(-boxWidth / 2, -boxHeight / 2);
    } else {
      ctx.translate(layer.transform.x, layer.transform.y);
    }

    ctx.scale(scaleX, scaleY);
    // A negative scale mirrors the drawing about the origin, which would send
    // it off to the other side; shifting by one layer width brings it back
    // into the box it is supposed to fill, flipped.
    if (scaleX < 0 || scaleY < 0) {
      ctx.translate(scaleX < 0 ? -naturalWidth : 0, scaleY < 0 ? -naturalHeight : 0);
    }

    if (layer.imageDataUrl) {
      const img = resolveImage(layer.imageDataUrl);
      if (img) {
        const width = layer.width ?? img.naturalWidth;
        const height = layer.height ?? img.naturalHeight;
        const adjusted = resolveAdjusted?.(layer.id, img, width, height);
        ctx.drawImage(adjusted ?? img, 0, 0, width, height);
      }
    } else if (layer.id === BACKGROUND_LAYER_ID && doc.background === 'white') {
      // The white sheet only exists in documents that ask for it: on a
      // transparent document it would cover the result of a cutout.
      ctx.fillStyle = theme.paper;
      ctx.fillRect(0, 0, doc.width, doc.height);
    }
    // Empty layers stay transparent: they must not veil the ones below.

    ctx.restore();
  }
}

/**
 * RenderEngine handles 2D compositing of the document layers, with panning,
 * zoom and the transparency checkerboard.
 *
 * `render()` is deliberately SYNCHRONOUS: images load in the background and
 * call `onImageLoaded` once ready to request a new frame. Awaiting inside
 * `render()` would corrupt the transformation stack whenever two frames
 * overlapped.
 */
export class RenderEngine {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private imageCache: Map<string, HTMLImageElement> = new Map();
  private loading: Set<string> = new Set();
  private theme: RenderTheme = RENDER_THEMES.dark;
  private onImageLoaded: (() => void) | null;

  private checkerPattern: CanvasPattern | null = null;
  private checkerKey = '';

  /** Half-width of the selection brush, in screen pixels. */
  private brushWidth = 26;

  private adjuster = sharedProcessor();
  /**
   * Processed layers, keyed by layer id. The signature guards against stale
   * results: re-running the shader on every frame would drop the frame rate
   * while panning, and reusing a stale one would ignore a moved slider.
   */
  private adjusted = new Map<
    string,
    { settings: string; scale: number; canvas: HTMLCanvasElement }
  >();
  /** When each layer's settings last changed, to tell a drag from a rest. */
  private lastChange = new Map<string, number>();
  private settleTimer: ReturnType<typeof setTimeout> | null = null;
  /** Halved copies of each photo, for drawing and processing it small. */
  private levels = new WeakMap<HTMLImageElement, HTMLCanvasElement[]>();
  private imageIds = new WeakMap<HTMLImageElement, number>();
  private nextImageId = 0;
  /** Zoom of the frame being drawn. */
  private zoom = 1;

  /**
   * Masks converted from brightness to alpha, keyed by source and size.
   * The conversion reads every pixel, so doing it once per mask rather than
   * once per frame is the difference between a live slider and a slideshow.
   */
  private maskAlpha = new Map<string, HTMLCanvasElement>();
  private maskCoverage = new Map<string, HTMLCanvasElement>();

  constructor(canvas: HTMLCanvasElement, onImageLoaded?: () => void) {
    this.canvas = canvas;
    // No willReadFrequently: nothing reads this canvas back, and the flag
    // moves it off the GPU, which made every frame of a large photo a CPU job.
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not acquire the Canvas 2D context');
    }
    this.ctx = context;
    this.onImageLoaded = onImageLoaded ?? null;
  }

  /** Sets the on-screen thickness of the selection brush. */
  public setBrushWidth(width: number): void {
    this.brushWidth = width;
  }

  public setTheme(theme: RenderTheme): void {
    this.theme = theme;
    this.checkerPattern = null;
  }

  /** Updates the physical canvas size, accounting for the device pixel ratio. */
  public resize(width: number, height: number): void {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(width * dpr));
    this.canvas.height = Math.max(1, Math.round(height * dpr));
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    // setTransform (not scale) avoids accumulating the dpr factor across resizes.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.checkerPattern = null; // the pattern is bound to the current context
  }

  /**
   * Returns the image if it is already decoded, otherwise starts loading it and
   * returns null: the next frame will draw it.
   */
  private getImage(src: string): HTMLImageElement | null {
    const cached = this.imageCache.get(src);
    if (cached) return cached;
    if (this.loading.has(src)) return null;

    this.loading.add(src);
    loadImage(src)
      .then((img) => {
        this.loading.delete(src);
        this.imageCache.set(src, img);
        this.onImageLoaded?.();
      })
      .catch(() => {
        this.loading.delete(src);
        console.warn('[Nabu] Could not load a layer image.');
      });

    return null;
  }

  /** Reusable checkerboard pattern, rebuilt only when the theme changes. */
  private getCheckerPattern(): CanvasPattern | null {
    const key = `${this.theme.checkerLight}|${this.theme.checkerDark}`;
    if (this.checkerPattern && this.checkerKey === key) return this.checkerPattern;

    const size = 12;
    const tile = window.document.createElement('canvas');
    tile.width = size * 2;
    tile.height = size * 2;
    const tctx = tile.getContext('2d');
    if (!tctx) return null;

    tctx.fillStyle = this.theme.checkerDark;
    tctx.fillRect(0, 0, size * 2, size * 2);
    tctx.fillStyle = this.theme.checkerLight;
    tctx.fillRect(0, 0, size, size);
    tctx.fillRect(size, size, size, size);

    this.checkerPattern = this.ctx.createPattern(tile, 'repeat');
    this.checkerKey = key;
    return this.checkerPattern;
  }

  /** Transparency checkerboard, counter-scaled to stay constant on screen. */
  private drawCheckerboard(width: number, height: number, zoom: number): void {
    const pattern = this.getCheckerPattern();
    this.ctx.save();
    if (pattern) {
      const matrix = new DOMMatrix();
      matrix.scaleSelf(1 / zoom, 1 / zoom);
      pattern.setTransform(matrix);
      this.ctx.fillStyle = pattern;
    } else {
      this.ctx.fillStyle = this.theme.checkerDark;
    }
    this.ctx.fillRect(0, 0, width, height);
    this.ctx.restore();
  }

  /**
   * Returns the adjusted version of a layer, rebuilding it only when the image
   * or one of its sliders actually changed.
   */
  private getAdjusted(
    doc: NabuDocument,
    layerId: string,
    image: HTMLImageElement,
    width: number,
    height: number,
    /**
     * Keeps a second document's results apart from the first: the before/after
     * split draws two versions of the same layer ids in every frame, and with a
     * single slot each would evict the other and both would re-run every time.
     */
    namespace = ''
  ): CanvasImageSource | null {
    const cacheId = namespace + layerId;
    const layer = doc.layers[layerId];
    if (!layer) {
      this.adjusted.delete(cacheId);
      return null;
    }

    // The screen shows the photo at some fraction of its size; working on more
    // pixels than that is paid for on every frame and seen by nobody.
    const display = this.displayScale(width, height);

    // Sliders and the chosen look collapse into one set of values before they
    // reach the shader, so a filter costs no extra pass.
    const adjustments = effectiveAdjustments(layer);
    const curves = layer.curves;
    const focus = layer.focus;
    const flatCurves = areCurvesNeutral(curves);
    const flatFocus = isFocusNeutral(focus);
    const lut = shaderLut(layer.lut);
    const masks = activeMasks(layer.masks);
    if (
      !adjustments &&
      flatCurves &&
      flatFocus &&
      isLocalNeutral(layer.local) &&
      !lut &&
      masks.length === 0
    ) {
      this.adjusted.delete(cacheId);
      return display < 1 ? this.level(image, display) : null;
    }

    // The mask goes through the same cache as any other image, so a frame
    // drawn before it has decoded simply comes back unfocused and the load
    // callback asks for another one.
    const mask =
      (focus?.mode === 'mask' || focus?.mode === 'depth') && focus.maskDataUrl
        ? this.getImage(focus.maskDataUrl)
        : null;

    const local = isLocalNeutral(layer.local) ? undefined : layer.local;
    const localMask = local ? this.getImage(local.maskDataUrl) : null;

    const key = [
      adjustmentKey(adjustments),
      curvesKey(curves),
      focusKey(focus),
      mask ? 1 : 0,
      localKey(local),
      localMask ? 1 : 0,
      lut?.key ?? '',
      masksKey(masks),
      // A selection mask that has not decoded yet must not be cached as final.
      masks.filter((m) => m.maskDataUrl && this.getImage(m.maskDataUrl)).length,
    ].join('#');
    // Identifies the pixels, not the settings: the processor keeps the photo
    // and its blurred copy on the GPU across calls about the same image, which
    // is what makes dragging a slider or a tilt-shift feel live.
    const pixelKey = `${layerId}|${this.imageId(image)}`;
    const settings = `${pixelKey}|${width}x${height}|${key}`;

    // While the settings are moving, work on a reduced copy and come back for
    // the screen-sized result once they rest: the same trade every raw
    // developer makes, because a slider that lags is worse than a preview that
    // sharpens a moment after it stops.
    const now = performance.now();
    const cached = this.adjusted.get(cacheId);
    if (cached?.settings !== settings) this.lastChange.set(cacheId, now);
    let scale = display;
    if (now - (this.lastChange.get(cacheId) ?? 0) < SETTLE_MS) {
      scale = Math.min(display, proxyScale(width, height));
      if (scale < display) this.scheduleSettle();
    }
    if (cached && cached.settings === settings && cached.scale >= scale) return cached.canvas;

    const source = this.level(image, scale);
    const w = scale < 1 ? (source as HTMLCanvasElement).width : width;
    const h = scale < 1 ? (source as HTMLCanvasElement).height : height;
    const run = (input: TexImageSource, values: Adjustments, withKey = true) =>
      this.adjuster.process(
        input,
        w,
        h,
        values,
        curves,
        focus,
        mask,
        withKey ? `${pixelKey}|${scale}` : undefined,
        lut,
        scale
      );

    // A bent curve or a focus effect on untouched sliders still has work to
    // do, so the neutral set is passed rather than skipping the pass.
    const base = adjustments ?? NEUTRAL_ADJUSTMENTS;
    let canvas = run(source, base);

    // A region is a second run of the same pipeline with the region's deltas
    // added, composited back through the mask. Two passes rather than a shader
    // that knows about regions: the pass is cached, and the alternative is
    // sixteen more uniforms and two code paths that can disagree.
    if (local && localMask) {
      const boosted = { ...base };
      for (const [name, value] of Object.entries(local.adjustments) as Array<
        [keyof Adjustments, number]
      >) {
        const [min, max] = ADJUSTMENT_RANGES[name];
        boosted[name] = Math.min(max, Math.max(min, boosted[name] + value * local.amount));
      }

      const region = run(source, boosted);
      const alpha = this.getMaskAlpha(local.maskDataUrl, localMask, w, h);

      if (region && alpha) {
        canvas = compositeThroughMask(canvas ?? source, region, alpha, w, h) ?? canvas;
      }
    }

    if (masks.length > 0) {
      const masked = applyMasks(
        canvas ?? source,
        w,
        h,
        masks,
        (input, values) =>
          this.adjuster.process(input, w, h, values, undefined, undefined, undefined, undefined, undefined, scale),
        (m) => this.getMaskCoverage(m, source, pixelKey, w, h)
      );
      if (masked !== source) canvas = masked as HTMLCanvasElement;
    }

    if (!canvas) {
      this.adjusted.delete(cacheId);
      return scale < 1 ? source : null;
    }

    this.adjusted.set(cacheId, { settings, scale, canvas });
    return canvas;
  }

  /** Power-of-two fraction of the photo's size that the screen actually shows. */
  private displayScale(width: number, height: number): number {
    const dpr = window.devicePixelRatio || 1;
    const shown = this.zoom * dpr;
    if (shown >= 1) return 1;
    // Never below what a 512-pixel side needs: tiny levels cost nothing anyway.
    const floor = Math.min(1, 512 / Math.max(width, height));
    return Math.max(2 ** Math.ceil(Math.log2(Math.max(shown, floor))), 1 / 64);
  }

  /** Stable number for a decoded image, for cache keys that must not collide. */
  private imageId(image: HTMLImageElement): number {
    let id = this.imageIds.get(image);
    if (id === undefined) {
      id = ++this.nextImageId;
      this.imageIds.set(image, id);
    }
    return id;
  }

  /**
   * The image reduced to `scale`, built by halving: each level is drawn from
   * the one above it, so the reduction stays sharp and each is made once.
   */
  private level(image: HTMLImageElement, scale: number): HTMLImageElement | HTMLCanvasElement {
    if (scale >= 1) return image;
    let levels = this.levels.get(image);
    if (!levels) {
      levels = [];
      this.levels.set(image, levels);
    }
    const depth = Math.round(Math.log2(1 / scale));
    let previous: HTMLImageElement | HTMLCanvasElement = levels[levels.length - 1] ?? image;
    let previousWidth = levels.length ? levels[levels.length - 1].width : image.naturalWidth;
    let previousHeight = levels.length ? levels[levels.length - 1].height : image.naturalHeight;
    while (levels.length < depth) {
      const next = window.document.createElement('canvas');
      next.width = Math.max(1, Math.ceil(previousWidth / 2));
      next.height = Math.max(1, Math.ceil(previousHeight / 2));
      const ctx = next.getContext('2d');
      if (!ctx) return image;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(previous, 0, 0, next.width, next.height);
      levels.push(next);
      previous = next;
      previousWidth = next.width;
      previousHeight = next.height;
    }
    return levels[depth - 1];
  }

  /** Asks for one more frame once the settings have stopped moving. */
  private scheduleSettle(): void {
    if (this.settleTimer) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      this.onImageLoaded?.();
    }, SETTLE_MS + 20);
  }

  /** Cached brightness-to-alpha conversion of a mask. */
  private getMaskAlpha(
    src: string,
    image: HTMLImageElement,
    width: number,
    height: number
  ): HTMLCanvasElement | null {
    const key = `${src.length}|${width}x${height}`;
    const cached = this.maskAlpha.get(key);
    if (cached) return cached;

    const built = alphaFromLuminance(image, width, height);
    if (built) this.maskAlpha.set(key, built);
    return built;
  }

  /**
   * Cached coverage of a hand-made mask. Keyed by shape and by the pixels it
   * reads, so dragging a slider inside a mask never redraws the mask itself.
   */
  private getMaskCoverage(
    mask: LocalMask,
    image: CanvasImageSource,
    pixelKey: string,
    width: number,
    height: number
  ): HTMLCanvasElement | null {
    const selection = mask.maskDataUrl ? this.getImage(mask.maskDataUrl) : null;
    if (mask.kind === 'selection' && !selection) return null;
    const key = `${mask.kind === 'luminance' ? pixelKey : ''}|${width}x${height}|${maskShapeKey(mask)}`;
    const cached = this.maskCoverage.get(key);
    if (cached) return cached;
    const built = rasteriseMask(mask, image, width, height, selection);
    if (built) {
      // Old shapes are useless once a handle has moved; keep the cache small.
      if (this.maskCoverage.size > 32) this.maskCoverage.clear();
      this.maskCoverage.set(key, built);
    }
    return built;
  }

  /** True when the GPU pipeline for adjustments is usable. */
  public get adjustmentsAvailable(): boolean {
    return this.adjuster.available;
  }

  /**
   * Outline and handles of the layer being placed.
   *
   * Sized in screen pixels rather than document ones, so a handle stays
   * grabbable at any zoom instead of shrinking away on a large photo.
   */
  private drawTransformBox(doc: NabuDocument, layerId: string, zoom: number): void {
    const layer = doc.layers[layerId];
    if (!layer) return;

    const { ctx } = this;
    const { width, height } = layerBoxSize(layer, doc);
    const { x, y, rotation } = layer.transform;

    ctx.save();

    // Work in the box's own space, so every handle below is placed on an
    // upright rectangle no matter how the layer is turned.
    ctx.translate(x + width / 2, y + height / 2);
    if (rotation) ctx.rotate(rotation);
    ctx.translate(-width / 2, -height / 2);

    // A hairline, not a frame. The box is there to say where the layer ends,
    // and a heavy outline over a photograph reads as damage to the photograph.
    ctx.lineWidth = 1 / zoom;
    ctx.strokeStyle = 'rgba(91, 183, 220, 0.55)';
    ctx.strokeRect(0, 0, width, height);

    const handle = TRANSFORM_HANDLE / zoom;
    const arm = handle * 0.9;

    // Corners are drawn as two short strokes meeting, the way a viewfinder
    // marks its frame — the corner is *implied* rather than covered by a block
    // of white sitting on the part of the picture you are trying to align.
    ctx.lineWidth = 2 / zoom;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
    ctx.shadowBlur = 3 / zoom;

    const corners: Array<[number, number, number, number]> = [
      [0, 0, 1, 1],
      [width, 0, -1, 1],
      [0, height, 1, -1],
      [width, height, -1, -1],
    ];

    for (const [cx, cy, dx, dy] of corners) {
      ctx.beginPath();
      ctx.moveTo(cx + dx * arm, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * arm);
      ctx.stroke();
    }

    // The edge handles stay grabbable but say almost nothing: a short tick on
    // the line rather than another object placed over the image.
    for (const { fx, fy } of SCALE_HANDLES) {
      if (fx !== 0.5 && fy !== 0.5) continue;
      const cx = fx * width;
      const cy = fy * height;
      const along = fx === 0.5 ? arm : 0;
      const down = fy === 0.5 ? arm : 0;
      ctx.beginPath();
      ctx.moveTo(cx - along, cy - down);
      ctx.lineTo(cx + along, cy + down);
      ctx.stroke();
    }

    // Rotation grip, on a short stem above the top edge: the classic position,
    // and far enough out that it never overlaps the corners.
    const stem = ROTATE_HANDLE_OFFSET / zoom;
    ctx.lineWidth = 1 / zoom;
    ctx.strokeStyle = 'rgba(91, 183, 220, 0.55)';
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, -stem);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(width / 2, -stem, handle * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
    ctx.shadowBlur = 3 / zoom;
    ctx.fill();

    // No centre grip any more: with the transform tool on, the whole layer is
    // the move target, so a dot in the middle of the picture bought nothing.

    ctx.restore();
  }

  /** Renders the document, applying pan, zoom and blend modes. */
  public render(
    doc: NabuDocument,
    viewport: ViewportTransform,
    containerWidth: number,
    containerHeight: number,
    selection?: Selection | null,
    /** True while the pointer is still down, so the stroke is painted live. */
    drawingSelection = false,
    /** Layer to show move and scale handles for, when one is being placed. */
    transformLayerId?: string | null,
    /**
     * The photo as it was opened, for comparing. With `split` null it replaces
     * the current picture outright; with a split, it fills the part of the frame
     * left of that fraction of the width.
     */
    compare?: { original: NabuDocument; split: number | null } | null
  ): void {
    const { ctx } = this;

    ctx.save();

    ctx.clearRect(0, 0, containerWidth, containerHeight);
    ctx.fillStyle = this.theme.viewportBg;
    ctx.fillRect(0, 0, containerWidth, containerHeight);

    // Viewport origin: centre of the container, shifted by the pan.
    ctx.translate(containerWidth / 2 + viewport.panX, containerHeight / 2 + viewport.panY);
    ctx.scale(viewport.zoom, viewport.zoom);

    if (compare && compare.split === null) {
      // The whole original, centred on its own size: after a crop the two are
      // different shapes, and the original is what the user asked to see.
      const { original } = compare;
      ctx.translate(-original.width / 2, -original.height / 2);
      this.drawDocument(original, viewport.zoom, 'before:');
      ctx.restore();
      return;
    }

    ctx.translate(-doc.width / 2, -doc.height / 2);
    this.drawDocument(doc, viewport.zoom);

    if (compare && compare.split !== null) {
      const x = doc.width * compare.split;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, x, doc.height);
      ctx.clip();
      ctx.fillStyle = this.theme.viewportBg;
      ctx.fillRect(0, 0, x, doc.height);
      this.drawDocument(compare.original, viewport.zoom, 'before:');
      ctx.restore();

      ctx.save();
      ctx.lineWidth = 2 / viewport.zoom;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = 4 / viewport.zoom;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, doc.height);
      ctx.stroke();
      ctx.restore();
    }

    // No frame around the document: the photo must not look like a sheet.

    if (selection && selection.path.length > 0) {
      this.drawSelection(selection, viewport.zoom, drawingSelection);
    }

    if (transformLayerId) {
      this.drawTransformBox(doc, transformLayerId, viewport.zoom);
    }

    ctx.restore();
  }

  /** Checkerboard and layers of one document, in document coordinates. */
  private drawDocument(doc: NabuDocument, zoom: number, namespace = ''): void {
    this.zoom = zoom;
    this.drawCheckerboard(doc.width, doc.height, zoom);
    paintLayers(
      this.ctx,
      doc,
      this.theme,
      (src) => this.getImage(src),
      (layerId, image, width, height) =>
        this.getAdjusted(doc, layerId, image, width, height, namespace)
    );
  }

  /**
   * Draws the selection.
   *
   * While the pointer is down it is painted as a thick, soft, glowing brush
   * stroke that follows the finger — the gesture reads as "smearing over what
   * interests me", not as "plotting a polygon". Once released it settles into a
   * filled region with a crisp outline.
   */
  private drawSelection(selection: Selection, zoom: number, drawing: boolean): void {
    const { ctx } = this;
    const path = selection.path;
    if (path.length === 0) return;

    ctx.save();

    const trace = () => {
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      for (const point of path.slice(1)) ctx.lineTo(point.x, point.y);
    };

    if (drawing) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Three passes of decreasing width build a soft halo without needing a
      // blur filter, which is expensive at high zoom.
      const passes: Array<[number, string]> = [
        [this.brushWidth * 2.0, 'rgba(91, 183, 220, 0.18)'],
        [this.brushWidth * 1.2, 'rgba(91, 183, 220, 0.34)'],
        [this.brushWidth * 0.5, 'rgba(222, 244, 252, 0.9)'],
      ];

      for (const [width, colour] of passes) {
        ctx.lineWidth = width / zoom;
        ctx.strokeStyle = colour;
        trace();
        ctx.stroke();
      }

      // A single tap still deserves a mark.
      if (path.length === 1) {
        ctx.beginPath();
        ctx.arc(path[0].x, path[0].y, this.brushWidth / zoom, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(91, 183, 220, 0.34)';
        ctx.fill();
      }

      ctx.restore();
      return;
    }

    // Settled selection: tinted fill plus a light-over-dark dashed outline that
    // stays legible on any image.
    trace();
    ctx.closePath();

    ctx.fillStyle = 'rgba(91, 183, 220, 0.16)';
    ctx.fill();

    ctx.lineWidth = 1.5 / zoom;
    ctx.setLineDash([6 / zoom, 4 / zoom]);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.stroke();
    ctx.lineDashOffset = 6 / zoom;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.stroke();

    ctx.restore();
  }
}

/**
 * Composes the document at full resolution on an offscreen canvas. Independent
 * of the zoom level and of the viewport crop; encoding it is the caller's
 * business, so every export format starts from the same pixels.
 */
export async function renderDocument(
  doc: NabuDocument,
  theme: RenderTheme
): Promise<HTMLCanvasElement> {
  const out = window.document.createElement('canvas');
  out.width = doc.width;
  out.height = doc.height;
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('Impossibile creare il canvas di esportazione');

  // Preload every image needed: an export must never be partial. Focus masks
  // count — a missing one would silently export the photo without the blur it
  // was showing on screen.
  const sources = Array.from(
    new Set(
      doc.rootLayerIds
        .flatMap((id) => [
          doc.layers[id]?.imageDataUrl,
          doc.layers[id]?.focus?.maskDataUrl,
          doc.layers[id]?.local?.maskDataUrl,
          ...(doc.layers[id]?.masks ?? []).map((mask) => mask.maskDataUrl),
        ])
        .filter((src): src is string => Boolean(src))
    )
  );

  const resolved = new Map<string, HTMLImageElement>();
  await Promise.all(
    sources.map((src) =>
      loadImage(src)
        .then((img) => {
          resolved.set(src, img);
        })
        .catch(() => undefined)
    )
  );

  // The exported file must match what is on screen, so the same adjustment
  // pipeline runs here too — filter and curves included.
  const adjuster = sharedProcessor();

  paintLayers(
    ctx,
    doc,
    theme,
    (src) => resolved.get(src) ?? null,
    (layerId, image, width, height) => {
      const layer = doc.layers[layerId];
      if (!layer) return null;

      const adjustments = effectiveAdjustments(layer);
      const local = isLocalNeutral(layer.local) ? undefined : layer.local;
      const lut = shaderLut(layer.lut);
      const masks = activeMasks(layer.masks);
      const flat =
        areCurvesNeutral(layer.curves) &&
        isFocusNeutral(layer.focus) &&
        !local &&
        !lut &&
        masks.length === 0;
      if (!adjustments && flat) return null;

      // Hand-made masks go on last, over whatever the rest of the pipeline made.
      const finish = (canvas: HTMLCanvasElement | null) => {
        if (masks.length === 0) return canvas;
        const masked = applyMasks(
          canvas ?? image,
          width,
          height,
          masks,
          (input, values) => adjuster.process(input, width, height, values),
          (m) =>
            rasteriseMask(
              m,
              image,
              width,
              height,
              m.maskDataUrl ? (resolved.get(m.maskDataUrl) ?? null) : null
            )
        );
        return masked === image ? canvas : (masked as HTMLCanvasElement);
      };

      const base = adjustments ?? NEUTRAL_ADJUSTMENTS;
      const focusMask = layer.focus?.maskDataUrl
        ? (resolved.get(layer.focus.maskDataUrl) ?? null)
        : null;

      const canvas = adjuster.process(
        image,
        width,
        height,
        base,
        layer.curves,
        layer.focus,
        focusMask,
        undefined,
        lut
      );

      // The export has to see the same regions the screen does, or the saved
      // file is a different photograph from the one that was approved.
      const localMask = local ? resolved.get(local.maskDataUrl) : undefined;
      if (!local || !localMask) return finish(canvas);

      const boosted = { ...base };
      for (const [name, value] of Object.entries(local.adjustments) as Array<
        [keyof Adjustments, number]
      >) {
        const [min, max] = ADJUSTMENT_RANGES[name];
        boosted[name] = Math.min(max, Math.max(min, boosted[name] + value * local.amount));
      }

      const region = adjuster.process(
        image,
        width,
        height,
        boosted,
        layer.curves,
        layer.focus,
        focusMask,
        undefined,
        lut
      );
      const alpha = alphaFromLuminance(localMask, width, height);
      if (!region || !alpha) return finish(canvas);

      return finish(compositeThroughMask(canvas ?? image, region, alpha, width, height) ?? canvas);
    }
  );

  return out;
}
