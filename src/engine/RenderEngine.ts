// src/engine/RenderEngine.ts
import {
  NabuDocument,
  ViewportTransform,
  BlendMode,
  ThemeMode,
  Selection,
  BACKGROUND_LAYER_ID,
} from '../types/document';

/** Colours used by the canvas, aligned with the active interface theme. */
export interface RenderTheme {
  viewportBg: string;
  checkerLight: string;
  checkerDark: string;
  paper: string;
  docBorder: string;
}

export const RENDER_THEMES: Record<ThemeMode, RenderTheme> = {
  dark: {
    viewportBg: '#0f0f13',
    checkerLight: '#2a2a32',
    checkerDark: '#1e1e24',
    paper: '#ffffff',
    docBorder: 'rgba(255, 255, 255, 0.22)',
  },
  light: {
    viewportBg: '#e9ecf2',
    checkerLight: '#ffffff',
    checkerDark: '#e2e6ee',
    paper: '#ffffff',
    docBorder: 'rgba(15, 23, 42, 0.28)',
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
    img.onerror = () => reject(new Error('Immagine non caricabile'));
    img.src = src;
  });
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
  resolveImage: (src: string) => HTMLImageElement | null
): void {
  // rootLayerIds runs top to bottom, so it is painted in reverse.
  const bottomToTop = [...doc.rootLayerIds].reverse();

  for (const layerId of bottomToTop) {
    const layer = doc.layers[layerId];
    if (!layer || !layer.visible || layer.opacity <= 0) continue;

    ctx.save();
    ctx.globalAlpha = layer.opacity;
    ctx.globalCompositeOperation = mapBlendMode(layer.blendMode);

    ctx.translate(layer.transform.x, layer.transform.y);
    if (layer.transform.rotation !== 0) ctx.rotate(layer.transform.rotation);
    ctx.scale(layer.transform.scaleX || 1, layer.transform.scaleY || 1);

    if (layer.imageDataUrl) {
      const img = resolveImage(layer.imageDataUrl);
      if (img) {
        ctx.drawImage(
          img,
          0,
          0,
          layer.width ?? img.naturalWidth,
          layer.height ?? img.naturalHeight
        );
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

  constructor(canvas: HTMLCanvasElement, onImageLoaded?: () => void) {
    this.canvas = canvas;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) {
      throw new Error('Impossibile acquisire il contesto Canvas 2D');
    }
    this.ctx = context;
    this.onImageLoaded = onImageLoaded ?? null;
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
        console.warn('[Nabu] Impossibile caricare il contenuto di un livello.');
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

  /** Renders the document, applying pan, zoom and blend modes. */
  public render(
    doc: NabuDocument,
    viewport: ViewportTransform,
    containerWidth: number,
    containerHeight: number,
    selection?: Selection | null
  ): void {
    const { ctx } = this;

    ctx.save();

    ctx.clearRect(0, 0, containerWidth, containerHeight);
    ctx.fillStyle = this.theme.viewportBg;
    ctx.fillRect(0, 0, containerWidth, containerHeight);

    // Viewport origin: centre of the container, shifted by the pan.
    ctx.translate(containerWidth / 2 + viewport.panX, containerHeight / 2 + viewport.panY);
    ctx.scale(viewport.zoom, viewport.zoom);
    ctx.translate(-doc.width / 2, -doc.height / 2);

    this.drawCheckerboard(doc.width, doc.height, viewport.zoom);
    paintLayers(ctx, doc, this.theme, (src) => this.getImage(src));

    // No frame around the document: the photo must not look like a sheet.

    if (selection && selection.path.length > 1) {
      this.drawSelection(selection, viewport.zoom);
    }

    ctx.restore();
  }

  /** Dashed lasso outline, of constant thickness on screen. */
  private drawSelection(selection: Selection, zoom: number): void {
    const { ctx } = this;
    ctx.save();

    ctx.beginPath();
    ctx.moveTo(selection.path[0].x, selection.path[0].y);
    for (const point of selection.path.slice(1)) {
      ctx.lineTo(point.x, point.y);
    }
    ctx.closePath();

    // Light-over-dark double stroke: stays legible on any image.
    ctx.lineWidth = 1.5 / zoom;
    ctx.setLineDash([6 / zoom, 4 / zoom]);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.stroke();
    ctx.lineDashOffset = 6 / zoom;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.stroke();

    ctx.restore();
  }
}

/**
 * Composes the document at full resolution on an offscreen canvas and returns a
 * PNG. Independent of the zoom level and of the viewport crop.
 */
export async function exportDocumentToPng(
  doc: NabuDocument,
  theme: RenderTheme
): Promise<string> {
  const out = window.document.createElement('canvas');
  out.width = doc.width;
  out.height = doc.height;
  const ctx = out.getContext('2d');
  if (!ctx) throw new Error('Impossibile creare il canvas di esportazione');

  // Preload every image needed: an export must never be partial.
  const sources = Array.from(
    new Set(
      doc.rootLayerIds
        .map((id) => doc.layers[id]?.imageDataUrl)
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

  paintLayers(ctx, doc, theme, (src) => resolved.get(src) ?? null);

  return out.toDataURL('image/png');
}
