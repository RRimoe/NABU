// src/engine/localMasks.ts
//
// Masks the user builds by hand: a graduated filter across the sky, an ellipse
// on a face, the highlights only, a selection. Each one is a coverage map plus
// a set of slider deltas, and both the screen and the export apply them here so
// they cannot drift apart.

import {
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
  type Adjustments,
  type LocalMask,
} from '../types/document';
import { alphaFromLuminance, compositeThroughMask } from './maskAlpha';

/** Pixels both a 2D canvas and the WebGL pipeline accept. */
export type Pixels = HTMLCanvasElement | HTMLImageElement;

/** Everything that changes what a mask covers, for cache keys. */
export function maskShapeKey(mask: LocalMask): string {
  const shape =
    mask.kind === 'linear'
      ? JSON.stringify(mask.linear)
      : mask.kind === 'radial'
        ? JSON.stringify(mask.radial)
        : mask.kind === 'luminance'
          ? JSON.stringify(mask.luminance)
          : String(mask.maskDataUrl?.length ?? 0);
  return `${mask.kind}:${mask.invert ? 1 : 0}:${shape}`;
}

/** Everything that changes the result of a stack of masks. */
export function masksKey(masks: LocalMask[]): string {
  return masks
    .map((mask) => {
      const values = Object.entries(mask.adjustments)
        .filter(([, value]) => value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([name, value]) => `${name}:${value}`)
        .join(',');
      return `${maskShapeKey(mask)}|${mask.amount.toFixed(3)}|${values}`;
    })
    .join(';');
}

/** The neutral set plus a mask's deltas, scaled and kept inside each range. */
export function maskAdjustments(mask: LocalMask): Adjustments {
  const out = { ...NEUTRAL_ADJUSTMENTS };
  for (const [name, value] of Object.entries(mask.adjustments) as Array<[keyof Adjustments, number]>) {
    if (!value) continue;
    const [min, max] = ADJUSTMENT_RANGES[name];
    out[name] = Math.min(max, Math.max(min, value * mask.amount));
  }
  return out;
}

const smoothstep = (edge0: number, edge1: number, x: number) => {
  if (edge1 <= edge0) return x < edge0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
};

function sheet(width: number, height: number) {
  const canvas = window.document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return ctx ? { canvas, ctx } : null;
}

/**
 * Coverage of a mask as an alpha channel, at the layer's size.
 *
 * `source` is the layer's own pixels, which a brightness mask reads; `selection`
 * is the decoded greyscale picture of a selection mask. Gradients are drawn by
 * the canvas itself, which is both faster and smoother than a pixel loop.
 */
export function rasteriseMask(
  mask: LocalMask,
  source: CanvasImageSource,
  width: number,
  height: number,
  selection: CanvasImageSource | null
): HTMLCanvasElement | null {
  let alpha: HTMLCanvasElement | null = null;

  if (mask.kind === 'selection') {
    if (!selection) return null;
    alpha = alphaFromLuminance(selection, width, height);
  } else {
    const made = sheet(width, height);
    if (!made) return null;
    const { canvas, ctx } = made;
    const w = canvas.width;
    const h = canvas.height;

    if (mask.kind === 'linear' && mask.linear) {
      const { x0, y0, x1, y1 } = mask.linear;
      const gradient = ctx.createLinearGradient(x0 * w, y0 * h, x1 * w, y1 * h);
      gradient.addColorStop(0, 'rgba(255,255,255,1)');
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, w, h);
    } else if (mask.kind === 'radial' && mask.radial) {
      const { cx, cy, rx, ry, feather } = mask.radial;
      // A circle drawn in a squashed space is an ellipse in the real one.
      ctx.save();
      ctx.translate(cx * w, cy * h);
      ctx.scale(Math.max(1, rx * w), Math.max(1, ry * h));
      const inner = Math.min(0.999, Math.max(0, 1 - feather));
      const gradient = ctx.createRadialGradient(0, 0, inner, 0, 0, 1);
      gradient.addColorStop(0, 'rgba(255,255,255,1)');
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(0, 0, 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else if (mask.kind === 'luminance' && mask.luminance) {
      const { low, high, smooth } = mask.luminance;
      ctx.drawImage(source, 0, 0, w, h);
      let pixels: ImageData;
      try {
        pixels = ctx.getImageData(0, 0, w, h);
      } catch {
        return null;
      }
      const data = pixels.data;
      for (let i = 0; i < data.length; i += 4) {
        const lum = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 2.55;
        const weight = smoothstep(low - smooth, low, lum) * (1 - smoothstep(high, high + smooth, lum));
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
        data[i + 3] = Math.round(weight * 255);
      }
      ctx.putImageData(pixels, 0, 0);
    } else {
      return null;
    }
    alpha = canvas;
  }

  if (!alpha || !mask.invert) return alpha;

  const flipped = sheet(alpha.width, alpha.height);
  if (!flipped) return alpha;
  flipped.ctx.fillStyle = '#ffffff';
  flipped.ctx.fillRect(0, 0, alpha.width, alpha.height);
  flipped.ctx.globalCompositeOperation = 'destination-out';
  flipped.ctx.drawImage(alpha, 0, 0);
  return flipped.canvas;
}

/**
 * Runs each mask over the picture in turn.
 *
 * A mask's deltas are applied to the picture as it stands after the masks
 * before it, not to the original, so two overlapping masks add up the way they
 * do in a photo editor instead of the later one hiding the earlier.
 */
export function applyMasks(
  start: Pixels,
  width: number,
  height: number,
  masks: LocalMask[],
  run: (input: Pixels, adjustments: Adjustments) => HTMLCanvasElement | null,
  coverage: (mask: LocalMask) => HTMLCanvasElement | null
): Pixels {
  let current: Pixels = start;
  for (const mask of masks) {
    const alpha = coverage(mask);
    if (!alpha) continue;
    const region = run(current, maskAdjustments(mask));
    if (!region) continue;
    current = compositeThroughMask(current, region, alpha, width, height) ?? current;
  }
  return current;
}
