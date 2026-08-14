// src/engine/curves.ts
//
// Tone curves: control points in, a lookup table out.
//
// The curve is evaluated on the CPU exactly once per change and handed to the
// shader as a 256-entry texture. Evaluating a spline per pixel would be both
// slower and pointless: an 8-bit channel can only take 256 values, so the table
// is the complete answer.

import {
  type CurveChannel,
  type CurvePoint,
  type Curves,
  CURVE_CHANNELS,
  isCurveNeutral,
} from '../types/document';

/** Entries in a channel table: one per possible 8-bit value. */
export const LUT_SIZE = 256;

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/**
 * Samples a curve into a table of `LUT_SIZE` values in 0..1.
 *
 * Interpolation is monotone cubic (Fritsch–Carlson): a plain Catmull-Rom spline
 * overshoots between close control points, which on a tone curve shows up as a
 * bright halo the user never asked for. Monotone tangents cannot overshoot, so
 * dragging a point only ever does what it looks like it does.
 */
export function sampleCurve(points: CurvePoint[]): Float32Array {
  const table = new Float32Array(LUT_SIZE);

  // Sorted, clamped, and with duplicate x values dropped: two points on the
  // same x would divide by zero when the slopes are computed.
  const sorted = [...points]
    .map((p) => ({ x: clamp01(p.x), y: clamp01(p.y) }))
    .sort((a, b) => a.x - b.x)
    .filter((p, index, all) => index === 0 || p.x > all[index - 1].x);

  if (sorted.length === 0) {
    for (let i = 0; i < LUT_SIZE; i += 1) table[i] = i / (LUT_SIZE - 1);
    return table;
  }

  if (sorted.length === 1) {
    table.fill(sorted[0].y);
    return table;
  }

  const count = sorted.length;

  // Secant slopes between consecutive points.
  const secants = new Array<number>(count - 1);
  for (let i = 0; i < count - 1; i += 1) {
    secants[i] = (sorted[i + 1].y - sorted[i].y) / (sorted[i + 1].x - sorted[i].x);
  }

  // Tangents: the average of the neighbouring secants, then corrected so no
  // segment can turn back on itself.
  const tangents = new Array<number>(count);
  tangents[0] = secants[0];
  tangents[count - 1] = secants[count - 2];
  for (let i = 1; i < count - 1; i += 1) {
    tangents[i] = secants[i - 1] * secants[i] <= 0 ? 0 : (secants[i - 1] + secants[i]) / 2;
  }

  for (let i = 0; i < count - 1; i += 1) {
    if (secants[i] === 0) {
      tangents[i] = 0;
      tangents[i + 1] = 0;
      continue;
    }
    const alpha = tangents[i] / secants[i];
    const beta = tangents[i + 1] / secants[i];
    const magnitude = Math.hypot(alpha, beta);
    if (magnitude > 3) {
      const scale = 3 / magnitude;
      tangents[i] = scale * alpha * secants[i];
      tangents[i + 1] = scale * beta * secants[i];
    }
  }

  let segment = 0;
  for (let i = 0; i < LUT_SIZE; i += 1) {
    const x = i / (LUT_SIZE - 1);

    // Outside the control points the curve holds its end value: the user drew
    // no opinion there, and extrapolating a tangent would invent one.
    if (x <= sorted[0].x) {
      table[i] = sorted[0].y;
      continue;
    }
    if (x >= sorted[count - 1].x) {
      table[i] = sorted[count - 1].y;
      continue;
    }

    while (segment < count - 2 && x > sorted[segment + 1].x) segment += 1;

    const left = sorted[segment];
    const right = sorted[segment + 1];
    const span = right.x - left.x;
    const t = (x - left.x) / span;
    const t2 = t * t;
    const t3 = t2 * t;

    // Hermite basis.
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    table[i] = clamp01(
      h00 * left.y + h10 * span * tangents[segment] + h01 * right.y + h11 * span * tangents[segment + 1]
    );
  }

  return table;
}

/**
 * Flattens the four curves into one RGBA texture the shader can sample.
 *
 * The per-channel curve runs first and the master afterwards, which is the
 * order every photo editor uses: the master is a final grade over whatever the
 * individual channels did.
 */
export function buildCurveLut(curves: Curves): Uint8Array {
  const master = sampleCurve(curves.rgb ?? []);
  const perChannel: Record<'r' | 'g' | 'b', Float32Array> = {
    r: sampleCurve(curves.r ?? []),
    g: sampleCurve(curves.g ?? []),
    b: sampleCurve(curves.b ?? []),
  };

  /** Runs the master over a channel result, still in table space. */
  const compose = (value: number) => master[Math.round(clamp01(value) * (LUT_SIZE - 1))];

  const data = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i += 1) {
    data[i * 4 + 0] = Math.round(compose(perChannel.r[i]) * 255);
    data[i * 4 + 1] = Math.round(compose(perChannel.g[i]) * 255);
    data[i * 4 + 2] = Math.round(compose(perChannel.b[i]) * 255);
    data[i * 4 + 3] = 255;
  }
  return data;
}

/** Stable key for caching a processed result, matching `adjustmentKey`. */
export function curvesKey(curves?: Curves): string {
  if (!curves) return '';
  return CURVE_CHANNELS.filter((channel) => !isCurveNeutral(curves[channel]))
    .map(
      (channel) =>
        `${channel}:${(curves[channel] ?? [])
          .map((p) => `${p.x.toFixed(4)},${p.y.toFixed(4)}`)
          .join(';')}`
    )
    .join('|');
}

/**
 * Inserts a point, keeping the list sorted by x.
 * Returns the index it landed at, so the caller can keep dragging it.
 */
export function insertCurvePoint(
  points: CurvePoint[],
  point: CurvePoint
): { points: CurvePoint[]; index: number } {
  const inserted = { x: clamp01(point.x), y: clamp01(point.y) };
  const next = [...points, inserted].sort((a, b) => a.x - b.x);
  return { points: next, index: next.indexOf(inserted) };
}

/**
 * Moves one point.
 *
 * The two ends stay pinned to their edge of the graph — a curve that started at
 * x = 0.3 would leave the darkest tones undefined — and every other point is
 * kept strictly between its neighbours so the curve cannot fold over itself.
 */
export function moveCurvePoint(
  points: CurvePoint[],
  index: number,
  target: CurvePoint
): CurvePoint[] {
  if (index < 0 || index >= points.length) return points;

  const next = points.map((p) => ({ ...p }));
  const y = clamp01(target.y);

  if (index === 0) {
    next[0] = { x: 0, y };
  } else if (index === points.length - 1) {
    next[index] = { x: 1, y };
  } else {
    const gap = 1 / 255;
    const low = next[index - 1].x + gap;
    const high = next[index + 1].x - gap;
    next[index] = { x: Math.min(high, Math.max(low, clamp01(target.x))), y };
  }

  return next;
}

/** Removes a point. The two ends are permanent and are silently kept. */
export function removeCurvePoint(points: CurvePoint[], index: number): CurvePoint[] {
  if (index <= 0 || index >= points.length - 1) return points;
  return points.filter((_, i) => i !== index);
}

/** Index of the point within `radius` of a position, or -1. */
export function findCurvePoint(
  points: CurvePoint[],
  at: CurvePoint,
  radius: number
): number {
  let best = -1;
  let bestDistance = radius;
  points.forEach((point, index) => {
    const distance = Math.hypot(point.x - at.x, point.y - at.y);
    if (distance <= bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

/**
 * Per-channel histogram of an image, normalised to 0..1.
 *
 * Read from a downscaled copy: the shape of the distribution is the same, and
 * a full-resolution `getImageData` would freeze the interface for a moment
 * every time the panel opens.
 */
export function computeHistogram(
  image: HTMLImageElement,
  sampleSize = 240
): Record<CurveChannel, Float32Array> | null {
  const width = Math.max(1, Math.min(sampleSize, image.naturalWidth));
  const height = Math.max(
    1,
    Math.round((image.naturalHeight / Math.max(1, image.naturalWidth)) * width)
  );

  const canvas = window.document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.drawImage(image, 0, 0, width, height);

  let pixels: Uint8ClampedArray;
  try {
    pixels = ctx.getImageData(0, 0, width, height).data;
  } catch {
    // A cross-origin image taints the canvas; the editor can live without the
    // histogram, so this is not worth surfacing.
    return null;
  }

  const bins: Record<CurveChannel, Float32Array> = {
    rgb: new Float32Array(LUT_SIZE),
    r: new Float32Array(LUT_SIZE),
    g: new Float32Array(LUT_SIZE),
    b: new Float32Array(LUT_SIZE),
  };

  for (let i = 0; i < pixels.length; i += 4) {
    // Fully transparent pixels carry no colour: counting them would spike the
    // black end of the histogram of every cutout.
    if (pixels[i + 3] < 8) continue;
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    bins.r[r] += 1;
    bins.g[g] += 1;
    bins.b[b] += 1;
    bins.rgb[Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b)] += 1;
  }

  for (const channel of CURVE_CHANNELS) {
    const data = bins[channel];
    let peak = 0;
    for (const value of data) peak = Math.max(peak, value);
    if (peak > 0) for (let i = 0; i < data.length; i += 1) data[i] /= peak;
  }

  return bins;
}
