// src/engine/harmonize.ts
//
// Making a pasted piece belong to the photo it was pasted into.
//
// What gives a composite away is almost never the outline — it is that the
// subject was lit by a different sun. This measures the light of the
// destination around where the object landed and moves the object's own light
// towards it: brightness, contrast, colour cast, and the grain of the sensor
// that took the picture. Then it puts a shadow under it, because an object with
// nothing beneath it reads as a sticker no matter how well its colours match.
//
// All of it is deterministic arithmetic on pixels. It cannot relight a subject
// that faces the wrong way, and it cannot cast a shadow that follows the
// geometry of the scene — those need a generative model, and saying so is more
// useful than pretending otherwise.

import { loadImage, paintLayers, RENDER_THEMES } from './RenderEngine';
import type { NabuDocument, LayerNode } from '../types/document';

export interface HarmonizeSettings {
  /** How far the object's light is moved towards the destination's, 0..1. */
  strength: number;
  /** How much of the destination's grain is matched, 0..1. */
  grain: number;
  /** Opacity of the contact shadow, 0..1. Zero means none. */
  shadow: number;
  /**
   * How hard the object is re-lit from the direction the scene is lit from,
   * 0..1. This is the one step that changes *where* the light falls rather
   * than how much of it there is.
   */
  relight: number;
}

export const DEFAULT_HARMONIZE: HarmonizeSettings = {
  strength: 0.7,
  grain: 0.6,
  shadow: 0.35,
  relight: 0.5,
};

/** A direction the light comes from, in image space, plus how sure we are. */
interface LightSource {
  /** Unit vector pointing towards the light. Y grows downwards. */
  x: number;
  y: number;
  /** 0 = flat light with no discernible side, 1 = strongly one-sided. */
  confidence: number;
}

export interface HarmonizeResult {
  /** The pasted layer, relit. */
  imageDataUrl: string;
  /** A document-sized shadow to place underneath, or null when not asked for. */
  shadowDataUrl: string | null;
}

/** Longest side used when measuring. Statistics do not need resolution. */
const SAMPLE_SIDE = 420;

/** How far past the object the destination is sampled, as a share of its size. */
const RING = 0.38;

/** Bounds on how much the contrast may be stretched, to keep it believable. */
const MIN_RATIO = 0.65;
const MAX_RATIO = 1.55;

const LUMA_R = 0.2126;
const LUMA_G = 0.7152;
const LUMA_B = 0.0722;

interface Stats {
  /** Mean luminance, 0..255. */
  luma: number;
  /** Standard deviation of luminance. */
  lumaSpread: number;
  /** Mean of each channel. */
  meanR: number;
  meanG: number;
  meanB: number;
  /** Standard deviation of each channel. */
  spreadR: number;
  spreadG: number;
  spreadB: number;
  /** Mean of each channel minus mean luminance: the colour cast, not the level. */
  castR: number;
  castG: number;
  castB: number;
  /** Mean absolute difference between neighbouring pixels: the grain. */
  noise: number;
  count: number;
}

const EMPTY: Stats = {
  luma: 0,
  lumaSpread: 0,
  meanR: 0,
  meanG: 0,
  meanB: 0,
  spreadR: 0,
  spreadG: 0,
  spreadB: 0,
  castR: 0,
  castG: 0,
  castB: 0,
  noise: 0,
  count: 0,
};

function newCanvas(width: number, height: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = window.document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D context unavailable');
  return [canvas, ctx];
}

/**
 * Measures a region.
 *
 * `accept` decides which pixels count, which is how the same routine measures
 * both an object (its opaque pixels) and the ring of background around it
 * (everything except where the object sits).
 */
function measure(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  accept: (x: number, y: number, alpha: number) => boolean
): Stats {
  let count = 0;
  let sum = 0;
  let sumSquares = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sumR2 = 0;
  let sumG2 = 0;
  let sumB2 = 0;
  let noise = 0;
  let noisePairs = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const alpha = pixels[i + 3];
      if (!accept(x, y, alpha)) continue;

      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const luma = LUMA_R * r + LUMA_G * g + LUMA_B * b;

      count += 1;
      sum += luma;
      sumSquares += luma * luma;
      sumR += r;
      sumG += g;
      sumB += b;
      sumR2 += r * r;
      sumG2 += g * g;
      sumB2 += b * b;

      // Grain, read as the difference from the pixel to the right. Only when
      // that neighbour is part of the same region, or an edge would read as
      // noise.
      if (x + 1 < width) {
        const j = i + 4;
        if (accept(x + 1, y, pixels[j + 3])) {
          const other = LUMA_R * pixels[j] + LUMA_G * pixels[j + 1] + LUMA_B * pixels[j + 2];
          noise += Math.abs(luma - other);
          noisePairs += 1;
        }
      }
    }
  }

  if (count === 0) return EMPTY;

  const meanLuma = sum / count;
  const meanR = sumR / count;
  const meanG = sumG / count;
  const meanB = sumB / count;

  return {
    luma: meanLuma,
    lumaSpread: Math.sqrt(Math.max(0, sumSquares / count - meanLuma * meanLuma)),
    meanR,
    meanG,
    meanB,
    spreadR: Math.sqrt(Math.max(0, sumR2 / count - meanR * meanR)),
    spreadG: Math.sqrt(Math.max(0, sumG2 / count - meanG * meanG)),
    spreadB: Math.sqrt(Math.max(0, sumB2 / count - meanB * meanB)),
    castR: meanR - meanLuma,
    castG: meanG - meanLuma,
    castB: meanB - meanLuma,
    noise: noisePairs > 0 ? noise / noisePairs : 0,
    count,
  };
}

/**
 * Which way the light is coming from, read off the destination.
 *
 * The brightness of the scene is treated as a mass and its centre found: if the
 * left of the frame carries more light than the right, the light is on the
 * left. It is a coarse instrument and it says so through `confidence` — an
 * evenly lit scene gives no direction at all, and the caller then leaves the
 * object alone rather than inventing a sun.
 */
function estimateLight(pixels: Uint8ClampedArray, width: number, height: number): LightSource {
  let weight = 0;
  let sumX = 0;
  let sumY = 0;
  let total = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      if (pixels[i + 3] < 24) continue;

      const luma =
        LUMA_R * pixels[i] + LUMA_G * pixels[i + 1] + LUMA_B * pixels[i + 2];
      total += 1;

      // Only the bright half pulls: the shadows of a scene say much less about
      // where its light is than its highlights do.
      const lit = Math.max(0, luma - 128) / 127;
      const w = lit * lit;
      weight += w;
      sumX += (x / Math.max(1, width - 1) - 0.5) * w;
      sumY += (y / Math.max(1, height - 1) - 0.5) * w;
    }
  }

  if (total === 0 || weight < 1e-6) return { x: 0, y: -1, confidence: 0 };

  const cx = sumX / weight;
  const cy = sumY / weight;
  const length = Math.hypot(cx, cy);

  if (length < 1e-4) {
    // Perfectly balanced: default to overhead, which is where light usually is,
    // but with no confidence so nothing acts on it strongly.
    return { x: 0, y: -1, confidence: 0 };
  }

  return {
    x: cx / length,
    y: cy / length,
    // A centroid a tenth of the frame off centre is already a clear side.
    confidence: Math.min(1, length / 0.12),
  };
}

/**
 * Surface normals for a cutout from its smoothed silhouette.
 *
 * Approximates the object as a 3D volumetric surface facing the camera at the center
 * and curving backwards towards the contour edges.
 */
function bulgeNormals(
  alpha: Float32Array,
  width: number,
  height: number
): { nx: Float32Array; ny: Float32Array; nz: Float32Array } {
  const nx = new Float32Array(width * height);
  const ny = new Float32Array(width * height);
  const nz = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      // Negative gradient of coverage points inward into the surface.
      const gx = -(alpha[i + 1] - alpha[i - 1]) * 0.5;
      const gy = -(alpha[i + width] - alpha[i - width]) * 0.5;

      const lenSq = gx * gx + gy * gy;
      if (lenSq < 1e-6) {
        nx[i] = 0;
        ny[i] = 0;
        nz[i] = 1;
      } else {
        const clampedLenSq = Math.min(0.96, lenSq * 4.0);
        const scale = Math.sqrt(clampedLenSq / lenSq);
        nx[i] = gx * scale;
        ny[i] = gy * scale;
        nz[i] = Math.sqrt(Math.max(0.04, 1 - clampedLenSq));
      }
    }
  }

  return { nx, ny, nz };
}

/** Box the layer occupies in document coordinates. */
function layerBox(layer: LayerNode, doc: NabuDocument) {
  const width = Math.abs((layer.width ?? doc.width) * (layer.transform.scaleX || 1));
  const height = Math.abs((layer.height ?? doc.height) * (layer.transform.scaleY || 1));
  return { x: layer.transform.x, y: layer.transform.y, width, height };
}

/**
 * Paints everything below `layerId` — that is what the object has to match.
 * Layers above it are irrelevant: they are in front, not behind.
 */
async function renderBackdrop(
  doc: NabuDocument,
  layerId: string,
  width: number,
  height: number
): Promise<HTMLCanvasElement> {
  const index = doc.rootLayerIds.indexOf(layerId);
  const below = index < 0 ? doc.rootLayerIds : doc.rootLayerIds.slice(index + 1);

  const subset: NabuDocument = { ...doc, rootLayerIds: below };
  const [canvas, ctx] = newCanvas(width, height);

  const resolved = new Map<string, HTMLImageElement>();
  await Promise.all(
    below.map(async (id) => {
      const src = doc.layers[id]?.imageDataUrl;
      if (!src || resolved.has(src)) return;
      try {
        resolved.set(src, await loadImage(src));
      } catch {
        // A layer that will not decode simply does not contribute.
      }
    })
  );

  ctx.save();
  ctx.scale(width / doc.width, height / doc.height);
  paintLayers(ctx, subset, RENDER_THEMES.dark, (src) => resolved.get(src) ?? null);
  ctx.restore();

  return canvas;
}

/**
 * Relights and color-harmonizes the layer to match where it was dropped.
 *
 * Employs tone-preserving color transfer, 3D volumetric hemispherical shading,
 * and dual-component contact shadows.
 */
export async function harmonizeLayer(
  doc: NabuDocument,
  layerId: string,
  settings: HarmonizeSettings
): Promise<HarmonizeResult | null> {
  const layer = doc.layers[layerId];
  if (!layer?.imageDataUrl) return null;
  if (doc.rootLayerIds.indexOf(layerId) === doc.rootLayerIds.length - 1) return null;

  const image = await loadImage(layer.imageDataUrl);
  const layerWidth = layer.width ?? doc.width;
  const layerHeight = layer.height ?? doc.height;
  const box = layerBox(layer, doc);

  // --- Measure the destination, in a ring around where the object landed ---
  const scale = Math.min(1, SAMPLE_SIDE / Math.max(doc.width, doc.height));
  const backdrop = await renderBackdrop(
    doc,
    layerId,
    doc.width * scale,
    doc.height * scale
  );
  const backdropCtx = backdrop.getContext('2d', { willReadFrequently: true });
  if (!backdropCtx) return null;

  const inner = {
    x: box.x * scale,
    y: box.y * scale,
    width: box.width * scale,
    height: box.height * scale,
  };
  const pad = Math.max(inner.width, inner.height) * RING;
  const outer = {
    x: Math.max(0, inner.x - pad),
    y: Math.max(0, inner.y - pad),
    right: Math.min(backdrop.width, inner.x + inner.width + pad),
    bottom: Math.min(backdrop.height, inner.y + inner.height + pad),
  };

  const regionWidth = Math.max(1, Math.round(outer.right - outer.x));
  const regionHeight = Math.max(1, Math.round(outer.bottom - outer.y));
  const region = backdropCtx.getImageData(
    Math.round(outer.x),
    Math.round(outer.y),
    regionWidth,
    regionHeight
  );

  const innerLeft = inner.x - outer.x;
  const innerTop = inner.y - outer.y;
  const innerRight = innerLeft + inner.width;
  const innerBottom = innerTop + inner.height;

  let destination = measure(
    region.data,
    regionWidth,
    regionHeight,
    (x, y, alpha) =>
      alpha > 24 &&
      (x < innerLeft || x > innerRight || y < innerTop || y > innerBottom)
  );

  if (destination.count < 200) {
    const whole = backdropCtx.getImageData(0, 0, backdrop.width, backdrop.height);
    destination = measure(whole.data, backdrop.width, backdrop.height, (_x, _y, a) => a > 24);
  }
  if (destination.count < 50) return null;

  const light = estimateLight(region.data, regionWidth, regionHeight);

  // --- Measure the object ---
  const sampleScale = Math.min(1, SAMPLE_SIDE / Math.max(layerWidth, layerHeight));
  const [sample, sampleCtx] = newCanvas(layerWidth * sampleScale, layerHeight * sampleScale);
  sampleCtx.drawImage(image, 0, 0, sample.width, sample.height);
  const sampled = sampleCtx.getImageData(0, 0, sample.width, sample.height);
  const source = measure(sampled.data, sample.width, sample.height, (_x, _y, a) => a > 200);
  if (source.count < 50) return null;

  // --- Apply at full resolution ---
  const [output, outputCtx] = newCanvas(layerWidth, layerHeight);
  outputCtx.drawImage(image, 0, 0, layerWidth, layerHeight);
  const target = outputCtx.getImageData(0, 0, layerWidth, layerHeight);
  const data = target.data;

  const strength = Math.max(0, Math.min(1, settings.strength));

  // Contrast stretch ratio
  const spreadRatio =
    source.lumaSpread > 4
      ? Math.max(MIN_RATIO, Math.min(MAX_RATIO, destination.lumaSpread / source.lumaSpread))
      : 1;
  const ratio = 1 + (spreadRatio - 1) * strength;
  const targetLuma = source.luma + (destination.luma - source.luma) * strength;

  // Color cast adaptation
  const castR = (destination.castR - source.castR) * strength;
  const castG = (destination.castG - source.castG) * strength;
  const castB = (destination.castB - source.castB) * strength;

  // Grain matching
  const noiseGap = Math.max(0, destination.noise - source.noise) * settings.grain;
  const noiseAmplitude = noiseGap * 1.4;

  // 3D volumetric relighting
  const relighting = settings.relight > 0 && light.confidence > 0.15;
  const shading = relighting ? shadingField(image, layerWidth, layerHeight) : null;
  const relightScale = relighting ? settings.relight * light.confidence * 45 : 0;
  // Key light with zenith incline (Lz = 0.5)
  const lz = 0.5;
  const lLen = Math.hypot(light.x, light.y, lz);
  const lx = light.x / lLen;
  const ly = light.y / lLen;
  const lzNorm = lz / lLen;

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;

    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const luma = LUMA_R * r + LUMA_G * g + LUMA_B * b;

    // Tonal protection curve: smooth bell curve (peaks in midtones, preserves deep shadows and specular highlights)
    const normLuma = luma / 255;
    const midtoneWeight = Math.sin(Math.PI * normLuma);

    // Luminance contrast and brightness transfer
    const newLuma = (luma - source.luma) * ratio + targetLuma;
    let lift = (newLuma - luma) * (0.35 + 0.65 * midtoneWeight);

    // 3D hemispherical diffuse shading
    if (shading) {
      const p = i >> 2;
      const dot = shading.nx[p] * lx + shading.ny[p] * ly + shading.nz[p] * lzNorm;
      // Lambertian diffuse modulation relative to ambient level (0.5)
      const diffuse = (dot - 0.5) * relightScale;
      lift += diffuse * midtoneWeight;
    }

    // Channel color cast applied with tone-aware saturation protection
    const toneCastWeight = 0.4 + 0.6 * midtoneWeight;
    let outR = r + lift + castR * toneCastWeight;
    let outG = g + lift + castG * toneCastWeight;
    let outB = b + lift + castB * toneCastWeight;

    if (noiseAmplitude > 0.01) {
      const n = (Math.sin(i * 12.9898) * 43758.5453) % 1;
      const jitter = (n - Math.floor(n) - 0.5) * noiseAmplitude;
      outR += jitter;
      outG += jitter;
      outB += jitter;
    }

    data[i] = outR < 0 ? 0 : outR > 255 ? 255 : outR;
    data[i + 1] = outG < 0 ? 0 : outG > 255 ? 255 : outG;
    data[i + 2] = outB < 0 ? 0 : outB > 255 ? 255 : outB;
  }

  outputCtx.putImageData(target, 0, 0);

  return {
    imageDataUrl: output.toDataURL('image/png'),
    shadowDataUrl:
      settings.shadow > 0
        ? await buildContactShadow(doc, layer, image, settings.shadow, light)
        : null,
  };
}

/**
 * The object's coverage, blurred, turned into a 3D field of surface normal directions.
 */
function shadingField(
  image: HTMLImageElement,
  width: number,
  height: number
): { nx: Float32Array; ny: Float32Array; nz: Float32Array } | null {
  const [, ctx] = newCanvas(width, height);
  const radius = Math.max(3, Math.min(width, height) * 0.1);
  ctx.filter = `blur(${radius.toFixed(1)}px)`;
  ctx.drawImage(image, 0, 0, width, height);
  ctx.filter = 'none';

  let pixels: Uint8ClampedArray;
  try {
    pixels = ctx.getImageData(0, 0, width, height).data;
  } catch {
    return null;
  }

  const alpha = new Float32Array(width * height);
  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    alpha[p] = pixels[i + 3] / 255;
  }

  return bulgeNormals(alpha, width, height);
}

/**
 * Realistic dual-component contact shadow:
 * 1. Occlusion core (tight, dark contact right under the base).
 * 2. Diffused penumbra (soft, directional shadow following light angle).
 */
async function buildContactShadow(
  doc: NabuDocument,
  layer: LayerNode,
  image: HTMLImageElement,
  opacity: number,
  light: LightSource
): Promise<string> {
  const [canvas, ctx] = newCanvas(doc.width, doc.height);
  const box = layerBox(layer, doc);

  const reach = box.width * 0.35 * light.confidence;
  const lean = -light.x * reach;
  const rotation = layer.transform.rotation || 0;

  // Pass 1: Diffuse Penumbra (soft, elongated shadow)
  ctx.save();
  ctx.translate(box.x + box.width / 2 + lean, box.y + box.height);
  if (rotation) ctx.rotate(rotation);
  ctx.transform(1.08, 0, light.confidence > 0.15 ? -light.x * 0.5 : 0, 0.22, 0, 0);
  ctx.translate(-box.width / 2, -box.height * 0.55);

  const penumbraBlur = Math.max(3, Math.min(box.width, box.height) * 0.07);
  ctx.filter = `blur(${penumbraBlur.toFixed(1)}px)`;
  ctx.globalAlpha = opacity * 0.65;
  ctx.drawImage(image, 0, 0, box.width, box.height);
  ctx.restore();

  // Pass 2: Occlusion Core (sharp, dense contact shadow directly under base)
  ctx.save();
  ctx.translate(box.x + box.width / 2 + lean * 0.2, box.y + box.height);
  if (rotation) ctx.rotate(rotation);
  ctx.transform(1.02, 0, 0, 0.08, 0, 0);
  ctx.translate(-box.width / 2, -box.height * 0.85);

  const coreBlur = Math.max(1.5, Math.min(box.width, box.height) * 0.02);
  ctx.filter = `blur(${coreBlur.toFixed(1)}px)`;
  ctx.globalAlpha = opacity * 0.85;
  ctx.drawImage(image, 0, 0, box.width, box.height);
  ctx.restore();

  // Colorize shadow to deep realistic neutral black
  ctx.globalCompositeOperation = 'source-in';
  ctx.fillStyle = 'rgba(10, 12, 16, 0.92)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  return canvas.toDataURL('image/png');
}
