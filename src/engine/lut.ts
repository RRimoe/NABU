// src/engine/lut.ts
//
// 3D colour lookup tables in the .cube format: the way colourists, cameras and
// other editors hand a grade around. Nabu reads the file once, keeps the table
// on the layer as 8-bit RGB, and the shader looks every pixel up in it.

import type { LayerLut } from '../types/document';

/** Largest table accepted; 65 is the biggest size cameras and editors export. */
const MAX_SIZE = 65;

export class LutError extends Error {}

/** Parses the text of a .cube file into an RGB8 table, red varying fastest. */
export function parseCube(text: string): { size: number; rgb: Uint8Array; title?: string } {
  let size = 0;
  let title: string | undefined;
  let domainMin = [0, 0, 0];
  let domainMax = [1, 1, 1];
  const values: number[] = [];

  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const [keyword, ...rest] = line.split(/\s+/);
    if (keyword === 'TITLE') {
      title = line.slice(5).trim().replace(/^"|"$/g, '');
    } else if (keyword === 'LUT_3D_SIZE') {
      size = Number(rest[0]);
    } else if (keyword === 'LUT_1D_SIZE') {
      throw new LutError('1D tables are not supported');
    } else if (keyword === 'DOMAIN_MIN') {
      domainMin = rest.slice(0, 3).map(Number);
    } else if (keyword === 'DOMAIN_MAX') {
      domainMax = rest.slice(0, 3).map(Number);
    } else if (/^[-+.\d]/.test(keyword)) {
      values.push(Number(keyword), Number(rest[0]), Number(rest[1]));
    }
  }

  if (!Number.isInteger(size) || size < 2 || size > MAX_SIZE) {
    throw new LutError(`unsupported table size ${size}`);
  }
  if (values.length !== size * size * size * 3 || values.some((v) => !Number.isFinite(v))) {
    throw new LutError('the table is incomplete');
  }

  const rgb = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const channel = i % 3;
    const span = domainMax[channel] - domainMin[channel] || 1;
    const normalised = (values[i] - domainMin[channel]) / span;
    rgb[i] = Math.round(Math.min(1, Math.max(0, normalised)) * 255);
  }
  return { size, rgb, title };
}

export function encodeTable(rgb: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < rgb.length; i += 0x8000) {
    binary += String.fromCharCode(...rgb.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

const decoded = new Map<string, Uint8Array>();

/**
 * The table of a layer's LUT, decoded once and remembered: the renderer asks
 * for it on every frame, and base64 is not something to decode sixty times a
 * second.
 */
export function decodeTable(lut: LayerLut): Uint8Array | null {
  const hit = decoded.get(lut.data);
  if (hit) return hit;
  try {
    const binary = atob(lut.data);
    const rgb = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) rgb[i] = binary.charCodeAt(i);
    if (rgb.length !== lut.size ** 3 * 3) return null;
    if (decoded.size > 8) decoded.clear();
    decoded.set(lut.data, rgb);
    return rgb;
  } catch {
    return null;
  }
}

/** What the shader needs, or undefined when the layer has no usable LUT. */
export function shaderLut(lut?: LayerLut) {
  if (!lut || lut.amount <= 0) return undefined;
  const rgb = decodeTable(lut);
  return rgb ? { size: lut.size, rgb, amount: Math.min(1, lut.amount), key: lutKey(lut) } : undefined;
}

/** Cheap identity for caches: same name, size and data length is the same table. */
export function lutKey(lut?: LayerLut) {
  return lut ? `${lut.name}|${lut.size}|${lut.data.length}|${lut.amount}` : '';
}
