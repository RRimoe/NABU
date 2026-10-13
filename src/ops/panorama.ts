// src/ops/panorama.ts
//
// Talking to the stitcher in the desktop backend. The photos go over as one
// binary message (length-prefixed files) and the panorama comes back as a
// small JSON header followed by a PNG: camera files are large, and base64
// inside JSON would add a third to them and copy them several times.

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export type PanoramaProjection = 'auto' | 'rectilinear' | 'cylindrical' | 'spherical' | 'planar';

export interface PanoramaOptions {
  projection: PanoramaProjection;
  /** Crop to the largest rectangle with picture everywhere. */
  crop: boolean;
  /** Largest output in megapixels; bigger panoramas are scaled down. */
  maxMegapixels: number;
  /** The photos are in shooting order: match each with its neighbours. */
  inOrder?: boolean;
}

export const DEFAULT_PANORAMA: PanoramaOptions = {
  projection: 'auto',
  crop: true,
  maxMegapixels: 60,
};

export interface PanoramaInfo {
  width: number;
  height: number;
  /** Indices into the files sent: which made it in, which did not connect. */
  used: number[];
  dropped: number[];
  projection: Exclude<PanoramaProjection, 'auto'>;
  /** Alignment error, root mean square, in pixels of the original photos. */
  rms: number;
  horizontalFov: number;
  verticalFov: number;
}

export interface PanoramaProgress {
  stage: string;
  percent: number;
}

/** True inside the desktop application, where the stitcher runs. */
export const stitcherAvailable = () =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

async function pack(files: File[]): Promise<Uint8Array> {
  const buffers = await Promise.all(files.map((file) => file.arrayBuffer()));
  const total = buffers.reduce((sum, b) => sum + 4 + b.byteLength, 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  let offset = 0;
  for (const buffer of buffers) {
    view.setUint32(offset, buffer.byteLength, true);
    out.set(new Uint8Array(buffer), offset + 4);
    offset += 4 + buffer.byteLength;
  }
  return out;
}

function unpack(buffer: ArrayBuffer): { info: PanoramaInfo; png: Blob } {
  const view = new DataView(buffer);
  const length = view.getUint32(0, true);
  const header = new TextDecoder().decode(new Uint8Array(buffer, 4, length));
  return {
    info: JSON.parse(header) as PanoramaInfo,
    png: new Blob([new Uint8Array(buffer, 4 + length)], { type: 'image/png' }),
  };
}

const toDataUrl = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });

/** Stitches the photos; progress is reported as it arrives. */
export async function stitchPanorama(
  files: File[],
  options: PanoramaOptions,
  onProgress: (progress: PanoramaProgress) => void
): Promise<{ info: PanoramaInfo; dataUrl: string }> {
  const unlisten = await listen<PanoramaProgress>('panorama-progress', (event) =>
    onProgress(event.payload)
  );
  try {
    onProgress({ stage: 'reading', percent: 0 });
    const body = await pack(files);
    const response = await invoke<ArrayBuffer>('stitch_panorama', body, {
      headers: { 'x-nabu-options': JSON.stringify(options) },
    });
    const { info, png } = unpack(response);
    return { info, dataUrl: await toDataUrl(png) };
  } finally {
    unlisten();
  }
}

export const cancelPanorama = () => invoke('cancel_panorama').catch(() => undefined);
