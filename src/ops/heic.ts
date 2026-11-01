// src/ops/heic.ts
//
// iPhone photos. The webview cannot show HEIC (outside Safari's engine), so the
// desktop backend decodes it and sends back a PNG; from there it is an
// ordinary photo. In the browser there is no backend, but Safari on iPhone,
// iPad and the Mac decodes HEIC itself, so the browser's own decoder is used.

import { invoke, isDesktop } from '../platform';
import { blobToDataUrl } from '../commands/files';

export const HEIC_EXTENSIONS = ['heic', 'heif'];

export const isHeicName = (name: string) => /\.(heic|heif)$/i.test(name);

/**
 * Draws HEIC bytes with the browser's own decoder, as a PNG, at most `side`
 * pixels on the long edge. Rejects where the browser cannot read HEIC.
 */
async function decodeInBrowser(bytes: Uint8Array | ArrayBuffer, side?: number): Promise<Blob> {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const bitmap = await createImageBitmap(new Blob([view as BlobPart], { type: 'image/heic' }));
  const scale = side ? Math.min(1, side / Math.max(bitmap.width, bitmap.height)) : 1;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('HEIC'))), 'image/png')
  );
}

/** Decodes HEIC bytes into a PNG data URL, upright. */
export async function heicToDataUrl(bytes: Uint8Array | ArrayBuffer): Promise<string> {
  if (!isDesktop()) return blobToDataUrl(await decodeInBrowser(bytes));
  const png = await invoke<ArrayBuffer>('decode_heic', bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
  return blobToDataUrl(new Blob([png], { type: 'image/png' }));
}

/** A small preview of a HEIC file, as an object URL (revoke it when done). */
export async function heicThumbnailUrl(file: Blob, side = 256): Promise<string> {
  if (!isDesktop()) return URL.createObjectURL(await decodeInBrowser(await file.arrayBuffer(), side));
  const bytes = new Uint8Array(await file.arrayBuffer());
  const png = await invoke<ArrayBuffer>('decode_heic', bytes, {
    headers: { 'x-nabu-max-side': String(side) },
  });
  return URL.createObjectURL(new Blob([png], { type: 'image/png' }));
}
