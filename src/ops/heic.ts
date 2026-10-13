// src/ops/heic.ts
//
// iPhone photos. The webview cannot show HEIC (outside Safari's engine), so the
// desktop backend decodes it and sends back a PNG; from there it is an
// ordinary photo.

import { invoke } from '@tauri-apps/api/core';
import { blobToDataUrl } from '../commands/files';

export const HEIC_EXTENSIONS = ['heic', 'heif'];

export const isHeicName = (name: string) => /\.(heic|heif)$/i.test(name);

/** Decodes HEIC bytes into a PNG data URL, upright. Desktop app only. */
export async function heicToDataUrl(bytes: Uint8Array | ArrayBuffer): Promise<string> {
  if (!('__TAURI_INTERNALS__' in window)) throw new Error('HEIC needs the desktop app');
  const png = await invoke<ArrayBuffer>('decode_heic', bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
  return blobToDataUrl(new Blob([png], { type: 'image/png' }));
}
