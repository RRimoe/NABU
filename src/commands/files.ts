// src/commands/files.ts
//
// Reading and writing files by path, for the commands that scripts and agents
// call. The webview cannot touch the disk itself; the backend does it and the
// bytes cross the bridge raw, without a base64 detour.

import { invoke } from '@tauri-apps/api/core';

export const desktopAvailable = () =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export async function readFile(path: string): Promise<Uint8Array> {
  const buffer = await invoke<ArrayBuffer>('read_file', { path });
  return new Uint8Array(buffer);
}

export async function writeFile(path: string, bytes: Uint8Array): Promise<void> {
  await invoke('write_file', bytes, { headers: { 'x-nabu-path': encodeURIComponent(path) } });
}

/** Last component of a path, whichever separator the system uses. */
export function baseName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

export function stem(path: string): string {
  return baseName(path).replace(/\.[^.]+$/, '');
}

export function extension(path: string): string {
  const match = /\.([^.\\/]+)$/.exec(path);
  return match ? match[1].toLowerCase() : '';
}

/** Joins a folder and a file name with the folder's own separator. */
export function joinPath(folder: string, name: string): string {
  const separator = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  return folder.replace(/[\\/]+$/, '') + separator + name;
}

const IMAGE_TYPES: Record<string, string> = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
  gif: 'image/gif',
  bmp: 'image/bmp',
  avif: 'image/avif',
  // Decoded by the backend (src/ops/heic.ts), not by the webview.
  heic: 'image/heic',
  heif: 'image/heif',
};

export const imageType = (path: string): string | undefined => IMAGE_TYPES[extension(path)];

export const blobToDataUrl = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
