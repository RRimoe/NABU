// src/ops/export.ts
//
// Turning the composed picture into a file someone can send. The composition is
// done once (renderDocument); everything here is encoding, so changing the
// format or the quality never re-runs the adjustment pipeline.

export type ExportFormat = 'png' | 'jpeg' | 'webp';

export interface ExportOptions {
  format: ExportFormat;
  /** 0..1, ignored by PNG, which is lossless. */
  quality: number;
  /** Output size relative to the document, 0..1. */
  scale: number;
}

export const EXPORT_FORMATS: Record<ExportFormat, { mime: string; extension: string; lossy: boolean; alpha: boolean }> = {
  png: { mime: 'image/png', extension: 'png', lossy: false, alpha: true },
  jpeg: { mime: 'image/jpeg', extension: 'jpg', lossy: true, alpha: false },
  webp: { mime: 'image/webp', extension: 'webp', lossy: true, alpha: true },
};

export const DEFAULT_EXPORT: ExportOptions = { format: 'jpeg', quality: 0.9, scale: 1 };

/** Smallest side an export may shrink to: below this nothing is recognisable. */
const MIN_EXPORT_SIDE = 16;

export function exportSize(width: number, height: number, scale: number) {
  return {
    width: Math.max(MIN_EXPORT_SIDE, Math.round(width * scale)),
    height: Math.max(MIN_EXPORT_SIDE, Math.round(height * scale)),
  };
}

/**
 * Encodes a composed canvas.
 *
 * JPEG has no transparency, and a browser encoding a transparent pixel to JPEG
 * writes it black, which turns every cutout into a silhouette on a black sheet.
 * Transparent areas are flattened onto white instead: it is what a print, a
 * document or an email shows behind a picture anyway.
 */
export async function encodeCanvas(source: HTMLCanvasElement, options: ExportOptions): Promise<Blob> {
  const spec = EXPORT_FORMATS[options.format];
  const { width, height } = exportSize(source.width, source.height, options.scale);

  let target = source;
  if (width !== source.width || height !== source.height || !spec.alpha) {
    target = document.createElement('canvas');
    target.width = width;
    target.height = height;
    const ctx = target.getContext('2d');
    if (!ctx) throw new Error('Canvas unavailable');
    if (!spec.alpha) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, width, height);
  }

  const blob = await new Promise<Blob | null>((resolve) =>
    target.toBlob(resolve, spec.mime, spec.lossy ? options.quality : undefined)
  );
  if (!blob) throw new Error(`Encoding to ${options.format.toUpperCase()} failed`);

  // A webview without a WebP encoder silently hands back a PNG. Saving that
  // under a .webp name would produce a file that lies about what it is.
  if (blob.type && blob.type !== spec.mime) {
    throw new Error(`${options.format.toUpperCase()} is not supported here`);
  }
  return blob;
}

/** Hands a blob to the user as a download. */
export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = fileName;
  link.href = url;
  link.click();
  // The click has started the download; the URL can go once it has been read.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** File name without characters the common file systems refuse. */
export function safeFileName(title: string, extension: string) {
  const base = title.replace(/[\\/:*?"<>|]+/g, '-').trim() || 'nabu';
  return `${base}.${extension}`;
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
