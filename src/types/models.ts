// src/types/models.ts
// Mirrors the structures serialised by src-tauri/src/ai.

export type ModelTaskKind = 'background-removal' | 'upscale' | 'segmentation';

export type InstallState = 'not-downloaded' | 'partial' | 'ready';

export interface ModelStatus {
  id: string;
  name: string;
  task: ModelTaskKind;
  /** Translation key; the text lives in the language catalogue. */
  descriptionKey: string;
  license: string;
  commercialUse: boolean;
  vramRequiredMb: number;
  totalBytes: number;
  state: InstallState;
  bytesOnDisk: number;
  scaleFactor: number | null;
  dir: string;
}

export interface DownloadProgress {
  modelId: string;
  fileName: string;
  fileIndex: number;
  fileCount: number;
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  bytesPerSecond: number;
}

export interface DownloadFinished {
  modelId: string;
  success: boolean;
  cancelled: boolean;
  error: unknown;
}

/** Translation keys for the task groups shown in the model manager. */
export const TASK_KEYS: Record<ModelTaskKind, string> = {
  'background-removal': 'task.backgroundRemoval',
  upscale: 'task.upscale',
  segmentation: 'task.segmentation',
};

/**
 * Formats a byte count using binary units.
 *
 * Locale-aware on purpose: the decimal separator differs between languages.
 */
export const formatBytes = (bytes: number, locale?: string): string => {
  if (bytes <= 0) return '0 MB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** i;
  const digits = value >= 100 || i <= 1 ? 0 : 1;
  return `${value.toLocaleString(locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} ${units[i]}`;
};
