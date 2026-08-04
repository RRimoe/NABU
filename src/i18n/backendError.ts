// src/i18n/backendError.ts
//
// The Rust backend never sends user-facing prose: it sends a translation key
// plus the values that key needs. Turning that into readable text is the
// frontend's job, so the same failure reads correctly in every language.

import { t } from './index';

export interface BackendError {
  key: string;
  params?: Record<string, string>;
  /** Untranslated technical cause, useful in logs and bug reports. */
  detail?: string;
}

function isBackendError(value: unknown): value is BackendError {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as BackendError).key === 'string'
  );
}

/** Human-readable message for anything thrown by an `invoke` call. */
export function describeError(error: unknown): string {
  if (isBackendError(error)) {
    return t(error.key, error.params);
  }
  // Not one of ours: a thrown Error, or a string from the Tauri bridge.
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Technical detail, when the backend attached one. */
export function errorDetail(error: unknown): string | undefined {
  return isBackendError(error) ? error.detail : undefined;
}
