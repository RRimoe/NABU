// src/platform/index.ts
//
// Where the backend lives. In the desktop app it is the Rust side of Tauri;
// in the browser (the web and iPhone/iPad version) there is no Rust process,
// so a call either goes to the Nabu service named by VITE_NABU_SERVICE_URL or
// fails with a translated "not available here" error instead of hanging.
//
// Every module talks to the backend through `invoke` and `listen` from here,
// never from '@tauri-apps/api' directly: that is the one seam where a web
// implementation of a command (WebAssembly, a GPU service, a cloud model) is
// plugged in later without touching the callers.

import { invoke as tauriInvoke, type InvokeArgs, type InvokeOptions } from '@tauri-apps/api/core';
import { listen as tauriListen, type EventCallback, type UnlistenFn } from '@tauri-apps/api/event';
import type { BackendError } from '../i18n/backendError';

export type { InvokeArgs, InvokeOptions, EventCallback, UnlistenFn };

/** True inside the Tauri desktop app. */
export const isDesktop = (): boolean =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/** Base URL of the Nabu web service, when one is configured at build time. */
export const serviceUrl = (): string | undefined => {
  const url = import.meta.env.VITE_NABU_SERVICE_URL as string | undefined;
  return url ? url.replace(/\/+$/, '') : undefined;
};

/** The error a browser build throws for a command nothing here can run. */
export function webUnavailable(command: string): BackendError {
  return { key: 'error.webUnavailable', detail: command };
}

/**
 * Runs a backend command. Same signature as Tauri's `invoke`, so callers do
 * not care which platform answers.
 */
export async function invoke<T>(
  command: string,
  args?: InvokeArgs,
  options?: InvokeOptions
): Promise<T> {
  if (isDesktop()) return tauriInvoke<T>(command, args, options);

  const base = serviceUrl();
  if (!base) throw webUnavailable(command);
  return callService<T>(base, command, args, options);
}

/**
 * Listens to a backend event. In the browser no backend emits events yet, so
 * this resolves to a no-op unlisten: progress bars simply do not move.
 */
export async function listen<T>(event: string, handler: EventCallback<T>): Promise<UnlistenFn> {
  if (isDesktop()) return tauriListen<T>(event, handler);
  return () => undefined;
}

/**
 * The service contract mirrors the Tauri one: `POST {base}/invoke/{command}`
 * with JSON arguments, or raw bytes plus the same `x-nabu-*` headers when the
 * call carries binary data. The answer is JSON, or bytes when the service
 * replies with a binary content type. Errors come back as a BackendError.
 */
async function callService<T>(
  base: string,
  command: string,
  args?: InvokeArgs,
  options?: InvokeOptions
): Promise<T> {
  const binary = args instanceof Uint8Array || args instanceof ArrayBuffer;
  const headers = new Headers(options?.headers as HeadersInit | undefined);
  headers.set('content-type', binary ? 'application/octet-stream' : 'application/json');

  let response: Response;
  try {
    response = await fetch(`${base}/invoke/${encodeURIComponent(command)}`, {
      method: 'POST',
      headers,
      body: binary ? (args as BodyInit) : JSON.stringify(args ?? {}),
      credentials: 'include',
    });
  } catch (err) {
    throw { key: 'error.serviceUnreachable', detail: String(err) } satisfies BackendError;
  }

  const type = response.headers.get('content-type') ?? '';
  if (!response.ok) {
    const body = type.includes('json') ? await response.json().catch(() => null) : null;
    if (body && typeof body.key === 'string') throw body as BackendError;
    throw { key: 'error.serviceFailed', detail: `${response.status} ${command}` } satisfies BackendError;
  }
  if (type.includes('json')) return (await response.json()) as T;
  return (await response.arrayBuffer()) as T;
}
