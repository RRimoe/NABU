// src/store/depthStore.ts
//
// The depth map of each open photo.
//
// Kept apart from the scene: detection is the fast pass that runs on every
// image, depth is heavier and only worth computing when something asks for it.
// Same caching rule though — one map per layer image, thrown away the moment
// the pixels change, because a depth map of an older version of the photo is
// worse than none.

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from './documentStore';

interface DepthEntry {
  /** Greyscale PNG data URL: white is near, black is far. */
  mapDataUrl: string;
  /** The layer image it was computed from, used as a staleness check. */
  sourceUrl: string;
}

interface DepthState {
  entries: Record<string, DepthEntry>;
  pending: Record<string, Promise<string | null>>;
  /** Set once a run fails, so the interface can stop offering it. */
  unavailable: boolean;

  /** Depth map of the active document, or null when absent or stale. */
  activeDepth: () => string | null;
  /** Computes it, reusing a fresh result or a run already in flight. */
  analyzeActive: () => Promise<string | null>;
  /** Drops the cached map, so the next request recomputes. */
  invalidate: (sessionId?: string) => void;
}

function activeLayerUrl(): { sessionId: string; url: string } | null {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  if (!layer?.imageDataUrl) return null;
  return { sessionId: state.activeId, url: layer.imageDataUrl };
}

export const useDepthStore = create<DepthState>((set, get) => ({
  entries: {},
  pending: {},
  unavailable: false,

  activeDepth: () => {
    const active = activeLayerUrl();
    if (!active) return null;
    const entry = get().entries[active.sessionId];
    if (!entry || entry.sourceUrl !== active.url) return null;
    return entry.mapDataUrl;
  },

  analyzeActive: async () => {
    const active = activeLayerUrl();
    if (!active) return null;

    const fresh = get().activeDepth();
    if (fresh) return fresh;

    const running = get().pending[active.sessionId];
    if (running) return running;

    const task = (async () => {
      try {
        const mapDataUrl = await invoke<string>('analyze_depth', {
          imageDataUrl: active.url,
        });
        set((s) => ({
          entries: {
            ...s.entries,
            [active.sessionId]: { mapDataUrl, sourceUrl: active.url },
          },
          unavailable: false,
        }));
        return mapDataUrl;
      } catch (err) {
        // No depth model installed, or running outside Tauri. Every effect
        // that reads depth has a geometric fallback, so this is a missing
        // upgrade rather than a failure.
        console.warn('[Nabu] Depth estimation unavailable:', err);
        set({ unavailable: true });
        return null;
      } finally {
        set((s) => {
          const pending = { ...s.pending };
          delete pending[active.sessionId];
          return { pending };
        });
      }
    })();

    set((s) => ({ pending: { ...s.pending, [active.sessionId]: task } }));
    return task;
  },

  invalidate: (sessionId) => {
    const id = sessionId ?? useDocumentStore.getState().activeId;
    set((s) => {
      const entries = { ...s.entries };
      delete entries[id];
      return { entries };
    });
  },
}));
