// src/store/sceneStore.ts
//
// Holds the scene model per document and decides when to (re)build it.
//
// The analysis is lazy and cached: a scene is computed at most once per layer
// image, reused until the pixels change, and never blocks the interface — a
// missing detection model simply means Nabu cannot see, and the editor keeps
// working exactly as before.

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from './documentStore';
import type { SceneModel } from '../types/scene';

interface SceneEntry {
  scene: SceneModel;
  /** The layer image the scene was computed from, used as a staleness check. */
  sourceUrl: string;
}

interface SceneState {
  entries: Record<string, SceneEntry>;
  pending: Record<string, Promise<SceneModel | null>>;

  /** Scene of the active document, or null when absent or stale. */
  activeScene: () => SceneModel | null;
  /** Analyses the active layer, reusing a fresh result or a run in flight. */
  analyzeActive: () => Promise<SceneModel | null>;
}

function activeLayerUrl(): { sessionId: string; url: string } | null {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  if (!layer?.imageDataUrl) return null;
  return { sessionId: state.activeId, url: layer.imageDataUrl };
}

export const useSceneStore = create<SceneState>((set, get) => ({
  entries: {},
  pending: {},

  activeScene: () => {
    const active = activeLayerUrl();
    if (!active) return null;
    const entry = get().entries[active.sessionId];
    if (!entry || entry.sourceUrl !== active.url) return null;
    return entry.scene;
  },

  analyzeActive: async () => {
    const active = activeLayerUrl();
    if (!active) return null;

    const fresh = get().activeScene();
    if (fresh) return fresh;

    const running = get().pending[active.sessionId];
    if (running) return running;

    const task = (async () => {
      try {
        const scene = await invoke<SceneModel>('analyze_scene', {
          imageDataUrl: active.url,
        });
        set((s) => ({
          entries: {
            ...s.entries,
            [active.sessionId]: { scene, sourceUrl: active.url },
          },
        }));
        return scene;
      } catch (err) {
        // No detector installed, or running outside Tauri: not an error the
        // user needs to act on. Requests simply fall back to working blind.
        console.warn('[Nabu] Scene analysis unavailable:', err);
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
}));
