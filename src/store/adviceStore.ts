// src/store/adviceStore.ts
//
// The second tier of the suggestions panel.
//
// The first tier is arithmetic: it measures the photo and proposes fixes, and
// it is on screen before the user has let go of the mouse. This one asks a
// small language model to look at the same facts and say something about them
// in a sentence — which takes seconds on a CPU, and is therefore never allowed
// to be something anyone waits for.
//
// The model's answer is not trusted. It picks a recipe from a closed list and
// refers to objects by id; anything outside the list, and any id that is not in
// the scene, is dropped. What survives is a sentence and a name — no operation
// the model wrote itself ever reaches the pixels.

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from './documentStore';
import { useSceneStore } from './sceneStore';
import { isRecipeId, type RecipeId } from '../ai/recipes';
import type { ImageStats } from '../engine/imageStats';
import type { SceneModel } from '../types/scene';
import { useI18nStore } from '../i18n';

export interface Advice {
  /** One sentence about this photo, in the user's language. */
  observation: string;
  recipe: RecipeId;
  /** Box of the object the recipe applies to, in layer pixels. */
  target?: [number, number, number, number];
  /** Class of that object, for the label. */
  targetClass?: string;
}

interface AdviceEntry {
  advice: Advice | null;
  sourceUrl: string;
}

interface AdviceState {
  entries: Record<string, AdviceEntry>;
  pending: Record<string, boolean>;
  unavailable: boolean;

  activeAdvice: () => Advice | null;
  /** Whether the model is thinking about the active photo right now. */
  isThinking: () => boolean;
  /** Asks for advice in the background. Safe to call repeatedly. */
  request: (stats: ImageStats | null) => void;
  clear: (sessionId?: string) => void;
}

function activeLayerUrl(): { sessionId: string; url: string } | null {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  if (!layer?.imageDataUrl) return null;
  return { sessionId: state.activeId, url: layer.imageDataUrl };
}

/**
 * The facts, written out for the model.
 *
 * English on purpose, and numbers turned into words: these models reason far
 * better about "dark" than about 0.24, and the observation comes back in the
 * user's language regardless because the prompt asks for it separately.
 */
function describe(scene: SceneModel | null, stats: ImageStats | null): string {
  const lines: string[] = ['OBJECTS:'];

  if (scene && scene.objects.length > 0) {
    for (const object of scene.objects.slice(0, 8)) {
      const traits: string[] = [object.position, object.prominence];
      if (object.color) traits.push(object.color);
      lines.push(`#${object.id} ${object.class} (${traits.join(', ')})`);
    }
  } else {
    lines.push('(none recognised)');
  }

  if (stats) {
    const brightness =
      stats.meanLuma < 0.38 ? 'dark' : stats.meanLuma > 0.66 ? 'bright' : 'well exposed';
    const contrast = stats.contrast < 0.5 ? 'flat' : 'contrasty';
    const colour = stats.saturation < 0.17 ? 'muted colours' : 'strong colours';
    const focus = stats.detail < 0.3 ? 'soft' : 'sharp';

    lines.push('', 'MEASUREMENTS:');
    lines.push(`brightness: ${brightness}`);
    lines.push(`contrast: ${contrast}`);
    lines.push(`colour: ${colour}`);
    lines.push(`detail: ${focus}`);
    if (stats.clippedHighlights > 0.03) lines.push('some highlights are burnt out');
    if (stats.clippedShadows > 0.05) lines.push('some shadows are crushed to black');
  }

  return lines.join('\n');
}

export const useAdviceStore = create<AdviceState>((set, get) => ({
  entries: {},
  pending: {},
  unavailable: false,

  activeAdvice: () => {
    const active = activeLayerUrl();
    if (!active) return null;
    const entry = get().entries[active.sessionId];
    if (!entry || entry.sourceUrl !== active.url) return null;
    return entry.advice;
  },

  isThinking: () => {
    const active = activeLayerUrl();
    return active ? Boolean(get().pending[active.sessionId]) : false;
  },

  request: (stats) => {
    const active = activeLayerUrl();
    if (!active || get().unavailable) return;

    const entry = get().entries[active.sessionId];
    if (entry?.sourceUrl === active.url) return;
    if (get().pending[active.sessionId]) return;

    set((s) => ({ pending: { ...s.pending, [active.sessionId]: true } }));

    const scene = useSceneStore.getState().entries[active.sessionId]?.scene ?? null;
    const language = useI18nStore.getState().current;

    void (async () => {
      try {
        const raw = await invoke<string>('advise_scene', {
          scene: describe(scene, stats),
          language,
        });

        set((s) => ({
          entries: {
            ...s.entries,
            [active.sessionId]: { advice: validate(raw, scene), sourceUrl: active.url },
          },
        }));
      } catch (err) {
        // No language model installed, or no LLM in this build. The first tier
        // is already on screen and loses nothing by this never arriving.
        console.warn('[Nabu] Advice unavailable:', err);
        set({ unavailable: true });
      } finally {
        set((s) => {
          const pending = { ...s.pending };
          delete pending[active.sessionId];
          return { pending };
        });
      }
    })();
  },

  clear: (sessionId) => {
    const id = sessionId ?? useDocumentStore.getState().activeId;
    set((s) => {
      const entries = { ...s.entries };
      delete entries[id];
      return { entries };
    });
  },
}));

/** Longest observation worth showing. Beyond this the model is rambling. */
const MAX_OBSERVATION = 160;

/**
 * Everything the model said, checked against what exists.
 *
 * A recipe name outside the list is dropped; an object id that is not in the
 * scene is dropped; a sentence longer than a sentence is dropped. What is left
 * is either a usable proposal or nothing at all, and nothing at all is a
 * perfectly good outcome — the measured tier is still there.
 */
function validate(raw: string, scene: SceneModel | null): Advice | null {
  if (!raw.trim()) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== 'object') return null;
  const record = parsed as Record<string, unknown>;

  const observation = typeof record.observation === 'string' ? record.observation.trim() : '';
  if (!observation || observation.length > MAX_OBSERVATION) return null;
  if (!isRecipeId(record.recipe)) return null;

  const advice: Advice = { observation, recipe: record.recipe };

  // The id is resolved against the scene rather than believed. This is the
  // lesson the router taught: a 1.5B model invents ids cheerfully, and the
  // lookup is what makes the answer honest.
  const id = typeof record.target === 'number' ? record.target : null;
  if (id !== null && scene) {
    const object = scene.objects.find((candidate) => candidate.id === id);
    if (object) {
      advice.target = object.bbox;
      advice.targetClass = object.class;
    }
  }

  return advice;
}
