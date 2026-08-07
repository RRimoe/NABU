// src/ai/router.ts
//
// Turns a typed sentence into an action.
//
// Two paths lead here. When the language model is installed, the sentence goes
// to it and comes back as a structured tool call, which understands any
// phrasing in any language. When it is not, the keyword matcher in `intents.ts`
// takes over: it understands far less, but it costs nothing and always answers.

import { invoke } from '@tauri-apps/api/core';
import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { useSceneStore } from '../store/sceneStore';
import { sceneSummary } from '../types/scene';
import { type Adjustments, NEUTRAL_ADJUSTMENTS } from '../types/document';
import { interpret as matchKeywords } from './intents';
import { removeBackground, upscale, extractObject } from './operations';
import { resizeByPercent, resizeLongestSide, resizeDocument } from '../ops/resize';
import { t } from '../i18n';

/** Mirrors `ToolCall` in src-tauri/src/ai/tools.rs. */
export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface RouteOutcome {
  /** Translation key describing what was done, or why nothing was. */
  messageKey: string;
  params?: Record<string, string | number>;
  handled: boolean;
}

const ADJUSTMENT_KEYS = Object.keys(NEUTRAL_ADJUSTMENTS) as Array<keyof Adjustments>;

/** Applies one adjustment to the active layer, relative to its current value. */
function applyAdjustment(property: keyof Adjustments, amount: number): RouteOutcome {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const layerId = doc?.activeLayerId;
  if (!doc || !layerId) {
    return { messageKey: 'error.noImageOpen', handled: false };
  }

  const current = { ...NEUTRAL_ADJUSTMENTS, ...doc.layers[layerId]?.adjustments };
  // Requests are relative: "brighter" after "brighter" should keep going.
  const next = Math.max(-100, Math.min(100, current[property] + amount));

  state.setAdjustment(layerId, property, next);
  state.commitAdjustments(layerId);

  return {
    messageKey: 'router.applied',
    params: { property: t(`adjust.${property}`), value: next > 0 ? `+${next}` : `${next}` },
    handled: true,
  };
}

/** Carries out a tool call, whatever produced it. */
export async function runToolCall(call: ToolCall): Promise<RouteOutcome> {
  switch (call.name) {
    case 'remove_background':
      await removeBackground();
      return { messageKey: 'router.done', handled: true };

    case 'cut_out_selection':
      await extractObject();
      return { messageKey: 'router.done', handled: true };

    case 'select_object': {
      // The id refers to the scene model built by the vision engine; the
      // object's box becomes a normal selection, so everything that works on
      // a selection (cut out, refine) now works on "the red car".
      const id = Number(call.arguments.id);
      const scene = useSceneStore.getState().activeScene();
      const target = scene?.objects.find((o) => o.id === id);
      if (!target) return { messageKey: 'command.notUnderstood', handled: false };

      const state = useDocumentStore.getState();
      const session = state.sessions[state.activeId];
      const doc = session?.document;
      const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
      if (!doc || !layer) return { messageKey: 'error.noImageOpen', handled: false };

      const [x0, y0, x1, y1] = target.bbox;
      const ox = layer.transform.x;
      const oy = layer.transform.y;
      state.setSelection({
        path: [
          { x: x0 + ox, y: y0 + oy },
          { x: x1 + ox, y: y0 + oy },
          { x: x1 + ox, y: y1 + oy },
          { x: x0 + ox, y: y1 + oy },
        ],
      });
      return { messageKey: 'router.selected', params: { id }, handled: true };
    }

    case 'upscale':
      await upscale();
      return { messageKey: 'router.done', handled: true };

    case 'resize': {
      const percent = Number(call.arguments.percent);
      const width = Number(call.arguments.width);
      const height = Number(call.arguments.height);

      if (Number.isFinite(percent) && percent > 0) return resizeByPercent(percent);
      if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
        return resizeDocument(width, height);
      }
      // One side given: keep the proportions rather than guessing the other.
      const side = Number.isFinite(width) && width > 0 ? width : height;
      if (Number.isFinite(side) && side > 0) return resizeLongestSide(side);

      return { messageKey: 'resize.unchanged', handled: false };
    }

    case 'reset_adjustments': {
      const state = useDocumentStore.getState();
      const layerId = state.sessions[state.activeId]?.document.activeLayerId;
      if (!layerId) return { messageKey: 'error.noImageOpen', handled: false };
      state.resetAdjustments(layerId);
      return { messageKey: 'router.reset', handled: true };
    }

    case 'adjust': {
      const property = String(call.arguments.property ?? '') as keyof Adjustments;
      const amount = Number(call.arguments.amount ?? 0);

      if (!ADJUSTMENT_KEYS.includes(property)) {
        return { messageKey: 'command.notUnderstood', handled: false };
      }
      // The model answers 0 when it has nothing to change; saying so is more
      // honest than silently doing nothing.
      if (!Number.isFinite(amount) || amount === 0) {
        return { messageKey: 'router.noChange', handled: false };
      }
      return applyAdjustment(property, amount);
    }

    default:
      return { messageKey: 'command.notUnderstood', handled: false };
  }
}

/** True when the language model is installed and this build can use it. */
export async function routerAvailable(): Promise<boolean> {
  try {
    return await invoke<boolean>('router_available');
  } catch {
    return false;
  }
}

/**
 * Interprets a sentence and carries it out.
 *
 * Tier 1 (Multilingual Smart Engine) runs first for 0ms execution on any device.
 * If Tier 1 does not match, Tier 2 (Ollama / Tauri Local LLM) is tried as a non-blocking fallback.
 */
export async function route(phrase: string): Promise<RouteOutcome> {
  const state = useDocumentStore.getState();
  const hasSelection = Boolean(state.sessions[state.activeId]?.selection);
  const activity = useActivityStore.getState();

  // Tier 1: Zero-latency Multilingual Pattern Engine (0ms execution, 0 MB VRAM)
  const intent = matchKeywords(phrase);
  if (intent.kind === 'tool') {
    return await runToolCall(intent.call);
  }

  // Early return for recognized generative features not yet implemented
  if (intent.kind === 'unknown' && intent.reasonKey === 'command.needsGenerative') {
    return { messageKey: intent.reasonKey, handled: false };
  }

  // Lazy vision: the scene is computed only now that a free-form request
  // needs it, and reused for every request after that. Without it the
  // language tiers still work, they just cannot resolve "the red car".
  const scene = await useSceneStore.getState().analyzeActive();

  // Tier 2: Non-blocking fallback to local Ollama API if running on localhost
  const ollamaCall = await queryOllama(phrase);
  if (ollamaCall) {
    return await runToolCall(ollamaCall);
  }

  // Tier 2: Fallback to built-in Tauri LLM router (if installed)
  if (await routerAvailable()) {
    const job = activity.begin('activity.understanding');
    state.setAIBusy(true, 'routing', 10);
    try {
      activity.step(job, 'activity.routing');
      const call = await invoke<ToolCall>('interpret_command', {
        text: phrase,
        hasSelection,
        scene: scene ? sceneSummary(scene) : null,
      });
      activity.end(job, 'done');
      state.setAIBusy(false);
      return await runToolCall(call);
    } catch (err) {
      console.warn('[Nabu] Built-in language router failed:', err);
      activity.end(job, 'failed', err);
      state.setAIBusy(false);
    }
  }

  return { messageKey: intent.reasonKey, handled: false };
}

/** Tries to query a local Ollama instance on port 11434 with a short 1.5s timeout. */
async function queryOllama(phrase: string): Promise<ToolCall | null> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1500);
    const res = await fetch('http://127.0.0.1:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: 'qwen2.5:1.5b',
        prompt: `Convert this image editor command into JSON tool call format: {"name": "adjust"|"remove_background"|"upscale"|"resize"|"reset_adjustments", "arguments": {}}: ${phrase}`,
        stream: false,
        format: 'json',
      }),
    });
    clearTimeout(timeoutId);
    if (res.ok) {
      const data = (await res.json()) as { response?: string };
      if (data.response) {
        const parsed = JSON.parse(data.response) as ToolCall;
        if (parsed?.name) return parsed;
      }
    }
  } catch {
    // Silent fallback if Ollama is not installed or unreachable
  }
  return null;
}

