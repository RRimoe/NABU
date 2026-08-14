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
import {
  removeBackground,
  upscale,
  extractObject,
  removeObjectByBbox,
  removeSelectedRegion,
} from './operations';
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

    case 'remove_object': {
      // The id refers to the scene model; the object's box drives the
      // segmenter → inpainter chain on the backend. A small model sometimes
      // invents an id when nothing matches — this lookup is the guarantee the
      // prompt cannot give, and the honest message is the right answer then.
      const id = Number(call.arguments.id);
      const scene = useSceneStore.getState().activeScene();
      const target = scene?.objects.find((o) => o.id === id);
      if (!target) return { messageKey: 'command.needsGenerative', handled: false };

      await removeObjectByBbox(target.bbox);
      return { messageKey: 'router.removed', params: { id }, handled: true };
    }

    case 'remove_selection':
      await removeSelectedRegion();
      return { messageKey: 'router.removedSelection', handled: true };

    case 'report_unsupported':
      // The router understood, and what it understood is not possible yet.
      // Saying so beats pretending: the user learns the limit, not a bug.
      return { messageKey: 'command.needsGenerative', handled: false };

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

// ---------------------------------------------------------------------------
// Scene grounding without a language model
// ---------------------------------------------------------------------------

/** Everyday words for the detector's classes, so "auto" finds a `car`. */
const CLASS_WORDS: Record<string, string[]> = {
  person: ['person', 'people', 'persona', 'persone', 'uomo', 'donna', 'ragazzo', 'ragazza'],
  car: ['car', 'auto', 'macchina'],
  bicycle: ['bicycle', 'bike', 'bici', 'bicicletta'],
  motorbike: ['motorbike', 'motorcycle', 'moto', 'motocicletta'],
  bus: ['bus', 'autobus'],
  train: ['train', 'treno'],
  truck: ['truck', 'camion', 'furgone'],
  boat: ['boat', 'barca'],
  bird: ['bird', 'uccello'],
  cat: ['cat', 'gatto'],
  dog: ['dog', 'cane'],
  horse: ['horse', 'cavallo'],
  umbrella: ['umbrella', 'ombrello'],
  bottle: ['bottle', 'bottiglia'],
  cup: ['cup', 'tazza'],
  chair: ['chair', 'sedia'],
  sofa: ['sofa', 'couch', 'divano'],
  laptop: ['laptop', 'portatile', 'computer'],
  'cell phone': ['phone', 'telefono', 'cellulare', 'smartphone'],
  book: ['book', 'libro'],
  clock: ['clock', 'orologio'],
};

const COLOR_WORDS: Record<string, string[]> = {
  red: ['red', 'rosso', 'rossa'],
  orange: ['orange', 'arancione'],
  yellow: ['yellow', 'giallo', 'gialla'],
  green: ['green', 'verde'],
  cyan: ['cyan', 'ciano', 'azzurro', 'azzurra'],
  blue: ['blue', 'blu'],
  purple: ['purple', 'viola'],
  pink: ['pink', 'rosa'],
  brown: ['brown', 'marrone'],
  black: ['black', 'nero', 'nera'],
  white: ['white', 'bianco', 'bianca'],
  gray: ['gray', 'grey', 'grigio', 'grigia'],
};

const POSITION_WORDS: Record<string, string[]> = {
  left: ['left', 'sinistra'],
  right: ['right', 'destra'],
  center: ['center', 'centre', 'centro'],
};

const SELECT_VERBS = /(seleziona|selezionare|select|scegli|evidenzia|pick)/i;
const REMOVE_VERBS = /(elimin|rimuov|togli|leva|cancell|remove|delete|erase|get\s+rid)/i;

function mentioned(phrase: string, words: string[]): boolean {
  return words.some((w) => phrase.includes(w));
}

/**
 * Resolves "select the red car" or "remove the dog" against the scene without
 * any language model.
 *
 * Deliberately conservative: it needs an explicit verb and an unambiguous
 * class match, and leaves everything else to the model tiers. Its value is
 * the build without a language engine, where it is the only grounding there
 * is — and it costs nothing.
 */
function matchSceneSelection(
  phrase: string,
  scene: import('../types/scene').SceneModel | null
): ToolCall | null {
  if (!scene) return null;
  const wantsRemoval = REMOVE_VERBS.test(phrase);
  if (!wantsRemoval && !SELECT_VERBS.test(phrase)) return null;
  const lower = phrase.toLowerCase();

  let candidates = scene.objects.filter((o) => {
    const words = CLASS_WORDS[o.class] ?? [o.class.toLowerCase()];
    return mentioned(lower, words);
  });
  if (candidates.length === 0) return null;

  // A colour or position in the phrase narrows the field.
  for (const [color, words] of Object.entries(COLOR_WORDS)) {
    if (mentioned(lower, words)) {
      const narrowed = candidates.filter((o) => o.color === color);
      if (narrowed.length > 0) candidates = narrowed;
      break;
    }
  }
  for (const [position, words] of Object.entries(POSITION_WORDS)) {
    if (mentioned(lower, words)) {
      const narrowed = candidates.filter((o) => o.position === position);
      if (narrowed.length > 0) candidates = narrowed;
      break;
    }
  }

  // Still ambiguous: the most confident detection is the best guess, and the
  // result is visible, so a wrong guess is obvious and cheap to undo.
  const best = candidates.reduce((a, b) => (b.confidence > a.confidence ? b : a));
  return {
    name: wantsRemoval ? 'remove_object' : 'select_object',
    arguments: { id: best.id },
  };
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

  // Lazy vision: the scene is computed only now that a free-form request
  // needs it, and reused for every request after that. Without it the
  // language tiers still work, they just cannot resolve "the red car".
  const scene = await useSceneStore.getState().analyzeActive();

  // Tier 1.5: explicit selection and removal phrases resolve against the
  // scene directly, which keeps "select the red car" and "remove the dog"
  // working even with no language model.
  const grounded = matchSceneSelection(phrase, scene);
  if (grounded) {
    return await runToolCall(grounded);
  }

  // A removal the scene could not ground, but with an active selection: the
  // user has already pointed at the thing — no detector needed.
  if (hasSelection && REMOVE_VERBS.test(phrase) && !SELECT_VERBS.test(phrase)) {
    return await runToolCall({ name: 'remove_selection', arguments: {} });
  }

  // A recognised generative request that the scene could not ground: answer
  // honestly rather than letting the fallback say "not understood". Checked
  // only after grounding, because "remove the dog" IS possible when a dog is
  // in the scene.
  if (intent.kind === 'unknown' && intent.reasonKey === 'command.needsGenerative') {
    return { messageKey: intent.reasonKey, handled: false };
  }

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

