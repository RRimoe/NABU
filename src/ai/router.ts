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
import { interpret as matchKeywords } from './intents';
import { findCommand, validate, coerceArgs } from '../commands';

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

/**
 * Carries out a tool call, whatever produced it.
 *
 * The language model's vocabulary is a subset of the command registry, so a
 * call runs exactly the code a script or an agent would run. Run directly
 * rather than through `execute`: the person who typed is looking at the
 * screen, and a failure should reach them as it does from a button.
 */
export async function runToolCall(call: ToolCall): Promise<RouteOutcome> {
  // The router understood, and what it understood is not possible yet.
  // Saying so beats pretending: the user learns the limit, not a bug.
  if (call.name === 'report_unsupported') {
    return { messageKey: 'command.needsGenerative', handled: false };
  }

  const command = findCommand(call.name);
  if (!command) return { messageKey: 'command.notUnderstood', handled: false };
  const args = coerceArgs(command, call.arguments ?? {});
  if (validate(command, args)) return { messageKey: 'command.notUnderstood', handled: false };

  const outcome = await command.run(args);
  // A small model sometimes invents an object id when nothing matches; the
  // honest answer then is that the thing was not found in the photo.
  if (outcome.messageKey === 'commands.noObject' && call.name === 'remove_object') {
    return { messageKey: 'command.needsGenerative', handled: false };
  }
  if (outcome.messageKey === 'commands.noObject') {
    return { messageKey: 'command.notUnderstood', handled: false };
  }
  return outcome;
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

