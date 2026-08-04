// src/i18n/index.ts
//
// Translation catalogue.
//
// A language is just a JSON file of flat `key: text` pairs. There are two ways
// to add one, and neither requires touching this file:
//
//   1. Drop `<code>.json` into `src/locales/` and rebuild — it is picked up by
//      the glob below, so contributors only ever send a translation file.
//   2. Drop `<code>.json` into `<app data>/languages/` — the desktop app reads
//      that folder at startup, so a translator needs no toolchain at all.
//
// English is the base language: any key a translation omits falls back to it,
// so a partial translation degrades gracefully instead of showing blanks.

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

export interface LanguageMeta {
  code: string;
  name: string;
  credit?: string;
}

export type Catalogue = Record<string, string>;

export interface Language {
  meta: LanguageMeta;
  strings: Catalogue;
  /** True for languages loaded from the user folder rather than bundled. */
  external: boolean;
}

const STORAGE_KEY = 'nabu.language';
export const BASE_LANGUAGE = 'en';

/** Splits the `$meta` header from the translated strings. */
function parseLanguage(raw: unknown, fallbackCode: string, external: boolean): Language | null {
  if (!raw || typeof raw !== 'object') return null;

  const record = raw as Record<string, unknown>;
  const meta = (record.$meta ?? {}) as Partial<LanguageMeta>;

  const strings: Catalogue = {};
  for (const [key, value] of Object.entries(record)) {
    if (key === '$meta') continue;
    if (typeof value === 'string') strings[key] = value;
  }

  const code = meta.code ?? fallbackCode;
  return {
    meta: { code, name: meta.name ?? code, credit: meta.credit },
    strings,
    external,
  };
}

/** Languages compiled into the application. */
function loadBundled(): Record<string, Language> {
  const modules = import.meta.glob('../locales/*.json', { eager: true }) as Record<
    string,
    { default: unknown }
  >;

  const languages: Record<string, Language> = {};
  for (const [path, module] of Object.entries(modules)) {
    const fallbackCode = path.split('/').pop()?.replace('.json', '') ?? 'unknown';
    const language = parseLanguage(module.default, fallbackCode, false);
    if (language) languages[language.meta.code] = language;
  }
  return languages;
}

/** Picks the best starting language: saved choice, then system, then English. */
function detectInitial(available: Record<string, Language>): string {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved && available[saved]) return saved;
  } catch {
    // Storage unavailable; fall through to detection.
  }

  for (const tag of navigator.languages ?? [navigator.language]) {
    if (!tag) continue;
    // "it-CH" should match the "it" catalogue.
    const primary = tag.toLowerCase().split('-')[0];
    if (available[tag]) return tag;
    if (available[primary]) return primary;
  }
  return BASE_LANGUAGE;
}

interface I18nState {
  languages: Record<string, Language>;
  current: string;
  setLanguage: (code: string) => void;
  /** Loads translations dropped into the user folder. */
  loadExternal: () => Promise<void>;
}

const bundled = loadBundled();

export const useI18nStore = create<I18nState>((set, get) => ({
  languages: bundled,
  current: detectInitial(bundled),

  setLanguage: (code) => {
    if (!get().languages[code]) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, code);
    } catch {
      // Preference not persisted; it still applies for this session.
    }
    set({ current: code });
  },

  loadExternal: async () => {
    try {
      const files = await invoke<Array<{ code: string; content: string }>>('list_languages');
      const extra: Record<string, Language> = {};

      for (const file of files) {
        try {
          const language = parseLanguage(JSON.parse(file.content), file.code, true);
          if (language) extra[language.meta.code] = language;
        } catch {
          console.warn(`[Nabu] Translation "${file.code}" is not valid JSON and was skipped.`);
        }
      }

      // User files win over bundled ones, so a translation can be corrected
      // without rebuilding the application.
      set((state) => ({ languages: { ...state.languages, ...extra } }));
    } catch {
      // Not running inside Tauri: only bundled languages are available.
    }
  },
}));

/** Replaces `{name}` placeholders with the given values. */
function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in params ? String(params[name]) : match
  );
}

/**
 * Resolves a key against the active language, falling back to English and
 * finally to the key itself, which makes a missing string obvious rather than
 * silently empty.
 */
export function translate(
  languages: Record<string, Language>,
  current: string,
  key: string,
  params?: Record<string, string | number>
): string {
  const template =
    languages[current]?.strings[key] ?? languages[BASE_LANGUAGE]?.strings[key] ?? key;
  return interpolate(template, params);
}

/** Hook used across the interface: `const t = useT(); t('rail.open')`. */
export function useT() {
  const languages = useI18nStore((s) => s.languages);
  const current = useI18nStore((s) => s.current);
  return (key: string, params?: Record<string, string | number>) =>
    translate(languages, current, key, params);
}

/** Translates outside React, for code that is not a component. */
export function t(key: string, params?: Record<string, string | number>): string {
  const { languages, current } = useI18nStore.getState();
  return translate(languages, current, key, params);
}
