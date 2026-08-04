// src/store/activityStore.ts
import { create } from 'zustand';

export interface ActivityStep {
  /** Translation key, resolved when the panel renders. */
  labelKey: string;
  params?: Record<string, string | number>;
  startedAt: number;
  /** Real elapsed milliseconds, measured with a clock: never estimated. */
  durationMs?: number;
  status: 'running' | 'done' | 'failed';
}

export interface ActivityEntry {
  id: string;
  titleKey: string;
  startedAt: number;
  steps: ActivityStep[];
  /** Whatever the failing call threw, translated at render time. */
  error?: unknown;
}

interface ActivityState {
  entries: ActivityEntry[];
  begin: (titleKey: string) => string;
  /** Closes the running step and opens a new one. */
  step: (id: string, labelKey: string, params?: Record<string, string | number>) => void;
  /** Closes the operation, marking the step still in flight. */
  end: (id: string, status: 'done' | 'failed', error?: unknown) => void;
  clear: () => void;
}

const closeRunning = (steps: ActivityStep[], status: 'done' | 'failed'): ActivityStep[] =>
  steps.map((step) =>
    step.status === 'running'
      ? { ...step, status, durationMs: Date.now() - step.startedAt }
      : step
  );

/**
 * Record of what the application actually did.
 *
 * A log with invented timings would be worse than no log at all: every duration
 * here is the difference between two real instants.
 */
export const useActivityStore = create<ActivityState>((set) => ({
  entries: [],

  begin: (titleKey) => {
    const id = `act-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    set((state) => ({
      entries: [{ id, titleKey, startedAt: Date.now(), steps: [] }, ...state.entries].slice(0, 20),
    }));
    return id;
  },

  step: (id, labelKey, params) =>
    set((state) => ({
      entries: state.entries.map((entry) =>
        entry.id === id
          ? {
              ...entry,
              steps: [
                ...closeRunning(entry.steps, 'done'),
                { labelKey, params, startedAt: Date.now(), status: 'running' as const },
              ],
            }
          : entry
      ),
    })),

  end: (id, status, error) =>
    set((state) => ({
      entries: state.entries.map((entry) =>
        entry.id === id ? { ...entry, steps: closeRunning(entry.steps, status), error } : entry
      ),
    })),

  clear: () => set({ entries: [] }),
}));
