// src/store/modelStore.ts
import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type {
  ModelStatus,
  DownloadProgress,
  DownloadFinished,
  ModelTaskKind,
} from '../types/models';

interface ModelState {
  models: ModelStatus[];
  /** Progress keyed by model id; present only while downloading. */
  progress: Record<string, DownloadProgress>;
  /**
   * Failure per model, kept untranslated: the backend sends a key plus values,
   * and the interface renders it in the active language when it draws.
   */
  errors: Record<string, unknown>;
  loading: boolean;
  /** True when not running inside Tauri (for example `npm run dev` in a browser). */
  unavailable: boolean;

  refresh: () => Promise<void>;
  startDownload: (modelId: string) => Promise<void>;
  cancelDownload: (modelId: string) => Promise<void>;
  removeModel: (modelId: string) => Promise<void>;
  revealFolder: () => Promise<void>;
  subscribe: () => Promise<UnlistenFn>;

  isReady: (modelId: string) => boolean;
  readyFor: (task: ModelTaskKind) => ModelStatus | undefined;

  /** Label of the accelerator detected by the backend, e.g. "GPU (DirectML)". */
  accelerator: string;
  loadAccelerator: () => Promise<void>;
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  progress: {},
  errors: {},
  loading: false,
  unavailable: false,

  refresh: async () => {
    set({ loading: true });
    try {
      const models = await invoke<ModelStatus[]>('list_models');
      set({ models, loading: false, unavailable: false });
    } catch (err) {
      // Outside Tauri the command does not exist: say so rather than showing
      // an empty list with no explanation.
      console.warn('[Nabu] Model catalogue unavailable:', err);
      set({ loading: false, unavailable: true });
    }
  },

  startDownload: async (modelId) => {
    set((state) => {
      const errors = { ...state.errors };
      delete errors[modelId];
      return {
        errors,
        progress: {
          ...state.progress,
          [modelId]: {
            modelId,
            fileName: '',
            fileIndex: 0,
            fileCount: 1,
            downloadedBytes: 0,
            totalBytes: 0,
            percent: 0,
            bytesPerSecond: 0,
          },
        },
      };
    });

    try {
      await invoke('download_model', { modelId });
    } catch (err) {
      set((state) => {
        const progress = { ...state.progress };
        delete progress[modelId];
        return { progress, errors: { ...state.errors, [modelId]: err } };
      });
    }
    await get().refresh();
  },

  cancelDownload: async (modelId) => {
    try {
      await invoke('cancel_model_download', { modelId });
    } catch (err) {
      console.warn('[Nabu] Cancellation failed:', err);
    }
  },

  removeModel: async (modelId) => {
    try {
      await invoke('delete_model', { modelId });
    } catch (err) {
      set((state) => ({ errors: { ...state.errors, [modelId]: err } }));
    }
    await get().refresh();
  },

  revealFolder: async () => {
    try {
      await invoke('reveal_models_directory');
    } catch (err) {
      console.warn('[Nabu] Could not open the folder:', err);
    }
  },

  /** Subscribes to progress events. Returns the cleanup function. */
  subscribe: async () => {
    const unlistenProgress = await listen<DownloadProgress>(
      'model-download-progress',
      (event) => {
        set((state) => ({
          progress: { ...state.progress, [event.payload.modelId]: event.payload },
        }));
      }
    );

    const unlistenFinished = await listen<DownloadFinished>(
      'model-download-finished',
      (event) => {
        const { modelId, error } = event.payload;
        set((state) => {
          const progress = { ...state.progress };
          delete progress[modelId];
          const errors = { ...state.errors };
          if (error) errors[modelId] = error;
          else delete errors[modelId];
          return { progress, errors };
        });
        void get().refresh();
      }
    );

    return () => {
      unlistenProgress();
      unlistenFinished();
    };
  },

  isReady: (modelId) => get().models.some((m) => m.id === modelId && m.state === 'ready'),

  readyFor: (task) => get().models.find((m) => m.task === task && m.state === 'ready'),

  accelerator: '',

  loadAccelerator: async () => {
    try {
      set({ accelerator: await invoke<string>('accelerator_label') });
    } catch {
      // Not running inside Tauri: there is no accelerator to report.
      set({ accelerator: '' });
    }
  },
}));
