// src/store/documentStore.ts
import { create } from 'zustand';
import {
  NabuDocument,
  LayerNode,
  ToolType,
  ViewportTransform,
  BlendMode,
  ThemeMode,
  Adjustments,
  NEUTRAL_ADJUSTMENTS,
  Selection,
  RefineMode,
  BACKGROUND_LAYER_ID,
} from '../types/document';

const THEME_STORAGE_KEY = 'nabu.theme';

/** Reads the saved theme, falling back to the system preference. */
const readInitialTheme = (): ThemeMode => {
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {
    // localStorage unavailable: carry on with the default.
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
};

/**
 * Everything belonging to a single open document.
 *
 * Zoom, history and selection live here rather than in global state: switching
 * between tabs must leave each one exactly as you left it.
 */
interface DocumentSession {
  document: NabuDocument;
  viewport: ViewportTransform;
  history: NabuDocument[];
  historyIndex: number;
  selection: Selection | null;
}

interface DocumentState {
  sessions: Record<string, DocumentSession>;
  /** Tab order. */
  order: string[];
  activeId: string;

  // Appearance and tools, shared by every document
  theme: ThemeMode;
  activeTool: ToolType;
  brushSize: number;
  brushOpacity: number;
  brushColor: string;

  /** Active only after a cutout: brush that corrects the transparency. */
  refineMode: RefineMode | null;
  refineSize: number;
  /** On-screen thickness of the selection brush, in pixels. */
  selectionBrush: number;

  /** The opening bubble is shown once per document, until dismissed. */
  welcomeDismissed: boolean;
  dismissWelcome: () => void;

  // AI state
  isAIBusy: boolean;
  aiProgress: number;
  aiStatusMessage: string;

  fitRequestId: number;

  // Documents
  newDocument: (width?: number, height?: number) => void;
  openImageAsDocument: (
    name: string,
    imageDataUrl: string,
    imageWidth: number,
    imageHeight: number
  ) => void;
  closeDocument: (id: string) => void;
  setActiveDocument: (id: string) => void;

  // Viewport
  setViewport: (updater: (prev: ViewportTransform) => ViewportTransform) => void;
  resetViewport: () => void;
  requestFitToScreen: () => void;

  // Appearance and tools
  setTheme: (theme: ThemeMode) => void;
  toggleTheme: () => void;
  setActiveTool: (tool: ToolType) => void;
  setBrushSize: (size: number) => void;
  setBrushOpacity: (opacity: number) => void;
  setBrushColor: (color: string) => void;

  // Selection
  setSelection: (selection: Selection | null) => void;

  // Refinement
  setRefineMode: (mode: RefineMode | null) => void;
  setRefineSize: (size: number) => void;

  // Layers
  addLayer: (type?: LayerNode['type'], name?: string, imageDataUrl?: string) => void;
  addImageLayer: (name: string, imageDataUrl: string, width: number, height: number) => string;
  removeLayer: (layerId: string) => void;
  setActiveLayer: (layerId: string) => void;
  updateLayer: (layerId: string, partial: Partial<LayerNode>) => void;
  toggleLayerVisibility: (layerId: string) => void;
  toggleLayerLock: (layerId: string) => void;
  setLayerBlendMode: (layerId: string, blendMode: BlendMode) => void;
  /**
   * Moves one adjustment slider. Dragging must not fill the history with a
   * step per pixel of travel, so this does not record one; `commitAdjustments`
   * closes the gesture with a single undoable step.
   */
  setAdjustment: (layerId: string, key: keyof Adjustments, value: number) => void;
  commitAdjustments: (layerId: string) => void;
  resetAdjustments: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  reorderLayers: (sourceIndex: number, destIndex: number) => void;

  // AI
  setAIBusy: (busy: boolean, message?: string, progress?: number) => void;
  updateAIProgress: (progress: number, message?: string) => void;

  // History
  undo: () => void;
  redo: () => void;
}

let documentCounter = 0;

const createDocument = (
  width = 1920,
  height = 1080,
  title?: string,
  background: NabuDocument['background'] = 'white'
): NabuDocument => {
  documentCounter += 1;
  const now = new Date().toISOString();
  return {
    id: `doc-${Date.now()}-${documentCounter}`,
    title: title || `Senza Titolo ${documentCounter}`,
    width,
    height,
    dpi: 300,
    colorSpace: 'srgb',
    background,
    rootLayerIds: [BACKGROUND_LAYER_ID],
    layers: {
      [BACKGROUND_LAYER_ID]: {
        id: BACKGROUND_LAYER_ID,
        name: 'Sfondo',
        type: 'raster',
        visible: true,
        locked: false,
        opacity: 1.0,
        blendMode: 'normal',
        transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, skewX: 0, skewY: 0 },
        width,
        height,
      },
    },
    activeLayerId: BACKGROUND_LAYER_ID,
    selectedLayerIds: [BACKGROUND_LAYER_ID],
    createdAt: now,
    updatedAt: now,
  };
};

const createSession = (document: NabuDocument): DocumentSession => ({
  document,
  viewport: { zoom: 1.0, panX: 0, panY: 0 },
  history: [document],
  historyIndex: 0,
  selection: null,
});

/** Maximum number of undo steps kept per document. */
const HISTORY_LIMIT = 30;

export const useDocumentStore = create<DocumentState>((set, get) => {
  /** Applies a change to the active session. */
  const patchSession = (
    updater: (session: DocumentSession) => Partial<DocumentSession>
  ) =>
    set((state) => {
      const current = state.sessions[state.activeId];
      if (!current) return state;
      return {
        sessions: {
          ...state.sessions,
          [state.activeId]: { ...current, ...updater(current) },
        },
      };
    });

  /**
   * Replaces the active document while recording a history step.
   * Every structural change goes through here, so undo/redo stays consistent
   * without each action having to remember it.
   */
  const commit = (
    mutate: (doc: NabuDocument) => NabuDocument,
    options: { history?: boolean } = {}
  ) =>
    patchSession((session) => {
      const next = { ...mutate(session.document), updatedAt: new Date().toISOString() };
      if (options.history === false) {
        return { document: next };
      }

      const history = session.history.slice(0, session.historyIndex + 1);
      history.push(next);
      if (history.length > HISTORY_LIMIT) history.shift();

      return { document: next, history, historyIndex: history.length - 1 };
    });

  const activeSession = () => get().sessions[get().activeId];

  return {
    // No document at startup: no blank canvas, just the invitation to open one.
    sessions: {},
    order: [],
    activeId: '',

    theme: readInitialTheme(),
    activeTool: 'select',
    brushSize: 20,
    brushOpacity: 1.0,
    brushColor: '#00f0ff',

    refineMode: null,
    refineSize: 40,
    selectionBrush: 26,

    welcomeDismissed: false,
    dismissWelcome: () => set({ welcomeDismissed: true }),

    isAIBusy: false,
    aiProgress: 0,
    aiStatusMessage: '',

    fitRequestId: 0,

    // --- Documenti -------------------------------------------------------

    newDocument: (width = 1920, height = 1080) => {
      const doc = createDocument(width, height);
      set((state) => ({
        sessions: { ...state.sessions, [doc.id]: createSession(doc) },
        order: [...state.order, doc.id],
        activeId: doc.id,
        refineMode: null,
      }));
      get().requestFitToScreen();
    },

    /** Every opened image becomes its own document, not another layer. */
    openImageAsDocument: (name, imageDataUrl, imageWidth, imageHeight) => {
      const doc = createDocument(imageWidth, imageHeight, name, 'transparent');
      const layerId = `layer-${Date.now()}`;

      const layer: LayerNode = {
        id: layerId,
        name: name || 'Immagine',
        type: 'raster',
        visible: true,
        locked: false,
        opacity: 1.0,
        blendMode: 'normal',
        transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, skewX: 0, skewY: 0 },
        imageDataUrl,
        width: imageWidth,
        height: imageHeight,
      };

      // The image is the document's content: it replaces the empty background.
      const withImage: NabuDocument = {
        ...doc,
        rootLayerIds: [layerId],
        layers: { [layerId]: layer },
        activeLayerId: layerId,
        selectedLayerIds: [layerId],
      };

      set((state) => ({
        sessions: { ...state.sessions, [withImage.id]: createSession(withImage) },
        order: [...state.order, withImage.id],
        activeId: withImage.id,
        refineMode: null,
        welcomeDismissed: false,
      }));
      get().requestFitToScreen();
    },

    closeDocument: (id) => {
      const { order, sessions, activeId } = get();
      if (!sessions[id]) return;

      const newOrder = order.filter((docId) => docId !== id);
      const newSessions = { ...sessions };
      delete newSessions[id];

      // Closing the active tab moves to the previous one.
      let newActive = activeId;
      if (activeId === id) {
        const removedAt = order.indexOf(id);
        newActive = newOrder[Math.max(0, removedAt - 1)] ?? '';
      }

      set({ sessions: newSessions, order: newOrder, activeId: newActive, refineMode: null });
      if (activeId === id) get().requestFitToScreen();
    },

    setActiveDocument: (id) => {
      if (!get().sessions[id]) return;
      set({ activeId: id, refineMode: null, welcomeDismissed: false });
    },

    // --- Viewport --------------------------------------------------------

    setViewport: (updater) =>
      patchSession((session) => ({ viewport: updater(session.viewport) })),

    resetViewport: () => patchSession(() => ({ viewport: { zoom: 1.0, panX: 0, panY: 0 } })),

    requestFitToScreen: () => set((state) => ({ fitRequestId: state.fitRequestId + 1 })),

    // --- Aspetto e strumenti --------------------------------------------

    setTheme: (theme) => {
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, theme);
      } catch {
        // Preference not persistable: it still applies for this session.
      }
      set({ theme });
    },

    toggleTheme: () => get().setTheme(get().theme === 'dark' ? 'light' : 'dark'),

    setActiveTool: (tool) => set({ activeTool: tool }),
    setBrushSize: (size) => set({ brushSize: Math.max(1, Math.min(500, size)) }),
    setBrushOpacity: (opacity) => set({ brushOpacity: Math.max(0, Math.min(1, opacity)) }),
    setBrushColor: (color) => set({ brushColor: color }),

    // --- Selezione -------------------------------------------------------

    setSelection: (selection) => patchSession(() => ({ selection })),

    // --- Rifinitura ------------------------------------------------------

    setRefineMode: (mode) => set({ refineMode: mode }),
    setRefineSize: (size) => set({ refineSize: Math.max(4, Math.min(400, size)) }),

    // --- Livelli ---------------------------------------------------------

    addLayer: (type = 'raster', name, imageDataUrl) => {
      const doc = activeSession()?.document;
      if (!doc) return;
      const id = `layer-${Date.now()}`;

      commit((current) => ({
        ...current,
        rootLayerIds: [id, ...current.rootLayerIds],
        layers: {
          ...current.layers,
          [id]: {
            id,
            name: name || `Livello ${Object.keys(current.layers).length + 1}`,
            type,
            visible: true,
            locked: false,
            opacity: 1.0,
            blendMode: 'normal',
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, skewX: 0, skewY: 0 },
            imageDataUrl,
            width: current.width,
            height: current.height,
          },
        },
        activeLayerId: id,
        selectedLayerIds: [id],
      }));
    },

    /** Like addLayer but with its own dimensions; returns the new id. */
    addImageLayer: (name, imageDataUrl, width, height) => {
      const id = `layer-${Date.now()}`;
      commit((current) => ({
        ...current,
        rootLayerIds: [id, ...current.rootLayerIds],
        layers: {
          ...current.layers,
          [id]: {
            id,
            name,
            type: 'raster',
            visible: true,
            locked: false,
            opacity: 1.0,
            blendMode: 'normal',
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, skewX: 0, skewY: 0 },
            imageDataUrl,
            width,
            height,
          },
        },
        activeLayerId: id,
        selectedLayerIds: [id],
      }));
      return id;
    },

    removeLayer: (layerId) => {
      const doc = activeSession()?.document;
      if (!doc || doc.rootLayerIds.length <= 1) return;

      commit((current) => {
        const rootLayerIds = current.rootLayerIds.filter((id) => id !== layerId);
        const layers = { ...current.layers };
        delete layers[layerId];
        const nextActive = rootLayerIds[0] ?? null;
        return {
          ...current,
          rootLayerIds,
          layers,
          activeLayerId: nextActive,
          selectedLayerIds: nextActive ? [nextActive] : [],
        };
      });
    },

    setActiveLayer: (layerId) =>
      commit(
        (current) => ({ ...current, activeLayerId: layerId, selectedLayerIds: [layerId] }),
        { history: false }
      ),

    updateLayer: (layerId, partial) => {
      const doc = activeSession()?.document;
      if (!doc?.layers[layerId]) return;
      commit((current) => ({
        ...current,
        layers: { ...current.layers, [layerId]: { ...current.layers[layerId], ...partial } },
      }));
    },

    toggleLayerVisibility: (layerId) => {
      const layer = activeSession()?.document.layers[layerId];
      if (!layer) return;
      get().updateLayer(layerId, { visible: !layer.visible });
    },

    toggleLayerLock: (layerId) => {
      const layer = activeSession()?.document.layers[layerId];
      if (!layer) return;
      get().updateLayer(layerId, { locked: !layer.locked });
    },

    setLayerBlendMode: (layerId, blendMode) => get().updateLayer(layerId, { blendMode }),

    setAdjustment: (layerId, key, value) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer) return current;
          const adjustments = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments, [key]: value };
          return {
            ...current,
            layers: { ...current.layers, [layerId]: { ...layer, adjustments } },
          };
        },
        { history: false }
      ),

    commitAdjustments: (layerId) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        // Re-writing the same values is enough: `commit` records the step.
        return {
          ...current,
          layers: { ...current.layers, [layerId]: { ...layer } },
        };
      }),

    resetAdjustments: (layerId) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        delete next.adjustments;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),
    setLayerOpacity: (layerId, opacity) => get().updateLayer(layerId, { opacity }),

    reorderLayers: (sourceIndex, destIndex) =>
      commit((current) => {
        const rootLayerIds = [...current.rootLayerIds];
        const [moved] = rootLayerIds.splice(sourceIndex, 1);
        rootLayerIds.splice(destIndex, 0, moved);
        return { ...current, rootLayerIds };
      }),

    // --- AI --------------------------------------------------------------

    setAIBusy: (busy, message = '', progress = 0) =>
      set({ isAIBusy: busy, aiStatusMessage: message, aiProgress: progress }),

    updateAIProgress: (progress, message) =>
      set((state) => ({
        aiProgress: progress,
        aiStatusMessage: message ?? state.aiStatusMessage,
      })),

    // --- Cronologia ------------------------------------------------------

    undo: () =>
      patchSession((session) =>
        session.historyIndex > 0
          ? {
              document: session.history[session.historyIndex - 1],
              historyIndex: session.historyIndex - 1,
            }
          : {}
      ),

    redo: () =>
      patchSession((session) =>
        session.historyIndex < session.history.length - 1
          ? {
              document: session.history[session.historyIndex + 1],
              historyIndex: session.historyIndex + 1,
            }
          : {}
      ),
  };
});

// --- Selectors ---------------------------------------------------------------
// Components read from here instead of digging into `sessions` themselves.

export const useHasDocuments = () => useDocumentStore((s) => s.order.length > 0);

export const useActiveDocument = () =>
  useDocumentStore((s) => s.sessions[s.activeId]?.document ?? null);

export const useViewport = () =>
  useDocumentStore((s) => s.sessions[s.activeId]?.viewport ?? null);

export const useSelection = () =>
  useDocumentStore((s) => s.sessions[s.activeId]?.selection ?? null);

export const useCanUndo = () =>
  useDocumentStore((s) => (s.sessions[s.activeId]?.historyIndex ?? 0) > 0);

export const useCanRedo = () =>
  useDocumentStore((s) => {
    const session = s.sessions[s.activeId];
    return session ? session.historyIndex < session.history.length - 1 : false;
  });
