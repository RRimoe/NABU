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
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
  Curves,
  CurveChannel,
  CurvePoint,
  NEUTRAL_CURVES,
  FocusEffect,
  LocalAdjustment,
  areCurvesNeutral,
  isFocusNeutral,
  isLocalNeutral,
  Selection,
  SelectionShape,
  RefineMode,
  RefinePoint,
  BACKGROUND_LAYER_ID,
} from '../types/document';

const THEME_STORAGE_KEY = 'nabu.theme';

/**
 * The part of a layer that is "how it was developed" rather than what it shows:
 * what copy and paste of settings carries from one photo to another.
 */
export interface DevelopSettings {
  adjustments?: Adjustments;
  curves?: Curves;
  filter?: LayerNode['filter'];
}

/** Sections the right-hand dock can show, one at a time. */
export type DockPanel = 'suggestions' | 'adjust' | 'filters';

/** A piece of picture waiting to be pasted, in its own pixels. */
export interface ClipboardItem {
  name: string;
  imageDataUrl: string;
  width: number;
  height: number;
  /** Document it was taken from, so the offer to paste can appear elsewhere. */
  sourceId: string;
}

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
  /**
   * The document as it was opened, for before/after. Kept apart from the
   * history, whose oldest steps are dropped once it fills up.
   */
  original: NabuDocument;
  selection: Selection | null;
  /** Clicks correcting the selection; the cut below is derived from them. */
  refinePoints: RefinePoint[];
  /** The corrected cut, as the source carrying its mask in the alpha channel. */
  refineCutDataUrl: string | null;
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

  /**
   * Before/after: `hold` shows the original while a key is held, `split` shows
   * it left of a draggable divider at `compareSplit` (0..1 of the width).
   */
  compare: 'off' | 'hold' | 'split';
  compareSplit: number;
  setCompare: (mode: 'off' | 'hold' | 'split') => void;
  setCompareSplit: (split: number) => void;

  // AI state
  isAIBusy: boolean;
  aiProgress: number;
  aiStatusMessage: string;

  fitRequestId: number;

  /**
   * Size of the canvas area on screen, in CSS pixels.
   *
   * Published by the viewport because the navigator needs it: the rectangle it
   * draws is "how much of the photo currently fits", which cannot be worked out
   * from the zoom alone.
   */
  viewportSize: { width: number; height: number };
  setViewportSize: (width: number, height: number) => void;

  // Documents
  newDocument: (width?: number, height?: number) => void;
  openImageAsDocument: (
    name: string,
    imageDataUrl: string,
    imageWidth: number,
    imageHeight: number
  ) => void;
  /** Opens a saved project as a new tab, with its layers and edits live. */
  openProjectDocument: (doc: NabuDocument) => void;
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

  /** Click-to-correct on the current selection. */
  pointRefine: boolean;
  setPointRefine: (on: boolean) => void;
  addRefinePoint: (point: RefinePoint) => void;
  undoRefinePoint: () => void;
  clearRefinePoints: () => void;
  setRefineCut: (dataUrl: string | null) => void;
  /** Shape the next drag traces: freehand, rectangle or ellipse. */
  selectionShape: SelectionShape;
  setSelectionShape: (shape: SelectionShape) => void;

  // Refinement
  setRefineMode: (mode: RefineMode | null) => void;
  setRefineSize: (size: number) => void;

  /**
   * Which section the right-hand dock is showing.
   *
   * The rail chooses it; suggestions are what the dock falls back to, because
   * with no prompt to type they are the app's opening move.
   */
  dockPanel: DockPanel;
  setDockPanel: (panel: DockPanel) => void;

  /**
   * What was last copied, shared by every open photo.
   *
   * It lives outside the sessions on purpose: copying from one picture and
   * pasting into another is the whole point, and a per-document clipboard
   * would make that impossible.
   */
  clipboard: ClipboardItem | null;
  setClipboard: (item: ClipboardItem | null) => void;
  /** Settings copied from a layer, waiting to be pasted onto another. */
  developClipboard: DevelopSettings | null;
  setDevelopClipboard: (settings: DevelopSettings | null) => void;

  // Layers
  addLayer: (type?: LayerNode['type'], name?: string, imageDataUrl?: string) => void;
  /** Copies a layer, pixels included, and selects the copy. */
  duplicateLayer: (layerId: string) => void;
  /** Inserts a raster layer directly underneath another. Returns its id. */
  addLayerBelow: (
    siblingId: string,
    name: string,
    imageDataUrl: string,
    width: number,
    height: number
  ) => string;
  addImageLayer: (name: string, imageDataUrl: string, width: number, height: number) => string;
  removeLayer: (layerId: string) => void;
  setActiveLayer: (layerId: string) => void;
  updateLayer: (layerId: string, partial: Partial<LayerNode>) => void;
  toggleLayerVisibility: (layerId: string) => void;
  toggleLayerLock: (layerId: string) => void;
  setLayerBlendMode: (layerId: string, blendMode: BlendMode) => void;
  /**
   * Replaces the document and every layer at once, for a resample.
   * A single call so the whole resize is one undoable step.
   */
  applyResize: (
    width: number,
    height: number,
    layers: Record<string, Partial<LayerNode>>
  ) => void;
  /** Moves, scales or rotates a layer without touching its pixels. */
  setLayerTransform: (layerId: string, partial: Partial<LayerNode['transform']>) => void;
  /** Closes a move, scale or rotate gesture with one history step. */
  commitTransform: (layerId: string) => void;
  /** Puts a layer back where it started, at its original size and angle. */
  resetTransform: (layerId: string) => void;
  /**
   * Moves one adjustment slider. Dragging must not fill the history with a
   * step per pixel of travel, so this does not record one; `commitAdjustments`
   * closes the gesture with a single undoable step.
   */
  setAdjustment: (layerId: string, key: keyof Adjustments, value: number) => void;
  commitAdjustments: (layerId: string) => void;
  resetAdjustments: (layerId: string) => void;
  /**
   * Adds a set of deltas to the sliders in one undoable step.
   * This is how a suggestion is applied: it nudges what is there rather than
   * replacing it, so a hand correction and a proposal can coexist.
   */
  applyAdjustmentDelta: (layerId: string, delta: Partial<Adjustments>) => void;
  /**
   * Rewrites one channel of the tone curve. Like the sliders, dragging a
   * control point records nothing; `commitCurves` closes the gesture.
   */
  setCurve: (layerId: string, channel: CurveChannel, points: CurvePoint[]) => void;
  commitCurves: (layerId: string) => void;
  /** Straightens one channel, or every channel when none is named. */
  resetCurves: (layerId: string, channel?: CurveChannel) => void;
  /**
   * Rewrites the selective focus. Dragging a band records nothing; the gesture
   * is closed with `commitFocus`, like the sliders and the curves.
   */
  setFocus: (layerId: string, focus: FocusEffect, options?: { history?: boolean }) => void;
  commitFocus: (layerId: string) => void;
  clearFocus: (layerId: string) => void;
  /**
   * Applies a compound edit in one undoable step.
   *
   * A recipe is several operations that only make sense together, so it has to
   * come back in one piece too: four undos to reverse one button is worse than
   * not having the button.
   */
  applyRecipe: (
    layerId: string,
    patch: { delta?: Partial<Adjustments>; focus?: FocusEffect; local?: LocalAdjustment }
  ) => void;
  /**
   * True while the sharp zone is being placed on the photo.
   *
   * The guide is a mode with a beginning and an end, like the crop frame: it
   * appears when the focus is being worked on and goes away when attention
   * moves elsewhere. Tying it to whether the effect is *active* instead left
   * two lines drawn across the photograph for as long as the blur existed.
   */
  focusEditing: boolean;
  setFocusEditing: (editing: boolean) => void;
  /** Confines a set of corrections to a region. */
  setLocalAdjustment: (layerId: string, local: LocalAdjustment) => void;
  /** Strength of the region, 0..1. No history until committed. */
  setLocalAmount: (layerId: string, amount: number) => void;
  commitLocal: (layerId: string) => void;
  clearLocal: (layerId: string) => void;
  /** Chooses a look, or clears it when the id is null. */
  /** `adjustments` carries the values of a look the user saved themselves. */
  setLayerFilter: (
    layerId: string,
    filterId: string | null,
    adjustments?: Partial<Adjustments>
  ) => void;
  /** Replaces a layer's sliders, curves and look in one step, as pasting does. */
  pasteDevelop: (layerId: string, develop: DevelopSettings) => void;
  /** Strength of the chosen look, 0..1. No history until committed. */
  setFilterAmount: (layerId: string, amount: number) => void;
  commitFilter: (layerId: string) => void;
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
  original: document,
  selection: null,
  refinePoints: [],
  refineCutDataUrl: null,
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

  /**
   * Closes a gesture, recording a step only if something actually changed.
   *
   * Every gesture ends by committing: the pointer goes up, the slider is
   * released. Most of those gestures moved something, but plenty did not — a
   * click on a handle without a drag, a slider nudged and put back, a tool
   * selected and abandoned. Recording those leaves the history holding two
   * identical documents in a row, and the first undo then appears to do
   * nothing at all, which is exactly what it looks like when undo is broken.
   *
   * `signature` decides what "changed" means for this particular gesture, so
   * the comparison stays cheap: a move compares one transform, not a document
   * carrying several megabytes of mask.
   */
  const commitIfChanged = (
    signature: (doc: NabuDocument) => string,
    mutate: (doc: NabuDocument) => NabuDocument
  ) =>
    patchSession((session) => {
      const next = { ...mutate(session.document), updatedAt: new Date().toISOString() };
      const previous = session.history[session.historyIndex];

      if (previous && signature(previous) === signature(next)) {
        return { document: next };
      }

      const history = session.history.slice(0, session.historyIndex + 1);
      history.push(next);
      if (history.length > HISTORY_LIMIT) history.shift();

      return { document: next, history, historyIndex: history.length - 1 };
    });

  /** Signatures for the gestures that can end without having done anything. */
  const transformSignature = (layerId: string) => (doc: NabuDocument) => {
    const t = doc.layers[layerId]?.transform;
    return t ? `${t.x},${t.y},${t.scaleX},${t.scaleY},${t.rotation}` : '';
  };

  const adjustmentSignature = (layerId: string) => (doc: NabuDocument) => {
    const layer = doc.layers[layerId];
    return JSON.stringify([layer?.adjustments ?? null, layer?.filter ?? null]);
  };

  const curveSignature = (layerId: string) => (doc: NabuDocument) =>
    JSON.stringify(doc.layers[layerId]?.curves ?? null);

  /** The mask is compared by length: two different masks never share one. */
  const focusSignature = (layerId: string) => (doc: NabuDocument) => {
    const focus = doc.layers[layerId]?.focus;
    if (!focus) return '';
    const { maskDataUrl, ...rest } = focus;
    return `${JSON.stringify(rest)}|${maskDataUrl?.length ?? 0}`;
  };

  const localSignature = (layerId: string) => (doc: NabuDocument) => {
    const local = doc.layers[layerId]?.local;
    if (!local) return '';
    return `${JSON.stringify(local.adjustments)}|${local.amount}|${local.maskDataUrl.length}`;
  };

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
    brushColor: '#ffffff',

    refineMode: null,
    refineSize: 40,
    selectionBrush: 26,

    welcomeDismissed: false,
    dismissWelcome: () => set({ welcomeDismissed: true }),

    compare: 'off',
    compareSplit: 0.5,
    setCompare: (mode) => set({ compare: mode }),
    setCompareSplit: (split) => set({ compareSplit: Math.min(0.98, Math.max(0.02, split)) }),

    isAIBusy: false,
    aiProgress: 0,
    aiStatusMessage: '',

    fitRequestId: 0,

    viewportSize: { width: 0, height: 0 },
    setViewportSize: (width, height) =>
      set((state) =>
        state.viewportSize.width === width && state.viewportSize.height === height
          ? state
          : { viewportSize: { width, height } }
      ),

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

    openProjectDocument: (saved) => {
      // A fresh id: the same project can be opened twice, and two tabs sharing
      // an id would share a history.
      documentCounter += 1;
      const doc: NabuDocument = { ...saved, id: `doc-${Date.now()}-${documentCounter}` };
      set((state) => ({
        sessions: { ...state.sessions, [doc.id]: createSession(doc) },
        order: [...state.order, doc.id],
        activeId: doc.id,
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
      // Both of these describe work on the photo being left behind.
      set({ activeId: id, refineMode: null, focusEditing: false, welcomeDismissed: false });
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

    // A new selection is a new question, so the corrections to the old one go
    // with it rather than quietly applying to something else.
    setSelection: (selection) =>
      patchSession(() => ({ selection, refinePoints: [], refineCutDataUrl: null })),

    pointRefine: false,
    setPointRefine: (on) => {
      set({ pointRefine: on });
      if (!on) patchSession(() => ({ refinePoints: [], refineCutDataUrl: null }));
    },
    addRefinePoint: (point) =>
      patchSession((session) => ({ refinePoints: [...session.refinePoints, point] })),
    undoRefinePoint: () =>
      patchSession((session) => ({ refinePoints: session.refinePoints.slice(0, -1) })),
    clearRefinePoints: () => patchSession(() => ({ refinePoints: [], refineCutDataUrl: null })),
    setRefineCut: (dataUrl) => patchSession(() => ({ refineCutDataUrl: dataUrl })),

    selectionShape: 'free',
    setSelectionShape: (shape) => set({ selectionShape: shape }),

    // --- Rifinitura ------------------------------------------------------

    setRefineMode: (mode) => set({ refineMode: mode }),
    setRefineSize: (size) => set({ refineSize: Math.max(4, Math.min(400, size)) }),

    dockPanel: 'suggestions',
    setDockPanel: (panel) => set({ dockPanel: panel }),

    clipboard: null,
    setClipboard: (item) => set({ clipboard: item }),
    developClipboard: null,
    setDevelopClipboard: (settings) => set({ developClipboard: settings }),

    // --- Livelli ---------------------------------------------------------

    addLayerBelow: (siblingId, name, imageDataUrl, width, height) => {
      const id = `layer-${Date.now()}`;
      commit((current) => {
        const index = current.rootLayerIds.indexOf(siblingId);
        const rootLayerIds = [...current.rootLayerIds];
        // Later in the array means lower in the stack, so "below" is after.
        rootLayerIds.splice(index < 0 ? rootLayerIds.length : index + 1, 0, id);

        return {
          ...current,
          rootLayerIds,
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
        };
      });
      return id;
    },

    duplicateLayer: (layerId) => {
      const doc = activeSession()?.document;
      const source = doc?.layers[layerId];
      if (!doc || !source) return;

      const id = `layer-${Date.now()}`;
      commit((current) => ({
        ...current,
        // Directly above the original, where a copy belongs.
        rootLayerIds: current.rootLayerIds.flatMap((existing) =>
          existing === layerId ? [id, existing] : [existing]
        ),
        layers: {
          ...current.layers,
          [id]: {
            ...source,
            id,
            name: `${source.name} 2`,
            // A copy is its own thing: it must not inherit the link a cutout
            // keeps to the layer it was cut from.
            sourceLayerId: undefined,
          },
        },
        activeLayerId: id,
        selectedLayerIds: [id],
      }));
    },

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

    applyResize: (width, height, layers) =>
      commit((current) => {
        const next = { ...current.layers };
        for (const [id, partial] of Object.entries(layers)) {
          if (next[id]) next[id] = { ...next[id], ...partial };
        }
        return { ...current, width, height, layers: next };
      }),

    setLayerTransform: (layerId, partial) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer) return current;
          return {
            ...current,
            layers: {
              ...current.layers,
              [layerId]: { ...layer, transform: { ...layer.transform, ...partial } },
            },
          };
        },
        // Dragging a handle must not record a step per pixel of travel.
        { history: false }
      ),

    commitTransform: (layerId) =>
      commitIfChanged(transformSignature(layerId), (current) => {
        const layer = current.layers[layerId];
        return layer
          ? { ...current, layers: { ...current.layers, [layerId]: { ...layer } } }
          : current;
      }),

    resetTransform: (layerId) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        return {
          ...current,
          layers: {
            ...current.layers,
            [layerId]: {
              ...layer,
              transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, skewX: 0, skewY: 0 },
            },
          },
        };
      }),

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
      commitIfChanged(adjustmentSignature(layerId), (current) => {
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

    applyAdjustmentDelta: (layerId, delta) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;

        const adjustments = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };
        for (const [key, value] of Object.entries(delta) as Array<
          [keyof Adjustments, number]
        >) {
          const [min, max] = ADJUSTMENT_RANGES[key];
          adjustments[key] = Math.min(max, Math.max(min, adjustments[key] + value));
        }

        return {
          ...current,
          layers: { ...current.layers, [layerId]: { ...layer, adjustments } },
        };
      }),

    setCurve: (layerId, channel, points) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer) return current;
          const curves: Curves = { ...NEUTRAL_CURVES, ...layer.curves, [channel]: points };
          return {
            ...current,
            layers: { ...current.layers, [layerId]: { ...layer, curves } },
          };
        },
        { history: false }
      ),

    commitCurves: (layerId) =>
      commitIfChanged(curveSignature(layerId), (current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        // A curve dragged back to the diagonal is no curve at all: dropping it
        // keeps the renderer's "is there anything to do" check honest.
        if (areCurvesNeutral(next.curves)) delete next.curves;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    resetCurves: (layerId, channel) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };

        if (!channel) {
          delete next.curves;
        } else {
          const curves: Curves = { ...NEUTRAL_CURVES, ...layer.curves, [channel]: NEUTRAL_CURVES[channel] };
          if (areCurvesNeutral(curves)) delete next.curves;
          else next.curves = curves;
        }

        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    setFocus: (layerId, focus, options = {}) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer) return current;
          return {
            ...current,
            layers: { ...current.layers, [layerId]: { ...layer, focus } },
          };
        },
        // Dragging the band across the frame must not record a step per pixel.
        { history: options.history ?? false }
      ),

    commitFocus: (layerId) =>
      commitIfChanged(focusSignature(layerId), (current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        // Switched off means gone: the renderer's "is there anything to do"
        // check should not have to look past a disabled effect.
        if (isFocusNeutral(next.focus)) delete next.focus;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    applyRecipe: (layerId, patch) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;

        const next = { ...layer };

        if (patch.delta) {
          const adjustments = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };
          for (const [key, value] of Object.entries(patch.delta) as Array<
            [keyof Adjustments, number]
          >) {
            const [min, max] = ADJUSTMENT_RANGES[key];
            adjustments[key] = Math.min(max, Math.max(min, adjustments[key] + value));
          }
          next.adjustments = adjustments;
        }

        if (patch.focus) next.focus = patch.focus;
        if (patch.local) next.local = patch.local;

        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    focusEditing: false,
    setFocusEditing: (editing) =>
      set((state) => (state.focusEditing === editing ? state : { focusEditing: editing })),

    setLocalAdjustment: (layerId, local) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        return {
          ...current,
          layers: { ...current.layers, [layerId]: { ...layer, local } },
        };
      }),

    setLocalAmount: (layerId, amount) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer?.local) return current;
          return {
            ...current,
            layers: {
              ...current.layers,
              [layerId]: {
                ...layer,
                local: { ...layer.local, amount: Math.max(0, Math.min(1, amount)) },
              },
            },
          };
        },
        { history: false }
      ),

    commitLocal: (layerId) =>
      commitIfChanged(localSignature(layerId), (current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        if (isLocalNeutral(next.local)) delete next.local;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    clearLocal: (layerId) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        delete next.local;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    clearFocus: (layerId) => {
      // Nothing left to place, so the guide has nothing to describe.
      set({ focusEditing: false });
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        delete next.focus;
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      });
    },

    setLayerFilter: (layerId, filterId, adjustments) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next = { ...layer };
        if (!filterId) {
          delete next.filter;
        } else {
          // Re-picking the same look keeps its strength; a new one starts full,
          // because that is what tapping a look is asking for.
          const amount = layer.filter?.id === filterId ? layer.filter.amount : 1;
          next.filter = adjustments
            ? { id: filterId, amount, adjustments: { ...adjustments } }
            : { id: filterId, amount };
        }
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    pasteDevelop: (layerId, develop) =>
      commit((current) => {
        const layer = current.layers[layerId];
        if (!layer) return current;
        const next: LayerNode = { ...layer };
        delete next.adjustments;
        delete next.curves;
        delete next.filter;
        if (develop.adjustments) next.adjustments = { ...develop.adjustments };
        if (develop.curves) next.curves = structuredClone(develop.curves);
        if (develop.filter) next.filter = structuredClone(develop.filter);
        return { ...current, layers: { ...current.layers, [layerId]: next } };
      }),

    setFilterAmount: (layerId, amount) =>
      commit(
        (current) => {
          const layer = current.layers[layerId];
          if (!layer?.filter) return current;
          return {
            ...current,
            layers: {
              ...current.layers,
              [layerId]: {
                ...layer,
                filter: { ...layer.filter, amount: Math.max(0, Math.min(1, amount)) },
              },
            },
          };
        },
        { history: false }
      ),

    commitFilter: (layerId) =>
      commitIfChanged(adjustmentSignature(layerId), (current) => {
        const layer = current.layers[layerId];
        return layer
          ? { ...current, layers: { ...current.layers, [layerId]: { ...layer } } }
          : current;
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

export const useOriginal = () =>
  useDocumentStore((s) => s.sessions[s.activeId]?.original ?? null);

export const useCanUndo = () =>
  useDocumentStore((s) => (s.sessions[s.activeId]?.historyIndex ?? 0) > 0);

export const useCanRedo = () =>
  useDocumentStore((s) => {
    const session = s.sessions[s.activeId];
    return session ? session.historyIndex < session.history.length - 1 : false;
  });
