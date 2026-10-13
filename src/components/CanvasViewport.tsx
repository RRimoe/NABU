// src/components/CanvasViewport.tsx
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  useDocumentStore,
  useActiveDocument,
  useViewport,
  useSelection,
  useOriginal, useDocumentFields } from '../store/documentStore';
import {
  RenderEngine,
  RENDER_THEMES,
  loadImage,
  layerBoxSize,
  toLayerBox,
  SCALE_HANDLES,
  TRANSFORM_HANDLE,
  ROTATE_HANDLE_OFFSET,
  type TransformHandle,
} from '../engine/RenderEngine';
import type { Point2D, SelectionShape } from '../types/document';
import { refineByClick } from '../ai/operations';
import { ActionBar } from './ActionBar';
import { CropOverlay } from './CropOverlay';
import { FocusGuide } from './FocusGuide';
import { MaskGuide } from './MaskGuide';
import { PasteChip } from './PasteChip';
import { TransformBar } from './TransformBar';
import { BrushBar } from './BrushBar';
import { canPaint, paintStroke } from '../ops/paint';
import { useT } from '../i18n';

/** Minimum distance between two lasso points, in document pixels. */
const LASSO_STEP = 3;

/** Points used to draw an ellipse: enough that no straight edge is visible. */
const ELLIPSE_STEPS = 64;

/** Smallest side a layer can be scaled down to, in document pixels. */
const MIN_LAYER_SIDE = 8;

/** Rotation snap while Shift is held, in degrees. */
const ROTATE_SNAP = 15;

/** State captured when a transform handle is grabbed. */
interface Grab {
  handle: TransformHandle;
  startPoint: Point2D;
  /** Transform as it was when the drag started. */
  origin: { x: number; y: number; scaleX: number; scaleY: number; rotation: number };
  /** Unscaled layer size. */
  base: { width: number; height: number };
  /** Scaled box size when the drag started. */
  box: { width: number; height: number };
  /** Pointer angle around the centre at the start, for rotation. */
  startAngle: number;
}

/**
 * View of the active document.
 *
 * `App` only mounts it when a document is open, so `doc` and `viewport` can be
 * treated as always present in here.
 */
export const CanvasViewport: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<RenderEngine | null>(null);

  const doc = useActiveDocument();
  const viewport = useViewport();
  const selection = useSelection();
  const original = useOriginal();
  const t = useT();

  const {
    theme,
    activeTool,
    setActiveTool,
    refineMode,
    refineSize,
    selectionShape,
    focusEditing,
    setFocusEditing,
    setViewport,
    setSelection,
    updateLayer,
    fitRequestId,
    activeId,
    selectionBrush,
    setLayerTransform,
    commitTransform,
    pointRefine,
    setPointRefine,
    brushSize,
    brushOpacity,
    brushColor,
    compare,
    compareSplit,
    setCompareSplit,
  } = useDocumentFields(
    'theme',
    'activeTool',
    'setActiveTool',
    'refineMode',
    'refineSize',
    'selectionShape',
    'focusEditing',
    'setFocusEditing',
    'setViewport',
    'setSelection',
    'updateLayer',
    'fitRequestId',
    'activeId',
    'selectionBrush',
    'setLayerTransform',
    'commitTransform',
    'pointRefine',
    'setPointRefine',
    'brushSize',
    'brushOpacity',
    'brushColor',
    'compare',
    'compareSplit',
    'setCompareSplit',
  );

  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isSpaceDown, setIsSpaceDown] = useState(false);
  const [draftSelection, setDraftSelection] = useState<Point2D[] | null>(null);
  const [grabbing, setGrabbing] = useState(false);

  const startPanPos = useRef({ x: 0, y: 0 });
  const grabRef = useRef<Grab | null>(null);
  const lassoRef = useRef<Point2D[] | null>(null);
  /** First corner of a rectangular or elliptical selection. */
  const shapeStartRef = useRef<Point2D | null>(null);
  const refineStrokeRef = useRef<Point2D[] | null>(null);
  /** Brush or eraser stroke in progress, in document coordinates. */
  const paintStrokeRef = useRef<Point2D[] | null>(null);
  const [draftStroke, setDraftStroke] = useState<Point2D[] | null>(null);
  /** Where the brush outline is drawn, in container pixels; null off the canvas. */
  const [brushAt, setBrushAt] = useState<Point2D | null>(null);

  const docRef = useRef(doc);
  docRef.current = doc;

  const transforming = activeTool === 'transform';
  const cropping = activeTool === 'crop';
  const painting = (activeTool === 'brush' || activeTool === 'eraser') && !refineMode;

  /**
   * What the comparison shows this frame. A split needs both pictures to be the
   * same shape; after a crop or a resize only the whole original makes sense.
   */
  const sameShape = original && original.width === doc.width && original.height === doc.height;
  const compareView =
    original && compare === 'hold'
      ? { original, split: null }
      : original && compare === 'split' && sameShape
        ? { original, split: compareSplit }
        : null;
  const splitting = compareView?.split != null;
  const splitDragRef = useRef(false);

  /**
   * The layer whose handles are shown.
   *
   * Always while the transform tool is on, and otherwise only when there is
   * something to place: on a single-layer document there is nothing to move it
   * relative to, and the box would just be in the way.
   */
  /**
   * The layer whose handles are shown — only while the transform tool is on.
   *
   * They used to appear whenever a document had more than one layer, which
   * meant a box drawn over the photograph most of the time, for a gesture the
   * user had not asked for. Handles are the vocabulary of one tool, so they
   * belong to that tool and to no other moment.
   */
  const transformLayerId =
    transforming && doc.activeLayerId && !refineMode && !cropping
      ? doc.activeLayerId
      : null;

  const [frameTick, setFrameTick] = useState(0);
  const requestFrame = useCallback(() => setFrameTick((t) => t + 1), []);

  // Engine setup and container size observation
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;
    const engine = new RenderEngine(canvasRef.current, requestFrame);
    engineRef.current = engine;

    const container = containerRef.current;
    const syncSize = () => {
      const { clientWidth, clientHeight } = container;
      engine.resize(clientWidth, clientHeight);
      setContainerSize({ width: clientWidth, height: clientHeight });
      // The navigator needs it too, and it lives in the other column.
      useDocumentStore.getState().setViewportSize(clientWidth, clientHeight);
    };

    syncSize();
    const observer = new ResizeObserver(syncSize);
    observer.observe(container);

    return () => {
      observer.disconnect();
      engineRef.current = null;
    };
  }, [requestFrame]);

  useEffect(() => {
    engineRef.current?.setTheme(RENDER_THEMES[theme]);
    requestFrame();
  }, [theme, requestFrame]);

  useEffect(() => {
    engineRef.current?.setBrushWidth(selectionBrush);
    requestFrame();
  }, [selectionBrush, requestFrame]);

  // Frame drawing: synchronous, so two frames can never overlap.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || containerSize.width === 0 || containerSize.height === 0) return;
    const shown = draftSelection ? { path: draftSelection } : selection;
    // Only the freehand gesture is painted as a brush stroke: a rectangle drawn
    // with a fat glowing line reads as a smear, not as a rectangle.
    const asStroke = Boolean(draftSelection) && selectionShape === 'free';
    engine.render(
      doc,
      viewport,
      containerSize.width,
      containerSize.height,
      shown,
      asStroke,
      transformLayerId,
      compareView
    );
  }, [
    compareView,
    doc,
    viewport,
    containerSize,
    frameTick,
    selection,
    draftSelection,
    selectionShape,
    transformLayerId,
  ]);

  // Before/after from the keyboard: backslash shows the original for as long
  // as it is held, Y toggles the split. Both are what photo editors use.
  useEffect(() => {
    const typing = () => {
      const tag = window.document.activeElement?.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA';
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (typing() || e.ctrlKey || e.metaKey || e.altKey) return;
      const { compare: mode, setCompare: change } = useDocumentStore.getState();
      if (e.key === '\\' && mode !== 'hold') {
        e.preventDefault();
        change('hold');
      } else if (e.key.toLowerCase() === 'y' && !e.repeat) {
        e.preventDefault();
        change(mode === 'split' ? 'off' : 'split');
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const { compare: mode, setCompare: change } = useDocumentStore.getState();
      if (e.key === '\\' && mode === 'hold') change('off');
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // "Fit to screen" requested by the store
  const handledFitRef = useRef(0);
  useEffect(() => {
    if (fitRequestId === 0 || fitRequestId === handledFitRef.current) return;
    const { width, height } = containerSize;
    if (width === 0 || height === 0) return;

    handledFitRef.current = fitRequestId;
    const current = docRef.current;
    const zoom = Math.min(
      10,
      Math.max(0.05, Math.min(width / current.width, height / current.height) * 0.92)
    );
    setViewport(() => ({ zoom, panX: 0, panY: 0 }));
  }, [fitRequestId, containerSize, setViewport]);

  // Switching document means refitting the view to the new canvas.
  useEffect(() => {
    useDocumentStore.getState().requestFitToScreen();
  }, [activeId]);

  // Space bar: temporary pan
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      const tag = window.document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      e.preventDefault();
      setIsSpaceDown(true);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') setIsSpaceDown(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // Arrow keys nudge the layer being placed: the only way to move something by
  // an exact pixel, and the reason nobody drags a layer twice to line it up.
  useEffect(() => {
    if (!transforming || !transformLayerId) return;

    const onKeyDown = (e: KeyboardEvent) => {
      const tag = window.document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      const step = e.shiftKey ? 10 : 1;
      const delta =
        e.key === 'ArrowLeft'
          ? { x: -step, y: 0 }
          : e.key === 'ArrowRight'
            ? { x: step, y: 0 }
            : e.key === 'ArrowUp'
              ? { x: 0, y: -step }
              : e.key === 'ArrowDown'
                ? { x: 0, y: step }
                : null;
      if (!delta) return;

      e.preventDefault();
      const layer = docRef.current.layers[transformLayerId];
      if (!layer) return;
      setLayerTransform(transformLayerId, {
        x: layer.transform.x + delta.x,
        y: layer.transform.y + delta.y,
      });
      commitTransform(transformLayerId);
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [transforming, transformLayerId, setLayerTransform, commitTransform]);

  // Wheel zoom, anchored at the pointer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Native non-passive listener: React's onWheel cannot call preventDefault.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const pointerX = e.clientX - rect.left - rect.width / 2;
      const pointerY = e.clientY - rect.top - rect.height / 2;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;

      setViewport((prev) => {
        const zoom = Math.min(10, Math.max(0.05, prev.zoom * factor));
        const ratio = zoom / prev.zoom;
        return {
          zoom,
          panX: pointerX - (pointerX - prev.panX) * ratio,
          panY: pointerY - (pointerY - prev.panY) * ratio,
        };
      });
    };

    container.addEventListener('wheel', onWheel, { passive: false });
    return () => container.removeEventListener('wheel', onWheel);
  }, [setViewport]);

  /** Pointer coordinates converted into document coordinates. */
  const toDocumentCoords = (clientX: number, clientY: number): Point2D => {
    const container = containerRef.current;
    if (!container) return { x: 0, y: 0 };
    const rect = container.getBoundingClientRect();
    const originX = rect.width / 2 + viewport.panX;
    const originY = rect.height / 2 + viewport.panY;
    return {
      x: (clientX - rect.left - originX) / viewport.zoom + doc.width / 2,
      y: (clientY - rect.top - originY) / viewport.zoom + doc.height / 2,
    };
  };

  /** Which handle, if any, sits under a point. Reach is in screen pixels. */
  const handleUnder = (point: Point2D): TransformHandle | null => {
    if (!transformLayerId) return null;
    const layer = doc.layers[transformLayerId];
    if (!layer) return null;

    const { width, height } = layerBoxSize(layer, doc);
    const local = toLayerBox(point, layer, doc);
    const reach = TRANSFORM_HANDLE / viewport.zoom;

    const near = (px: number, py: number) =>
      Math.abs(local.x - px) <= reach && Math.abs(local.y - py) <= reach;

    if (near(width / 2, -ROTATE_HANDLE_OFFSET / viewport.zoom)) return 'rotate';

    for (const { id, fx, fy } of SCALE_HANDLES) {
      if (near(fx * width, fy * height)) return id;
    }

    // The whole layer is the move target. The handles only appear with the
    // transform tool on, so there is no other gesture here to compete with.
    if (local.x >= 0 && local.y >= 0 && local.x <= width && local.y <= height) {
      return 'move';
    }

    return null;
  };

  const wantsPan = (button: number) => button === 1 || activeTool === 'pan' || isSpaceDown;

  const handleMouseDown = (e: React.MouseEvent) => {
    // The divider sits on top of every tool: grabbing it never paints, selects
    // or moves anything.
    if (splitting && e.button === 0) {
      const dividerX = toScreen({ x: doc.width * compareSplit, y: 0 }).x;
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect && Math.abs(e.clientX - rect.left - dividerX) <= 8) {
        splitDragRef.current = true;
        return;
      }
    }

    // Reaching for the photograph ends the placing of the sharp zone. The
    // guide stops its own mouse events, so this only fires for presses that
    // really did land on the picture.
    if (focusEditing) setFocusEditing(false);

    // The crop frame owns the canvas while it is up: a drag there is resizing
    // the frame, never starting a selection underneath it.
    if (cropping && e.button === 0) return;

    if (wantsPan(e.button)) {
      setIsPanning(true);
      startPanPos.current = { x: e.clientX - viewport.panX, y: e.clientY - viewport.panY };
      return;
    }
    if (e.button !== 0) return;

    const point = toDocumentCoords(e.clientX, e.clientY);

    // Correcting a selection by pointing at it. The click is the whole
    // gesture — no drag, no confirmation — so it takes the press and stops.
    if (pointRefine && selection) {
      void refineByClick(point, e.altKey);
      return;
    }

    if (refineMode) {
      refineStrokeRef.current = [point];
      return;
    }

    if (painting) {
      const layer = doc.activeLayerId ? doc.layers[doc.activeLayerId] : undefined;
      if (!canPaint(layer)) return;
      paintStrokeRef.current = [point];
      setDraftStroke([point]);
      return;
    }

    // A handle wins over the selection, so placing a layer never starts one.
    const handle = handleUnder(point);
    if (handle && transformLayerId) {
      const layer = doc.layers[transformLayerId];
      const box = layerBoxSize(layer, doc);
      const centreX = layer.transform.x + box.width / 2;
      const centreY = layer.transform.y + box.height / 2;

      grabRef.current = {
        handle,
        startPoint: point,
        origin: {
          x: layer.transform.x,
          y: layer.transform.y,
          scaleX: layer.transform.scaleX || 1,
          scaleY: layer.transform.scaleY || 1,
          rotation: layer.transform.rotation || 0,
        },
        base: {
          width: layer.width ?? doc.width,
          height: layer.height ?? doc.height,
        },
        box,
        startAngle: Math.atan2(point.y - centreY, point.x - centreX),
      };
      setGrabbing(true);
      return;
    }

    if (selectionShape === 'free') {
      // Dragging draws the lasso: it is the editor's primary interaction.
      lassoRef.current = [point];
      setDraftSelection([point]);
    } else {
      shapeStartRef.current = point;
      setDraftSelection(shapePath(selectionShape, point, point, e.shiftKey));
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (splitDragRef.current) {
      setCompareSplit(toDocumentCoords(e.clientX, e.clientY).x / doc.width);
      return;
    }
    if (isPanning) {
      setViewport((prev) => ({
        ...prev,
        panX: e.clientX - startPanPos.current.x,
        panY: e.clientY - startPanPos.current.y,
      }));
      return;
    }

    const point = toDocumentCoords(e.clientX, e.clientY);

    if (painting) {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) setBrushAt({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }

    const stroke = paintStrokeRef.current;
    if (stroke) {
      // Spacing relative to the brush keeps a big brush light and a small one
      // smooth; a fixed step would be one or the other.
      const last = stroke[stroke.length - 1];
      if (Math.hypot(point.x - last.x, point.y - last.y) < Math.max(1, brushSize / 10)) return;
      stroke.push(point);
      setDraftStroke([...stroke]);
      return;
    }

    const grab = grabRef.current;
    if (grab) {
      applyGrab(grab, point, e.shiftKey, e.altKey);
      return;
    }

    if (refineStrokeRef.current) {
      refineStrokeRef.current.push(point);
      return;
    }

    const start = shapeStartRef.current;
    if (start) {
      setDraftSelection(shapePath(selectionShape, start, point, e.shiftKey));
      return;
    }

    const lasso = lassoRef.current;
    if (!lasso) return;

    // Points too close together add weight, not shape.
    const last = lasso[lasso.length - 1];
    if (Math.hypot(point.x - last.x, point.y - last.y) < LASSO_STEP) return;

    lasso.push(point);
    setDraftSelection([...lasso]);
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    if (splitDragRef.current) {
      splitDragRef.current = false;
      return;
    }

    if (grabRef.current) {
      grabRef.current = null;
      setGrabbing(false);
      if (transformLayerId) commitTransform(transformLayerId);
      return;
    }

    if (refineStrokeRef.current) {
      const stroke = refineStrokeRef.current;
      refineStrokeRef.current = null;
      if (stroke.length > 0) void applyRefineStroke(stroke);
      return;
    }

    if (paintStrokeRef.current) {
      const stroke = paintStrokeRef.current;
      paintStrokeRef.current = null;
      void applyPaintStroke(stroke);
      return;
    }

    const shapeStart = shapeStartRef.current;
    const lasso = lassoRef.current;
    const path = shapeStart ? draftSelection : lasso;

    shapeStartRef.current = null;
    lassoRef.current = null;
    setDraftSelection(null);
    if (!path) return;

    // A very short stroke is a click, not a selection: clear it instead.
    if (path.length < 4 || pathArea(path) < 64) {
      setSelection(null);
      return;
    }
    setSelection({ path, shape: selectionShape });

    // Drawing with the smart lasso opens the correction loop straight away: the
    // segmenter has just guessed, and the moment to say where it guessed wrong
    // is now, not after the removal has already used it.
    if (activeTool === 'ai-lasso') setPointRefine(true);
  };

  /**
   * Moves, scales or rotates the layer while a handle is dragged.
   *
   * All of it is worked out in the box's own unrotated space and converted back
   * at the end: that way one piece of arithmetic covers every angle, and the
   * corner opposite the one being dragged stays exactly where it is.
   */
  const applyGrab = (grab: Grab, point: Point2D, shift: boolean, alt: boolean) => {
    if (!transformLayerId) return;

    const centreX = grab.origin.x + grab.box.width / 2;
    const centreY = grab.origin.y + grab.box.height / 2;

    if (grab.handle === 'move') {
      let dx = point.x - grab.startPoint.x;
      let dy = point.y - grab.startPoint.y;
      // Shift keeps the movement on one axis, the usual way to slide something
      // along an edge without drifting off it.
      if (shift) {
        if (Math.abs(dx) > Math.abs(dy)) dy = 0;
        else dx = 0;
      }
      setLayerTransform(transformLayerId, {
        x: grab.origin.x + dx,
        y: grab.origin.y + dy,
      });
      return;
    }

    if (grab.handle === 'rotate') {
      const angle = Math.atan2(point.y - centreY, point.x - centreX);
      let rotation = grab.origin.rotation + (angle - grab.startAngle);
      if (shift) {
        const step = (ROTATE_SNAP * Math.PI) / 180;
        rotation = Math.round(rotation / step) * step;
      }
      // Rotation pivots on the centre, which does not move, so x and y stay.
      setLayerTransform(transformLayerId, { rotation });
      return;
    }

    const spec = SCALE_HANDLES.find((h) => h.id === grab.handle);
    if (!spec) return;

    const cos = Math.cos(-grab.origin.rotation);
    const sin = Math.sin(-grab.origin.rotation);
    const dx = point.x - centreX;
    const dy = point.y - centreY;
    const local = {
      x: dx * cos - dy * sin + grab.box.width / 2,
      y: dx * sin + dy * cos + grab.box.height / 2,
    };

    const movesX = spec.fx !== 0.5;
    const movesY = spec.fy !== 0.5;

    // Alt scales around the centre instead of the opposite corner.
    let width = grab.box.width;
    let height = grab.box.height;

    if (movesX) {
      width = alt
        ? Math.abs(local.x - grab.box.width / 2) * 2
        : spec.fx === 1
          ? local.x
          : grab.box.width - local.x;
    }
    if (movesY) {
      height = alt
        ? Math.abs(local.y - grab.box.height / 2) * 2
        : spec.fy === 1
          ? local.y
          : grab.box.height - local.y;
    }

    // Corners keep the proportions unless Shift asks for a free stretch: a
    // photo distorted by accident is the classic way to ruin a composition.
    if (movesX && movesY && !shift) {
      const ratioX = width / grab.box.width;
      const ratioY = height / grab.box.height;
      const ratio = Math.abs(ratioX - 1) > Math.abs(ratioY - 1) ? ratioX : ratioY;
      width = grab.box.width * ratio;
      height = grab.box.height * ratio;
    }

    width = Math.max(MIN_LAYER_SIDE, width);
    height = Math.max(MIN_LAYER_SIDE, height);

    // Where the pinned point sits, as a fraction of the box.
    const anchorX = alt ? 0.5 : movesX ? 1 - spec.fx : 0.5;
    const anchorY = alt ? 0.5 : movesY ? 1 - spec.fy : 0.5;

    // That same point in document space, from the box as it was.
    const forwardCos = Math.cos(grab.origin.rotation);
    const forwardSin = Math.sin(grab.origin.rotation);
    const offsetX = (anchorX - 0.5) * grab.box.width;
    const offsetY = (anchorY - 0.5) * grab.box.height;
    const anchorDocX = centreX + offsetX * forwardCos - offsetY * forwardSin;
    const anchorDocY = centreY + offsetX * forwardSin + offsetY * forwardCos;

    // Put the new box back so that point lands in the same place.
    const newOffsetX = (anchorX - 0.5) * width;
    const newOffsetY = (anchorY - 0.5) * height;
    const newCentreX = anchorDocX - (newOffsetX * forwardCos - newOffsetY * forwardSin);
    const newCentreY = anchorDocY - (newOffsetX * forwardSin + newOffsetY * forwardCos);

    // The sign carries the mirroring, so it has to survive a resize.
    const signX = grab.origin.scaleX < 0 ? -1 : 1;
    const signY = grab.origin.scaleY < 0 ? -1 : 1;

    setLayerTransform(transformLayerId, {
      scaleX: (signX * width) / grab.base.width,
      scaleY: (signY * height) / grab.base.height,
      x: newCentreX - width / 2,
      y: newCentreY - height / 2,
    });
  };

  /**
   * Applies one refinement-brush stroke to the active layer.
   *
   * "Erase" removes opacity; "restore" repaints pixels from the source layer,
   * which is why a cutout keeps a reference to its original.
   */
  const applyRefineStroke = async (stroke: Point2D[]) => {
    const current = docRef.current;
    const layerId = current.activeLayerId;
    if (!layerId) return;
    const layer = current.layers[layerId];
    if (!layer?.imageDataUrl) return;

    const width = layer.width ?? current.width;
    const height = layer.height ?? current.height;

    const canvas = window.document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    try {
      ctx.drawImage(await loadImage(layer.imageDataUrl), 0, 0, width, height);
    } catch {
      return;
    }

    /** Traces the brush path, in layer coordinates. */
    const traceStroke = (target: CanvasRenderingContext2D) => {
      target.translate(-layer.transform.x, -layer.transform.y);
      target.lineCap = 'round';
      target.lineJoin = 'round';
      target.lineWidth = refineSize;
      target.beginPath();
      target.moveTo(stroke[0].x, stroke[0].y);
      for (const point of stroke.slice(1)) target.lineTo(point.x, point.y);
      // A single tap would produce no stroke at all: nudge it a hair.
      if (stroke.length === 1) target.lineTo(stroke[0].x + 0.01, stroke[0].y);
    };

    if (refineMode === 'erase') {
      ctx.save();
      traceStroke(ctx);
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = '#000';
      ctx.stroke();
      ctx.restore();
    } else {
      const source = layer.sourceLayerId ? current.layers[layer.sourceLayerId] : undefined;
      if (!source?.imageDataUrl) return;

      // Clip the original pixels to the stroke shape on a separate canvas and
      // then overlay them: clipping directly here would be undone by restore,
      // repainting the whole image and wiping out the cutout.
      const [stencil, stencilCtx] = (() => {
        const c = window.document.createElement('canvas');
        c.width = width;
        c.height = height;
        return [c, c.getContext('2d')] as const;
      })();
      if (!stencilCtx) return;

      stencilCtx.save();
      traceStroke(stencilCtx);
      stencilCtx.strokeStyle = '#fff';
      stencilCtx.stroke();
      stencilCtx.restore();

      // `source-in` keeps the original only where the stroke left pixels.
      stencilCtx.globalCompositeOperation = 'source-in';
      try {
        stencilCtx.drawImage(await loadImage(source.imageDataUrl), 0, 0, width, height);
      } catch {
        return;
      }

      ctx.drawImage(stencil, 0, 0);
    }

    updateLayer(layerId, { imageDataUrl: canvas.toDataURL('image/png') });
  };

  /**
   * Bakes a finished brush or eraser stroke into the active layer. The draft
   * stays on screen until the new pixels are in, so the stroke never blinks.
   */
  const applyPaintStroke = async (stroke: Point2D[]) => {
    const current = docRef.current;
    const layerId = current.activeLayerId;
    const layer = layerId ? current.layers[layerId] : undefined;
    try {
      if (layerId && canPaint(layer)) {
        const painted = await paintStroke(layer, current, stroke, {
          mode: activeTool === 'eraser' ? 'erase' : 'paint',
          size: brushSize,
          opacity: brushOpacity,
          color: brushColor,
          clip: selection?.path,
        });
        if (painted) updateLayer(layerId, { imageDataUrl: painted });
      }
    } finally {
      setDraftStroke(null);
    }
  };

  /** Document point to container pixels, for drawing over the canvas. */
  const toScreen = (point: Point2D) => ({
    x: (point.x - doc.width / 2) * viewport.zoom + containerSize.width / 2 + viewport.panX,
    y: (point.y - doc.height / 2) * viewport.zoom + containerSize.height / 2 + viewport.panY,
  });

  const panMode = isSpaceDown || activeTool === 'pan';
  const cursor = isPanning
    ? 'grabbing'
    : panMode
      ? 'grab'
      : grabbing
        ? 'grabbing'
        : transforming
          ? 'move'
          : painting
            ? 'none'
            : 'crosshair';

  return (
    <div
      ref={containerRef}
      className="nabu-viewport-container"
      style={{ cursor }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        setBrushAt(null);
        handleMouseUp();
      }}
    >
      <canvas ref={canvasRef} className="main-render-canvas" />

      {cropping && (
        <CropOverlay
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
          onClose={() => setActiveTool('select')}
        />
      )}

      {/* The sharp zone is placed by dragging it onto the thing that should
          stay sharp, not by typing two numbers. It is a mode with an end:
          tied to the panel being open it could not be moved after applying,
          and tied to the effect being active it never went away. */}
      {focusEditing && !cropping && !refineMode && !transforming && (
        <FocusGuide
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
        />
      )}

      {/* Handles of the gradient mask being edited in the Masks panel. */}
      {!cropping && !refineMode && !transforming && !painting && (
        <MaskGuide
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
        />
      )}

      {!refineMode && !cropping && <PasteChip />}

      {/* One bar in one place, rather than a bubble that lands wherever the
          selection happens to be: the actions are always where you left them. */}
      {selection && !refineMode && !transforming && !cropping && !painting && <ActionBar />}

      {/* The stroke being drawn and the brush outline. The real pixels arrive
          when the stroke ends; until then this is what the user is drawing. */}
      {painting && !panMode && (
        <svg className="brush-overlay" width={containerSize.width} height={containerSize.height}>
          {draftStroke && (
            <polyline
              points={draftStroke
                .map(toScreen)
                .map((p) => `${p.x},${p.y}`)
                .join(' ')}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={Math.max(1, brushSize * viewport.zoom)}
              stroke={activeTool === 'eraser' ? 'rgba(255, 255, 255, 0.55)' : brushColor}
              strokeOpacity={activeTool === 'eraser' ? 1 : brushOpacity}
            />
          )}
          {brushAt && (
            <>
              <circle
                cx={brushAt.x}
                cy={brushAt.y}
                r={Math.max(2, (brushSize * viewport.zoom) / 2)}
                className="brush-ring dark"
              />
              <circle
                cx={brushAt.x}
                cy={brushAt.y}
                r={Math.max(2, (brushSize * viewport.zoom) / 2)}
                className="brush-ring"
              />
            </>
          )}
        </svg>
      )}

      {painting && <BrushBar onDone={() => setActiveTool('select')} />}

      {compareView && (
        <div className="compare-labels">
          {splitting ? (
            <>
              <span
                className="compare-label"
                style={{ left: toScreen({ x: doc.width * compareSplit, y: 0 }).x - 8, transform: 'translateX(-100%)' }}
              >
                {t('compare.before')}
              </span>
              <span
                className="compare-label"
                style={{ left: toScreen({ x: doc.width * compareSplit, y: 0 }).x + 8 }}
              >
                {t('compare.after')}
              </span>
            </>
          ) : (
            <span className="compare-label centred">{t('compare.original')}</span>
          )}
        </div>
      )}

      {transforming && transformLayerId && (
        <TransformBar layerId={transformLayerId} onDone={() => setActiveTool('select')} />
      )}
    </div>
  );
};

/**
 * Builds the outline of a rectangular or elliptical drag.
 *
 * Both come out as a plain list of points, so the renderer and the segmenter
 * keep treating every selection the same way.
 */
function shapePath(
  shape: SelectionShape,
  start: Point2D,
  end: Point2D,
  constrain: boolean
): Point2D[] {
  let width = end.x - start.x;
  let height = end.y - start.y;

  // Shift turns the rectangle into a square and the ellipse into a circle.
  if (constrain) {
    const side = Math.max(Math.abs(width), Math.abs(height));
    width = Math.sign(width || 1) * side;
    height = Math.sign(height || 1) * side;
  }

  const x0 = Math.min(start.x, start.x + width);
  const y0 = Math.min(start.y, start.y + height);
  const x1 = Math.max(start.x, start.x + width);
  const y1 = Math.max(start.y, start.y + height);

  if (shape === 'rect') {
    return [
      { x: x0, y: y0 },
      { x: x1, y: y0 },
      { x: x1, y: y1 },
      { x: x0, y: y1 },
    ];
  }

  const centreX = (x0 + x1) / 2;
  const centreY = (y0 + y1) / 2;
  const radiusX = (x1 - x0) / 2;
  const radiusY = (y1 - y0) / 2;

  return Array.from({ length: ELLIPSE_STEPS }, (_, index) => {
    const angle = (index / ELLIPSE_STEPS) * Math.PI * 2;
    return {
      x: centreX + Math.cos(angle) * radiusX,
      y: centreY + Math.sin(angle) * radiusY,
    };
  });
}

/** Signed polygon area: tells a real selection from an accidental click. */
function pathArea(path: Point2D[]): number {
  let sum = 0;
  for (let i = 0; i < path.length; i += 1) {
    const a = path[i];
    const b = path[(i + 1) % path.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}
