// src/components/CanvasViewport.tsx
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  useDocumentStore,
  useActiveDocument,
  useViewport,
  useSelection,
} from '../store/documentStore';
import {
  RenderEngine,
  RENDER_THEMES,
  loadImage,
  TRANSFORM_HANDLE,
} from '../engine/RenderEngine';
import type { Point2D } from '../types/document';
import { SelectionActions } from './SelectionActions';


/** Minimum distance between two lasso points, in document pixels. */
const LASSO_STEP = 3;

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

  const {
    theme,
    activeTool,
    refineMode,
    refineSize,
    setViewport,
    setSelection,
    updateLayer,
    fitRequestId,
    activeId,
    selectionBrush,
    setLayerTransform,
    commitTransform,
  } = useDocumentStore();

  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isSpaceDown, setIsSpaceDown] = useState(false);
  const [draftLasso, setDraftLasso] = useState<Point2D[] | null>(null);

  const startPanPos = useRef({ x: 0, y: 0 });
  /** Grab in progress on the transform box, if any. */
  const grabRef = useRef<{
    corner: 'nw' | 'ne' | 'sw' | 'se' | 'move';
    startPoint: Point2D;
    origin: { x: number; y: number; scaleX: number; scaleY: number };
    size: { width: number; height: number };
  } | null>(null);
  const lassoRef = useRef<Point2D[] | null>(null);
  const refineStrokeRef = useRef<Point2D[] | null>(null);

  const docRef = useRef(doc);
  docRef.current = doc;

  /**
   * The layer whose handles are shown.
   *
   * Only when there is something to place: on a single-layer document there is
   * nothing to move it relative to, and the box would just be in the way.
   */
  const transformLayerId =
    doc.rootLayerIds.length > 1 && doc.activeLayerId && !refineMode
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
    const shown = draftLasso ? { path: draftLasso } : selection;
    engine.render(
      doc,
      viewport,
      containerSize.width,
      containerSize.height,
      shown,
      Boolean(draftLasso),
      transformLayerId
    );
  }, [doc, viewport, containerSize, frameTick, selection, draftLasso, transformLayerId]);

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
  const handleUnder = (point: Point2D) => {
    if (!transformLayerId) return null;
    const layer = doc.layers[transformLayerId];
    if (!layer) return null;

    const width = (layer.width ?? doc.width) * (layer.transform.scaleX || 1);
    const height = (layer.height ?? doc.height) * (layer.transform.scaleY || 1);
    const { x, y } = layer.transform;
    const reach = TRANSFORM_HANDLE / viewport.zoom;

    const near = (px: number, py: number) =>
      Math.abs(point.x - px) <= reach && Math.abs(point.y - py) <= reach;

    if (near(x + width / 2, y + height / 2)) return 'move' as const;
    if (near(x, y)) return 'nw' as const;
    if (near(x + width, y)) return 'ne' as const;
    if (near(x, y + height)) return 'sw' as const;
    if (near(x + width, y + height)) return 'se' as const;
    return null;
  };

  const wantsPan = (button: number) => button === 1 || activeTool === 'pan' || isSpaceDown;

  const handleMouseDown = (e: React.MouseEvent) => {
    if (wantsPan(e.button)) {
      setIsPanning(true);
      startPanPos.current = { x: e.clientX - viewport.panX, y: e.clientY - viewport.panY };
      return;
    }
    if (e.button !== 0) return;

    const point = toDocumentCoords(e.clientX, e.clientY);

    if (refineMode) {
      refineStrokeRef.current = [point];
      return;
    }

    // A handle wins over the lasso, so placing a layer never starts a selection.
    const corner = handleUnder(point);
    if (corner && transformLayerId) {
      const layer = doc.layers[transformLayerId];
      grabRef.current = {
        corner,
        startPoint: point,
        origin: {
          x: layer.transform.x,
          y: layer.transform.y,
          scaleX: layer.transform.scaleX || 1,
          scaleY: layer.transform.scaleY || 1,
        },
        size: {
          width: layer.width ?? doc.width,
          height: layer.height ?? doc.height,
        },
      };
      return;
    }

    // Dragging draws the lasso: it is the editor's primary interaction.
    lassoRef.current = [point];
    setDraftLasso([point]);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setViewport((prev) => ({
        ...prev,
        panX: e.clientX - startPanPos.current.x,
        panY: e.clientY - startPanPos.current.y,
      }));
      return;
    }

    const point = toDocumentCoords(e.clientX, e.clientY);

    const grab = grabRef.current;
    if (grab) {
      applyGrab(grab, point);
      return;
    }

    if (refineStrokeRef.current) {
      refineStrokeRef.current.push(point);
      return;
    }

    const lasso = lassoRef.current;
    if (!lasso) return;

    // Points too close together add weight, not shape.
    const last = lasso[lasso.length - 1];
    if (Math.hypot(point.x - last.x, point.y - last.y) < LASSO_STEP) return;

    lasso.push(point);
    setDraftLasso([...lasso]);
  };

  const handleMouseUp = () => {
    setIsPanning(false);

    if (grabRef.current) {
      grabRef.current = null;
      if (transformLayerId) commitTransform(transformLayerId);
      return;
    }

    if (refineStrokeRef.current) {
      const stroke = refineStrokeRef.current;
      refineStrokeRef.current = null;
      if (stroke.length > 0) void applyRefineStroke(stroke);
      return;
    }

    const lasso = lassoRef.current;
    lassoRef.current = null;
    setDraftLasso(null);
    if (!lasso) return;

    // A very short stroke is a click, not a lasso: clear the selection.
    if (lasso.length < 4 || pathArea(lasso) < 64) {
      setSelection(null);
      return;
    }
    setSelection({ path: lasso });
  };

  /** Moves or scales the layer while a handle is dragged. */
  const applyGrab = (grab: NonNullable<typeof grabRef.current>, point: Point2D) => {
    if (!transformLayerId) return;
    const dx = point.x - grab.startPoint.x;
    const dy = point.y - grab.startPoint.y;

    if (grab.corner === 'move') {
      setLayerTransform(transformLayerId, {
        x: grab.origin.x + dx,
        y: grab.origin.y + dy,
      });
      return;
    }

    const width = grab.size.width * grab.origin.scaleX;
    const height = grab.size.height * grab.origin.scaleY;

    // Dragging a corner pins the opposite one, which is what everyone expects.
    // The aspect ratio follows whichever axis moved more, so the layer never
    // distorts into something the user did not ask for.
    const growX = grab.corner === 'ne' || grab.corner === 'se' ? dx : -dx;
    const growY = grab.corner === 'sw' || grab.corner === 'se' ? dy : -dy;
    const factor = Math.max(
      0.02,
      1 + (Math.abs(growX) > Math.abs(growY) ? growX / width : growY / height)
    );

    const scaleX = grab.origin.scaleX * factor;
    const scaleY = grab.origin.scaleY * factor;
    const newWidth = grab.size.width * scaleX;
    const newHeight = grab.size.height * scaleY;

    const anchorRight = grab.corner === 'nw' || grab.corner === 'sw';
    const anchorBottom = grab.corner === 'nw' || grab.corner === 'ne';

    setLayerTransform(transformLayerId, {
      scaleX,
      scaleY,
      x: anchorRight ? grab.origin.x + (width - newWidth) : grab.origin.x,
      y: anchorBottom ? grab.origin.y + (height - newHeight) : grab.origin.y,
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

  const docLeft = containerSize.width / 2 + viewport.panX - (viewport.zoom * doc.width) / 2;
  const docTop = containerSize.height / 2 + viewport.panY - (viewport.zoom * doc.height) / 2;

  /** Anchor for the bubble: above the centre of the lasso, in screen pixels. */
  const anchor = selection
    ? (() => {
        const xs = selection.path.map((p) => p.x);
        const ys = selection.path.map((p) => p.y);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        const top = Math.min(...ys);
        return { x: docLeft + cx * viewport.zoom, y: docTop + top * viewport.zoom };
      })()
    : null;

  const panMode = isSpaceDown || activeTool === 'pan';
  const cursor = isPanning ? 'grabbing' : panMode ? 'grab' : 'crosshair';

  return (
    <div
      ref={containerRef}
      className="nabu-viewport-container"
      style={{ cursor }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <canvas ref={canvasRef} className="main-render-canvas" />

      {anchor && !refineMode && <SelectionActions x={anchor.x} y={anchor.y} />}
    </div>
  );
};


/** Signed polygon area: tells a real lasso from an accidental click. */
function pathArea(path: Point2D[]): number {
  let sum = 0;
  for (let i = 0; i < path.length; i += 1) {
    const a = path[i];
    const b = path[(i + 1) % path.length];
    sum += a.x * b.y - b.x * a.y;
  }
  return Math.abs(sum) / 2;
}

