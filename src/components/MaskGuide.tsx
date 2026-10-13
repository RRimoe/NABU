// src/components/MaskGuide.tsx
import { useEffect, useRef, useState } from 'react';
import { useActiveDocument, useViewport, useDocumentFields } from '../store/documentStore';
import { layerToDocument } from '../ops/paint';
import type { LocalMask } from '../types/document';

interface Props {
  containerWidth: number;
  containerHeight: number;
}

type Grip = 'start' | 'end' | 'move' | 'rx' | 'ry' | 'centre';

/**
 * Handles for the open gradient mask, drawn on the photo.
 *
 * Mask geometry lives in fractions of the layer, so everything here goes
 * through the layer's own transform: a gradient placed on a rotated or scaled
 * layer stays on the part of the picture it was put on.
 */
export const MaskGuide: React.FC<Props> = ({ containerWidth, containerHeight }) => {
  const doc = useActiveDocument();
  const viewport = useViewport();
  const { updateMask, commitMasks, editingMaskId } = useDocumentFields('updateMask', 'commitMasks', 'editingMaskId');

  const gripRef = useRef<{ grip: Grip; from: { x: number; y: number }; mask: LocalMask } | null>(
    null
  );
  const [dragging, setDragging] = useState(false);
  const pendingRef = useRef<Partial<LocalMask> | null>(null);
  const frameRef = useRef(0);

  useEffect(
    () => () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    },
    []
  );

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  const mask = layer?.masks?.find((m) => m.id === editingMaskId);
  if (!doc || !viewport || !layerId || !layer || !mask) return null;
  if (!(mask.kind === 'linear' && mask.linear) && !(mask.kind === 'radial' && mask.radial)) {
    return null;
  }

  const { zoom } = viewport;
  const originX = containerWidth / 2 + viewport.panX - (zoom * doc.width) / 2;
  const originY = containerHeight / 2 + viewport.panY - (zoom * doc.height) / 2;
  const naturalWidth = layer.width ?? doc.width;
  const naturalHeight = layer.height ?? doc.height;

  // Layer fractions → screen pixels, and back.
  const toScreenMatrix = new DOMMatrix()
    .translate(originX, originY)
    .scale(zoom, zoom)
    .multiply(layerToDocument(layer, doc))
    .scale(naturalWidth, naturalHeight);
  const fromScreenMatrix = toScreenMatrix.inverse();
  const toScreen = (x: number, y: number) => {
    const p = toScreenMatrix.transformPoint(new DOMPoint(x, y));
    return { x: p.x, y: p.y };
  };

  const schedule = (patch: Partial<LocalMask>) => {
    pendingRef.current = patch;
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      const value = pendingRef.current;
      pendingRef.current = null;
      if (value) updateMask(layerId, mask.id, value);
    });
  };

  const toFraction = (event: React.PointerEvent) => {
    const rect = (event.currentTarget as Element)
      .closest('.nabu-viewport-container')
      ?.getBoundingClientRect();
    if (!rect) return null;
    const p = fromScreenMatrix.transformPoint(
      new DOMPoint(event.clientX - rect.left, event.clientY - rect.top)
    );
    return { x: p.x, y: p.y };
  };

  const start = (grip: Grip) => (event: React.PointerEvent) => {
    event.stopPropagation();
    event.preventDefault();
    const from = toFraction(event);
    if (!from) return;
    (event.currentTarget as Element).setPointerCapture(event.pointerId);
    gripRef.current = { grip, from, mask };
    setDragging(true);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const drag = gripRef.current;
    if (!drag) return;
    const at = toFraction(event);
    if (!at) return;
    const dx = at.x - drag.from.x;
    const dy = at.y - drag.from.y;
    const { linear, radial } = drag.mask;

    if (linear) {
      if (drag.grip === 'start') schedule({ linear: { ...linear, x0: at.x, y0: at.y } });
      else if (drag.grip === 'end') schedule({ linear: { ...linear, x1: at.x, y1: at.y } });
      else
        schedule({
          linear: { x0: linear.x0 + dx, y0: linear.y0 + dy, x1: linear.x1 + dx, y1: linear.y1 + dy },
        });
    } else if (radial) {
      if (drag.grip === 'centre') schedule({ radial: { ...radial, cx: radial.cx + dx, cy: radial.cy + dy } });
      else if (drag.grip === 'rx')
        schedule({ radial: { ...radial, rx: Math.max(0.01, Math.abs(at.x - radial.cx)) } });
      else if (drag.grip === 'ry')
        schedule({ radial: { ...radial, ry: Math.max(0.01, Math.abs(at.y - radial.cy)) } });
    }
  };

  const end = (event: React.PointerEvent) => {
    if (!gripRef.current) return;
    gripRef.current = null;
    setDragging(false);
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
    }
    if (pendingRef.current) {
      updateMask(layerId, mask.id, pendingRef.current);
      pendingRef.current = null;
    }
    const target = event.currentTarget as Element;
    if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
    commitMasks(layerId);
  };

  const handle = (grip: Grip, x: number, y: number, kind: 'main' | 'small' = 'main') => {
    const p = toScreen(x, y);
    return (
      <circle
        key={grip}
        className={`mask-handle ${kind}`}
        data-grip={grip}
        cx={p.x}
        cy={p.y}
        r={kind === 'main' ? 7 : 5}
        onPointerDown={start(grip)}
      />
    );
  };

  const matrix = `matrix(${toScreenMatrix.a} ${toScreenMatrix.b} ${toScreenMatrix.c} ${toScreenMatrix.d} ${toScreenMatrix.e} ${toScreenMatrix.f})`;

  let body: React.ReactNode = null;
  if (mask.linear) {
    const { x0, y0, x1, y1 } = mask.linear;
    // Lines across the frame, perpendicular to the gradient, at both ends.
    const nx = -(y1 - y0);
    const ny = x1 - x0;
    const length = Math.hypot(nx, ny) || 1;
    const ux = (nx / length) * 3;
    const uy = (ny / length) * 3;
    const across = (x: number, y: number, dashed: boolean) => (
      <line
        className={`mask-line ${dashed ? 'soft' : ''}`}
        x1={x - ux}
        y1={y - uy}
        x2={x + ux}
        y2={y + uy}
        vectorEffect="non-scaling-stroke"
      />
    );
    body = (
      <>
        <g transform={matrix}>
          {across(x0, y0, false)}
          {across(x1, y1, true)}
          <line
            className="mask-line axis"
            x1={x0}
            y1={y0}
            x2={x1}
            y2={y1}
            vectorEffect="non-scaling-stroke"
          />
        </g>
        {handle('start', x0, y0)}
        {handle('end', x1, y1, 'small')}
        {handle('move', (x0 + x1) / 2, (y0 + y1) / 2, 'small')}
      </>
    );
  } else if (mask.radial) {
    const { cx, cy, rx, ry, feather } = mask.radial;
    const inner = Math.max(0, 1 - feather);
    body = (
      <>
        <g transform={matrix}>
          <ellipse className="mask-line" cx={cx} cy={cy} rx={rx} ry={ry} vectorEffect="non-scaling-stroke" />
          <ellipse
            className="mask-line soft"
            cx={cx}
            cy={cy}
            rx={rx * inner}
            ry={ry * inner}
            vectorEffect="non-scaling-stroke"
          />
        </g>
        {handle('centre', cx, cy)}
        {handle('rx', cx + rx, cy, 'small')}
        {handle('ry', cx, cy + ry, 'small')}
      </>
    );
  }

  return (
    <svg
      className={`mask-guide ${dragging ? 'dragging' : ''}`}
      width={containerWidth}
      height={containerHeight}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      // See FocusGuide: the canvas listens to mouse events, which pointer
      // handlers do not stop on their own.
      onMouseDown={(event) => event.stopPropagation()}
      onMouseMove={(event) => dragging && event.stopPropagation()}
      onMouseUp={(event) => event.stopPropagation()}
    >
      {body}
    </svg>
  );
};
