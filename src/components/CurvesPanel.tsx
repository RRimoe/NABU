// src/components/CurvesPanel.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import {
  type CurveChannel,
  type CurvePoint,
  CURVE_CHANNELS,
  NEUTRAL_CURVE,
  isCurveNeutral,
} from '../types/document';
import {
  computeHistogram,
  findCurvePoint,
  insertCurvePoint,
  moveCurvePoint,
  removeCurvePoint,
  sampleCurve,
  LUT_SIZE,
} from '../engine/curves';
import { loadImage } from '../engine/RenderEngine';
import { useT } from '../i18n';

/** Logical side of the graph, in CSS pixels. */
const GRAPH_SIZE = 268;

/** Grab radius for a control point, in CSS pixels. */
const POINT_REACH = 11;

/** Colour of each channel's line, and of its histogram. */
const CHANNEL_COLOURS: Record<CurveChannel, string> = {
  rgb: '#5bb7dc',
  r: '#f87171',
  g: '#4ade80',
  b: '#60a5fa',
};

/**
 * Tone curves with the control every other panel hides.
 *
 * A slider says "more contrast"; a curve says *where*. It is the one place in
 * Nabu that asks the user to think in input and output rather than in nouns,
 * which is exactly why it lives behind Tab with the other professional panels
 * instead of on the canvas.
 */
export const CurvesPanel: React.FC = () => {
  const doc = useActiveDocument();
  const { setCurve, commitCurves, resetCurves } = useDocumentFields('setCurve', 'commitCurves', 'resetCurves');
  const t = useT();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [channel, setChannel] = useState<CurveChannel>('rgb');
  const [histogram, setHistogram] = useState<Record<CurveChannel, Float32Array> | null>(null);
  /** Index of the point being dragged, or -1. */
  const draggingRef = useRef(-1);
  const [hovered, setHovered] = useState(-1);

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  const source = layer?.imageDataUrl;

  const points = useMemo<CurvePoint[]>(
    () => layer?.curves?.[channel] ?? NEUTRAL_CURVE,
    [layer?.curves, channel]
  );

  // The histogram belongs to the pixels, not to the curve: it is read once per
  // image and then left alone, so dragging a point stays free.
  useEffect(() => {
    let cancelled = false;
    if (!source) {
      setHistogram(null);
      return;
    }

    void loadImage(source)
      .then((image) => {
        if (!cancelled) setHistogram(computeHistogram(image));
      })
      .catch(() => {
        if (!cancelled) setHistogram(null);
      });

    return () => {
      cancelled = true;
    };
  }, [source]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const size = GRAPH_SIZE;
    if (canvas.width !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const toCanvas = (point: CurvePoint) => ({
      x: point.x * size,
      y: (1 - point.y) * size,
    });

    // --- Histogram, behind everything ------------------------------------
    const bins = histogram?.[channel];
    if (bins) {
      ctx.beginPath();
      ctx.moveTo(0, size);
      for (let i = 0; i < LUT_SIZE; i += 1) {
        // The square root tames the peak: a photo with a large flat sky would
        // otherwise flatten every other tone into an invisible line.
        const height = Math.sqrt(bins[i]) * size * 0.92;
        ctx.lineTo((i / (LUT_SIZE - 1)) * size, size - height);
      }
      ctx.lineTo(size, size);
      ctx.closePath();
      ctx.fillStyle =
        channel === 'rgb' ? 'rgba(148, 163, 184, 0.22)' : `${CHANNEL_COLOURS[channel]}2e`;
      ctx.fill();
    }

    // --- Grid and the neutral diagonal -----------------------------------
    ctx.strokeStyle = 'rgba(128, 128, 148, 0.22)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) {
      const offset = (i / 4) * size;
      ctx.beginPath();
      ctx.moveTo(offset, 0);
      ctx.lineTo(offset, size);
      ctx.moveTo(0, offset);
      ctx.lineTo(size, offset);
      ctx.stroke();
    }

    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, size);
    ctx.lineTo(size, 0);
    ctx.stroke();
    ctx.setLineDash([]);

    // --- The curve itself, sampled exactly as the shader will see it ------
    const table = sampleCurve(points);
    ctx.beginPath();
    for (let i = 0; i < LUT_SIZE; i += 1) {
      const x = (i / (LUT_SIZE - 1)) * size;
      const y = (1 - table[i]) * size;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = CHANNEL_COLOURS[channel];
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // --- Control points ---------------------------------------------------
    points.forEach((point, index) => {
      const { x, y } = toCanvas(point);
      const active = index === hovered || index === draggingRef.current;
      ctx.beginPath();
      ctx.arc(x, y, active ? 5.5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = active ? CHANNEL_COLOURS[channel] : '#ffffff';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = CHANNEL_COLOURS[channel];
      ctx.stroke();
    });
  }, [channel, histogram, hovered, points]);

  useEffect(draw, [draw]);

  if (!doc || !layerId || !layer) return null;

  /** Pointer position as curve coordinates: 0..1 left to right, bottom to top. */
  const toCurveCoords = (event: React.PointerEvent): CurvePoint => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: 1 - (event.clientY - rect.top) / rect.height,
    };
  };

  const reachFor = (event: React.PointerEvent) =>
    POINT_REACH / event.currentTarget.getBoundingClientRect().width;

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const at = toCurveCoords(event);
    const existing = findCurvePoint(points, at, reachFor(event));

    // Right-click or Alt-click removes: the two conventions people try.
    if ((event.button === 2 || event.altKey) && existing > 0 && existing < points.length - 1) {
      setCurve(layerId, channel, removeCurvePoint(points, existing));
      commitCurves(layerId);
      return;
    }
    if (event.button !== 0) return;

    event.currentTarget.setPointerCapture(event.pointerId);

    if (existing >= 0) {
      draggingRef.current = existing;
      return;
    }

    // Clicking empty graph adds a point there and starts dragging it, so one
    // gesture both creates and places it.
    const { points: next, index } = insertCurvePoint(points, at);
    draggingRef.current = index;
    setCurve(layerId, channel, next);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const at = toCurveCoords(event);
    const index = draggingRef.current;

    if (index < 0) {
      setHovered(findCurvePoint(points, at, reachFor(event)));
      return;
    }

    setCurve(layerId, channel, moveCurvePoint(points, index, at));
  };

  const endDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (draggingRef.current < 0) return;
    draggingRef.current = -1;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    // The gesture ends here, so this is where one undo step belongs.
    commitCurves(layerId);
  };

  const anyBent = CURVE_CHANNELS.some((c) => !isCurveNeutral(layer.curves?.[c]));

  return (
    <div className="curves-panel">
      <div className="curves-tabs">
        {CURVE_CHANNELS.map((option) => (
          <button
            key={option}
            className={`curves-tab ${option === channel ? 'on' : ''} ${
              isCurveNeutral(layer.curves?.[option]) ? '' : 'bent'
            }`}
            style={{ '--channel-colour': CHANNEL_COLOURS[option] } as React.CSSProperties}
            onClick={() => setChannel(option)}
          >
            {t(`curves.channel.${option}`)}
          </button>
        ))}
      </div>

      <canvas
        ref={canvasRef}
        className="curves-graph"
        style={{ width: '100%', aspectRatio: '1 / 1' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => setHovered(-1)}
        onContextMenu={(e) => e.preventDefault()}
      />

      <p className="curves-hint">{t('curves.hint')}</p>

      <div className="curves-actions">
        <button
          onClick={() => resetCurves(layerId, channel)}
          disabled={isCurveNeutral(layer.curves?.[channel])}
        >
          {t('curves.resetChannel', { channel: t(`curves.channel.${channel}`) })}
        </button>
        <button onClick={() => resetCurves(layerId)} disabled={!anyBent}>
          <RotateCcw size={13} />
          <span>{t('curves.resetAll')}</span>
        </button>
      </div>
    </div>
  );
};
