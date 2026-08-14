// src/components/FocusGuide.tsx
import { useEffect, useRef, useState } from 'react';
import { useDocumentStore, useActiveDocument, useViewport } from '../store/documentStore';
import { NEUTRAL_FOCUS, type FocusEffect } from '../types/document';

interface Props {
  containerWidth: number;
  containerHeight: number;
}

/** What the pointer is currently changing. */
type Grip = 'centre' | 'size' | 'feather';

/** Smallest sharp zone, as a fraction of the frame height. */
const MIN_SIZE = 0.02;

/**
 * Where the sharp zone is, drawn over the photo and moved by hand.
 *
 * Two sliders can describe a band, but nobody positions a tilt-shift by typing
 * numbers — you put the sharp strip on the row of houses you care about, and
 * you widen it until it covers them. So both the placing and the widening are
 * drags, and the sliders stay for the parts a drag is bad at.
 *
 * Only the geometric modes get a guide: a mask and a depth map already show
 * where they are, by blurring everything that is not the subject.
 */
export const FocusGuide: React.FC<Props> = ({ containerWidth, containerHeight }) => {
  const doc = useActiveDocument();
  const viewport = useViewport();
  const { setFocus, commitFocus } = useDocumentStore();

  const gripRef = useRef<Grip | null>(null);
  const [active, setActive] = useState<Grip | null>(null);

  /**
   * The next value the drag wants, and the frame that will apply it.
   *
   * A pointer can report a thousand times a second, and every report used to
   * become a store update — which re-runs the whole GPU pipeline, blur
   * included, on a twelve-megapixel photo. Coalescing to one update per frame
   * is the difference between a drag that follows the finger and one that
   * arrives a second later.
   */
  const pendingRef = useRef<FocusEffect | null>(null);
  const frameRef = useRef(0);

  useEffect(
    () => () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    },
    []
  );

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  const focus = layer?.focus;

  if (!doc || !viewport || !layerId || !focus) return null;
  if (focus.mode !== 'linear' && focus.mode !== 'radial') return null;

  const settings = { ...NEUTRAL_FOCUS, ...focus };
  const { zoom } = viewport;
  const originX = containerWidth / 2 + viewport.panX - (zoom * doc.width) / 2;
  const originY = containerHeight / 2 + viewport.panY - (zoom * doc.height) / 2;

  const width = doc.width * zoom;
  const height = doc.height * zoom;
  const centreX = originX + settings.centreX * width;
  const centreY = originY + settings.centreY * height;

  /** Queues a change for the next frame instead of applying it immediately. */
  const schedule = (next: FocusEffect) => {
    pendingRef.current = next;
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      const value = pendingRef.current;
      pendingRef.current = null;
      if (value) setFocus(layerId, value);
    });
  };

  /** Pointer position relative to the frame, 0..1 on each axis. */
  const toFraction = (event: React.PointerEvent) => {
    const rect = event.currentTarget.closest('.nabu-viewport-container')?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: (event.clientX - rect.left - originX) / width,
      y: (event.clientY - rect.top - originY) / height,
    };
  };

  const start = (grip: Grip) => (event: React.PointerEvent) => {
    event.stopPropagation();
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    gripRef.current = grip;
    setActive(grip);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const grip = gripRef.current;
    if (!grip) return;
    const at = toFraction(event);
    if (!at) return;

    if (grip === 'centre') {
      schedule({
        ...settings,
        centreX: Math.max(0, Math.min(1, at.x)),
        centreY: Math.max(0, Math.min(1, at.y)),
      });
      return;
    }

    // Everything is measured in fractions of the frame's height, which is the
    // unit the shader works in too — that is what keeps this outline on top of
    // the blur it is describing rather than near it.
    const reach = Math.hypot(
      (at.x - settings.centreX) * (width / height),
      at.y - settings.centreY
    );

    if (grip === 'size') {
      schedule({ ...settings, size: Math.max(MIN_SIZE, reach) });
    } else {
      // The softness handle sits outside the sharp edge, so what it sets is
      // the gap between the two.
      schedule({ ...settings, feather: Math.max(0, reach - settings.size) });
    }
  };

  const end = (event: React.PointerEvent) => {
    if (!gripRef.current) return;
    gripRef.current = null;
    setActive(null);

    // Whatever the last frame did not get to apply, apply now.
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
    }
    if (pendingRef.current) {
      setFocus(layerId, pendingRef.current);
      pendingRef.current = null;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    commitFocus(layerId);
  };

  const band = settings.size * height;
  const soft = settings.feather * height;
  const radiusY = settings.size * height;
  const radiusX = radiusY * settings.aspect;

  return (
    <div
      className={`focus-guide ${active ? 'dragging' : ''}`}
      style={{ left: centreX, top: centreY }}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      /*
       * Pointer events and mouse events are two families, and stopping one does
       * not stop the other: the browser fires a compatibility `mousedown`
       * separately, which reached the canvas underneath and started a lasso
       * every time the guide was grabbed. The canvas listens to mouse events,
       * so the mouse events have to be stopped here too.
       */
      onMouseDown={(event) => event.stopPropagation()}
      onMouseMove={(event) => active && event.stopPropagation()}
      onMouseUp={(event) => event.stopPropagation()}
    >
      {settings.mode === 'linear' ? (
        <div
          className="focus-band"
          style={{ transform: `translate(-50%, -50%) rotate(${settings.angle}deg)` }}
        >
          {/* The sharp edges are the handle: the line you can see is the line
              you grab, which is one fewer thing to explain than a dot. */}
          <span
            className="focus-line grab"
            style={{ top: -band }}
            onPointerDown={start('size')}
          />
          <span
            className="focus-line grab"
            style={{ top: band }}
            onPointerDown={start('size')}
          />
          <span
            className="focus-line soft grab"
            style={{ top: -(band + soft) }}
            onPointerDown={start('feather')}
          />
          <span
            className="focus-line soft grab"
            style={{ top: band + soft }}
            onPointerDown={start('feather')}
          />
        </div>
      ) : (
        <>
          {/* The ellipse itself stays untouchable: a rounded box is hit across
              its whole area, so making it grabbable would swallow every click
              inside the sharp zone — the part of the photograph being looked
              at. Two small handles instead. */}
          <span
            className="focus-ellipse"
            style={{ width: radiusX * 2, height: radiusY * 2 }}
          />
          <span
            className="focus-ellipse soft"
            style={{ width: (radiusX + soft) * 2, height: (radiusY + soft) * 2 }}
          />

          <span
            className="focus-grip"
            style={{ top: radiusY }}
            onPointerDown={start('size')}
          />
          <span
            className="focus-grip feather"
            style={{ top: radiusY + soft }}
            onPointerDown={start('feather')}
          />
        </>
      )}

      <span className="focus-centre" onPointerDown={start('centre')} />
    </div>
  );
};
