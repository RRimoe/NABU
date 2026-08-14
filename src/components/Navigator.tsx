// src/components/Navigator.tsx
import { useEffect, useRef, useState } from 'react';
import { useDocumentStore, useActiveDocument, useViewport } from '../store/documentStore';
import { loadImage, paintLayers, RENDER_THEMES } from '../engine/RenderEngine';
import { useT } from '../i18n';

/** Longest side of the thumbnail, in CSS pixels. */
const THUMB = 260;

/**
 * The whole photo, small, with a box showing what is on screen.
 *
 * Zoomed in past the point where the picture fits, the canvas stops telling you
 * where you are — you can be on an ear with no way to know whose. Scrollbars
 * answer that question badly and take up an edge each; a map answers it at a
 * glance and can be dragged, which is also the fastest way to get somewhere.
 *
 * It appears only when it has something to say: at a zoom where everything is
 * visible, a box around the whole thumbnail is decoration.
 */
export const Navigator: React.FC = () => {
  const doc = useActiveDocument();
  const viewport = useViewport();
  const viewportSize = useDocumentStore((s) => s.viewportSize);
  const setViewport = useDocumentStore((s) => s.setViewport);
  const theme = useDocumentStore((s) => s.theme);
  const t = useT();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draggingRef = useRef(false);
  const [ready, setReady] = useState(false);

  // The map is redrawn when the pixels change, not when the view moves: panning
  // must not re-composite the document sixty times a second.
  const signature = doc
    ? doc.rootLayerIds
        .map((id) => {
          const layer = doc.layers[id];
          return `${id}:${layer?.imageDataUrl?.length ?? 0}:${layer?.visible ? 1 : 0}`;
        })
        .join('|')
    : '';

  // Read through a ref so the effect below does not list `doc` as a dependency:
  // the document object is replaced on every edit, including a pan, and
  // recompositing the map each time would defeat the point of the signature.
  const docRef = useRef(doc);
  docRef.current = doc;

  useEffect(() => {
    const canvas = canvasRef.current;
    const doc = docRef.current;
    if (!canvas || !doc) return;

    let cancelled = false;
    const scale = THUMB / Math.max(doc.width, doc.height);
    const width = Math.max(1, Math.round(doc.width * scale));
    const height = Math.max(1, Math.round(doc.height * scale));

    void (async () => {
      const sources = Array.from(
        new Set(
          doc.rootLayerIds
            .map((id) => doc.layers[id]?.imageDataUrl)
            .filter((src): src is string => Boolean(src))
        )
      );

      const resolved = new Map<string, HTMLImageElement>();
      await Promise.all(
        sources.map((src) =>
          loadImage(src)
            .then((image) => {
              resolved.set(src, image);
            })
            .catch(() => undefined)
        )
      );
      if (cancelled) return;

      // Only the backing store is set here. The element's *box* comes from the
      // stylesheet, which knows the photo's aspect ratio from the first paint —
      // sizing it in an async effect left a moment where the map was the
      // browser's default 300x150 whatever shape the photo was.
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.scale(scale, scale);
      // No adjustment pass: this is a map, and it is redrawn often enough that
      // running the whole grade through it would cost more than it tells.
      paintLayers(ctx, doc, RENDER_THEMES[theme], (src) => resolved.get(src) ?? null);
      ctx.restore();

      setReady(true);
    })();

    return () => {
      cancelled = true;
    };
  }, [signature, theme, doc?.id, doc?.width, doc?.height]);

  if (!doc || !viewport) return null;

  // What the canvas is showing, in document coordinates, then as fractions of
  // the photo. Fractions rather than pixels: the map takes whatever width the
  // panel gives it, and a rectangle in percentages stays right at any size.
  const visibleW = viewportSize.width / viewport.zoom;
  const visibleH = viewportSize.height / viewport.zoom;
  const centreX = doc.width / 2 - viewport.panX / viewport.zoom;
  const centreY = doc.height / 2 - viewport.panY / viewport.zoom;

  const covers = visibleW >= doc.width && visibleH >= doc.height;

  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  const left = clamp((centreX - visibleW / 2) / doc.width);
  const top = clamp((centreY - visibleH / 2) / doc.height);
  const right = clamp((centreX + visibleW / 2) / doc.width);
  const bottom = clamp((centreY + visibleH / 2) / doc.height);

  /** Centres the view on the point that was clicked. */
  const goTo = (event: React.PointerEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const docX = ((event.clientX - rect.left) / rect.width) * doc.width;
    const docY = ((event.clientY - rect.top) / rect.height) * doc.height;

    setViewport((previous) => ({
      ...previous,
      panX: (doc.width / 2 - docX) * previous.zoom,
      panY: (doc.height / 2 - docY) * previous.zoom,
    }));
  };

  return (
    <div className="navigator">
      <div
        className="navigator-map"
        // The map is the photo's shape, always: a square photo gets a square
        // map, and it takes whatever width the panel has to give.
        style={{ aspectRatio: `${doc.width} / ${doc.height}` }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          draggingRef.current = true;
          goTo(event);
        }}
        onPointerMove={(event) => {
          if (draggingRef.current) goTo(event);
        }}
        onPointerUp={(event) => {
          draggingRef.current = false;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
      >
        <canvas ref={canvasRef} className={ready ? 'ready' : ''} />

        {/* Only worth drawing when it is smaller than the picture. */}
        {!covers && (
          <span
            className="navigator-view"
            style={{
              left: `${left * 100}%`,
              top: `${top * 100}%`,
              width: `${(right - left) * 100}%`,
              height: `${(bottom - top) * 100}%`,
            }}
          />
        )}
      </div>

      <p className="navigator-hint">
        {covers ? t('navigator.whole') : t('navigator.hint')}
      </p>
    </div>
  );
};
