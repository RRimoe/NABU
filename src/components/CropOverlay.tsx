// src/components/CropOverlay.tsx
import { useEffect, useRef, useState } from 'react';
import { Check, X } from 'lucide-react';
import { useActiveDocument, useViewport } from '../store/documentStore';
import { clampCrop, cropDocument, fitRatio, CROP_RATIOS, MIN_CROP, type CropRect } from '../ops/crop';
import { useT } from '../i18n';

interface Props {
  /** Size of the viewport container, in CSS pixels. */
  containerWidth: number;
  containerHeight: number;
  onClose: () => void;
}

type Grip = 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w' | 'move';

const GRIPS: Array<{ id: Grip; fx: number; fy: number }> = [
  { id: 'nw', fx: 0, fy: 0 },
  { id: 'n', fx: 0.5, fy: 0 },
  { id: 'ne', fx: 1, fy: 0 },
  { id: 'e', fx: 1, fy: 0.5 },
  { id: 'se', fx: 1, fy: 1 },
  { id: 's', fx: 0.5, fy: 1 },
  { id: 'sw', fx: 0, fy: 1 },
  { id: 'w', fx: 0, fy: 0.5 },
];

/**
 * The crop frame, drawn in HTML over the canvas rather than inside it.
 *
 * Everything here is chrome, not picture: keeping it in the DOM means the
 * handles are real focusable elements at a constant size, and the renderer
 * stays a thing that draws documents.
 *
 * The frame is only ever applied on confirmation. Cropping throws pixels away,
 * and an operation that destroys something on the way past is not one people
 * forgive.
 */
export const CropOverlay: React.FC<Props> = ({ containerWidth, containerHeight, onClose }) => {
  const doc = useActiveDocument();
  const viewport = useViewport();
  const t = useT();

  const [rect, setRect] = useState<CropRect | null>(null);
  const [ratio, setRatio] = useState<number | null>(null);
  const [ratioId, setRatioId] = useState('free');
  const [busy, setBusy] = useState(false);

  const dragRef = useRef<{ grip: Grip; startX: number; startY: number; origin: CropRect } | null>(
    null
  );

  // A new document starts with the whole picture framed: the first drag then
  // takes something away, which is what cropping is.
  useEffect(() => {
    if (!doc) return;
    setRect({ x: 0, y: 0, width: doc.width, height: doc.height });
    setRatio(null);
    setRatioId('free');
  }, [doc?.id, doc?.width, doc?.height]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
      if (event.key === 'Enter') void apply();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (!doc || !viewport || !rect) return null;

  const { zoom } = viewport;
  const originX = containerWidth / 2 + viewport.panX - (zoom * doc.width) / 2;
  const originY = containerHeight / 2 + viewport.panY - (zoom * doc.height) / 2;

  const toScreen = (value: number) => value * zoom;

  const chooseRatio = (id: string, value: number | null) => {
    setRatioId(id);
    // "Original" means the document's own proportions, which is the one ratio
    // that cannot be written down in advance.
    const resolved = value === 0 ? doc.width / doc.height : value;
    setRatio(resolved);
    if (resolved !== null) setRect(fitRatio(resolved, doc.width, doc.height));
  };

  const startDrag = (grip: Grip) => (event: React.PointerEvent) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      grip,
      startX: event.clientX,
      startY: event.clientY,
      origin: rect,
    };
  };

  const onDrag = (event: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;

    const dx = (event.clientX - drag.startX) / zoom;
    const dy = (event.clientY - drag.startY) / zoom;
    const origin = drag.origin;

    if (drag.grip === 'move') {
      setRect(
        clampCrop(
          { ...origin, x: origin.x + dx, y: origin.y + dy },
          doc.width,
          doc.height
        )
      );
      return;
    }

    const spec = GRIPS.find((g) => g.id === drag.grip);
    if (!spec) return;

    let left = origin.x;
    let top = origin.y;
    let right = origin.x + origin.width;
    let bottom = origin.y + origin.height;

    if (spec.fx === 0) left = Math.min(right - MIN_CROP, origin.x + dx);
    if (spec.fx === 1) right = Math.max(left + MIN_CROP, right + dx);
    if (spec.fy === 0) top = Math.min(bottom - MIN_CROP, origin.y + dy);
    if (spec.fy === 1) bottom = Math.max(top + MIN_CROP, bottom + dy);

    let width = right - left;
    let height = bottom - top;

    // A locked ratio wins over the exact pointer position: the side the user
    // is not holding follows, anchored on the opposite edge.
    if (ratio) {
      if (spec.fx === 0.5) width = height * ratio;
      else height = width / ratio;

      if (spec.fx === 0) left = right - width;
      else right = left + width;
      if (spec.fy === 0) top = bottom - height;
      else bottom = top + height;
    }

    setRect(
      clampCrop(
        { x: left, y: top, width: right - left, height: bottom - top },
        doc.width,
        doc.height
      )
    );
  };

  const endDrag = (event: React.PointerEvent) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  };

  const apply = async () => {
    setBusy(true);
    await cropDocument(rect);
    setBusy(false);
    onClose();
  };

  const untouched =
    Math.round(rect.width) === doc.width && Math.round(rect.height) === doc.height;

  const frame = {
    left: originX + toScreen(rect.x),
    top: originY + toScreen(rect.y),
    width: toScreen(rect.width),
    height: toScreen(rect.height),
  };

  return (
    <div className="crop-layer" onPointerMove={onDrag} onPointerUp={endDrag}>
      {/* Four panels rather than one box-shadow: they dim exactly the discarded
          area even when the photo runs past the edge of the viewport. */}
      <div className="crop-shade" style={{ left: 0, top: 0, right: 0, height: frame.top }} />
      <div
        className="crop-shade"
        style={{ left: 0, top: frame.top + frame.height, right: 0, bottom: 0 }}
      />
      <div
        className="crop-shade"
        style={{ left: 0, top: frame.top, width: frame.left, height: frame.height }}
      />
      <div
        className="crop-shade"
        style={{
          left: frame.left + frame.width,
          top: frame.top,
          right: 0,
          height: frame.height,
        }}
      />

      <div
        className="crop-frame"
        style={frame}
        onPointerDown={startDrag('move')}
      >
        {/* Thirds, the composition guide every camera draws. */}
        <span className="crop-third v" style={{ left: '33.33%' }} />
        <span className="crop-third v" style={{ left: '66.66%' }} />
        <span className="crop-third h" style={{ top: '33.33%' }} />
        <span className="crop-third h" style={{ top: '66.66%' }} />

        {GRIPS.map(({ id, fx, fy }) => (
          <span
            key={id}
            className={`crop-grip ${id}`}
            style={{ left: `${fx * 100}%`, top: `${fy * 100}%` }}
            onPointerDown={startDrag(id)}
          />
        ))}
      </div>

      <div className="crop-bar" onPointerDown={(e) => e.stopPropagation()}>
        <span className="crop-readout">
          {Math.round(rect.width)} x {Math.round(rect.height)}
        </span>

        <span className="action-sep" />

        {CROP_RATIOS.map(({ id, ratio: value }) => (
          <button
            key={id}
            className={ratioId === id ? 'on' : ''}
            onClick={() => chooseRatio(id, value)}
          >
            {t(`crop.ratio.${id}`)}
          </button>
        ))}

        <span className="action-sep" />

        <button className="crop-cancel" onClick={onClose} title={t('crop.cancel.hint')}>
          <X size={14} />
        </button>
        <button
          className="crop-apply"
          onClick={() => void apply()}
          disabled={busy || untouched}
          title={t('crop.apply.hint')}
        >
          <Check size={14} />
          <span>{t('crop.apply')}</span>
        </button>
      </div>
    </div>
  );
};
