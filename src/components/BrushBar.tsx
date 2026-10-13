// src/components/BrushBar.tsx
import { Check } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import { canPaint } from '../ops/paint';
import { useT } from '../i18n';

interface Props {
  onDone: () => void;
}

export const MIN_BRUSH = 1;
export const MAX_BRUSH = 500;

/**
 * Size, strength and colour of the brush or eraser, shown only while one of
 * them is in hand. Size is in photo pixels, so a stroke covers the same part of
 * the picture whatever the zoom.
 */
export const BrushBar: React.FC<Props> = ({ onDone }) => {
  const doc = useActiveDocument();
  const {
    activeTool,
    brushSize,
    brushOpacity,
    brushColor,
    setBrushSize,
    setBrushOpacity,
    setBrushColor,
  } = useDocumentFields(
    'activeTool',
    'brushSize',
    'brushOpacity',
    'brushColor',
    'setBrushSize',
    'setBrushOpacity',
    'setBrushColor',
  );
  const t = useT();

  const erasing = activeTool === 'eraser';
  const healing = activeTool === 'heal';
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : undefined;

  return (
    <div
      className="transform-bar brush-bar"
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
    >
      <span className="transform-readout">{t(healing ? 'brush.heal' : erasing ? 'brush.eraser' : 'brush.brush')}</span>

      <span className="transform-sep" />

      <label className="brush-field" title={t('brush.size.hint')}>
        <span>{t('brush.size')}</span>
        <input
          type="range"
          min={MIN_BRUSH}
          max={MAX_BRUSH}
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
        />
        <b>{brushSize}</b>
      </label>

      {healing ? (
        <span className="transform-readout dim">{t('brush.heal.howto')}</span>
      ) : (
        <label className="brush-field">
          <span>{t('brush.opacity')}</span>
          <input
            type="range"
            min={5}
            max={100}
            value={Math.round(brushOpacity * 100)}
            onChange={(e) => setBrushOpacity(Number(e.target.value) / 100)}
          />
          <b>{Math.round(brushOpacity * 100)}</b>
        </label>
      )}

      {!erasing && !healing && (
        <label className="brush-colour" title={t('brush.colour')}>
          <input type="color" value={brushColor} onChange={(e) => setBrushColor(e.target.value)} />
        </label>
      )}

      {!canPaint(layer) && <span className="transform-readout dim">{t('brush.cannotPaint')}</span>}

      <span className="transform-sep" />

      <button className="transform-done" onClick={onDone}>
        <Check size={14} />
        <span>{t('transform.done')}</span>
      </button>
    </div>
  );
};
