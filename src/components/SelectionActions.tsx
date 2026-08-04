// src/components/SelectionActions.tsx
import { Sparkles, TrendingUp, Brush, X, Lightbulb } from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { removeBackground, upscale } from '../ai/operations';
import { useT } from '../i18n';

interface Props {
  x: number;
  y: number;
}

/**
 * Contextual actions next to the selection.
 *
 * This replaces the toolbar: instead of hunting for a command in a menu, you
 * point at *what* and choose *how* right where you are already looking.
 */
export const SelectionActions: React.FC<Props> = ({ x, y }) => {
  const doc = useActiveDocument();
  const { setSelection, setRefineMode, isAIBusy } = useDocumentStore();
  const t = useT();

  const activeLayer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  // Refining only makes sense on a layer that still knows its original.
  const canRefine = Boolean(activeLayer?.sourceLayerId);

  return (
    <div
      className="selection-bubble"
      style={{ left: x, top: y }}
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
    >
      <div className="bubble-head">
        <Lightbulb size={14} className="bubble-icon" />
        <span>{t('bubble.question')}</span>
        <button
          className="bubble-dismiss"
          onClick={() => setSelection(null)}
          title={t('bubble.dismiss')}
        >
          <X size={13} />
        </button>
      </div>

      <div className="bubble-actions">
        <button onClick={() => void removeBackground()} disabled={isAIBusy}>
          <Sparkles size={14} />
          <span>{t('bubble.remove')}</span>
        </button>

        <button onClick={() => void upscale()} disabled={isAIBusy}>
          <TrendingUp size={14} />
          <span>{t('bubble.enhance')}</span>
        </button>

        {canRefine && (
          <button onClick={() => setRefineMode('erase')}>
            <Brush size={14} />
            <span>{t('bubble.refine')}</span>
          </button>
        )}
      </div>
    </div>
  );
};
