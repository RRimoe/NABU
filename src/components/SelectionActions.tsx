// src/components/SelectionActions.tsx
import { Scissors, Brush, X, Lightbulb, Eraser } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import { extractObject, removeSelectedRegion } from '../ai/operations';
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
  const { setSelection, setRefineMode, isAIBusy } = useDocumentFields('setSelection', 'setRefineMode', 'isAIBusy');
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
        {/* Only actions scoped to the selection belong here. Removing the
            background and enlarging work on the whole photo, so they live in
            the opening bubble and in the command line instead. */}
        <button
          onClick={() => void extractObject()}
          disabled={isAIBusy}
          title={t('bubble.extract.hint')}
        >
          <Scissors size={14} />
          <span>{t('bubble.extract')}</span>
        </button>

        <button
          onClick={() => void removeSelectedRegion()}
          disabled={isAIBusy}
          title={t('bubble.remove.hint')}
        >
          <Eraser size={14} />
          <span>{t('bubble.remove')}</span>
        </button>

        {canRefine && (
          <button onClick={() => setRefineMode('erase')} title={t('bubble.refine.hint')}>
            <Brush size={14} />
            <span>{t('bubble.refine')}</span>
          </button>
        )}
      </div>
    </div>
  );
};
