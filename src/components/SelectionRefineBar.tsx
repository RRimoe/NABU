// src/components/SelectionRefineBar.tsx
import { Check, Undo2, X } from 'lucide-react';
import { useDocumentStore, useSelection } from '../store/documentStore';
import { useT } from '../i18n';
import { refineSelection } from '../ai/operations';

/**
 * Controls for correcting a selection by pointing at it.
 *
 * It says what a click does and offers a way back, and then gets out of the
 * way: there is no confirmation between the click and the result. Correcting a
 * selection is a loop — point, look, point again — and a question in the middle
 * of that loop asks the user to restate what they had already decided by
 * clicking. So the sign of a click is read off the pixels instead: on something
 * already selected it can only mean "not that", and outside it can only mean
 * "this too".
 *
 * Like the edge brush, it replaces the command line only while it is needed and
 * leaves nothing permanent behind.
 */
export const SelectionRefineBar: React.FC = () => {
  const {
    pointRefine,
    setPointRefine,
    undoRefinePoint,
    clearRefinePoints,
    setRefineCut,
    sessions,
    activeId,
  } = useDocumentStore();
  const selection = useSelection();
  const t = useT();

  if (!pointRefine || !selection) return null;

  const points = sessions[activeId]?.refinePoints ?? [];

  const undo = async () => {
    undoRefinePoint();
    const left = useDocumentStore.getState().sessions[activeId]?.refinePoints ?? [];
    setRefineCut(left.length ? await refineSelection(left) : null);
  };

  return (
    <div className="nabu-refine">
      <span className="refine-title">{t('pointRefine.title')}</span>
      <span className="refine-hint">{t('pointRefine.hint')}</span>

      <div className="refine-modes">
        <button onClick={undo} disabled={points.length === 0} title={t('pointRefine.undo.hint')}>
          <Undo2 size={14} />
          <span>{t('pointRefine.undo')}</span>
        </button>
        <button
          onClick={() => clearRefinePoints()}
          disabled={points.length === 0}
          title={t('pointRefine.reset.hint')}
        >
          <X size={14} />
          <span>{t('pointRefine.reset')}</span>
        </button>
      </div>

      <button className="refine-done" onClick={() => setPointRefine(false)}>
        <Check size={14} />
        <span>{t('pointRefine.done')}</span>
      </button>
    </div>
  );
};
