// src/components/RefineBar.tsx
import { Eraser, Undo2, Check } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { useT } from '../i18n';

/**
 * Controls for the edge-refinement brush.
 *
 * It replaces the command line only while an edge is being corrected, and
 * disappears as soon as the user confirms: nothing permanent is added to the
 * interface.
 */
export const RefineBar: React.FC = () => {
  const { refineMode, setRefineMode, refineSize, setRefineSize } = useDocumentStore();
  const t = useT();

  return (
    <div className="nabu-refine">
      <span className="refine-title">{t('refine.title')}</span>

      <div className="refine-modes">
        <button
          className={refineMode === 'restore' ? 'active' : ''}
          onClick={() => setRefineMode('restore')}
          title={t('refine.restore.hint')}
        >
          <Undo2 size={14} />
          <span>{t('refine.restore')}</span>
        </button>
        <button
          className={refineMode === 'erase' ? 'active' : ''}
          onClick={() => setRefineMode('erase')}
          title={t('refine.erase.hint')}
        >
          <Eraser size={14} />
          <span>{t('refine.erase')}</span>
        </button>
      </div>

      <label className="refine-size">
        <span>{t('refine.size')}</span>
        <input
          type="range"
          min={4}
          max={300}
          value={refineSize}
          onChange={(e) => setRefineSize(Number(e.target.value))}
        />
        <span className="refine-size-value">{refineSize}</span>
      </label>

      <button className="refine-done" onClick={() => setRefineMode(null)} title={t('refine.done.hint')}>
        <Check size={14} />
        <span>{t('refine.done')}</span>
      </button>
    </div>
  );
};
