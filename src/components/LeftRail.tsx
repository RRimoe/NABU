// src/components/LeftRail.tsx
import { FolderOpen, Hand, Sparkles, Share, Settings } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { useT } from '../i18n';

interface Props {
  onOpen: () => void;
  onExport: () => void;
  onSettings: () => void;
  onEdit: () => void;
  hasPhoto: boolean;
}

/**
 * Side rail reduced to the essentials: what you want to do, not which tool.
 *
 * There is no "select" entry, because dragging on the photo already selects —
 * a button for something that happens by itself is dead interface. "Move" is
 * here instead, since panning a zoomed photo has no other visible affordance.
 *
 * The letters "AI" never appear: people want to edit a photo, not to be told
 * which technology edits it.
 */
export const LeftRail: React.FC<Props> = ({
  onOpen,
  onExport,
  onSettings,
  onEdit,
  hasPhoto,
}) => {
  const { activeTool, setActiveTool, isAIBusy } = useDocumentStore();
  const t = useT();

  const panning = activeTool === 'pan';

  return (
    <nav className="nabu-rail">
      <div className="rail-brand">
        <span className="rail-logo" />
        <span className="rail-wordmark">NABU</span>
      </div>

      <div className="rail-items">
        <button className="rail-item" onClick={onOpen} title={t('rail.open.hint')}>
          <FolderOpen size={20} />
          <span>{t('rail.open')}</span>
        </button>

        <button
          className={`rail-item ${panning ? 'active' : ''}`}
          onClick={() => setActiveTool(panning ? 'select' : 'pan')}
          title={t('rail.move.hint')}
          disabled={!hasPhoto}
        >
          <Hand size={20} />
          <span>{t('rail.move')}</span>
        </button>

        <button
          className={`rail-item accent ${isAIBusy ? 'busy' : ''}`}
          onClick={onEdit}
          title={t('rail.edit.hint')}
          disabled={!hasPhoto}
        >
          <Sparkles size={20} />
          <span>{t('rail.edit')}</span>
        </button>

        <button
          className="rail-item"
          onClick={onExport}
          title={t('rail.export.hint')}
          disabled={!hasPhoto}
        >
          <Share size={20} />
          <span>{t('rail.export')}</span>
        </button>
      </div>

      <div className="rail-footer">
        <button className="rail-icon" onClick={onSettings} title={t('settings.title')}>
          <Settings size={18} />
        </button>
      </div>
    </nav>
  );
};
