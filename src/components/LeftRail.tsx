// src/components/LeftRail.tsx
import { FolderOpen, Lasso, Sparkles, Share, Settings } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { useT } from '../i18n';

interface Props {
  onOpen: () => void;
  onExport: () => void;
  onSettings: () => void;
  canExport: boolean;
}

/**
 * Side rail reduced to the essentials: what you want to do, not which tool.
 *
 * The letters "AI" never appear here — people want to edit a photo, not to be
 * told which technology edits it.
 */
export const LeftRail: React.FC<Props> = ({ onOpen, onExport, onSettings, canExport }) => {
  const { activeTool, setActiveTool, isAIBusy } = useDocumentStore();
  const t = useT();

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
          className={`rail-item ${activeTool === 'ai-lasso' ? 'active' : ''}`}
          onClick={() => setActiveTool('ai-lasso')}
          title={t('rail.select.hint')}
          disabled={!canExport}
        >
          <Lasso size={20} />
          <span>{t('rail.select')}</span>
        </button>

        <button
          className={`rail-item accent ${isAIBusy ? 'busy' : ''}`}
          onClick={() => setActiveTool('ai-lasso')}
          title={t('rail.edit.hint')}
          disabled={!canExport}
        >
          <Sparkles size={20} />
          <span>{t('rail.edit')}</span>
        </button>

        <button
          className="rail-item"
          onClick={onExport}
          title={t('rail.export.hint')}
          disabled={!canExport}
        >
          <Share size={20} />
          <span>{t('rail.export')}</span>
        </button>
      </div>

      <div className="rail-footer">
        <button className="rail-icon" onClick={onSettings} title={t('rail.settings')}>
          <Settings size={18} />
        </button>
      </div>
    </nav>
  );
};
