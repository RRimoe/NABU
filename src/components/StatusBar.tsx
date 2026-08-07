// src/components/StatusBar.tsx
import { useState } from 'react';
import { RotateCcw, RotateCw, Maximize2, PanelRight } from 'lucide-react';
import {
  useDocumentStore,
  useViewport,
  useActiveDocument,
  useCanUndo,
  useCanRedo,
} from '../store/documentStore';
import { ResizePopover } from './ResizePopover';
import { useModelStore } from '../store/modelStore';
import { useT } from '../i18n';

interface Props {
  panelsOpen: boolean;
  /** False while there is nothing to put in the panels. */
  panelsAvailable: boolean;
  onTogglePanels: () => void;
}

export const StatusBar: React.FC<Props> = ({ panelsOpen, panelsAvailable, onTogglePanels }) => {
  const viewport = useViewport();
  const doc = useActiveDocument();
  const [resizing, setResizing] = useState(false);
  const { undo, redo, setViewport, requestFitToScreen } = useDocumentStore();
  const canUndo = useCanUndo();
  const canRedo = useCanRedo();
  const accelerator = useModelStore((s) => s.accelerator);
  const t = useT();

  return (
    <footer className="nabu-status">
      <div className="status-left">
        <span className="status-version">NABU 0.1.0</span>
        <span className="status-dot">-</span>
        <span>{t('status.allLocal')}</span>
        <span className="status-dot">-</span>
        <span>{t('status.yourData')}</span>
        {accelerator && (
          <>
            <span className="status-dot">-</span>
            <span>{accelerator}</span>
          </>
        )}
      </div>

      <div className="status-right">
        {doc && (
          <div className="status-size-wrap">
            {resizing && <ResizePopover onClose={() => setResizing(false)} />}
            <button
              className={`status-size ${resizing ? 'on' : ''}`}
              onClick={() => setResizing((v) => !v)}
              title={t('resize.title')}
            >
              {doc.width} x {doc.height}
            </button>
          </div>
        )}

        <button className="status-btn" onClick={undo} disabled={!canUndo} title={t('status.undo')}>
          <RotateCcw size={15} />
        </button>
        <button className="status-btn" onClick={redo} disabled={!canRedo} title={t('status.redo')}>
          <RotateCw size={15} />
        </button>

        {viewport && (
          <>
            <input
              className="status-zoom-slider"
              type="range"
              min={5}
              max={400}
              value={Math.round(viewport.zoom * 100)}
              onChange={(e) => setViewport((v) => ({ ...v, zoom: Number(e.target.value) / 100 }))}
              title={t('status.zoom')}
            />
            <span className="status-zoom">{Math.round(viewport.zoom * 100)}%</span>

            <button className="status-btn" onClick={requestFitToScreen} title={t('status.fit')}>
              <Maximize2 size={15} />
            </button>
          </>
        )}

        {panelsAvailable && (
          <button
            className={`status-btn ${panelsOpen ? 'on' : ''}`}
            onClick={onTogglePanels}
            title={t('status.panels')}
          >
            <PanelRight size={15} />
          </button>
        )}
      </div>
    </footer>
  );
};
