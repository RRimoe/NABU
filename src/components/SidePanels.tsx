// src/components/SidePanels.tsx
import { Check, X, Loader, History, Layers as LayersIcon, SlidersHorizontal } from 'lucide-react';
import { useActivityStore } from '../store/activityStore';
import { useActiveDocument } from '../store/documentStore';
import { LayerManager } from './LayerManager';
import { AdjustPanel } from './AdjustPanel';
import { useT } from '../i18n';
import { describeError } from '../i18n/backendError';

const formatDuration = (ms?: number) => {
  if (ms === undefined) return '';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
};

/**
 * Column of advanced panels: hidden by default, revealed with Tab.
 *
 * It holds only real information — the history records operations that were
 * actually performed, with measured timings.
 */
export const SidePanels: React.FC = () => {
  const entries = useActivityStore((s) => s.entries);
  const doc = useActiveDocument();
  const t = useT();

  // An empty panel is noise: each section appears only when it has content.
  const showLayers = Boolean(doc && doc.rootLayerIds.length > 0);
  const showActivity = entries.length > 0;
  if (!showLayers && !showActivity) return null;

  return (
    <aside className="nabu-panels">
      {showLayers && (
        <section className="side-panel">
          <header className="side-panel-header">
            <SlidersHorizontal size={14} />
            <span>{t('panels.adjust')}</span>
          </header>
          <AdjustPanel />
        </section>
      )}

      {showLayers && (
        <section className="side-panel">
          <header className="side-panel-header">
            <LayersIcon size={14} />
            <span>{t('panels.layers')}</span>
          </header>
          <LayerManager />
        </section>
      )}

      {showActivity && (
        <section className="side-panel">
          <header className="side-panel-header">
            <History size={14} />
            <span>{t('panels.history')}</span>
          </header>

          <div className="activity-list">
            {entries.map((entry) => (
              <article key={entry.id} className="activity-entry">
                <div className="activity-title">{t(entry.titleKey)}</div>

                {entry.steps.map((step, index) => (
                  <div key={index} className={`activity-step ${step.status}`}>
                    {step.status === 'running' && <Loader size={12} className="spin-icon" />}
                    {step.status === 'done' && <Check size={12} />}
                    {step.status === 'failed' && <X size={12} />}
                    <span className="activity-step-label">{t(step.labelKey, step.params)}</span>
                    <span className="activity-step-time">{formatDuration(step.durationMs)}</span>
                  </div>
                ))}

                {entry.error !== undefined && (
                  <div className="activity-error">{describeError(entry.error)}</div>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
    </aside>
  );
};
