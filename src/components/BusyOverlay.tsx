// src/components/BusyOverlay.tsx
import { useDocumentFields } from '../store/documentStore';
import { useT } from '../i18n';

/**
 * Shown while a model is running.
 *
 * Local inference can take seconds on CPU, and without a sign of life the
 * application looks frozen. The sweeping band conveys "working"; the bar below
 * carries the real percentage reported by the backend, so it never invents
 * progress that is not happening.
 */
export const BusyOverlay: React.FC = () => {
  const { isAIBusy, aiProgress, aiStatusMessage } = useDocumentFields('isAIBusy', 'aiProgress', 'aiStatusMessage');
  const t = useT();

  if (!isAIBusy) return null;

  // The backend sends a stage identifier; unknown stages fall back to a
  // neutral label rather than showing a raw key.
  const stageKey = aiStatusMessage ? `activity.stage.${aiStatusMessage}` : '';
  const translated = stageKey ? t(stageKey) : '';
  const label = translated && translated !== stageKey ? translated : t('busy.working');

  return (
    <div className="nabu-busy" role="status" aria-live="polite">
      <div className="busy-sweep" />

      <div className="busy-card">
        <div className="busy-orbit">
          <span />
          <span />
          <span />
        </div>

        <div className="busy-text">
          <span className="busy-label">{label}</span>
          <div className="busy-bar">
            <div
              className="busy-bar-fill"
              style={{ width: `${Math.max(4, Math.min(100, aiProgress))}%` }}
            />
          </div>
        </div>

        <span className="busy-percent">{Math.round(aiProgress)}%</span>
      </div>
    </div>
  );
};
