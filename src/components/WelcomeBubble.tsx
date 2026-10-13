// src/components/WelcomeBubble.tsx
import { Sparkles, TrendingUp, X } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import { removeBackground, upscale } from '../ai/operations';
import { useT } from '../i18n';

interface Props {
  x: number;
  y: number;
}

/**
 * Bubble shown right after a photo is opened.
 *
 * It offers what Nabu can genuinely do to the whole photo, and points at the
 * gesture that unlocks everything else. It deliberately does not claim to have
 * understood the picture: nothing here has looked at its content.
 */
export const WelcomeBubble: React.FC<Props> = ({ x, y }) => {
  const doc = useActiveDocument();
  const { isAIBusy, dismissWelcome, welcomeDismissed } = useDocumentFields('isAIBusy', 'dismissWelcome', 'welcomeDismissed');
  const t = useT();

  const hasImage = Boolean(
    doc && doc.rootLayerIds.some((id) => doc.layers[id]?.imageDataUrl)
  );
  if (!hasImage || welcomeDismissed) return null;

  return (
    <div className="welcome-bubble" style={{ left: x, top: y }}>
      <div className="bubble-head">
        <Sparkles size={14} className="bubble-icon accent" />
        <span>{t('welcome.question')}</span>
        <button className="bubble-dismiss" onClick={dismissWelcome} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      <div className="bubble-actions">
        <button
          onClick={() => void removeBackground()}
          disabled={isAIBusy}
          title={t('bubble.remove.hint')}
        >
          <Sparkles size={14} />
          <span>{t('bubble.removeBackground')}</span>
        </button>

        <button
          onClick={() => void upscale()}
          disabled={isAIBusy}
          title={t('bubble.enhance.hint')}
        >
          <TrendingUp size={14} />
          <span>{t('bubble.enhance')}</span>
        </button>
      </div>

      <p className="bubble-hint">{t('welcome.hint')}</p>
    </div>
  );
};
