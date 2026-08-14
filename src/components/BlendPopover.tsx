// src/components/BlendPopover.tsx
import { useState } from 'react';
import { Check, Info, X } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { blendActiveLayer } from '../ops/blend';
import { DEFAULT_HARMONIZE, type HarmonizeSettings } from '../engine/harmonize';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
}

const CONTROLS: Array<{ key: keyof HarmonizeSettings }> = [
  { key: 'strength' },
  { key: 'relight' },
  { key: 'grain' },
  { key: 'shadow' },
];

/**
 * Settings for making a pasted piece belong to the photo it landed in.
 *
 * The note at the bottom is not a disclaimer bolted on afterwards — it is the
 * honest boundary of what this does. It matches light, colour, grain and adds
 * a contact shadow. It cannot turn a subject lit from the left into one lit
 * from the right, and someone who knows that in advance will point the camera
 * differently next time instead of blaming the tool.
 */
export const BlendPopover: React.FC<Props> = ({ onClose }) => {
  const isAIBusy = useDocumentStore((s) => s.isAIBusy);
  const [settings, setSettings] = useState<HarmonizeSettings>(DEFAULT_HARMONIZE);
  const t = useT();

  const apply = async () => {
    const done = await blendActiveLayer(settings);
    if (done) onClose();
  };

  return (
    <div className="blend-popover" onPointerDown={(e) => e.stopPropagation()}>
      <div className="blend-head">
        <span>{t('blend.title')}</span>
        <button className="bubble-dismiss" onClick={onClose} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      {CONTROLS.map(({ key }) => (
        <div key={key} className="adjust-row">
          <label className="adjust-name" htmlFor={`blend-${key}`}>
            {t(`blend.${key}`)}
          </label>
          <input
            id={`blend-${key}`}
            className="adjust-slider"
            type="range"
            min={0}
            max={100}
            value={Math.round(settings[key] * 100)}
            onChange={(e) =>
              setSettings((current) => ({ ...current, [key]: Number(e.target.value) / 100 }))
            }
          />
          <span className="adjust-value">{Math.round(settings[key] * 100)}%</span>
        </div>
      ))}

      <p className="blend-note">
        <Info size={12} />
        <span>{t('blend.limits')}</span>
      </p>

      <button className="blend-apply" onClick={() => void apply()} disabled={isAIBusy}>
        <Check size={14} />
        <span>{t('blend.apply')}</span>
      </button>
    </div>
  );
};
