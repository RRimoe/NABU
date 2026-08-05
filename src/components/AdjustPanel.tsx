// src/components/AdjustPanel.tsx
import { RotateCcw } from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { type Adjustments, NEUTRAL_ADJUSTMENTS, isNeutral } from '../types/document';
import { useT } from '../i18n';

interface Slider {
  key: keyof Adjustments;
  min: number;
  max: number;
}

interface Group {
  titleKey: string;
  sliders: Slider[];
}

/**
 * Grouped the way a photographer expects to find them, not the way the shader
 * happens to evaluate them.
 */
const GROUPS: Group[] = [
  {
    titleKey: 'adjust.group.tone',
    sliders: [
      { key: 'exposure', min: -100, max: 100 },
      { key: 'contrast', min: -100, max: 100 },
      { key: 'highlights', min: -100, max: 100 },
      { key: 'shadows', min: -100, max: 100 },
      { key: 'whites', min: -100, max: 100 },
      { key: 'blacks', min: -100, max: 100 },
    ],
  },
  {
    titleKey: 'adjust.group.colour',
    sliders: [
      { key: 'temperature', min: -100, max: 100 },
      { key: 'tint', min: -100, max: 100 },
      { key: 'vibrance', min: -100, max: 100 },
      { key: 'saturation', min: -100, max: 100 },
      { key: 'hue', min: -180, max: 180 },
    ],
  },
  {
    titleKey: 'adjust.group.presence',
    sliders: [
      { key: 'clarity', min: -100, max: 100 },
      { key: 'sharpness', min: 0, max: 100 },
      { key: 'blur', min: 0, max: 20 },
    ],
  },
  {
    titleKey: 'adjust.group.effects',
    sliders: [
      { key: 'vignette', min: -100, max: 100 },
      { key: 'grain', min: 0, max: 100 },
    ],
  },
];

export const AdjustPanel: React.FC = () => {
  const doc = useActiveDocument();
  const { setAdjustment, commitAdjustments, resetAdjustments } = useDocumentStore();
  const t = useT();

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  if (!layerId || !layer) return null;

  const values: Adjustments = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };
  const touched = !isNeutral(layer.adjustments);

  return (
    <div className="adjust-panel">
      {GROUPS.map((group) => (
        <section key={group.titleKey} className="adjust-group">
          <h4 className="adjust-group-title">{t(group.titleKey)}</h4>

          {group.sliders.map(({ key, min, max }) => {
            const value = values[key];
            const modified = value !== 0;

            return (
              <div key={key} className={`adjust-row ${modified ? 'modified' : ''}`}>
                <label className="adjust-name" htmlFor={`adj-${key}`}>
                  {t(`adjust.${key}`)}
                </label>

                <input
                  id={`adj-${key}`}
                  className="adjust-slider"
                  type="range"
                  min={min}
                  max={max}
                  step={1}
                  value={value}
                  onChange={(e) => setAdjustment(layerId, key, Number(e.target.value))}
                  // The gesture ends here, so this is where one undo step belongs.
                  onPointerUp={() => commitAdjustments(layerId)}
                  onKeyUp={() => commitAdjustments(layerId)}
                  // Double-click returns a single slider to neutral, the
                  // convention every photo editor already uses.
                  onDoubleClick={() => {
                    setAdjustment(layerId, key, 0);
                    commitAdjustments(layerId);
                  }}
                />

                <span className="adjust-value">{value > 0 ? `+${value}` : value}</span>
              </div>
            );
          })}
        </section>
      ))}

      <button
        className="adjust-reset"
        onClick={() => resetAdjustments(layerId)}
        disabled={!touched}
      >
        <RotateCcw size={13} />
        <span>{t('adjust.reset')}</span>
      </button>
    </div>
  );
};
