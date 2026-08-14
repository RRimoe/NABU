// src/components/AdjustPanel.tsx
import { RotateCcw } from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import {
  type Adjustments,
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
  isNeutral,
} from '../types/document';
import { useT } from '../i18n';

interface Group {
  titleKey: string;
  sliders: Array<keyof Adjustments>;
}

/**
 * Grouped the way a photographer expects to find them, not the way the shader
 * happens to evaluate them. The ranges come from the type, so a slider and the
 * filter engine can never disagree about how far a value may go.
 */
const GROUPS: Group[] = [
  {
    titleKey: 'adjust.group.tone',
    sliders: ['exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks'],
  },
  {
    titleKey: 'adjust.group.colour',
    sliders: ['temperature', 'tint', 'vibrance', 'saturation', 'hue'],
  },
  {
    titleKey: 'adjust.group.presence',
    sliders: ['clarity', 'sharpness', 'blur'],
  },
  {
    titleKey: 'adjust.group.effects',
    sliders: ['vignette', 'grain', 'bloom', 'aberration'],
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

          {group.sliders.map((key) => {
            const [min, max] = ADJUSTMENT_RANGES[key];
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
