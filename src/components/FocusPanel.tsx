// src/components/FocusPanel.tsx
import { Aperture, Circle, Layers, RectangleHorizontal, RotateCcw, Scan } from 'lucide-react';
import { useDocumentStore, useActiveDocument, useSelection } from '../store/documentStore';
import { useDepthStore } from '../store/depthStore';
import { focusOnDepth, focusOnSelection } from '../ops/focus';
import { NEUTRAL_FOCUS, type FocusEffect, type FocusMode } from '../types/document';
import { useT } from '../i18n';

/** The four ways of saying where the picture stays sharp. */
const MODES: Array<{ id: Exclude<FocusMode, 'off'>; Icon: typeof Circle }> = [
  { id: 'depth', Icon: Layers },
  { id: 'mask', Icon: Scan },
  { id: 'linear', Icon: RectangleHorizontal },
  { id: 'radial', Icon: Circle },
];

interface Slider {
  key: 'amount' | 'feather' | 'size' | 'aspect' | 'angle' | 'plane';
  min: number;
  max: number;
  step: number;
  /** Modes the control applies to. */
  modes: FocusMode[];
  /** Shown to the user; the stored value is not always the readable one. */
  format: (value: number) => string;
}

const SLIDERS: Slider[] = [
  {
    key: 'amount',
    min: 0,
    max: 100,
    step: 1,
    modes: ['mask', 'linear', 'radial', 'depth'],
    format: (v) => String(Math.round(v)),
  },
  {
    key: 'plane',
    min: 0,
    max: 100,
    step: 1,
    modes: ['depth'],
    format: (v) => `${Math.round(v)}%`,
  },
  {
    key: 'feather',
    min: 0,
    max: 60,
    step: 1,
    modes: ['mask', 'linear', 'radial', 'depth'],
    format: (v) => String(Math.round(v)),
  },
  {
    key: 'size',
    min: 2,
    max: 70,
    step: 1,
    modes: ['linear', 'radial', 'depth'],
    format: (v) => `${Math.round(v)}%`,
  },
  {
    key: 'aspect',
    min: 30,
    max: 300,
    step: 5,
    modes: ['radial'],
    format: (v) => `${Math.round(v)}%`,
  },
  {
    key: 'angle',
    min: -90,
    max: 90,
    step: 1,
    modes: ['linear'],
    format: (v) => `${Math.round(v)}°`,
  },
];

/** Fractions are stored 0..1 but read better as percentages. */
const toSlider = (key: Slider['key'], focus: FocusEffect): number => {
  if (key === 'feather') return focus.feather * 100;
  if (key === 'size') return focus.size * 100;
  if (key === 'aspect') return focus.aspect * 100;
  if (key === 'plane') return focus.plane * 100;
  return focus[key];
};

const fromSlider = (key: Slider['key'], value: number): Partial<FocusEffect> => {
  if (key === 'feather') return { feather: value / 100 };
  if (key === 'size') return { size: value / 100 };
  if (key === 'aspect') return { aspect: value / 100 };
  if (key === 'plane') return { plane: value / 100 };
  return { [key]: value } as Partial<FocusEffect>;
};

/**
 * Selective focus: what stays sharp, and how far out of focus everything else
 * goes.
 *
 * The three modes are one effect, not three. A band across the frame is the
 * tilt-shift that makes a street look like a model railway; an ellipse is the
 * soft surround a portrait lens gives you; a mask follows a real object, which
 * is the one that needs the segmenter and the only one that can put a person
 * in front of a background that melts away behind them.
 */
export const FocusPanel: React.FC = () => {
  const doc = useActiveDocument();
  const selection = useSelection();
  const { setFocus, commitFocus, clearFocus, isAIBusy } = useDocumentStore();
  // A depth run that already failed once is not offered again: the model is
  // either installed or it is not, and retrying costs the user another wait.
  const depthUnavailable = useDepthStore((s) => s.unavailable);
  const t = useT();

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  if (!doc || !layerId || !layer) return null;

  const focus: FocusEffect = { ...NEUTRAL_FOCUS, ...layer.focus };
  const active = focus.mode !== 'off';

  const choose = (mode: Exclude<FocusMode, 'off'>) => {
    // Picking the shape again switches the whole effect off, so the panel needs
    // no separate on/off control.
    if (focus.mode === mode) {
      clearFocus(layerId);
      return;
    }

    if (mode === 'mask') {
      // A mask has to be drawn from something. With a selection in hand that
      // is what it follows; without one there is nothing to build it from.
      if (selection) void focusOnSelection();
      return;
    }

    if (mode === 'depth') {
      // Reuses a map already computed for this photo; the first time it costs
      // one model run, which is why it goes through the busy indicator.
      void focusOnDepth();
      return;
    }

    setFocus(layerId, { ...focus, mode }, { history: true });
  };

  const update = (partial: Partial<FocusEffect>) =>
    setFocus(layerId, { ...focus, ...partial });

  return (
    <div className="focus-panel">
      <div className="focus-modes">
        {MODES.map(({ id, Icon }) => {
          const unavailable =
            (id === 'mask' && focus.mode !== 'mask' && !selection) ||
            (id === 'depth' && focus.mode !== 'depth' && depthUnavailable);
          return (
            <button
              key={id}
              className={`focus-mode ${focus.mode === id ? 'on' : ''}`}
              onClick={() => choose(id)}
              disabled={unavailable || isAIBusy}
              title={t(`focus.mode.${id}.hint`)}
            >
              <Icon size={15} />
              <span>{t(`focus.mode.${id}`)}</span>
            </button>
          );
        })}
      </div>

      {active && (
        <>
          {SLIDERS.filter((slider) => slider.modes.includes(focus.mode)).map((slider) => {
            const value = toSlider(slider.key, focus);
            return (
              <div key={slider.key} className="adjust-row">
                <label className="adjust-name" htmlFor={`focus-${slider.key}`}>
                  {t(`focus.${slider.key}`)}
                </label>
                <input
                  id={`focus-${slider.key}`}
                  className="adjust-slider"
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={value}
                  onChange={(e) => update(fromSlider(slider.key, Number(e.target.value)))}
                  onPointerUp={() => commitFocus(layerId)}
                  onKeyUp={() => commitFocus(layerId)}
                />
                <span className="adjust-value">{slider.format(value)}</span>
              </div>
            );
          })}

          <label className="focus-invert">
            <input
              type="checkbox"
              checked={focus.invert}
              onChange={(e) => {
                update({ invert: e.target.checked });
                commitFocus(layerId);
              }}
            />
            <span>{t('focus.invert')}</span>
          </label>

          <button className="adjust-reset" onClick={() => clearFocus(layerId)}>
            <RotateCcw size={13} />
            <span>{t('focus.clear')}</span>
          </button>
        </>
      )}

      {!active && (
        <p className="focus-hint">
          <Aperture size={13} />
          <span>{selection ? t('focus.intro') : t('focus.maskHint')}</span>
        </p>
      )}
    </div>
  );
};
