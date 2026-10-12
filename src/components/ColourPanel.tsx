// src/components/ColourPanel.tsx
import { useRef, useState } from 'react';
import { FileUp, X } from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import {
  type Adjustments,
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
} from '../types/document';
import { MIX_BANDS } from '../engine/adjustments';
import { encodeTable, parseCube } from '../engine/lut';
import { useT } from '../i18n';

type Band = (typeof MIX_BANDS)[number];

/** Swatch shown for each band, so the choice is made by colour, not by name. */
const SWATCH: Record<Band, string> = {
  Red: '#e5484d',
  Orange: '#f58b2b',
  Yellow: '#f5d90a',
  Green: '#46a758',
  Aqua: '#12a594',
  Blue: '#3e63dd',
  Purple: '#8e4ec6',
  Magenta: '#d6409f',
};

const GRADES = [
  { hue: 'gradeShadowHue', sat: 'gradeShadowSat', label: 'colour.grade.shadows' },
  { hue: 'gradeMidHue', sat: 'gradeMidSat', label: 'colour.grade.midtones' },
  { hue: 'gradeHighlightHue', sat: 'gradeHighlightSat', label: 'colour.grade.highlights' },
] as const;

/**
 * Colour mixer and colour grading, the two tools that change one colour or one
 * range of tones without touching the rest.
 *
 * The mixer shows one band at a time behind a row of swatches: eight bands of
 * three sliders each would be a wall of twenty-four controls that nobody reads.
 * Grading uses a hue track painted with the colours it picks from, because a
 * hue in degrees means nothing to most people and a rainbow means everything.
 */
export const ColourPanel: React.FC = () => {
  const doc = useActiveDocument();
  const { setAdjustment, commitAdjustments, setLayerLut } = useDocumentStore();
  const [band, setBand] = useState<Band>('Blue');
  const [lutError, setLutError] = useState<string | null>(null);
  const lutInput = useRef<HTMLInputElement>(null);
  const t = useT();

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  if (!layerId || !layer) return null;

  const values: Adjustments = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };

  const slider = (key: keyof Adjustments, labelKey: string, track?: string) => {
    const [min, max] = ADJUSTMENT_RANGES[key];
    const value = values[key] ?? 0;
    return (
      <div key={key} className={`adjust-row ${value !== 0 ? 'modified' : ''}`}>
        <label className="adjust-name" htmlFor={`adj-${key}`}>
          {t(labelKey)}
        </label>
        <input
          id={`adj-${key}`}
          className={`adjust-slider ${track ? 'colour-track' : ''}`}
          style={track ? { background: track } : undefined}
          type="range"
          min={min}
          max={max}
          step={1}
          value={value}
          onChange={(e) => setAdjustment(layerId, key, Number(e.target.value))}
          onPointerUp={() => commitAdjustments(layerId)}
          onKeyUp={() => commitAdjustments(layerId)}
          onDoubleClick={() => {
            setAdjustment(layerId, key, 0);
            commitAdjustments(layerId);
          }}
        />
        <span className="adjust-value">{value > 0 && !track ? `+${value}` : value}</span>
      </div>
    );
  };

  /** Reads a .cube file and puts it on the layer at full strength. */
  const loadLut = async (file: File) => {
    try {
      const { size, rgb, title } = parseCube(await file.text());
      const name = title || file.name.replace(/\.cube$/i, '');
      setLayerLut(layerId, { name, size, data: encodeTable(rgb), amount: 1 });
      setLutError(null);
    } catch (err) {
      setLutError(t('colour.lut.unreadable', { name: file.name, reason: String((err as Error).message ?? err) }));
    }
  };

  const lut = layer.lut;

  const bandTouched = (b: Band) =>
    values[`mix${b}Hue`] !== 0 || values[`mix${b}Sat`] !== 0 || values[`mix${b}Lum`] !== 0;

  return (
    <div className="adjust-panel colour-panel">
      <section className="adjust-group">
        <h4 className="adjust-group-title">{t('colour.mixer')}</h4>
        <div className="colour-bands" role="tablist">
          {MIX_BANDS.map((b) => (
            <button
              key={b}
              role="tab"
              aria-selected={band === b}
              className={`colour-band ${band === b ? 'on' : ''} ${bandTouched(b) ? 'touched' : ''}`}
              style={{ background: SWATCH[b] }}
              onClick={() => setBand(b)}
              title={t(`colour.band.${b.toLowerCase()}`)}
            />
          ))}
        </div>
        {slider(`mix${band}Hue`, 'colour.hue')}
        {slider(`mix${band}Sat`, 'colour.saturation')}
        {slider(`mix${band}Lum`, 'colour.luminance')}
      </section>

      <section className="adjust-group">
        <h4 className="adjust-group-title">{t('colour.grading')}</h4>
        {GRADES.map((grade) => (
          <div key={grade.hue} className="colour-grade">
            <span className="colour-grade-name">
              <span
                className="colour-grade-dot"
                style={{
                  background:
                    values[grade.sat] > 0
                      ? `hsl(${values[grade.hue]} ${40 + values[grade.sat] * 0.6}% 55%)`
                      : 'transparent',
                }}
              />
              {t(grade.label)}
            </span>
            {slider(
              grade.hue,
              'colour.hue',
              'linear-gradient(to right, hsl(0 80% 55%), hsl(60 80% 55%), hsl(120 80% 45%), hsl(180 80% 45%), hsl(240 80% 60%), hsl(300 80% 55%), hsl(360 80% 55%))'
            )}
            {slider(grade.sat, 'colour.amount')}
          </div>
        ))}
        {slider('gradeBalance', 'colour.balance')}
      </section>

      <section className="adjust-group">
        <h4 className="adjust-group-title">{t('colour.lut')}</h4>
        <input
          ref={lutInput}
          type="file"
          accept=".cube"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void loadLut(file);
            e.target.value = '';
          }}
        />
        {lut ? (
          <>
            <div className="lut-row">
              <span className="lut-name" title={lut.name}>
                {lut.name}
              </span>
              <span className="lut-size">{lut.size}³</span>
              <button
                className="bubble-dismiss"
                onClick={() => setLayerLut(layerId, null)}
                title={t('colour.lut.remove')}
              >
                <X size={12} />
              </button>
            </div>
            <div className={`adjust-row modified`}>
              <label className="adjust-name" htmlFor="lut-amount">
                {t('colour.amount')}
              </label>
              <input
                id="lut-amount"
                className="adjust-slider"
                type="range"
                min={0}
                max={100}
                value={Math.round(lut.amount * 100)}
                onChange={(e) =>
                  setLayerLut(layerId, { ...lut, amount: Number(e.target.value) / 100 }, { history: false })
                }
                onPointerUp={() => setLayerLut(layerId, { ...lut })}
                onKeyUp={() => setLayerLut(layerId, { ...lut })}
              />
              <span className="adjust-value">{Math.round(lut.amount * 100)}</span>
            </div>
          </>
        ) : (
          <button className="lut-load" onClick={() => lutInput.current?.click()}>
            <FileUp size={13} />
            <span>{t('colour.lut.load')}</span>
          </button>
        )}
        {lutError && <p className="resize-note">{lutError}</p>}
      </section>
    </div>
  );
};
