// src/components/MasksPanel.tsx
import { Circle, FlipVertical2, Lasso, Sun, Trash2 } from 'lucide-react';
import { useDocumentStore, useActiveDocument, useSelection } from '../store/documentStore';
import {
  ADJUSTMENT_RANGES,
  MASK_ADJUSTMENTS,
  isMaskNeutral,
  type LocalMask,
  type LocalMaskKind,
} from '../types/document';
import { buildSelectionMask } from '../ops/mask';
import { useT } from '../i18n';

const KINDS: Array<{ kind: LocalMaskKind; Icon: typeof Circle }> = [
  { kind: 'linear', Icon: FlipVertical2 },
  { kind: 'radial', Icon: Circle },
  { kind: 'luminance', Icon: Sun },
  { kind: 'selection', Icon: Lasso },
];

const newId = () => `mask-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

/** A fresh mask of each kind, placed where it is most often wanted. */
function freshMask(kind: LocalMaskKind): LocalMask {
  const base = { id: newId(), kind, invert: false, amount: 1, adjustments: {} };
  switch (kind) {
    // The sky: full effect at the top edge, gone by the middle.
    case 'linear':
      return { ...base, linear: { x0: 0.5, y0: 0, x1: 0.5, y1: 0.5 } };
    case 'radial':
      return { ...base, radial: { cx: 0.5, cy: 0.5, rx: 0.25, ry: 0.25, feather: 0.5 } };
    // The highlights, the range people most often want to tame on its own.
    case 'luminance':
      return { ...base, luminance: { low: 65, high: 100, smooth: 15 } };
    case 'selection':
      return base;
  }
}

/**
 * Masked corrections, the way a photo editor's mask panel works: add a
 * gradient, an ellipse, a brightness range or the current selection, then move
 * the sliders for that part of the picture only. One mask is open at a time;
 * its shape is moved on the photo itself.
 */
export const MasksPanel: React.FC = () => {
  const doc = useActiveDocument();
  const selection = useSelection();
  const {
    addMask,
    updateMask,
    commitMasks,
    removeMask,
    editingMaskId,
    setEditingMask,
    setSelection,
  } = useDocumentStore();
  const t = useT();

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  if (!layerId || !layer) return null;

  const masks = layer.masks ?? [];
  const open = masks.find((mask) => mask.id === editingMaskId) ?? null;

  const add = async (kind: LocalMaskKind) => {
    const mask = freshMask(kind);
    if (kind === 'selection') {
      const built = await buildSelectionMask();
      if (!built) return;
      mask.maskDataUrl = built.maskDataUrl;
      setSelection(null);
    }
    addMask(layerId, mask);
  };

  const change = (patch: Partial<LocalMask>) => {
    if (open) updateMask(layerId, open.id, patch);
  };
  const done = () => commitMasks(layerId);

  const row = (
    id: string,
    label: string,
    value: number,
    min: number,
    max: number,
    set: (value: number) => void,
    reset: number
  ) => (
    <div key={id} className={`adjust-row ${value !== reset ? 'modified' : ''}`}>
      <label className="adjust-name" htmlFor={`mask-${id}`}>
        {label}
      </label>
      <input
        id={`mask-${id}`}
        className="adjust-slider"
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(e) => set(Number(e.target.value))}
        onPointerUp={done}
        onKeyUp={done}
        onDoubleClick={() => {
          set(reset);
          done();
        }}
      />
      <span className="adjust-value">{value > 0 && reset === 0 ? `+${value}` : value}</span>
    </div>
  );

  return (
    <div className="adjust-panel masks-panel">
      <div className="focus-modes">
        {KINDS.map(({ kind, Icon }) => (
          <button
            key={kind}
            className="focus-mode"
            onClick={() => void add(kind)}
            disabled={kind === 'selection' && !selection}
            title={t(`masks.add.${kind}.hint`)}
          >
            <Icon size={15} />
            <span>{t(`masks.kind.${kind}`)}</span>
          </button>
        ))}
      </div>

      {masks.length === 0 ? (
        <p className="focus-hint">{t('masks.empty')}</p>
      ) : (
        <ul className="mask-list">
          {masks.map((mask, index) => (
            <li key={mask.id}>
              <button
                className={`mask-item ${mask.id === editingMaskId ? 'on' : ''} ${
                  isMaskNeutral(mask) ? 'idle' : ''
                }`}
                onClick={() => setEditingMask(mask.id === editingMaskId ? null : mask.id)}
              >
                <span className="mask-index">{index + 1}</span>
                <span className="mask-name">
                  {t(`masks.kind.${mask.kind}`)}
                  {mask.invert ? ` · ${t('masks.inverted')}` : ''}
                </span>
              </button>
              <button
                className="bubble-dismiss"
                onClick={() => removeMask(layerId, mask.id)}
                title={t('masks.remove')}
              >
                <Trash2 size={12} />
              </button>
            </li>
          ))}
        </ul>
      )}

      {open && (
        <div className="mask-editor">
          <label className="mask-invert">
            <input
              type="checkbox"
              checked={open.invert}
              onChange={(e) => {
                updateMask(layerId, open.id, { invert: e.target.checked }, { history: true });
              }}
            />
            <span>{t('masks.invert')}</span>
          </label>

          {open.kind === 'radial' &&
            open.radial &&
            row(
              'feather',
              t('masks.feather'),
              Math.round(open.radial.feather * 100),
              0,
              100,
              (v) => change({ radial: { ...open.radial!, feather: v / 100 } }),
              50
            )}

          {open.kind === 'luminance' && open.luminance && (
            <>
              {row(
                'low',
                t('masks.range.low'),
                open.luminance.low,
                0,
                100,
                (v) => change({ luminance: { ...open.luminance!, low: Math.min(v, open.luminance!.high) } }),
                65
              )}
              {row(
                'high',
                t('masks.range.high'),
                open.luminance.high,
                0,
                100,
                (v) => change({ luminance: { ...open.luminance!, high: Math.max(v, open.luminance!.low) } }),
                100
              )}
              {row(
                'smooth',
                t('masks.range.smooth'),
                open.luminance.smooth,
                0,
                50,
                (v) => change({ luminance: { ...open.luminance!, smooth: v } }),
                15
              )}
            </>
          )}

          {MASK_ADJUSTMENTS.map((key) => {
            const [min, max] = ADJUSTMENT_RANGES[key];
            return row(
              key,
              t(`adjust.${key}`),
              open.adjustments[key] ?? 0,
              min,
              max,
              (v) => change({ adjustments: { ...open.adjustments, [key]: v } }),
              0
            );
          })}

          {row(
            'amount',
            t('colour.amount'),
            Math.round(open.amount * 100),
            0,
            100,
            (v) => change({ amount: v / 100 }),
            100
          )}

          {(open.kind === 'linear' || open.kind === 'radial') && (
            <p className="focus-hint">{t(`masks.drag.${open.kind}`)}</p>
          )}
        </div>
      )}
    </div>
  );
};
