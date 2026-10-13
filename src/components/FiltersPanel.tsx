// src/components/FiltersPanel.tsx
import { useEffect, useMemo, useRef, useState } from 'react';
import { X, Palette, Plus, Check } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import { FILTERS, FILTER_GROUPS, previewAdjustments } from '../engine/filters';
import { adjustmentKey, sharedProcessor } from '../engine/adjustments';
import { loadImage } from '../engine/RenderEngine';
import { NEUTRAL_ADJUSTMENTS, isNeutral, type Adjustments } from '../types/document';
import { useLookStore } from '../store/lookStore';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
}

/** Side of a thumbnail, in pixels. Small enough that all of them are cheap. */
const THUMB_SIZE = 128;

/** Thumbnails rendered per animation frame, so the panel opens immediately. */
const BATCH = 4;

/**
 * Looks, presented the way a phone camera presents them: the actual photo under
 * each name, not a swatch of an unrelated sunset.
 *
 * Every tile is the user's own image put through the preset at full strength,
 * which is the only preview that ever answers the question being asked. They
 * are rendered in small batches so opening the panel is instant even on a
 * twenty-megapixel photo.
 */
export const FiltersPanel: React.FC<Props> = ({ onClose }) => {
  const doc = useActiveDocument();
  const { setLayerFilter, setFilterAmount, commitFilter } = useDocumentFields('setLayerFilter', 'setFilterAmount', 'commitFilter');
  const t = useT();
  const { looks, save: saveLook, remove: removeLook } = useLookStore();
  /** Name being typed for a new look; null while the field is closed. */
  const [naming, setNaming] = useState<string | null>(null);
  const looksKey = looks.map((look) => look.id).join(',');
  const looksRef = useRef(looks);
  looksRef.current = looks;

  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [basePreview, setBasePreview] = useState<string | null>(null);
  const frameRef = useRef(0);

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  const source = layer?.imageDataUrl;

  // The manual sliders are part of the picture: a preview that ignored them
  // would promise a result the canvas then refuses to show.
  const base = useMemo<Adjustments>(
    () => ({ ...NEUTRAL_ADJUSTMENTS, ...layer?.adjustments }),
    [layer?.adjustments]
  );
  const baseKey = adjustmentKey(base);
  // Read through a ref: dragging a slider hands us a new object on every frame,
  // and rebuilding twenty thumbnails per frame would make the panel stutter.
  // The key changes only when a value really does.
  const baseRef = useRef(base);
  const thumbRef = useRef<{ source: string; canvas: HTMLCanvasElement } | null>(null);
  baseRef.current = base;

  useEffect(() => {
    if (!source) {
      setPreviews({});
      setBasePreview(null);
      return;
    }

    let cancelled = false;

    const build = async () => {
      // The thumbnail is cut once per photo. Decoding the full-size picture
      // again for every slider step was the panel's whole cost while dragging.
      let thumb = thumbRef.current?.source === source ? thumbRef.current.canvas : null;
      if (!thumb) {
        const image = await loadImage(source);
        // Square centre crop: the grid reads as a grid, and a panoramic photo
        // does not turn every tile into a letterbox.
        const side = Math.min(image.naturalWidth, image.naturalHeight);
        thumb = window.document.createElement('canvas');
        thumb.width = THUMB_SIZE;
        thumb.height = THUMB_SIZE;
        const ctx = thumb.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(
          image,
          (image.naturalWidth - side) / 2,
          (image.naturalHeight - side) / 2,
          side,
          side,
          0,
          0,
          THUMB_SIZE,
          THUMB_SIZE
        );
        thumbRef.current = { source, canvas: thumb };
      }
      if (cancelled) return;

      const processor = sharedProcessor();
      const render = (adjustments: Adjustments) => {
        const output = processor.process(thumb, THUMB_SIZE, THUMB_SIZE, adjustments);
        return (output ?? thumb).toDataURL('image/jpeg', 0.82);
      };

      const current = baseRef.current;
      setBasePreview(render(current));

      // The user's own looks first: they are the ones most likely to be wanted.
      const queue = [
        ...looksRef.current.map((look) => ({ id: look.id, adjustments: look.adjustments })),
        ...FILTERS.map((preset) => ({ id: preset.id, adjustments: undefined })),
      ];

      let index = 0;
      const step = () => {
        if (cancelled) return;
        const slice = queue.slice(index, index + BATCH);
        if (slice.length === 0) return;

        const produced: Record<string, string> = {};
        for (const item of slice) {
          produced[item.id] = render(previewAdjustments(current, item.id, 1, item.adjustments));
        }
        setPreviews((current) => ({ ...current, ...produced }));

        index += BATCH;
        frameRef.current = window.requestAnimationFrame(step);
      };

      step();
    };

    // Wait for the sliders to rest: the tiles follow the photo, and redoing
    // them for each step of a drag would compete with the drag itself.
    const timer = window.setTimeout(() => {
      void build().catch(() => {
        if (!cancelled) setPreviews({});
      });
    }, thumbRef.current?.source === source ? 250 : 0);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.cancelAnimationFrame(frameRef.current);
    };
  }, [source, baseKey, looksKey]);

  if (!doc || !layerId || !layer) return null;

  const active = layer.filter;
  const canSave = !isNeutral(layer.adjustments);

  /** Saves the current sliders as a look and puts it on, so the result is visible. */
  const confirmSave = () => {
    if (naming === null || !layer.adjustments) return;
    const values = Object.fromEntries(
      Object.entries(layer.adjustments).filter(([, value]) => value !== 0)
    ) as Partial<Adjustments>;
    saveLook(naming || t('looks.untitled'), values);
    setNaming(null);
  };

  return (
    <section className="side-panel filters-panel">
      <header className="side-panel-header">
        <Palette size={14} />
        <span>{t('panels.filters')}</span>
        <button className="bubble-dismiss" onClick={onClose} title={t('filters.close')}>
          <X size={13} />
        </button>
      </header>

      <div className="filters-scroll">
        <div className="filters-grid">
          <button
            className={`filter-tile ${!active ? 'on' : ''}`}
            onClick={() => setLayerFilter(layerId, null)}
          >
            <span className="filter-thumb">
              {basePreview && <img src={basePreview} alt="" draggable={false} />}
            </span>
            <span className="filter-name">{t('filters.none')}</span>
          </button>
        </div>

        <div className="filters-group">
          <h4 className="filters-group-title">{t('looks.mine')}</h4>
          <div className="filters-grid">
            {looks.map((look) => (
              <div key={look.id} className="filter-tile-wrap">
                <button
                  className={`filter-tile ${active?.id === look.id ? 'on' : ''}`}
                  onClick={() => setLayerFilter(layerId, look.id, look.adjustments)}
                  title={look.name}
                >
                  <span className="filter-thumb">
                    {previews[look.id] && <img src={previews[look.id]} alt="" draggable={false} />}
                  </span>
                  <span className="filter-name">{look.name}</span>
                </button>
                <button
                  className="filter-tile-remove"
                  onClick={() => removeLook(look.id)}
                  title={t('looks.remove')}
                >
                  <X size={11} />
                </button>
              </div>
            ))}

            <button
              className="filter-tile filter-tile-add"
              disabled={!canSave}
              onClick={() => setNaming('')}
              title={canSave ? t('looks.save.hint') : t('looks.save.empty')}
            >
              <span className="filter-thumb">
                <Plus size={20} />
              </span>
              <span className="filter-name">{t('looks.save')}</span>
            </button>
          </div>

          {naming !== null && (
            <form
              className="looks-name"
              onSubmit={(e) => {
                e.preventDefault();
                confirmSave();
              }}
            >
              <input
                autoFocus
                value={naming}
                placeholder={t('looks.name')}
                maxLength={40}
                onChange={(e) => setNaming(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') setNaming(null);
                }}
              />
              <button type="submit" title={t('looks.save')}>
                <Check size={14} />
              </button>
            </form>
          )}
        </div>

        {FILTER_GROUPS.map((group) => (
          <div key={group} className="filters-group">
            <h4 className="filters-group-title">{t(`filters.group.${group}`)}</h4>

            <div className="filters-grid">
              {FILTERS.filter((preset) => preset.group === group).map((preset) => (
                <button
                  key={preset.id}
                  className={`filter-tile ${active?.id === preset.id ? 'on' : ''}`}
                  onClick={() => setLayerFilter(layerId, preset.id)}
                  title={t(`filter.${preset.id}`)}
                >
                  <span className="filter-thumb">
                    {previews[preset.id] && (
                      <img src={previews[preset.id]} alt="" draggable={false} />
                    )}
                  </span>
                  <span className="filter-name">{t(`filter.${preset.id}`)}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {active && (
        <footer className="filters-strength">
          <label htmlFor="filter-amount">{t('filters.strength')}</label>
          <input
            id="filter-amount"
            type="range"
            min={0}
            max={100}
            value={Math.round(active.amount * 100)}
            onChange={(e) => setFilterAmount(layerId, Number(e.target.value) / 100)}
            onPointerUp={() => commitFilter(layerId)}
            onKeyUp={() => commitFilter(layerId)}
          />
          <span className="filters-strength-value">{Math.round(active.amount * 100)}%</span>
        </footer>
      )}
    </section>
  );
};
