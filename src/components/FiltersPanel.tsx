// src/components/FiltersPanel.tsx
import { useEffect, useMemo, useRef, useState } from 'react';
import { X, Palette } from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { FILTERS, FILTER_GROUPS, previewAdjustments } from '../engine/filters';
import { adjustmentKey, sharedProcessor } from '../engine/adjustments';
import { loadImage } from '../engine/RenderEngine';
import { NEUTRAL_ADJUSTMENTS, type Adjustments } from '../types/document';
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
  const { setLayerFilter, setFilterAmount, commitFilter } = useDocumentStore();
  const t = useT();

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
  baseRef.current = base;

  useEffect(() => {
    if (!source) {
      setPreviews({});
      setBasePreview(null);
      return;
    }

    let cancelled = false;

    const build = async () => {
      const image = await loadImage(source);

      // Square centre crop: the grid reads as a grid, and a panoramic photo
      // does not turn every tile into a letterbox.
      const side = Math.min(image.naturalWidth, image.naturalHeight);
      const thumb = window.document.createElement('canvas');
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

      if (cancelled) return;

      const processor = sharedProcessor();
      const render = (adjustments: Adjustments) => {
        const output = processor.process(thumb, THUMB_SIZE, THUMB_SIZE, adjustments);
        return (output ?? thumb).toDataURL('image/jpeg', 0.82);
      };

      const current = baseRef.current;
      setBasePreview(render(current));

      let index = 0;
      const step = () => {
        if (cancelled) return;
        const slice = FILTERS.slice(index, index + BATCH);
        if (slice.length === 0) return;

        const produced: Record<string, string> = {};
        for (const preset of slice) {
          produced[preset.id] = render(previewAdjustments(current, preset.id));
        }
        setPreviews((current) => ({ ...current, ...produced }));

        index += BATCH;
        frameRef.current = window.requestAnimationFrame(step);
      };

      step();
    };

    void build().catch(() => {
      if (!cancelled) setPreviews({});
    });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frameRef.current);
    };
  }, [source, baseKey]);

  if (!doc || !layerId || !layer) return null;

  const active = layer.filter;

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
