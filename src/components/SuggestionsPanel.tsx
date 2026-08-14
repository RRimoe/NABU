// src/components/SuggestionsPanel.tsx
import { useEffect, useMemo, useState } from 'react';
import {
  Check,
  ChevronDown,
  Crop,
  Download,
  Eraser,
  Scissors,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { useSceneStore } from '../store/sceneStore';
import { useModelStore } from '../store/modelStore';
import { useAdviceStore } from '../store/adviceStore';
import { buildSuggestions, classLabel, type Suggestion } from '../ai/suggestions';
import { runRecipe } from '../ai/recipes';
import { runSuggestion } from '../ai/operations';
import { measureImage, type ImageStats } from '../engine/imageStats';
import { squareThumbnail } from '../engine/thumbnail';
import { sharedProcessor } from '../engine/adjustments';
import { loadImage } from '../engine/RenderEngine';
import { NEUTRAL_ADJUSTMENTS, type Adjustments } from '../types/document';
import { useT } from '../i18n';

interface Props {
  /** Opens the model manager when a proposal needs something not installed. */
  onNeedsModel: () => void;
}

/** Side of a card thumbnail, in pixels. */
const THUMB_SIZE = 96;

/** Proposals shown before the fold. The rest live under the expander. */
const VISIBLE = 3;

/** Icon per operation, so a card that runs a model looks like what it does. */
const ACTION_ICONS = {
  'remove-background': Scissors,
  upscale: TrendingUp,
  'extract-object': Scissors,
  'remove-object': Eraser,
  'crop-to': Crop,
} as const;

/**
 * What Nabu proposes for this photo.
 *
 * This panel is what the typed prompt became. The difference is not cosmetic:
 * a request had to be phrased correctly, understood by a small model, and
 * mapped onto an operation — three places to fail. A proposal is measured from
 * the pixels, arrives instantly, reads the same in every language, and shows
 * its result before it is accepted.
 */
export const SuggestionsPanel: React.FC<Props> = ({ onNeedsModel }) => {
  const doc = useActiveDocument();
  const isAIBusy = useDocumentStore((s) => s.isAIBusy);
  const activeId = useDocumentStore((s) => s.activeId);
  const scene = useSceneStore((s) => s.entries[activeId]?.scene ?? null);
  const readyFor = useModelStore((s) => s.readyFor);
  const models = useModelStore((s) => s.models);
  const t = useT();

  const [stats, setStats] = useState<ImageStats | null>(null);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState(false);

  const advice = useAdviceStore((s) => s.entries[activeId]?.advice ?? null);
  const thinking = useAdviceStore((s) => Boolean(s.pending[activeId]));
  const requestAdvice = useAdviceStore((s) => s.request);

  const layerId = doc?.activeLayerId ?? null;
  const layer = layerId ? doc?.layers[layerId] : null;
  const source = layer?.imageDataUrl;

  const base = useMemo<Adjustments>(
    () => ({ ...NEUTRAL_ADJUSTMENTS, ...layer?.adjustments }),
    [layer?.adjustments]
  );

  // Measured from the stored pixels, never from the adjusted result: reading
  // the corrected image would make a fix disappear and reappear as the sliders
  // moved, which is the behaviour of a broken instrument.
  useEffect(() => {
    let cancelled = false;
    if (!source) {
      setStats(null);
      return;
    }
    void loadImage(source)
      .then((image) => {
        if (cancelled) return;
        setStats(measureImage(image, image.naturalWidth, image.naturalHeight));
      })
      .catch(() => {
        if (!cancelled) setStats(null);
      });
    return () => {
      cancelled = true;
    };
  }, [source]);

  // Asked for once the measurements exist, and then forgotten about. Nothing
  // below depends on it arriving.
  useEffect(() => {
    if (stats) requestAdvice(stats);
  }, [stats, requestAdvice, source]);

  const suggestions = useMemo(() => {
    if (!doc || !layer) return [];
    return buildSuggestions({
      stats,
      scene,
      current: base,
      width: layer.width ?? doc.width,
      height: layer.height ?? doc.height,
    });
  }, [doc, layer, stats, scene, base]);

  // A slider proposal can show exactly what it would do; an operation cannot be
  // previewed without running the model, so it keeps the plain photo.
  const previewKey = suggestions
    .filter((s) => s.adjustments)
    .map((s) => s.id)
    .join(',');

  useEffect(() => {
    let cancelled = false;
    if (!source) {
      setThumbs({});
      return;
    }

    void loadImage(source)
      .then((image) => {
        if (cancelled) return;
        const thumb = squareThumbnail(image, THUMB_SIZE);
        if (!thumb) return;

        const processor = sharedProcessor();
        const produced: Record<string, string> = {};

        const plain = thumb.toDataURL('image/jpeg', 0.8);
        produced.__base = plain;

        for (const suggestion of suggestions) {
          if (!suggestion.adjustments) continue;
          const values = { ...base };
          for (const [key, value] of Object.entries(suggestion.adjustments) as Array<
            [keyof Adjustments, number]
          >) {
            values[key] = values[key] + value;
          }
          const output = processor.process(thumb, THUMB_SIZE, THUMB_SIZE, values);
          produced[suggestion.id] = (output ?? thumb).toDataURL('image/jpeg', 0.8);
        }

        setThumbs(produced);
      })
      .catch(() => {
        if (!cancelled) setThumbs({});
      });

    return () => {
      cancelled = true;
    };
    // The proposals and the sliders are read through the closure on purpose:
    // both change identity on every keystroke, while `previewKey` changes only
    // when a thumbnail would actually come out different.
  }, [source, previewKey]);

  if (!doc || !layer) return null;

  const shown = expanded ? suggestions : suggestions.slice(0, VISIBLE);
  const hidden = suggestions.length - shown.length;

  /** True when the card needs a model that is not on disk yet. */
  const needsModel = (suggestion: Suggestion) =>
    Boolean(suggestion.requires) && !readyFor(suggestion.requires!);

  const label = (suggestion: Suggestion, key: string) => {
    const params = suggestion.params
      ? { ...suggestion.params, subject: classLabel(String(suggestion.params.subject ?? ''), t) }
      : undefined;
    return t(key, params);
  };

  return (
    <section className="side-panel suggestions-panel">
      <header className="side-panel-header">
        <Sparkles size={14} />
        <span>{t('panels.suggestions')}</span>
      </header>

      {/* The model's reading of the photo, when it arrives. It sits above the
          measured proposals because it is about *this* picture rather than
          about a number, and it is never waited for: the list below was
          already complete before this was asked for. */}
      {advice && (
        <button
          className="advice-card"
          disabled={isAIBusy}
          onClick={() => void runRecipe(advice.recipe, advice.target)}
        >
          <span className="advice-mark">
            <Sparkles size={13} />
          </span>
          <span className="suggestion-text">
            <span className="advice-observation">{advice.observation}</span>
            <span className="advice-action">
              {t(`recipe.${advice.recipe}`)}
              {advice.targetClass ? ` — ${classLabel(advice.targetClass, t)}` : ''}
            </span>
          </span>
          <span className="suggestion-apply">
            <Check size={14} />
          </span>
        </button>
      )}

      {!advice && thinking && <p className="advice-thinking">{t('advice.thinking')}</p>}

      {suggestions.length === 0 ? (
        <p className="suggestions-empty">{t('suggest.none')}</p>
      ) : (
        <div className="suggestions-list">
          {shown.map((suggestion) => {
            const missing = needsModel(suggestion);
            const Icon = suggestion.action
              ? ACTION_ICONS[suggestion.action.type]
              : Sparkles;

            return (
              <button
                key={suggestion.id}
                className={`suggestion-card ${missing ? 'needs-model' : ''}`}
                disabled={isAIBusy}
                onClick={() => {
                  if (missing) onNeedsModel();
                  else void runSuggestion(suggestion);
                }}
              >
                <span className="suggestion-thumb">
                  {(thumbs[suggestion.id] ?? thumbs.__base) && (
                    <img
                      src={thumbs[suggestion.id] ?? thumbs.__base}
                      alt=""
                      draggable={false}
                    />
                  )}
                  {!suggestion.adjustments && (
                    <span className="suggestion-thumb-badge">
                      <Icon size={12} />
                    </span>
                  )}
                </span>

                <span className="suggestion-text">
                  <span className="suggestion-title">
                    {label(suggestion, suggestion.titleKey)}
                  </span>
                  <span className="suggestion-detail">
                    {missing
                      ? t('suggest.needsModel')
                      : label(suggestion, suggestion.detailKey)}
                  </span>
                </span>

                <span className="suggestion-apply">
                  {missing ? <Download size={14} /> : <Check size={14} />}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {hidden > 0 && (
        <button className="suggestions-more" onClick={() => setExpanded(true)}>
          <span>{t('suggest.more', { count: hidden })}</span>
          <ChevronDown size={15} />
        </button>
      )}

      {expanded && suggestions.length > VISIBLE && (
        <button className="suggestions-more" onClick={() => setExpanded(false)}>
          <span>{t('suggest.less')}</span>
          <ChevronDown size={15} className="flipped" />
        </button>
      )}

      {models.length === 0 && (
        <p className="suggestions-note">{t('suggest.noModels')}</p>
      )}
    </section>
  );
};
