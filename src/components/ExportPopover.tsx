// src/components/ExportPopover.tsx
import { useEffect, useRef, useState } from 'react';
import { Download, X } from 'lucide-react';
import { useActiveDocument, useDocumentStore } from '../store/documentStore';
import { renderDocument, RENDER_THEMES } from '../engine/RenderEngine';
import {
  DEFAULT_EXPORT,
  EXPORT_FORMATS,
  ExportFormat,
  ExportOptions,
  downloadBlob,
  encodeCanvas,
  exportSize,
  formatBytes,
  safeFileName,
} from '../ops/export';
import { describeError } from '../i18n/backendError';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
}

const STORAGE_KEY = 'nabu.export';
const SCALES = [1, 0.5, 0.25];

/** The last choice, so a second export of the day is one click. */
const readSaved = (): ExportOptions => {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? 'null');
    if (saved && saved.format in EXPORT_FORMATS) return { ...DEFAULT_EXPORT, ...saved };
  } catch {
    // Nothing remembered: the defaults are fine.
  }
  return DEFAULT_EXPORT;
};

/**
 * Whether any pixel is see-through, judged on a small copy: a cutout shows up
 * at any size, and reading back a full-resolution canvas would cost far more
 * than the answer is worth.
 */
function hasTransparency(canvas: HTMLCanvasElement) {
  const probe = document.createElement('canvas');
  const scale = Math.min(1, 256 / Math.max(canvas.width, canvas.height));
  probe.width = Math.max(1, Math.round(canvas.width * scale));
  probe.height = Math.max(1, Math.round(canvas.height * scale));
  const ctx = probe.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;
  ctx.drawImage(canvas, 0, 0, probe.width, probe.height);
  const { data } = ctx.getImageData(0, 0, probe.width, probe.height);
  for (let i = 3; i < data.length; i += 4) if (data[i] < 250) return true;
  return false;
}

/**
 * Format, quality and size, with the weight of the resulting file shown before
 * it is written: "how big will this be" is the question people actually have
 * when they pick a quality, and a number answers it better than a description.
 *
 * The document is composed once when the popover opens; moving the quality
 * slider only re-encodes, which is quick enough to follow the drag.
 */
export const ExportPopover: React.FC<Props> = ({ onClose }) => {
  const doc = useActiveDocument();
  const theme = useDocumentStore((s) => s.theme);
  const t = useT();

  const [options, setOptions] = useState<ExportOptions>(readSaved);
  const [composed, setComposed] = useState<HTMLCanvasElement | null>(null);
  const [transparent, setTransparent] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const encoding = useRef(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  useEffect(() => {
    if (!doc) return;
    let cancelled = false;
    renderDocument(doc, RENDER_THEMES[theme])
      .then((canvas) => {
        if (cancelled) return;
        setComposed(canvas);
        setTransparent(hasTransparency(canvas));
      })
      .catch((err) => {
        if (!cancelled) setError(describeError(err));
      });
    return () => {
      cancelled = true;
    };
  }, [doc, theme]);

  // Re-encode after the slider settles; only the latest request may land.
  useEffect(() => {
    if (!composed) return;
    const ticket = ++encoding.current;
    setBlob(null);
    const timer = window.setTimeout(() => {
      encodeCanvas(composed, options)
        .then((result) => {
          if (ticket === encoding.current) {
            setBlob(result);
            setError(null);
          }
        })
        .catch((err) => {
          if (ticket === encoding.current) setError(describeError(err));
        });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [composed, options]);

  if (!doc) return null;

  const spec = EXPORT_FORMATS[options.format];
  const size = exportSize(doc.width, doc.height, options.scale);
  const update = (partial: Partial<ExportOptions>) => setOptions((prev) => ({ ...prev, ...partial }));

  const save = () => {
    if (!blob) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(options));
    } catch {
      // Not remembering the choice is no reason to fail the export.
    }
    downloadBlob(blob, safeFileName(doc.title, spec.extension));
    onClose();
  };

  return (
    <div className="resize-popover export-popover" onMouseDown={(e) => e.stopPropagation()}>
      <div className="resize-head">
        <span>{t('export.title')}</span>
        <button className="bubble-dismiss" onClick={onClose} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      <div className="export-label">{t('export.format')}</div>
      <div className="resize-presets export-choices">
        {(Object.keys(EXPORT_FORMATS) as ExportFormat[]).map((format) => (
          <button
            key={format}
            className={options.format === format ? 'on' : ''}
            onClick={() => update({ format })}
          >
            {format === 'jpeg' ? 'JPEG' : format.toUpperCase()}
          </button>
        ))}
      </div>

      {spec.lossy && (
        <label className="export-quality">
          <span className="export-label">
            {t('export.quality')} <b>{Math.round(options.quality * 100)}</b>
          </span>
          <input
            type="range"
            min={40}
            max={100}
            value={Math.round(options.quality * 100)}
            onChange={(e) => update({ quality: Number(e.target.value) / 100 })}
          />
        </label>
      )}

      <div className="export-label">{t('export.size')}</div>
      <div className="resize-presets export-choices">
        {SCALES.map((scale) => (
          <button
            key={scale}
            className={options.scale === scale ? 'on' : ''}
            onClick={() => update({ scale })}
          >
            {Math.round(scale * 100)}%
          </button>
        ))}
      </div>

      <p className="resize-note">
        {t('export.summary', {
          width: size.width,
          height: size.height,
          weight: blob ? formatBytes(blob.size) : '…',
        })}
        {!spec.alpha && transparent && <> {t('export.flattenNote')}</>}
      </p>

      {error && <p className="resize-note export-error">{t('error.exportFailed', { reason: error })}</p>}

      <button className="resize-apply" onClick={save} disabled={!blob}>
        <Download size={14} />
        <span>{t('export.save', { format: spec.extension.toUpperCase() })}</span>
      </button>
    </div>
  );
};
