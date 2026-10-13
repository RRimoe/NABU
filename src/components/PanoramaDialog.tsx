// src/components/PanoramaDialog.tsx
import { isHeicName } from '../ops/heic';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ImagePlus, Images, X } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import {
  DEFAULT_PANORAMA,
  PanoramaOptions,
  PanoramaProgress,
  PanoramaProjection,
  cancelPanorama,
  stitchPanorama,
  stitcherAvailable,
} from '../ops/panorama';
import { describeError } from '../i18n/backendError';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
  /** Photos to start with, for example dropped on the start screen. */
  initialFiles?: File[];
}

const STORAGE_KEY = 'nabu.panorama';
const PROJECTIONS: PanoramaProjection[] = ['auto', 'rectilinear', 'cylindrical', 'spherical'];
/** "Standard" keeps the result easy to edit; "full" keeps every pixel. */
const SIZES = [60, 120];

const readSaved = (): PanoramaOptions => {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? 'null');
    if (saved && PROJECTIONS.includes(saved.projection)) return { ...DEFAULT_PANORAMA, ...saved };
  } catch {
    // Nothing remembered.
  }
  return DEFAULT_PANORAMA;
};

/**
 * Photos in, one panorama out, as a new document.
 *
 * Order does not matter and nothing has to be lined up: the stitcher finds
 * which photos overlap by itself. What is left to choose is how the result is
 * flattened and whether its ragged edge is cropped, with sensible defaults
 * for both.
 */
export const PanoramaDialog: React.FC<Props> = ({ onClose, initialFiles }) => {
  const openImageAsDocument = useDocumentStore((s) => s.openImageAsDocument);
  const t = useT();
  const [files, setFiles] = useState<File[]>(initialFiles ?? []);
  const [options, setOptions] = useState<PanoramaOptions>(readSaved);
  const [progress, setProgress] = useState<PanoramaProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const running = progress !== null;
  const available = stitcherAvailable();

  const thumbs = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);
  useEffect(() => () => thumbs.forEach((url) => URL.revokeObjectURL(url)), [thumbs]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !running) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, running]);

  const add = (more: File[]) => {
    // HEIC goes through too: the stitcher decodes it itself.
    const images = more.filter((file) => file.type.startsWith('image/') || isHeicName(file.name));
    setFiles((prev) => [
      ...prev,
      // The same file twice adds nothing but a perfect, useless match.
      ...images.filter((f) => !prev.some((p) => p.name === f.name && p.size === f.size)),
    ]);
    setError(null);
  };

  const update = (partial: Partial<PanoramaOptions>) => setOptions((prev) => ({ ...prev, ...partial }));

  const run = async () => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(options));
    } catch {
      // Not remembering is no reason to fail.
    }
    setError(null);
    setProgress({ stage: 'reading', percent: 0 });
    try {
      const { info, dataUrl } = await stitchPanorama(files, options, setProgress);
      openImageAsDocument(t('panorama.documentName'), dataUrl, info.width, info.height);
      if (info.dropped.length > 0) {
        alert(t('panorama.dropped', { names: info.dropped.map((i) => files[i]?.name ?? i).join(', ') }));
      }
      onClose();
    } catch (err) {
      const message = describeError(err);
      // Cancelling is not an error worth a red line.
      if (!(typeof err === 'object' && err && (err as { key?: string }).key === 'backend.cancelled')) {
        setError(message);
      }
      setProgress(null);
    }
  };

  return (
    <div
      className="resize-popover panorama-dialog"
      onMouseDown={(e) => e.stopPropagation()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!running) add(Array.from(e.dataTransfer.files));
      }}
    >
      <div className="resize-head">
        <span>
          <Images size={14} /> {t('panorama.title')}
        </span>
        <button className="bubble-dismiss" onClick={onClose} disabled={running} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      <input
        ref={input}
        type="file"
        accept="image/*,.heic,.heif"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => {
          add(Array.from(e.target.files ?? []));
          e.target.value = '';
        }}
      />

      <div className="panorama-photos">
        {files.map((file, i) => (
          <div key={`${file.name}-${i}`} className="panorama-thumb" title={file.name}>
            <img src={thumbs[i]} alt="" />
            {!running && (
              <button
                className="panorama-remove"
                onClick={() => setFiles((prev) => prev.filter((_, k) => k !== i))}
                title={t('panorama.remove')}
              >
                <X size={11} />
              </button>
            )}
          </div>
        ))}
        {!running && (
          <button className="panorama-add" onClick={() => input.current?.click()}>
            <ImagePlus size={18} strokeWidth={1.6} />
            <span>{files.length === 0 ? t('panorama.addFirst') : t('panorama.add')}</span>
          </button>
        )}
      </div>

      <div className="export-label">{t('panorama.projection')}</div>
      <div className="resize-presets export-choices">
        {PROJECTIONS.map((projection) => (
          <button
            key={projection}
            className={options.projection === projection ? 'on' : ''}
            onClick={() => update({ projection })}
            disabled={running}
            title={t(`panorama.projection.${projection}.hint`)}
          >
            {t(`panorama.projection.${projection}`)}
          </button>
        ))}
      </div>

      <div className="export-label">{t('panorama.size')}</div>
      <div className="resize-presets export-choices">
        {SIZES.map((size) => (
          <button
            key={size}
            className={options.maxMegapixels === size ? 'on' : ''}
            onClick={() => update({ maxMegapixels: size })}
            disabled={running}
          >
            {t(size === SIZES[0] ? 'panorama.size.standard' : 'panorama.size.full', { megapixels: size })}
          </button>
        ))}
      </div>

      <label className="mask-invert panorama-crop">
        <input
          type="checkbox"
          checked={options.crop}
          disabled={running}
          onChange={(e) => update({ crop: e.target.checked })}
        />
        <span>{t('panorama.crop')}</span>
      </label>

      {running && progress && (
        <div className="panorama-progress">
          <div className="panorama-bar">
            <div style={{ width: `${progress.percent}%` }} />
          </div>
          <span>{t(`panorama.stage.${progress.stage}`)}</span>
        </div>
      )}

      {!available && <p className="resize-note">{t('panorama.desktopOnly')}</p>}
      {error && <p className="resize-note export-error">{error}</p>}

      {running ? (
        <button className="resize-apply secondary" onClick={() => void cancelPanorama()}>
          <span>{t('panorama.cancel')}</span>
        </button>
      ) : (
        <button className="resize-apply" onClick={() => void run()} disabled={!available || files.length < 2}>
          <Images size={14} />
          <span>
            {files.length < 2 ? t('panorama.needTwo') : t('panorama.create', { count: files.length })}
          </span>
        </button>
      )}
    </div>
  );
};
