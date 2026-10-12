// src/components/BatchDialog.tsx
import { useEffect, useRef, useState } from 'react';
import { FolderOpen, ImagePlus, Layers, X } from 'lucide-react';
import { open } from '@tauri-apps/plugin-dialog';
import { revealItemInDir } from '@tauri-apps/plugin-opener';
import { useLookStore } from '../store/lookStore';
import { FILTERS } from '../engine/filters';
import { activeDevelop } from '../ops/develop';
import type { ExportFormat } from '../ops/export';
import { runBatch, type BatchFileResult, type BatchProgress, type BatchStep } from '../commands/batch';
import { baseName, desktopAvailable } from '../commands/files';
import { describeError } from '../i18n/backendError';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
}

type Source = 'current' | 'look';

interface Settings {
  source: Source;
  look: string;
  /** Longest side in pixels, or 0 to keep each photo's size. */
  longSide: number;
  format: ExportFormat;
  quality: number;
  suffix: string;
  outputDir: string;
}

const STORAGE_KEY = 'nabu.batch';
const FORMATS: ExportFormat[] = ['jpeg', 'png', 'webp'];
const SIDES = [0, 3840, 2048, 1080];
const DEFAULTS: Settings = {
  source: 'current',
  look: FILTERS[0]?.id ?? '',
  longSide: 0,
  format: 'jpeg',
  quality: 0.9,
  suffix: '-nabu',
  outputDir: '',
};

const readSaved = (): Settings => {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? 'null');
    if (saved && typeof saved === 'object') return { ...DEFAULTS, ...saved };
  } catch {
    // Nothing remembered.
  }
  return DEFAULTS;
};

const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'avif'];

/**
 * One edit, many photos.
 *
 * The edit is either the one on the photo in front of the user (develop it
 * once, then apply it to the whole shoot) or a look; the photos are developed
 * and exported one by one into a folder, and the originals are never touched.
 * It runs the same batch command scripts and agents call.
 */
export const BatchDialog: React.FC<Props> = ({ onClose }) => {
  const t = useT();
  const looks = useLookStore((s) => s.looks);
  const [files, setFiles] = useState<string[]>([]);
  const [settings, setSettings] = useState<Settings>(readSaved);
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [results, setResults] = useState<BatchFileResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelled = useRef(false);
  const running = progress !== null;
  const available = desktopAvailable();
  // Read once: the batch must apply what was on screen when it was opened.
  const [develop] = useState(activeDevelop);
  const source: Source = settings.source === 'current' && !develop ? 'look' : settings.source;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !running) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, running]);

  const update = (partial: Partial<Settings>) => setSettings((prev) => ({ ...prev, ...partial }));

  const addPhotos = async () => {
    const picked = await open({
      multiple: true,
      filters: [{ name: t('batch.photos'), extensions: IMAGE_EXTENSIONS }],
    });
    if (!picked) return;
    const paths = Array.isArray(picked) ? picked : [picked];
    setFiles((prev) => [...prev, ...paths.filter((p) => !prev.includes(p))]);
    setResults(null);
  };

  const chooseFolder = async () => {
    const picked = await open({ directory: true, defaultPath: settings.outputDir || undefined });
    if (typeof picked === 'string') update({ outputDir: picked });
  };

  const steps = (): BatchStep[] => {
    const list: BatchStep[] = [];
    if (source === 'current' && develop) list.push({ command: 'apply_develop', args: { settings: develop } });
    if (source === 'look') list.push({ command: 'apply_look', args: { look: settings.look } });
    if (settings.longSide > 0) list.push({ command: 'resize', args: { long_side: settings.longSide } });
    return list;
  };

  const run = async () => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // Not remembering is no reason to fail.
    }
    cancelled.current = false;
    setError(null);
    setResults(null);
    setProgress({ index: 0, total: files.length, file: files[0] });
    try {
      const outcome = await runBatch(
        {
          files,
          steps: steps(),
          outputDir: settings.outputDir,
          format: settings.format,
          quality: settings.quality,
          scale: 1,
          suffix: settings.suffix,
        },
        setProgress,
        () => cancelled.current
      );
      setResults(outcome);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setProgress(null);
    }
  };

  const failures = results?.filter((r) => !r.ok) ?? [];
  const succeeded = results ? results.length - failures.length : 0;
  const ready = available && files.length > 0 && settings.outputDir !== '';

  return (
    <div className="resize-popover panorama-dialog batch-dialog" onMouseDown={(e) => e.stopPropagation()}>
      <div className="resize-head">
        <span>
          <Layers size={14} /> {t('batch.title')}
        </span>
        <button className="bubble-dismiss" onClick={onClose} disabled={running} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      <div className="batch-files">
        {files.map((file) => (
          <div key={file} className="batch-file" title={file}>
            <span>{baseName(file)}</span>
            {!running && (
              <button
                className="panorama-remove"
                onClick={() => setFiles((prev) => prev.filter((f) => f !== file))}
                title={t('panorama.remove')}
              >
                <X size={11} />
              </button>
            )}
          </div>
        ))}
        {!running && (
          <button className="panorama-add batch-add" onClick={() => void addPhotos()} disabled={!available}>
            <ImagePlus size={16} strokeWidth={1.6} />
            <span>{files.length === 0 ? t('batch.addFirst') : t('batch.add')}</span>
          </button>
        )}
      </div>

      <div className="export-label">{t('batch.apply')}</div>
      <div className="resize-presets export-choices">
        <button
          className={source === 'current' ? 'on' : ''}
          onClick={() => update({ source: 'current' })}
          disabled={running || !develop}
          title={develop ? undefined : t('batch.noDevelop')}
        >
          {t('batch.source.current')}
        </button>
        <button className={source === 'look' ? 'on' : ''} onClick={() => update({ source: 'look' })} disabled={running}>
          {t('batch.source.look')}
        </button>
      </div>
      {source === 'look' && (
        <select
          className="batch-select"
          value={settings.look}
          disabled={running}
          onChange={(e) => update({ look: e.target.value })}
        >
          {FILTERS.map((f) => (
            <option key={f.id} value={f.id}>
              {t(`filter.${f.id}`)}
            </option>
          ))}
          {looks.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>
      )}

      <div className="export-label">{t('batch.size')}</div>
      <div className="resize-presets export-choices">
        {SIDES.map((side) => (
          <button
            key={side}
            className={settings.longSide === side ? 'on' : ''}
            onClick={() => update({ longSide: side })}
            disabled={running}
          >
            {side === 0 ? t('batch.size.original') : `${side}px`}
          </button>
        ))}
      </div>

      <div className="export-label">{t('batch.format')}</div>
      <div className="resize-presets export-choices">
        {FORMATS.map((format) => (
          <button
            key={format}
            className={settings.format === format ? 'on' : ''}
            onClick={() => update({ format })}
            disabled={running}
          >
            {format.toUpperCase()}
          </button>
        ))}
      </div>
      {settings.format !== 'png' && (
        <label className="batch-quality">
          <span>{t('batch.quality', { value: Math.round(settings.quality * 100) })}</span>
          <input
            type="range"
            min={50}
            max={100}
            value={Math.round(settings.quality * 100)}
            disabled={running}
            onChange={(e) => update({ quality: Number(e.target.value) / 100 })}
          />
        </label>
      )}

      <div className="export-label">{t('batch.output')}</div>
      <button className="batch-folder" onClick={() => void chooseFolder()} disabled={running || !available}>
        <FolderOpen size={14} />
        <span>{settings.outputDir || t('batch.chooseFolder')}</span>
      </button>
      <label className="batch-suffix">
        <span>{t('batch.suffix')}</span>
        <input
          type="text"
          value={settings.suffix}
          disabled={running}
          onChange={(e) => update({ suffix: e.target.value })}
          spellCheck={false}
        />
      </label>

      {running && progress && (
        <div className="panorama-progress">
          <div className="panorama-bar">
            <div style={{ width: `${(progress.index / Math.max(1, progress.total)) * 100}%` }} />
          </div>
          <span>
            {t('batch.progress', { index: Math.min(progress.index + 1, progress.total), total: progress.total })}
            {progress.file ? ` · ${baseName(progress.file)}` : ''}
          </span>
        </div>
      )}

      {results && (
        <div className="batch-summary">
          <p>{t('batch.done', { done: succeeded, total: results.length })}</p>
          {failures.map((f) => (
            <p key={f.file} className="export-error">
              {baseName(f.file)}: {f.error}
            </p>
          ))}
          {succeeded > 0 && (
            <button
              className="batch-folder"
              onClick={() => void revealItemInDir(results.find((r) => r.ok)?.output ?? settings.outputDir)}
            >
              <FolderOpen size={14} />
              <span>{t('batch.reveal')}</span>
            </button>
          )}
        </div>
      )}

      {!available && <p className="resize-note">{t('panorama.desktopOnly')}</p>}
      {error && <p className="resize-note export-error">{error}</p>}

      {running ? (
        <button className="resize-apply secondary" onClick={() => (cancelled.current = true)}>
          <span>{t('panorama.cancel')}</span>
        </button>
      ) : (
        <button className="resize-apply" onClick={() => void run()} disabled={!ready}>
          <Layers size={14} />
          <span>{files.length === 0 ? t('batch.needPhotos') : t('batch.run', { count: files.length })}</span>
        </button>
      )}
    </div>
  );
};
