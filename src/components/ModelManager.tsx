// src/components/ModelManager.tsx
import { useEffect } from 'react';
import {
  X,
  Download,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  FolderOpen,
  HardDrive,
  Ban,
  RefreshCw,
} from 'lucide-react';
import { useModelStore } from '../store/modelStore';
import { TASK_KEYS, formatBytes, type ModelStatus } from '../types/models';
import { useI18nStore, useT } from '../i18n';
import { describeError } from '../i18n/backendError';

interface Props {
  open: boolean;
  onClose: () => void;
}

export const ModelManager: React.FC<Props> = ({ open, onClose }) => {
  const {
    models,
    progress,
    errors,
    loading,
    unavailable,
    refresh,
    startDownload,
    cancelDownload,
    removeModel,
    revealFolder,
  } = useModelStore();

  const t = useT();
  const locale = useI18nStore((s) => s.current);
  const size = (bytes: number) => formatBytes(bytes, locale);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (open) void refresh();
  }, [open, refresh]);

  if (!open) return null;

  const grouped = models.reduce<Record<string, ModelStatus[]>>((acc, model) => {
    (acc[model.task] ||= []).push(model);
    return acc;
  }, {});

  const totalOnDisk = models
    .filter((m) => m.state === 'ready')
    .reduce((sum, m) => sum + m.totalBytes, 0);

  return (
    <div className="nabu-modal-backdrop" onClick={onClose}>
      <div
        className="nabu-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={t('models.title')}
      >
        <header className="modal-header">
          <div className="panel-title">
            <HardDrive size={16} />
            <span>{t('models.title')}</span>
          </div>
          <div className="panel-actions">
            <button className="icon-btn" onClick={() => void refresh()} title={t('models.refresh')}>
              <RefreshCw size={15} className={loading ? 'spin-icon' : ''} />
            </button>
            <button className="icon-btn" onClick={() => void revealFolder()} title={t('models.openFolder')}>
              <FolderOpen size={15} />
            </button>
            <button className="icon-btn" onClick={onClose} title={t('models.close')}>
              <X size={16} />
            </button>
          </div>
        </header>

        <div className="modal-body">
          {unavailable && (
            <div className="model-warning">
              <AlertTriangle size={15} />
              <span>{t('models.desktopOnly', { command: 'npm run tauri dev' })}</span>
            </div>
          )}

          {!unavailable && (
            <p className="modal-intro">
              {t('models.intro')}
              {totalOnDisk > 0 && ` ${t('models.installed', { size: size(totalOnDisk) })}`}
            </p>
          )}

          {Object.entries(grouped).map(([task, entries]) => (
            <section key={task} className="model-group">
              <h3 className="model-group-title">
                {t(TASK_KEYS[task as keyof typeof TASK_KEYS] ?? task)}
              </h3>

              {entries.map((model) => {
                const active = progress[model.id];
                const error = errors[model.id];
                const isReady = model.state === 'ready';
                const isDownloading = Boolean(active);

                return (
                  <article key={model.id} className={`model-card ${isReady ? 'installed' : ''}`}>
                    <div className="model-card-main">
                      <div className="model-card-head">
                        <span className="model-name">{model.name}</span>
                        {isReady && (
                          <span className="model-badge ready">
                            <CheckCircle2 size={11} /> {t('models.ready')}
                          </span>
                        )}
                        {model.state === 'partial' && !isDownloading && (
                          <span className="model-badge partial">{t('models.partial')}</span>
                        )}
                        {!model.commercialUse && (
                          <span className="model-badge warn" title={model.license}>
                            {t('models.nonCommercial')}
                          </span>
                        )}
                      </div>

                      <p className="model-desc">{t(model.descriptionKey)}</p>

                      <div className="model-meta">
                        <span>{size(model.totalBytes)}</span>
                        <span className="dot">-</span>
                        <span>{t('models.vram', { mb: model.vramRequiredMb })}</span>
                        <span className="dot">-</span>
                        <span>{model.license}</span>
                      </div>

                      {isDownloading && (
                        <div className="model-progress">
                          <div className="model-progress-bar">
                            <div
                              className="model-progress-fill"
                              style={{ width: `${Math.max(2, active.percent)}%` }}
                            />
                          </div>
                          <span className="model-progress-label">
                            {active.totalBytes > 0
                              ? `${size(active.downloadedBytes)} / ${size(active.totalBytes)}`
                              : t('models.connecting')}
                            {active.bytesPerSecond > 0 && ` - ${size(active.bytesPerSecond)}/s`}
                            {active.fileCount > 1 &&
                              ` - ${t('models.fileProgress', {
                                current: Math.min(active.fileIndex + 1, active.fileCount),
                                total: active.fileCount,
                              })}`}
                          </span>
                        </div>
                      )}

                      {error !== undefined && (
                        <div className="model-error">
                          <AlertTriangle size={13} />
                          <span>{describeError(error)}</span>
                        </div>
                      )}
                    </div>

                    <div className="model-card-actions">
                      {isDownloading ? (
                        <button className="nav-btn" onClick={() => void cancelDownload(model.id)}>
                          <Ban size={14} />
                          <span>{t('models.cancel')}</span>
                        </button>
                      ) : isReady ? (
                        <button
                          className="icon-btn danger"
                          onClick={() => void removeModel(model.id)}
                          title={t('models.delete')}
                        >
                          <Trash2 size={15} />
                        </button>
                      ) : (
                        <button
                          className="nav-btn"
                          onClick={() => void startDownload(model.id)}
                          disabled={unavailable}
                        >
                          <Download size={14} />
                          <span>
                            {model.state === 'partial' ? t('models.resume') : t('models.download')}
                          </span>
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </section>
          ))}

          {!unavailable && models.length === 0 && !loading && (
            <p className="modal-intro">{t('models.empty')}</p>
          )}
        </div>
      </div>
    </div>
  );
};
