// src/components/SettingsPanel.tsx
import { useEffect } from 'react';
import { X, Moon, Sun, Languages, HardDrive, ChevronRight } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { useModelStore } from '../store/modelStore';
import { useI18nStore, useT } from '../i18n';
import { formatBytes } from '../types/models';

interface Props {
  open: boolean;
  onClose: () => void;
  onOpenModels: () => void;
}

/**
 * The only place with permanent settings, reached from the rail.
 *
 * It holds what a person changes once and forgets: theme, language, models.
 * Anything that belongs to editing lives next to the photo instead.
 */
export const SettingsPanel: React.FC<Props> = ({ open, onClose, onOpenModels }) => {
  const { theme, setTheme } = useDocumentStore();
  const models = useModelStore((s) => s.models);
  const { languages, current, setLanguage } = useI18nStore();
  const t = useT();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const installed = models.filter((m) => m.state === 'ready');
  const onDisk = installed.reduce((sum, m) => sum + m.totalBytes, 0);

  // Sorted by name so the list does not reshuffle when a translation is added.
  const available = Object.values(languages).sort((a, b) =>
    a.meta.name.localeCompare(b.meta.name)
  );

  return (
    <div className="nabu-modal-backdrop" onClick={onClose}>
      <div
        className="nabu-modal settings-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={t('settings.title')}
      >
        <header className="modal-header">
          <div className="panel-title">
            <span>{t('settings.title')}</span>
          </div>
          <button className="icon-btn" onClick={onClose} title={t('settings.close')}>
            <X size={16} />
          </button>
        </header>

        <div className="modal-body">
          <section className="settings-section">
            <h3 className="settings-label">{t('settings.appearance')}</h3>
            <div className="settings-row">
              <span className="settings-row-label">{t('settings.theme')}</span>
              <div className="segmented">
                <button
                  className={theme === 'light' ? 'active' : ''}
                  onClick={() => setTheme('light')}
                >
                  <Sun size={14} />
                  <span>{t('settings.theme.light')}</span>
                </button>
                <button
                  className={theme === 'dark' ? 'active' : ''}
                  onClick={() => setTheme('dark')}
                >
                  <Moon size={14} />
                  <span>{t('settings.theme.dark')}</span>
                </button>
              </div>
            </div>
          </section>

          <section className="settings-section">
            <h3 className="settings-label">
              <Languages size={13} /> {t('settings.language')}
            </h3>
            <div className="language-grid">
              {available.map((language) => (
                <button
                  key={language.meta.code}
                  className={`language-option ${language.meta.code === current ? 'active' : ''}`}
                  onClick={() => setLanguage(language.meta.code)}
                >
                  <span className="language-name">{language.meta.name}</span>
                  <span className="language-code">{language.meta.code}</span>
                </button>
              ))}
            </div>
            <p className="settings-hint">{t('settings.language.hint')}</p>
          </section>

          <section className="settings-section">
            <h3 className="settings-label">
              <HardDrive size={13} /> {t('settings.models')}
            </h3>
            <button className="settings-link" onClick={onOpenModels}>
              <span>
                {installed.length > 0
                  ? t('settings.models.hint', {
                      count: installed.length,
                      size: formatBytes(onDisk, current),
                    })
                  : t('settings.models.none')}
              </span>
              <span className="settings-link-action">
                {t('settings.manageModels')}
                <ChevronRight size={14} />
              </span>
            </button>
          </section>
        </div>
      </div>
    </div>
  );
};
