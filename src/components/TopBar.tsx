// src/components/TopBar.tsx
import { useEffect, useRef, useState } from 'react';
import {
  ClipboardCopy,
  ClipboardPaste,
  Copy,
  ClipboardCheck,
  Download,
  FolderOpen,
  MoreHorizontal,
  Moon,
  Save,
  Scaling,
  Settings,
  Sun,
  Images,
} from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { useSceneStore } from '../store/sceneStore';
import { copySelection, pasteClipboard } from '../ops/clipboard';
import { activeDevelop, copyDevelop, pasteDevelop } from '../ops/develop';
import { Logo } from './Logo';
import { useT } from '../i18n';

interface Props {
  onOpen: () => void;
  onExport: () => void;
  onPanorama: () => void;
  onSaveProject: () => void;
  onSettings: () => void;
  onResize: () => void;
  hasPhoto: boolean;
}

/**
 * Title bar: identity on the left, what Nabu understands in the middle,
 * everything that is not editing on the right.
 *
 * The centre chip is the only place the application reports what it has worked
 * out about the photo. It replaced the prompt box, and it has to be honest
 * about it: with no detector installed it says so rather than implying it can
 * see. The word "AI" never appears — people want their photo read, not a
 * technology announced.
 */
export const TopBar: React.FC<Props> = ({
  onOpen,
  onExport,
  onPanorama,
  onSaveProject,
  onSettings,
  onResize,
  hasPhoto,
}) => {
  const doc = useActiveDocument();
  const { theme, toggleTheme } = useDocumentStore();
  const clipboard = useDocumentStore((s) => s.clipboard);
  const developClipboard = useDocumentStore((s) => s.developClipboard);
  const analyzing = useSceneStore((s) => Object.keys(s.pending).length > 0);
  const entries = useSceneStore((s) => s.entries);
  const activeId = useDocumentStore((s) => s.activeId);
  const t = useT();

  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Clicking anywhere else puts the menu away, the way a menu is expected to.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMenuOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [menuOpen]);

  const scene = entries[activeId]?.scene ?? null;

  const status = (() => {
    if (!doc) return { key: 'scene.idle', params: undefined, tone: 'quiet' as const };
    if (analyzing) return { key: 'scene.reading', params: undefined, tone: 'busy' as const };
    if (!scene) return { key: 'scene.blind', params: undefined, tone: 'quiet' as const };
    if (scene.objects.length === 0) {
      return { key: 'scene.nothing', params: undefined, tone: 'quiet' as const };
    }
    return {
      key: 'scene.understood',
      params: { count: scene.objects.length },
      tone: 'ready' as const,
    };
  })();

  return (
    <header className="nabu-topbar">
      <div className="topbar-brand">
        <Logo size={22} />
        <span className="rail-wordmark">NABU</span>
      </div>

      <div className={`scene-chip ${status.tone}`}>
        <span className="scene-dot" />
        <span>{t(status.key, status.params)}</span>
      </div>

      <div className="topbar-actions" ref={menuRef}>
        <button
          className="topbar-icon"
          onClick={toggleTheme}
          title={t(theme === 'dark' ? 'settings.theme.light' : 'settings.theme.dark')}
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        <button
          className={`topbar-icon ${menuOpen ? 'on' : ''}`}
          onClick={() => setMenuOpen((open) => !open)}
          title={t('menu.more')}
        >
          <MoreHorizontal size={16} />
        </button>

        {menuOpen && (
          <div className="topbar-menu">
            <button
              onClick={() => {
                setMenuOpen(false);
                onOpen();
              }}
            >
              <FolderOpen size={14} />
              <span>{t('rail.open')}</span>
              <kbd>Ctrl+O</kbd>
            </button>

            <button
              onClick={() => {
                setMenuOpen(false);
                onPanorama();
              }}
            >
              <Images size={14} />
              <span>{t('panorama.menu')}</span>
            </button>

            <button
              disabled={!hasPhoto}
              onClick={() => {
                setMenuOpen(false);
                onSaveProject();
              }}
            >
              <Save size={14} />
              <span>{t('project.save')}</span>
              <kbd>Ctrl+S</kbd>
            </button>

            <button
              disabled={!hasPhoto}
              onClick={() => {
                setMenuOpen(false);
                onExport();
              }}
            >
              <Download size={14} />
              <span>{t('rail.export')}</span>
              <kbd>Ctrl+E</kbd>
            </button>

            <button
              disabled={!hasPhoto}
              onClick={() => {
                setMenuOpen(false);
                onResize();
              }}
            >
              <Scaling size={14} />
              <span>{t('resize.title')}</span>
            </button>

            <div className="topbar-menu-sep" />

            {/* Copy and paste exist as keystrokes, which is fine for the people
                who already know them and invisible to everyone else. */}
            <button
              disabled={!hasPhoto}
              onClick={() => {
                setMenuOpen(false);
                void copySelection();
              }}
            >
              <ClipboardCopy size={14} />
              <span>{t('action.copy')}</span>
              <kbd>Ctrl+C</kbd>
            </button>

            <button
              disabled={!hasPhoto || !clipboard}
              onClick={() => {
                setMenuOpen(false);
                void pasteClipboard();
              }}
            >
              <ClipboardPaste size={14} />
              <span>{t('paste.action')}</span>
              <kbd>Ctrl+V</kbd>
            </button>

            <button
              disabled={!hasPhoto || !activeDevelop()}
              title={!hasPhoto || activeDevelop() ? undefined : t('develop.nothing')}
              onClick={() => {
                setMenuOpen(false);
                copyDevelop();
              }}
            >
              <Copy size={14} />
              <span>{t('develop.copy')}</span>
              <kbd>Ctrl+Shift+C</kbd>
            </button>

            <button
              disabled={!hasPhoto || !developClipboard}
              onClick={() => {
                setMenuOpen(false);
                pasteDevelop();
              }}
            >
              <ClipboardCheck size={14} />
              <span>{t('develop.paste')}</span>
              <kbd>Ctrl+Shift+V</kbd>
            </button>

            <div className="topbar-menu-sep" />

            <button
              onClick={() => {
                setMenuOpen(false);
                onSettings();
              }}
            >
              <Settings size={14} />
              <span>{t('settings.title')}</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
