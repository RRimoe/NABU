// src/App.tsx
import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { LeftRail } from './components/LeftRail';
import { CanvasViewport } from './components/CanvasViewport';
import { CommandBar } from './components/CommandBar';
import { StatusBar } from './components/StatusBar';
import { SidePanels } from './components/SidePanels';
import { DocumentTabs } from './components/DocumentTabs';
import { ModelManager } from './components/ModelManager';
import { RefineBar } from './components/RefineBar';
import { EmptyState } from './components/EmptyState';
import { SettingsPanel } from './components/SettingsPanel';
import { BusyOverlay } from './components/BusyOverlay';
import { useDocumentStore, useActiveDocument, useHasDocuments } from './store/documentStore';
import { useModelStore } from './store/modelStore';
import { useSceneStore } from './store/sceneStore';
import { useI18nStore, useT } from './i18n';
import { describeError } from './i18n/backendError';
import { useActivityStore } from './store/activityStore';
import { exportDocumentToPng, RENDER_THEMES, loadImage } from './engine/RenderEngine';
import './App.css';

function App() {
  const doc = useActiveDocument();
  const hasDocuments = useHasDocuments();
  const {
    undo,
    redo,
    theme,
    isAIBusy,
    refineMode,
    setRefineMode,
    setSelection,
    openImageAsDocument,
    updateAIProgress,
  } = useDocumentStore();
  const { subscribe, refresh, loadAccelerator } = useModelStore();
  const loadExternalLanguages = useI18nStore((s) => s.loadExternal);
  const t = useT();
  const activityCount = useActivityStore((s) => s.entries.length);

  const [modelsOpen, setModelsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [panelsOpen, setPanelsOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Panels only make sense once they have something to show.
  const panelsAvailable = hasDocuments || activityCount > 0;
  const showPanels = panelsOpen && panelsAvailable;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  // Backend events: model downloads and inference progress.
  useEffect(() => {
    const disposers: Array<() => void> = [];
    let disposed = false;

    const attach = (fn: () => void) => {
      if (disposed) fn();
      else disposers.push(fn);
    };

    void subscribe().then(attach);
    // The backend reports a stage identifier, not prose: it is translated here.
    void listen<{ percent: number; stage: string }>('ai-task-progress', (event) => {
      updateAIProgress(event.payload.percent, event.payload.stage);
    }).then(attach);

    void refresh();
    void loadAccelerator();
    // Translations dropped into the user folder, loaded alongside the built-in ones.
    void loadExternalLanguages();

    return () => {
      disposed = true;
      disposers.forEach((fn) => fn());
    };
  }, [subscribe, refresh, loadAccelerator, loadExternalLanguages, updateAIProgress]);

  /** Opens one or more files, each as its own document. */
  const openFiles = useCallback(
    (files: File[]) => {
      files
        .filter((file) => file.type.startsWith('image/'))
        .forEach((file) => {
          const reader = new FileReader();
          reader.onload = async (event) => {
            const dataUrl = event.target?.result as string | undefined;
            if (!dataUrl) return;
            try {
              const image = await loadImage(dataUrl);
              openImageAsDocument(
                file.name.replace(/\.[^/.]+$/, ''),
                dataUrl,
                image.naturalWidth,
                image.naturalHeight
              );
              // Fast pass of the vision engine, in the background: by the time
              // the first request mentions an object, the scene already exists.
              void useSceneStore.getState().analyzeActive();
            } catch {
              alert(t('error.unreadableImage', { name: file.name }));
            }
          };
          reader.readAsDataURL(file);
        });
    },
    [openImageAsDocument, t]
  );

  // Global shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = document.activeElement?.tagName;
      const typing = tag === 'INPUT' || tag === 'TEXTAREA';

      // Tab switches between the bare interface and the professional panels.
      if (e.key === 'Tab' && !typing) {
        e.preventDefault();
        setPanelsOpen((open) => !open);
        return;
      }

      if (e.key === 'Escape') {
        if (refineMode) setRefineMode(null);
        else setSelection(null);
        return;
      }

      if (typing) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        fileInputRef.current?.click();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [undo, redo, refineMode, setRefineMode, setSelection]);

  const handleExport = useCallback(async () => {
    if (!doc) return;
    try {
      const dataUrl = await exportDocumentToPng(doc, RENDER_THEMES[theme]);
      const link = document.createElement('a');
      link.download = `${doc.title || 'nabu'}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      alert(t('error.exportFailed', { reason: describeError(err) }));
    }
  }, [doc, theme, t]);

  /** "Edit" in the rail puts the cursor in the prompt: that is its whole job. */
  const focusPrompt = useCallback(() => {
    const input = document.querySelector<HTMLInputElement>('.command-input');
    input?.focus();
  }, []);

  return (
    <div
      className={`nabu-app ${showPanels ? 'panels-open' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        // Only when the pointer really leaves the window, not a child element.
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        openFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => {
          openFiles(Array.from(e.target.files ?? []));
          e.target.value = '';
        }}
        accept="image/*"
        multiple
        style={{ display: 'none' }}
      />

      <LeftRail
        onOpen={() => fileInputRef.current?.click()}
        onExport={handleExport}
        onSettings={() => setSettingsOpen(true)}
        onEdit={focusPrompt}
        hasPhoto={hasDocuments}
      />

      <div className="nabu-stage">
        <DocumentTabs />

        <div className="nabu-canvas-area">
          <BusyOverlay />

          {hasDocuments ? (
            <CanvasViewport />
          ) : (
            <EmptyState onOpen={() => fileInputRef.current?.click()} dragging={dragging} />
          )}

          {refineMode ? (
            <RefineBar />
          ) : (
            <CommandBar busy={isAIBusy} disabled={!hasDocuments} />
          )}
        </div>
      </div>

      {showPanels && <SidePanels />}

      <StatusBar
        panelsOpen={showPanels}
        panelsAvailable={panelsAvailable}
        onTogglePanels={() => setPanelsOpen((o) => !o)}
      />

      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onOpenModels={() => {
          setSettingsOpen(false);
          setModelsOpen(true);
        }}
      />

      <ModelManager open={modelsOpen} onClose={() => setModelsOpen(false)} />
    </div>
  );
}

export default App;
