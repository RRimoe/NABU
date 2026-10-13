// src/App.tsx
import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { TopBar } from './components/TopBar';
import { LeftRail } from './components/LeftRail';
import { CanvasViewport } from './components/CanvasViewport';
import { StatusBar } from './components/StatusBar';
import { SidePanels } from './components/SidePanels';
import { SuggestionsPanel } from './components/SuggestionsPanel';
import { FiltersPanel } from './components/FiltersPanel';
import { AdjustPanel } from './components/AdjustPanel';
import { FocusPanel } from './components/FocusPanel';
import { Navigator } from './components/Navigator';
import { CurvesPanel } from './components/CurvesPanel';
import { DocumentTabs } from './components/DocumentTabs';
import { ModelManager } from './components/ModelManager';
import { RefineBar } from './components/RefineBar';
import { SelectionRefineBar } from './components/SelectionRefineBar';
import { EmptyState } from './components/EmptyState';
import { SettingsPanel } from './components/SettingsPanel';
import { ResizePopover } from './components/ResizePopover';
import { BusyOverlay } from './components/BusyOverlay';
import { useDocumentStore, useHasDocuments, useDocumentFields } from './store/documentStore';
import { useModelStore } from './store/modelStore';
import { heicToDataUrl, isHeicName } from './ops/heic';
import { useSceneStore } from './store/sceneStore';
import { useI18nStore, useT } from './i18n';
import { useActivityStore } from './store/activityStore';
import { copySelection, pasteClipboard } from './ops/clipboard';
import { loadImage } from './engine/RenderEngine';
import { ExportPopover } from './components/ExportPopover';
import { PanoramaDialog } from './components/PanoramaDialog';
import { BatchDialog } from './components/BatchDialog';
import { ColourPanel } from './components/ColourPanel';
import { MasksPanel } from './components/MasksPanel';
import { downloadBlob, safeFileName } from './ops/export';
import { copyDevelop, pasteDevelop } from './ops/develop';
import {
  isProjectFile,
  parseProject,
  projectBlob,
  ProjectError,
  PROJECT_EXTENSION,
} from './ops/project';
import { connectBridge } from './commands/bridge';
import './App.css';

function App() {
  const hasDocuments = useHasDocuments();
  const {
    undo,
    redo,
    theme,
    refineMode,
    setRefineMode,
    setSelection,
    activeTool,
    setActiveTool,
    selectionShape,
    setSelectionShape,
    focusEditing,
    setFocusEditing,
    dockPanel,
    setDockPanel,
    openImageAsDocument,
    openProjectDocument,
    updateAIProgress,
    brushSize,
    setBrushSize,
  } = useDocumentFields(
    'undo',
    'redo',
    'theme',
    'refineMode',
    'setRefineMode',
    'setSelection',
    'activeTool',
    'setActiveTool',
    'selectionShape',
    'setSelectionShape',
    'focusEditing',
    'setFocusEditing',
    'dockPanel',
    'setDockPanel',
    'openImageAsDocument',
    'openProjectDocument',
    'updateAIProgress',
    'brushSize',
    'setBrushSize',
  );
  const { subscribe, refresh, loadAccelerator } = useModelStore();
  const loadExternalLanguages = useI18nStore((s) => s.loadExternal);
  const t = useT();
  const activityCount = useActivityStore((s) => s.entries.length);

  const [modelsOpen, setModelsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [resizeOpen, setResizeOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [panoramaOpen, setPanoramaOpen] = useState(false);
  const [batchOpen, setBatchOpen] = useState(false);
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

    // Scripts and agents reach the registry once the installed models are
    // known: a call arriving earlier would be told a model is missing.
    void refresh()
      .catch(() => undefined)
      .then(() => connectBridge())
      .then(attach);
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
      // Projects first: they carry their own layers and are not images at all.
      files.filter(isProjectFile).forEach((file) => {
        file
          .text()
          .then((text) => openProjectDocument(parseProject(text)))
          .catch((err) => {
            const reason = err instanceof ProjectError ? err.reason : 'damaged';
            alert(t(`project.error.${reason}`, { name: file.name }));
          });
      });

      const open = async (file: File, dataUrl: string) => {
        try {
          const image = await loadImage(dataUrl);
          openImageAsDocument(
            file.name.replace(/\.[^/.]+$/, ''),
            dataUrl,
            image.naturalWidth,
            image.naturalHeight
          );
          // Reading the scene is now the opening move of the whole
          // application: the suggestions panel has nothing to say about
          // the contents of the photo until this comes back.
          void useSceneStore.getState().analyzeActive();
        } catch {
          alert(t('error.unreadableImage', { name: file.name }));
        }
      };

      files.forEach((file) => {
        // iPhone photos: the webview cannot read them, the backend can. Some
        // systems give them no type at all, so the name decides.
        if (isHeicName(file.name)) {
          file
            .arrayBuffer()
            .then(heicToDataUrl)
            .then((dataUrl) => open(file, dataUrl))
            .catch(() => alert(t('error.unreadableImage', { name: file.name })));
          return;
        }
        if (!file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (event) => {
          const dataUrl = event.target?.result as string | undefined;
          if (dataUrl) void open(file, dataUrl);
        };
        reader.readAsDataURL(file);
      });
    },
    [openImageAsDocument, openProjectDocument, t]
  );

  /** Writes the active document as a .nabu project, every edit still live. */
  const saveProject = useCallback(() => {
    // Read at the moment of saving: subscribing the whole application to the
    // document would re-render every panel on each step of a slider drag.
    const { sessions, activeId } = useDocumentStore.getState();
    const doc = sessions[activeId]?.document;
    if (!doc) return;
    downloadBlob(projectBlob(doc), safeFileName(doc.title, PROJECT_EXTENSION));
  }, []);

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
        else if (focusEditing) setFocusEditing(false);
        else if (activeTool !== 'select') setActiveTool('select');
        else setSelection(null);
        return;
      }

      if (typing) return;

      // Single letters, the way every editor assigns them: the tool you want,
      // one keystroke away, without having to aim at anything.
      if (!e.ctrlKey && !e.metaKey && !e.altKey && hasDocuments) {
        const key = e.key.toLowerCase();
        const tools: Record<string, () => void> = {
          v: () => setActiveTool('select'),
          h: () => setActiveTool('pan'),
          c: () => setActiveTool('crop'),
          b: () => setActiveTool(activeTool === 'brush' ? 'select' : 'brush'),
          e: () => setActiveTool(activeTool === 'eraser' ? 'select' : 'eraser'),
          j: () => setActiveTool(activeTool === 'heal' ? 'select' : 'heal'),
          // Brackets resize the brush, the convention every painting tool shares.
          '[': () => setBrushSize(Math.round(brushSize * 0.8)),
          ']': () => setBrushSize(Math.max(brushSize + 1, Math.round(brushSize * 1.25))),
          t: () => setActiveTool(activeTool === 'transform' ? 'select' : 'transform'),
          a: () => setDockPanel(dockPanel === 'adjust' ? 'suggestions' : 'adjust'),
          f: () => setDockPanel(dockPanel === 'filters' ? 'suggestions' : 'filters'),
          s: () => {
            const order = ['free', 'rect', 'ellipse'] as const;
            setSelectionShape(order[(order.indexOf(selectionShape) + 1) % order.length]);
          },
        };

        const run = tools[key];
        if (run) {
          e.preventDefault();
          run();
          return;
        }
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's' && hasDocuments) {
        e.preventDefault();
        saveProject();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'e' && hasDocuments) {
        e.preventDefault();
        setExportOpen(true);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        fileInputRef.current?.click();
      } else if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'c' && hasDocuments) {
        // With Shift, copy how the photo was developed rather than its pixels.
        e.preventDefault();
        copyDevelop();
      } else if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'v' && hasDocuments) {
        e.preventDefault();
        pasteDevelop();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c' && hasDocuments) {
        e.preventDefault();
        void copySelection();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v' && hasDocuments) {
        e.preventDefault();
        void pasteClipboard();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    undo,
    redo,
    refineMode,
    setRefineMode,
    setSelection,
    activeTool,
    setActiveTool,
    selectionShape,
    setSelectionShape,
    focusEditing,
    setFocusEditing,
    dockPanel,
    setDockPanel,
    hasDocuments,
    saveProject,
    brushSize,
    setBrushSize,
  ]);

  return (
    <div
      className={`nabu-app ${hasDocuments ? 'has-dock' : ''}`}
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
        accept={`image/*,.heic,.heif,.${PROJECT_EXTENSION}`}
        multiple
        style={{ display: 'none' }}
      />

      <TopBar
        onOpen={() => fileInputRef.current?.click()}
        onExport={() => setExportOpen(true)}
        onPanorama={() => setPanoramaOpen(true)}
        onBatch={() => setBatchOpen(true)}
        onSaveProject={saveProject}
        onSettings={() => setSettingsOpen(true)}
        onResize={() => setResizeOpen(true)}
        hasPhoto={hasDocuments}
      />

      <LeftRail hasPhoto={hasDocuments} />

      <div className="nabu-stage">
        <DocumentTabs />

        <div className="nabu-canvas-area">
          <BusyOverlay />

          {hasDocuments ? (
            <CanvasViewport />
          ) : (
            <EmptyState
              onOpen={() => fileInputRef.current?.click()}
              onPanorama={() => setPanoramaOpen(true)}
              dragging={dragging}
            />
          )}

          {refineMode && <RefineBar />}
          <SelectionRefineBar />
        </div>
      </div>

      {/* One dock on the right. What it shows is the rail's business; the
          professional panels stack underneath whatever is up there. */}
      {(hasDocuments || showPanels) && (
        <aside className="nabu-dock">
          {hasDocuments && dockPanel === 'suggestions' && (
            <SuggestionsPanel onNeedsModel={() => setModelsOpen(true)} />
          )}

          {hasDocuments && dockPanel === 'filters' && (
            <FiltersPanel onClose={() => setDockPanel('suggestions')} />
          )}

          {hasDocuments && dockPanel === 'adjust' && (
            <>
              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.navigator')}</span>
                </header>
                <Navigator />
              </section>

              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.adjust')}</span>
                </header>
                <AdjustPanel />
              </section>

              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.colour')}</span>
                </header>
                <ColourPanel />
              </section>

              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.masks')}</span>
                </header>
                <MasksPanel />
              </section>

              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.focus')}</span>
                </header>
                <FocusPanel />
              </section>

              <section className="side-panel">
                <header className="side-panel-header">
                  <span>{t('panels.curves')}</span>
                </header>
                <CurvesPanel />
              </section>
            </>
          )}

          {showPanels && <SidePanels />}
        </aside>
      )}

      <StatusBar
        panelsOpen={showPanels}
        panelsAvailable={panelsAvailable}
        onTogglePanels={() => setPanelsOpen((o) => !o)}
        onOpenModels={() => setModelsOpen(true)}
      />

      {resizeOpen && (
        <div className="nabu-modal-backdrop" onMouseDown={() => setResizeOpen(false)}>
          <div className="resize-centred" onMouseDown={(e) => e.stopPropagation()}>
            <ResizePopover onClose={() => setResizeOpen(false)} />
          </div>
        </div>
      )}

      {panoramaOpen && (
        <div className="nabu-modal-backdrop">
          <div className="resize-centred" onMouseDown={(e) => e.stopPropagation()}>
            <PanoramaDialog onClose={() => setPanoramaOpen(false)} />
          </div>
        </div>
      )}

      {batchOpen && (
        <div className="nabu-modal-backdrop">
          <div className="resize-centred" onMouseDown={(e) => e.stopPropagation()}>
            <BatchDialog onClose={() => setBatchOpen(false)} />
          </div>
        </div>
      )}

      {exportOpen && (
        <div className="nabu-modal-backdrop" onMouseDown={() => setExportOpen(false)}>
          <div className="resize-centred" onMouseDown={(e) => e.stopPropagation()}>
            <ExportPopover onClose={() => setExportOpen(false)} />
          </div>
        </div>
      )}

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
