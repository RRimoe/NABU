// src/components/StatusBar.tsx
import {
  Circle,
  Lasso,
  Maximize2,
  PanelRight,
  RotateCcw,
  RotateCw,
  Square,
} from 'lucide-react';
import {
  useDocumentStore,
  useViewport,
  useActiveDocument,
  useCanUndo,
  useCanRedo,
} from '../store/documentStore';
import { useSceneStore } from '../store/sceneStore';
import { useModelStore } from '../store/modelStore';
import { useAdviceStore } from '../store/adviceStore';
import { runRecipe } from '../ai/recipes';
import { Logo } from './Logo';
import type { SelectionShape } from '../types/document';
import { useT } from '../i18n';

interface Props {
  panelsOpen: boolean;
  /** False while there is nothing to put in the panels. */
  panelsAvailable: boolean;
  onTogglePanels: () => void;
  onOpenModels: () => void;
}

/**
 * How a drag on the photo is interpreted.
 *
 * Freehand comes first because it is the default and the one the rest of the
 * interface is built around; the other two are there for the jobs a hand-drawn
 * outline is simply wrong for, like squaring off a sign or a face.
 */
const SHAPES: Array<{ id: SelectionShape; Icon: typeof Lasso }> = [
  { id: 'free', Icon: Lasso },
  { id: 'rect', Icon: Square },
  { id: 'ellipse', Icon: Circle },
];

export const StatusBar: React.FC<Props> = ({
  panelsOpen,
  panelsAvailable,
  onTogglePanels,
  onOpenModels,
}) => {
  const viewport = useViewport();
  const doc = useActiveDocument();
  const { undo, redo, requestFitToScreen, selectionShape, setSelectionShape } =
    useDocumentStore();
  const canUndo = useCanUndo();
  const canRedo = useCanRedo();
  const models = useModelStore((s) => s.models);
  const unavailable = useModelStore((s) => s.unavailable);
  const accelerator = useModelStore((s) => s.accelerator);
  const analyzing = useSceneStore((s) => Object.keys(s.pending).length > 0);
  const activeId = useDocumentStore((s) => s.activeId);
  const isAIBusy = useDocumentStore((s) => s.isAIBusy);
  const advice = useAdviceStore((s) => s.entries[activeId]?.advice ?? null);
  const t = useT();

  const installed = models.filter((model) => model.state === 'ready').length;

  /**
   * Does the thing Nabu currently thinks is worth doing.
   *
   * The one round control in the application, in the middle of the window, so
   * it has to carry the most valuable action there is — and that is applying
   * the top proposal, not repeating an internal calculation. With no proposal
   * yet it falls back to reading the photo again, which is the only sensible
   * thing left to do.
   */
  const act = () => {
    if (!doc) return;

    if (advice) {
      void runRecipe(advice.recipe, advice.target);
      return;
    }

    const activeId = useDocumentStore.getState().activeId;
    useSceneStore.setState((state) => {
      const entries = { ...state.entries };
      delete entries[activeId];
      return { entries };
    });
    void useSceneStore.getState().analyzeActive();
  };

  return (
    <footer className="nabu-status">
      <div className="status-left">
        <button className="status-models" onClick={onOpenModels} title={t('status.models.hint')}>
          <span className={`status-led ${unavailable ? 'off' : installed > 0 ? 'on' : 'idle'}`} />
          <span>
            {unavailable
              ? t('status.models.unavailable')
              : t('status.models', { count: installed })}
          </span>
        </button>

        {accelerator && (
          <>
            <span className="status-dot">-</span>
            <span>{accelerator}</span>
          </>
        )}
      </div>

      {/* The one round button in the application, in the one place nothing else
          competes for. It carries the mark itself rather than an icon standing
          in for it: what it is about to do is said by the fill and the tooltip,
          not by swapping the identity out for a pictogram. */}
      <button
        className={`status-scan ${analyzing ? 'busy' : ''} ${advice ? 'ready' : ''}`}
        onClick={act}
        disabled={!doc || analyzing || isAIBusy}
        title={advice ? t(`recipe.${advice.recipe}`) : t('status.reanalyze')}
      >
        <Logo size={24} tone="accent" />
      </button>

      <div className="status-right">
        {doc && (
          <div className="status-shapes" title={t('selection.shape.hint')}>
            {SHAPES.map(({ id, Icon }) => (
              <button
                key={id}
                className={`status-btn ${selectionShape === id ? 'on' : ''}`}
                onClick={() => setSelectionShape(id)}
                title={t(`selection.shape.${id}`)}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>
        )}

        {doc && <span className="status-dimensions">{doc.width} x {doc.height}</span>}

        <button className="status-btn" onClick={undo} disabled={!canUndo} title={t('status.undo')}>
          <RotateCcw size={15} />
        </button>
        <button className="status-btn" onClick={redo} disabled={!canRedo} title={t('status.redo')}>
          <RotateCw size={15} />
        </button>

        {viewport && (
          <>
            <span className="status-zoom">{Math.round(viewport.zoom * 100)}%</span>
            <button className="status-btn" onClick={requestFitToScreen} title={t('status.fit')}>
              <Maximize2 size={15} />
            </button>
          </>
        )}

        {panelsAvailable && (
          <button
            className={`status-btn ${panelsOpen ? 'on' : ''}`}
            onClick={onTogglePanels}
            title={t('status.panels')}
          >
            <PanelRight size={15} />
          </button>
        )}
      </div>
    </footer>
  );
};
