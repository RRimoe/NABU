// src/components/ActionBar.tsx
import {
  Aperture,
  Bandage,
  Brush,
  ClipboardCopy,
  Crop,
  Eraser,
  Palette,
  Scissors,
  Sparkles,
  X,
} from 'lucide-react';
import { useDocumentStore, useActiveDocument, useSelection, useDocumentFields } from '../store/documentStore';
import { useModelStore } from '../store/modelStore';
import { useSceneStore } from '../store/sceneStore';
import { extractObject, removeSelectedRegion } from '../ai/operations';
import { actionsFor, classLabelOf, objectUnderSelection, type ActionId } from '../ai/semantics';
import { cropDocument } from '../ops/crop';
import { contentAwareFill, retouchAvailable } from '../ops/heal';
import { copySelection } from '../ops/clipboard';
import { focusOnSelection } from '../ops/focus';
import { enhanceSubject, recolourSelection } from '../ops/local';
import { selectionBounds } from '../engine/compositing';
import { useT } from '../i18n';

/** How far a recolour turns the hue. A third of the wheel reads as a change. */
const RECOLOUR_STEP = 120;

/**
 * What you can do with what you just selected — and it depends on what that is.
 *
 * A traditional editor shows the same tools whatever is under the cursor,
 * because it does not know. Here the detector named every object before the
 * drag even started, so selecting a person offers to brighten them and
 * selecting a plant offers to change its colour. The order differs too: for a
 * person the first thing anyone wants is to make them look better, for a stray
 * object in the corner it is to get rid of it.
 *
 * A button appears only when it can actually run. An action needing a model
 * that is not installed is not shown greyed out with an explanation nobody
 * reads — it is simply not offered, and the suggestions panel is where missing
 * models get named.
 */
export const ActionBar: React.FC = () => {
  const doc = useActiveDocument();
  const selection = useSelection();
  const activeId = useDocumentStore((s) => s.activeId);
  const { isAIBusy, setSelection, setRefineMode } = useDocumentFields('isAIBusy', 'setSelection', 'setRefineMode');
  const readyFor = useModelStore((s) => s.readyFor);
  const scene = useSceneStore((s) => s.entries[activeId]?.scene ?? null);
  const t = useT();

  if (!doc || !selection) return null;

  const layerId = doc.activeLayerId;
  const layer = layerId ? doc.layers[layerId] : null;
  if (!layerId || !layer) return null;

  const subject = objectUnderSelection(selection, scene, {
    x: layer.transform.x,
    y: layer.transform.y,
  });

  const canSegment = Boolean(readyFor('segmentation'));
  const canInpaint = Boolean(readyFor('inpainting'));
  // Refining only makes sense on a layer that still knows its original.
  const canRefine = Boolean(layer.sourceLayerId);

  /** Cuts the whole photo down to the selected area. */
  const cropHere = () => {
    const box = selectionBounds(selection, doc.width, doc.height, 0);
    void cropDocument(box);
  };

  const BUTTONS: Record<
    ActionId,
    { Icon: typeof Eraser; labelKey: string; hintKey: string; run: () => void; available: boolean }
  > = {
    remove: {
      Icon: Eraser,
      labelKey: 'bubble.remove',
      hintKey: 'bubble.remove.hint',
      run: () => void removeSelectedRegion(),
      available: canInpaint,
    },
    // Rebuilt from the photo itself, so it needs no model: the right tool for
    // texture (a wall, grass, sky, skin) and the stand-in for "remove" when
    // the inpainting model is not installed.
    fill: {
      Icon: Bandage,
      labelKey: 'action.fill',
      hintKey: 'action.fill.hint',
      run: () => void contentAwareFill(),
      available: retouchAvailable(),
    },
    extract: {
      Icon: Scissors,
      labelKey: 'bubble.extract',
      hintKey: 'bubble.extract.hint',
      run: () => void extractObject(),
      available: canSegment,
    },
    'blur-behind': {
      Icon: Aperture,
      labelKey: 'action.blurBehind',
      hintKey: 'action.blurBehind.hint',
      run: () => void focusOnSelection(),
      // Needs no model: without a segmenter the drawn shape is the mask.
      available: true,
    },
    'enhance-subject': {
      Icon: Sparkles,
      labelKey: 'action.enhanceSubject',
      hintKey: 'action.enhanceSubject.hint',
      run: () => void enhanceSubject(),
      available: true,
    },
    recolour: {
      Icon: Palette,
      labelKey: 'action.recolour',
      hintKey: 'action.recolour.hint',
      run: () => void recolourSelection(RECOLOUR_STEP),
      available: true,
    },
    'crop-here': {
      Icon: Crop,
      labelKey: 'action.cropHere',
      hintKey: 'action.cropHere.hint',
      run: cropHere,
      available: true,
    },
    copy: {
      Icon: ClipboardCopy,
      labelKey: 'action.copy',
      hintKey: 'action.copy.hint',
      run: () => void copySelection().then(() => setSelection(null)),
      available: true,
    },
  };

  const offered = actionsFor(subject).filter((id) => BUTTONS[id].available);

  return (
    <div
      className="nabu-actionbar"
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
    >
      {/* Naming what was recognised is not decoration: it is the only way the
          user can tell an informed list of actions from a generic one. */}
      {subject && (
        <>
          <span className="action-subject">{classLabelOf(subject.class, t)}</span>
          <span className="action-sep" />
        </>
      )}

      {offered.map((id, index) => {
        const { Icon, labelKey, hintKey, run } = BUTTONS[id];
        return (
          <button
            key={id}
            className={index === 0 ? 'action-primary' : ''}
            onClick={run}
            disabled={isAIBusy}
            title={t(hintKey)}
          >
            <Icon size={15} />
            <span>{t(labelKey)}</span>
          </button>
        );
      })}

      {canRefine && (
        <button onClick={() => setRefineMode('erase')} title={t('bubble.refine.hint')}>
          <Brush size={15} />
          <span>{t('bubble.refine')}</span>
        </button>
      )}

      <span className="action-sep" />

      <button
        className="action-dismiss"
        onClick={() => setSelection(null)}
        title={t('bubble.dismiss')}
      >
        <X size={14} />
      </button>
    </div>
  );
};
