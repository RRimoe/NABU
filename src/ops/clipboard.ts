// src/ops/clipboard.ts
//
// Taking a piece of one photo into another.
//
// The clipboard holds pixels, not a reference: the source document can be
// closed, resized or undone and what was copied stays exactly as it was when
// it was copied. That is the behaviour people expect from a clipboard, and the
// alternative — a live link to a layer — breaks the first time either side
// changes.

import { useDocumentStore } from '../store/documentStore';
import { useActivityStore } from '../store/activityStore';
import { loadImage } from '../engine/RenderEngine';
import { layerSize } from '../engine/compositing';
import { cutOut, trimToContent, PHOTO_MATTE } from '../engine/matte';
import { buildSelectionMask } from './mask';
import { t } from '../i18n';
import { describeError } from '../i18n/backendError';
import { report } from './notify';

/**
 * Copies the current selection out of the active layer, or the whole layer
 * when nothing is selected.
 *
 * The cut follows the outline rather than its bounding box: copying a dog and
 * getting a rectangle of lawn with a dog in it is not copying a dog.
 */
export async function copySelection(): Promise<boolean> {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document;
  const selection = session?.selection ?? null;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;

  if (!doc || !layer?.imageDataUrl) {
    report(t('error.noImageOpen'));
    return false;
  }

  const size = layerSize(layer, doc.width, doc.height);
  const image = await loadImage(layer.imageDataUrl);

  // Nothing selected: the layer itself is what was meant. A cutout copied this
  // way keeps its transparency, which is the common case after "Cut out".
  if (!selection) {
    state.setClipboard({
      name: layer.name,
      imageDataUrl: layer.imageDataUrl,
      width: size.width,
      height: size.height,
      sourceId: state.activeId,
    });
    return true;
  }

  const activity = useActivityStore.getState();
  const job = activity.begin('activity.copyCutout');
  state.setAIBusy(true, '', 5);

  try {
    // The outline is a rough gesture, and used as it stands it produces a
    // cutout with a rim of the old background — which is precisely what a
    // composite gives itself away by. So it goes through the segmenter first,
    // exactly as "Cut out" and "Blur behind" already do, and then through the
    // matte, which pulls the edge in, softens it and replaces the colours in
    // the transition band with the object's own.
    activity.step(job, 'activity.buildingMask');
    const built = await buildSelectionMask();
    if (!built) {
      activity.end(job, 'failed');
      return false;
    }

    activity.step(job, 'activity.refiningEdge');
    const maskImage = await loadImage(built.maskDataUrl);
    const cut = cutOut(image, maskImage, size.width, size.height, PHOTO_MATTE);
    const trimmed = cut ? trimToContent(cut) : null;

    if (!trimmed) {
      activity.end(job, 'failed');
      report(t('error.emptyCutout'));
      return false;
    }

    state.setClipboard({
      name: layer.name,
      imageDataUrl: trimmed.toDataURL('image/png'),
      width: trimmed.width,
      height: trimmed.height,
      sourceId: state.activeId,
    });
    activity.end(job, 'done');
    return true;
  } catch (err) {
    activity.end(job, 'failed', err);
    report(describeError(err));
    return false;
  } finally {
    setTimeout(() => state.setAIBusy(false), 200);
  }
}

/**
 * Drops what was copied into the active photo as a new layer.
 *
 * It arrives centred, at a size that fits, and with the transform tool already
 * on — because the very next thing anyone does after pasting is move it.
 */
export async function pasteClipboard(): Promise<boolean> {
  const state = useDocumentStore.getState();
  const item = state.clipboard;
  const doc = state.sessions[state.activeId]?.document;

  if (!doc) {
    report(t('error.noImageOpen'));
    return false;
  }
  if (!item) return false;

  const id = state.addImageLayer(item.name, item.imageDataUrl, item.width, item.height);

  // Something copied from a large photo into a small one would otherwise land
  // mostly outside the frame; two thirds of the shorter side is a size you can
  // see all of and then scale by hand.
  const fit = Math.min(1, (Math.min(doc.width, doc.height) * 0.66) / Math.max(item.width, item.height));

  state.setLayerTransform(id, {
    scaleX: fit,
    scaleY: fit,
    x: (doc.width - item.width * fit) / 2,
    y: (doc.height - item.height * fit) / 2,
  });
  state.commitTransform(id);
  state.setSelection(null);
  state.setActiveTool('transform');
  return true;
}
