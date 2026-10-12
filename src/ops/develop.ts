// src/ops/develop.ts
//
// Copying how one photo was developed onto another: the sliders, the curves and
// the look, without the pixels, masks or regions that only make sense on the
// photo they were drawn on.

import { useDocumentStore, type DevelopSettings } from '../store/documentStore';
import { areCurvesNeutral, isNeutral } from '../types/document';

/** The develop settings of the active layer, or null when it has none. */
export function activeDevelop(): DevelopSettings | null {
  const { sessions, activeId } = useDocumentStore.getState();
  const doc = sessions[activeId]?.document;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : undefined;
  if (!layer) return null;

  const settings: DevelopSettings = {};
  if (!isNeutral(layer.adjustments)) settings.adjustments = { ...layer.adjustments! };
  if (!areCurvesNeutral(layer.curves)) settings.curves = structuredClone(layer.curves);
  if (layer.filter) settings.filter = structuredClone(layer.filter);
  return Object.keys(settings).length > 0 ? settings : null;
}

export function copyDevelop(): boolean {
  const settings = activeDevelop();
  if (!settings) return false;
  useDocumentStore.getState().setDevelopClipboard(settings);
  return true;
}

export function pasteDevelop(): boolean {
  const state = useDocumentStore.getState();
  const settings = state.developClipboard;
  const doc = state.sessions[state.activeId]?.document;
  const layerId = doc?.activeLayerId;
  if (!settings || !layerId) return false;
  state.pasteDevelop(layerId, settings);
  return true;
}
