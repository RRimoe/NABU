// src/ops/project.ts
//
// The .nabu project file: the document exactly as the editor holds it, so a
// photo closed today reopens tomorrow with every layer, slider, curve, look and
// mask still live. Exporting a PNG keeps the picture; this keeps the work.
//
// The format is the document model serialised as JSON, wrapped in a small
// envelope that says what it is and which version wrote it. Layer pixels are
// already data URLs in the model, so the file is self-contained: one file, no
// sidecar folder to lose.

import { NabuDocument, LayerNode } from '../types/document';

export const PROJECT_EXTENSION = 'nabu';
export const PROJECT_MIME = 'application/x-nabu-project+json';

/** Marker identifying the file, so a random JSON file is not mistaken for one. */
const FORMAT = 'nabu-project';

/**
 * Bumped whenever a saved file would be read wrongly by an older Nabu. Newer
 * fields that an older reader can safely ignore do not need a bump.
 */
export const PROJECT_VERSION = 1;

interface ProjectFile {
  format: typeof FORMAT;
  version: number;
  savedAt: string;
  document: NabuDocument;
}

/** Why a file could not be opened, in terms the interface can translate. */
export class ProjectError extends Error {
  constructor(
    public readonly reason: 'notProject' | 'tooNew' | 'damaged',
    detail?: string
  ) {
    super(detail ?? reason);
    this.name = 'ProjectError';
  }
}

export function isProjectFile(file: File) {
  return file.name.toLowerCase().endsWith(`.${PROJECT_EXTENSION}`);
}

export function serializeProject(doc: NabuDocument): string {
  const file: ProjectFile = {
    format: FORMAT,
    version: PROJECT_VERSION,
    savedAt: new Date().toISOString(),
    document: doc,
  };
  return JSON.stringify(file);
}

export function projectBlob(doc: NabuDocument): Blob {
  return new Blob([serializeProject(doc)], { type: PROJECT_MIME });
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPositive = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

/**
 * Reads a project back, refusing anything that would put the editor in a state
 * it cannot render. A file that half-loads is worse than one that does not: the
 * user would keep working on a document that is silently missing layers.
 */
export function parseProject(text: string): NabuDocument {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new ProjectError('notProject');
  }

  if (!isObject(raw) || raw.format !== FORMAT) throw new ProjectError('notProject');
  if (typeof raw.version !== 'number' || raw.version > PROJECT_VERSION) {
    throw new ProjectError('tooNew', `version ${String(raw.version)}`);
  }

  const doc = raw.document;
  if (!isObject(doc) || !isPositive(doc.width) || !isPositive(doc.height)) {
    throw new ProjectError('damaged', 'document size');
  }
  if (!isObject(doc.layers) || !Array.isArray(doc.rootLayerIds)) {
    throw new ProjectError('damaged', 'layers');
  }

  const layers = doc.layers as Record<string, LayerNode>;
  const rootLayerIds = (doc.rootLayerIds as unknown[]).filter(
    (id): id is string => typeof id === 'string'
  );
  const missing = rootLayerIds.filter((id) => !isObject(layers[id]));
  if (missing.length > 0) throw new ProjectError('damaged', `missing layers ${missing.join(', ')}`);

  const active =
    typeof doc.activeLayerId === 'string' && layers[doc.activeLayerId]
      ? doc.activeLayerId
      : (rootLayerIds[rootLayerIds.length - 1] ?? null);

  return {
    ...(doc as unknown as NabuDocument),
    rootLayerIds,
    layers,
    activeLayerId: active,
    selectedLayerIds: active ? [active] : [],
    background: doc.background === 'white' ? 'white' : 'transparent',
  };
}
