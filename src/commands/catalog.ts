// src/commands/catalog.ts
//
// The commands themselves. Each one is a thin wrapper over an operation the
// interface already performs, so a script and a click produce the same edit,
// recorded in the same history.

import { useDocumentStore } from '../store/documentStore';
import { useSceneStore } from '../store/sceneStore';
import { useLookStore } from '../store/lookStore';
import {
  ADJUSTMENT_RANGES,
  NEUTRAL_ADJUSTMENTS,
  type Adjustments,
  type Point2D,
} from '../types/document';
import { FILTERS } from '../engine/filters';
import { parseCube, encodeTable } from '../engine/lut';
import { loadImage, renderDocument, RENDER_THEMES } from '../engine/RenderEngine';
import { removeBackground, upscale, extractObject, removeObjectByBbox, removeSelectedRegion } from '../ai/operations';
import { resizeByPercent, resizeLongestSide, resizeDocument } from '../ops/resize';
import { cropDocument } from '../ops/crop';
import { heicToDataUrl, isHeicName } from '../ops/heic';
import { contentAwareFill, healActive, pathShape, strokeShape } from '../ops/heal';
import { activeDevelop } from '../ops/develop';
import { encodeCanvas, EXPORT_FORMATS, type ExportFormat } from '../ops/export';
import { serializeProject, parseProject, PROJECT_EXTENSION } from '../ops/project';
import { stitchPanorama, type PanoramaProjection } from '../ops/panorama';
import { t } from '../i18n';
import { register, type Outcome } from './registry';
import { readFile, writeFile, baseName, stem, extension, imageType, blobToDataUrl } from './files';
import type { DevelopSettings } from '../store/documentStore';

const ADJUSTMENT_KEYS = Object.keys(NEUTRAL_ADJUSTMENTS) as Array<keyof Adjustments>;

const done = (data?: unknown): Outcome => ({ messageKey: 'router.done', handled: true, data });
const noImage: Outcome = { messageKey: 'error.noImageOpen', handled: false };

function active() {
  const state = useDocumentStore.getState();
  const session = state.sessions[state.activeId];
  const doc = session?.document ?? null;
  const layer = doc?.activeLayerId ? doc.layers[doc.activeLayerId] : null;
  return { state, session, doc, layer };
}

/** Rounds coordinates for the caller: nobody needs fifteen decimals of a pixel. */
const round = (value: number) => Math.round(value * 100) / 100;

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

/** Opens a photo or a .nabu project from disk as a new document. */
export async function openPath(path: string): Promise<Outcome> {
  const name = baseName(path);
  const bytes = await readFile(path);

  if (extension(path) === PROJECT_EXTENSION) {
    const doc = parseProject(new TextDecoder().decode(bytes));
    useDocumentStore.getState().openProjectDocument(doc);
    return { messageKey: 'commands.opened', params: { name }, handled: true, data: { id: useDocumentStore.getState().activeId } };
  }

  const type = imageType(path);
  if (!type) return { messageKey: 'commands.unsupportedFormat', params: { name }, handled: false };

  const dataUrl = isHeicName(path)
    ? await heicToDataUrl(bytes)
    : await blobToDataUrl(new Blob([bytes as BlobPart], { type }));
  let image: HTMLImageElement;
  try {
    image = await loadImage(dataUrl);
  } catch {
    return { messageKey: 'error.unreadableImage', params: { name }, handled: false };
  }
  const state = useDocumentStore.getState();
  state.openImageAsDocument(stem(path), dataUrl, image.naturalWidth, image.naturalHeight);
  const id = useDocumentStore.getState().activeId;
  return {
    messageKey: 'commands.opened',
    params: { name },
    handled: true,
    data: { id, width: image.naturalWidth, height: image.naturalHeight },
  };
}

function describe() {
  const { session, doc, layer } = active();
  if (!doc || !session) return null;
  return {
    id: doc.id,
    title: doc.title,
    width: doc.width,
    height: doc.height,
    activeLayerId: doc.activeLayerId,
    layers: doc.rootLayerIds
      .map((id) => doc.layers[id])
      .filter(Boolean)
      .map((l) => ({
        id: l.id,
        name: l.name,
        type: l.type,
        visible: l.visible,
        opacity: l.opacity,
        blendMode: l.blendMode,
        width: l.width,
        height: l.height,
        x: l.transform.x,
        y: l.transform.y,
      })),
    develop: layer ? activeDevelop() : null,
    selection: session.selection
      ? { points: session.selection.path.map((p) => ({ x: round(p.x), y: round(p.y) })) }
      : null,
    canUndo: session.historyIndex > 0,
    canRedo: session.historyIndex < session.history.length - 1,
  };
}

/** Renders the document as it would be exported, then encodes it. */
export async function exportActive(
  path: string,
  options: { format?: ExportFormat; quality?: number; scale?: number } = {}
): Promise<Outcome> {
  const { state, doc } = active();
  if (!doc) return noImage;

  const ext = extension(path);
  const format: ExportFormat =
    options.format ?? (ext === 'png' ? 'png' : ext === 'webp' ? 'webp' : 'jpeg');
  const canvas = await renderDocument(doc, RENDER_THEMES[state.theme]);
  const blob = await encodeCanvas(canvas, {
    format,
    quality: options.quality ?? 0.92,
    scale: options.scale ?? 1,
  });
  await writeFile(path, new Uint8Array(await blob.arrayBuffer()));
  return { messageKey: 'commands.saved', params: { path }, handled: true, data: { path, bytes: blob.size, format } };
}

register(
  {
    name: 'open_image',
    description:
      'Open a photo (JPEG, PNG, WebP, GIF, BMP, AVIF) or a .nabu project from an absolute path, as a new document that becomes the active one.',
    params: { path: { type: 'string', description: 'Absolute path of the file.', required: true } },
    run: (args) => openPath(String(args.path)),
  },
  {
    name: 'list_documents',
    description: 'List the open documents and say which one is active.',
    params: {},
    readOnly: true,
    run: () => {
      const { sessions, order, activeId } = useDocumentStore.getState();
      return done(
        order
          .map((id) => sessions[id]?.document)
          .filter(Boolean)
          .map((d) => ({ id: d.id, title: d.title, width: d.width, height: d.height, active: d.id === activeId }))
      );
    },
  },
  {
    name: 'select_document',
    description: 'Make an open document the active one; every other command works on the active document.',
    params: { id: { type: 'string', description: 'Document id from list_documents.', required: true } },
    run: (args) => {
      const id = String(args.id);
      const state = useDocumentStore.getState();
      if (!state.sessions[id]) return { messageKey: 'commands.noDocument', params: { id }, handled: false };
      state.setActiveDocument(id);
      return done();
    },
  },
  {
    name: 'close_document',
    description: 'Close a document without saving it: the active one, or the one named by id.',
    params: { id: { type: 'string', description: 'Document id; the active document when omitted.' } },
    run: (args) => {
      const state = useDocumentStore.getState();
      const id = typeof args.id === 'string' ? args.id : state.activeId;
      if (!state.sessions[id]) return { messageKey: 'commands.noDocument', params: { id }, handled: false };
      state.closeDocument(id);
      return done();
    },
  },
  {
    name: 'describe_document',
    description:
      'Describe the active document: size, layers, the develop settings of the active layer, the current selection and whether undo is possible.',
    params: {},
    readOnly: true,
    run: () => {
      const description = describe();
      return description ? done(description) : noImage;
    },
  },
  {
    name: 'get_preview',
    description:
      'Return a picture of the active document as it currently looks, edits included, to check a result or to locate things in it. Coordinates in the picture scale back to the document by width / preview width.',
    params: {
      max_size: {
        type: 'integer',
        description: 'Longest side of the preview in pixels (default 1024).',
        minimum: 64,
        maximum: 4096,
      },
    },
    readOnly: true,
    run: async (args) => {
      const { state, doc } = active();
      if (!doc) return noImage;
      const longest = Number(args.max_size ?? 1024);
      const canvas = await renderDocument(doc, RENDER_THEMES[state.theme]);
      const scale = Math.min(1, longest / Math.max(doc.width, doc.height));
      const blob = await encodeCanvas(canvas, { format: 'jpeg', quality: 0.85, scale });
      const dataUrl = await blobToDataUrl(blob);
      return {
        messageKey: 'commands.preview',
        params: { width: doc.width, height: doc.height },
        handled: true,
        data: { documentWidth: doc.width, documentHeight: doc.height, scale: round(scale) },
        image: { mimeType: 'image/jpeg', base64: dataUrl.slice(dataUrl.indexOf(',') + 1) },
      };
    },
  },
  {
    name: 'export_image',
    description:
      'Save the active document, with every edit applied, as an image file. The format follows the extension (.jpg, .png, .webp) unless given.',
    params: {
      path: { type: 'string', description: 'Absolute path to write; an existing file is replaced.', required: true },
      format: { type: 'string', description: 'Output format.', enum: ['jpeg', 'png', 'webp'] },
      quality: { type: 'number', description: 'Quality for JPEG and WebP, 0..1 (default 0.92).', minimum: 0.05, maximum: 1 },
      scale: { type: 'number', description: 'Output size relative to the document, 0..1 (default 1).', minimum: 0.01, maximum: 1 },
    },
    run: (args) =>
      exportActive(String(args.path), {
        format: args.format as ExportFormat | undefined,
        quality: args.quality as number | undefined,
        scale: args.scale as number | undefined,
      }),
  },
  {
    name: 'save_project',
    description: 'Save the active document as a .nabu project, keeping layers, adjustments and masks editable.',
    params: { path: { type: 'string', description: 'Absolute path ending in .nabu.', required: true } },
    run: async (args) => {
      const { doc } = active();
      if (!doc) return noImage;
      const path = String(args.path);
      await writeFile(path, new TextEncoder().encode(serializeProject(doc)));
      return { messageKey: 'commands.saved', params: { path }, handled: true };
    },
  },
  {
    name: 'undo',
    description: 'Undo the last edit on the active document.',
    params: {},
    run: () => {
      const { state, session } = active();
      if (!session) return noImage;
      if (session.historyIndex === 0) return { messageKey: 'router.noChange', handled: false };
      state.undo();
      return done();
    },
  },
  {
    name: 'redo',
    description: 'Redo the edit that was last undone.',
    params: {},
    run: () => {
      const { state, session } = active();
      if (!session) return noImage;
      if (session.historyIndex >= session.history.length - 1) return { messageKey: 'router.noChange', handled: false };
      state.redo();
      return done();
    },
  }
);

// ---------------------------------------------------------------------------
// Develop
// ---------------------------------------------------------------------------

const ADJUSTMENT_HELP = `One of: ${ADJUSTMENT_KEYS.join(', ')}.`;

register(
  {
    name: 'adjust',
    description:
      'Change one adjustment of the active layer relative to where it is now: "brighter" is exposure +20, "a bit warmer" is temperature +10. Most sliders run -100..100.',
    params: {
      property: { type: 'string', description: ADJUSTMENT_HELP, required: true, enum: ADJUSTMENT_KEYS },
      amount: { type: 'number', description: 'How much to add; negative to reduce.', required: true, minimum: -200, maximum: 200 },
    },
    run: (args) => {
      const { state, doc, layer } = active();
      if (!doc || !layer) return noImage;
      const property = String(args.property) as keyof Adjustments;
      const amount = Number(args.amount);
      if (amount === 0) return { messageKey: 'router.noChange', handled: false };
      const [min, max] = ADJUSTMENT_RANGES[property];
      const current = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments }[property];
      const next = Math.max(min, Math.min(max, current + amount));
      state.setAdjustment(layer.id, property, next);
      state.commitAdjustments(layer.id);
      return {
        messageKey: 'router.applied',
        params: { property: t(`adjust.${property}`), value: next > 0 ? `+${next}` : `${next}` },
        handled: true,
        data: { [property]: next },
      };
    },
  },
  {
    name: 'set_adjustments',
    description: `Set adjustments of the active layer to exact values, in one undoable step. Keys not given stay as they are. Ranges: ${ADJUSTMENT_KEYS.map((k) => `${k} ${ADJUSTMENT_RANGES[k][0]}..${ADJUSTMENT_RANGES[k][1]}`).join(', ')}.`,
    params: {
      values: { type: 'object', description: 'Map of adjustment name to value, e.g. {"exposure": 15, "contrast": 10}.', required: true },
    },
    run: (args) => {
      const { state, doc, layer } = active();
      if (!doc || !layer) return noImage;
      const values = args.values as Record<string, unknown>;
      const current = { ...NEUTRAL_ADJUSTMENTS, ...layer.adjustments };
      const delta: Partial<Adjustments> = {};
      for (const [key, raw] of Object.entries(values)) {
        if (!ADJUSTMENT_KEYS.includes(key as keyof Adjustments)) {
          return { messageKey: 'commands.unknownAdjustment', params: { name: key }, handled: false };
        }
        const value = Number(raw);
        if (!Number.isFinite(value)) return { messageKey: 'commands.unknownAdjustment', params: { name: key }, handled: false };
        const property = key as keyof Adjustments;
        const [min, max] = ADJUSTMENT_RANGES[property];
        delta[property] = Math.max(min, Math.min(max, value)) - current[property];
      }
      state.applyAdjustmentDelta(layer.id, delta);
      return done();
    },
  },
  {
    name: 'reset_adjustments',
    description: 'Put every adjustment of the active layer back to neutral.',
    params: {},
    run: () => {
      const { state, layer } = active();
      if (!layer) return noImage;
      state.resetAdjustments(layer.id);
      return { messageKey: 'router.reset', handled: true };
    },
  },
  {
    name: 'list_looks',
    description: 'List the looks (colour presets) that apply_look accepts: the built-in ones and those the user saved.',
    params: {},
    readOnly: true,
    run: () =>
      done([
        ...FILTERS.map((f) => ({ id: f.id, name: t(`filter.${f.id}`), group: f.group })),
        ...useLookStore.getState().looks.map((l) => ({ id: l.id, name: l.name, group: 'saved' })),
      ]),
  },
  {
    name: 'apply_look',
    description: 'Apply a look to the active layer, by id or by the name of a look the user saved. Use "none" to remove it.',
    params: {
      look: { type: 'string', description: 'Look id or saved name, from list_looks.', required: true },
      amount: { type: 'number', description: 'Strength 0..1 (default 1).', minimum: 0, maximum: 1 },
    },
    run: (args) => {
      const { state, layer } = active();
      if (!layer) return noImage;
      const wanted = String(args.look);
      if (wanted === 'none') {
        state.setLayerFilter(layer.id, null);
        return done();
      }
      const saved = useLookStore
        .getState()
        .looks.find((l) => l.id === wanted || l.name.toLowerCase() === wanted.toLowerCase());
      const builtIn = FILTERS.find((f) => f.id === wanted);
      if (!saved && !builtIn) return { messageKey: 'commands.unknownLook', params: { name: wanted }, handled: false };
      state.setLayerFilter(layer.id, saved?.id ?? builtIn!.id, saved?.adjustments);
      if (typeof args.amount === 'number') {
        state.setFilterAmount(layer.id, args.amount);
        state.commitFilter(layer.id);
      }
      return done();
    },
  },
  {
    name: 'apply_lut',
    description: 'Apply a colour lookup table from a .cube file to the active layer.',
    params: {
      path: { type: 'string', description: 'Absolute path of the .cube file.', required: true },
      amount: { type: 'number', description: 'Strength 0..1 (default 1).', minimum: 0, maximum: 1 },
    },
    run: async (args) => {
      const { state, layer } = active();
      if (!layer) return noImage;
      const path = String(args.path);
      const table = parseCube(new TextDecoder().decode(await readFile(path)));
      state.setLayerLut(layer.id, {
        name: table.title || stem(path),
        size: table.size,
        data: encodeTable(table.rgb),
        amount: typeof args.amount === 'number' ? args.amount : 1,
      });
      return done();
    },
  },
  {
    name: 'apply_develop',
    description:
      'Replace the develop settings of the active layer (adjustments, curves, look, LUT) with the ones given, in the shape describe_document returns under "develop". This is how one photo\'s edit is copied onto others.',
    params: { settings: { type: 'object', description: 'Develop settings object.', required: true } },
    run: (args) => {
      const { state, layer } = active();
      if (!layer) return noImage;
      state.pasteDevelop(layer.id, args.settings as DevelopSettings);
      return done();
    },
  }
);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

register(
  {
    name: 'resize',
    description:
      'Resize the document and every layer. Give a percentage, the longest side, or a width and/or height in pixels; with one side only, the proportions are kept.',
    params: {
      percent: { type: 'number', description: 'Scale in percent: 50 halves, 200 doubles.', minimum: 1, maximum: 1000 },
      long_side: { type: 'integer', description: 'New length of the longest side, in pixels.', minimum: 16, maximum: 20000 },
      width: { type: 'integer', description: 'New width in pixels.', minimum: 16, maximum: 20000 },
      height: { type: 'integer', description: 'New height in pixels.', minimum: 16, maximum: 20000 },
    },
    run: (args) => {
      const { doc } = active();
      if (!doc) return noImage;
      const { percent, long_side, width, height } = args as Record<string, number | undefined>;
      if (percent) return resizeByPercent(percent);
      if (long_side) return resizeLongestSide(long_side);
      if (width && height) return resizeDocument(width, height);
      if (width) return resizeDocument(width, Math.round((doc.height * width) / doc.width));
      if (height) return resizeDocument(Math.round((doc.width * height) / doc.height), height);
      return { messageKey: 'resize.unchanged', handled: false };
    },
  },
  {
    name: 'crop',
    description: 'Crop the document to a rectangle given in document pixels.',
    params: {
      x: { type: 'number', description: 'Left edge.', required: true, minimum: 0 },
      y: { type: 'number', description: 'Top edge.', required: true, minimum: 0 },
      width: { type: 'number', description: 'Width.', required: true, minimum: 16 },
      height: { type: 'number', description: 'Height.', required: true, minimum: 16 },
    },
    run: async (args) => {
      const ok = await cropDocument({
        x: Number(args.x),
        y: Number(args.y),
        width: Number(args.width),
        height: Number(args.height),
      });
      return ok ? done() : { messageKey: 'router.noChange', handled: false };
    },
  }
);

// ---------------------------------------------------------------------------
// Seeing and selecting
// ---------------------------------------------------------------------------

function objectById(id: number) {
  return useSceneStore.getState().activeScene()?.objects.find((o) => o.id === id) ?? null;
}

function selectBox(x0: number, y0: number, x1: number, y1: number, shape: 'rectangle' | 'ellipse' = 'rectangle') {
  const { state, layer } = active();
  if (!layer) return false;
  const ox = layer.transform.x;
  const oy = layer.transform.y;
  let path: Point2D[];
  if (shape === 'ellipse') {
    const cx = (x0 + x1) / 2 + ox;
    const cy = (y0 + y1) / 2 + oy;
    const rx = (x1 - x0) / 2;
    const ry = (y1 - y0) / 2;
    path = Array.from({ length: 64 }, (_, i) => {
      const a = (i / 64) * Math.PI * 2;
      return { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a) };
    });
  } else {
    path = [
      { x: x0 + ox, y: y0 + oy },
      { x: x1 + ox, y: y0 + oy },
      { x: x1 + ox, y: y1 + oy },
      { x: x0 + ox, y: y1 + oy },
    ];
  }
  state.setSelection({ path });
  return true;
}

register(
  {
    name: 'analyze_scene',
    description:
      'Detect the objects in the active photo. Returns each one with an id, class, colour, position and pixel box (x0, y0, x1, y1); the ids are what select_object and remove_object take. Needs an object-detection model installed.',
    params: {},
    readOnly: true,
    run: async () => {
      const { layer } = active();
      if (!layer?.imageDataUrl) return noImage;
      const scene = await useSceneStore.getState().analyzeActive();
      if (!scene) return { messageKey: 'commands.noScene', handled: false };
      return {
        messageKey: 'commands.sceneRead',
        params: { count: scene.objects.length },
        handled: true,
        data: scene,
      };
    },
  },
  {
    name: 'select_object',
    description: 'Select one object found by analyze_scene, by id.',
    params: { id: { type: 'integer', description: 'Object id from analyze_scene.', required: true, minimum: 1 } },
    run: (args) => {
      const id = Number(args.id);
      const target = objectById(id);
      if (!target) return { messageKey: 'commands.noObject', params: { id }, handled: false };
      const [x0, y0, x1, y1] = target.bbox;
      if (!selectBox(x0, y0, x1, y1)) return noImage;
      return { messageKey: 'router.selected', params: { id }, handled: true };
    },
  },
  {
    name: 'select_region',
    description:
      'Select a rectangle or ellipse, in pixels of the active layer. Use it for anything analyze_scene did not name: look at get_preview, then select around the thing.',
    params: {
      x: { type: 'number', description: 'Left edge.', required: true },
      y: { type: 'number', description: 'Top edge.', required: true },
      width: { type: 'number', description: 'Width.', required: true, minimum: 1 },
      height: { type: 'number', description: 'Height.', required: true, minimum: 1 },
      shape: { type: 'string', description: 'Outline shape (default rectangle).', enum: ['rectangle', 'ellipse'] },
    },
    run: (args) => {
      const x = Number(args.x);
      const y = Number(args.y);
      const ok = selectBox(x, y, x + Number(args.width), y + Number(args.height), args.shape === 'ellipse' ? 'ellipse' : 'rectangle');
      return ok ? done() : noImage;
    },
  },
  {
    name: 'clear_selection',
    description: 'Remove the current selection.',
    params: {},
    run: () => {
      useDocumentStore.getState().setSelection(null);
      return done();
    },
  }
);

// ---------------------------------------------------------------------------
// AI edits
// ---------------------------------------------------------------------------

register(
  {
    name: 'remove_object',
    description:
      'Erase one object found by analyze_scene and rebuild the background behind it. Pick the id whose class, colour and position match what should go. Needs the segmentation and inpainting models.',
    params: { id: { type: 'integer', description: 'Object id from analyze_scene.', required: true, minimum: 1 } },
    run: async (args) => {
      const id = Number(args.id);
      const target = objectById(id);
      if (!target) return { messageKey: 'commands.noObject', params: { id }, handled: false };
      await removeObjectByBbox(target.bbox);
      return { messageKey: 'router.removed', params: { id }, handled: true };
    },
  },
  {
    name: 'remove_selection',
    description: 'Erase whatever is inside the current selection and rebuild the background behind it.',
    params: {},
    run: async () => {
      await removeSelectedRegion();
      return { messageKey: 'router.removedSelection', handled: true };
    },
  },
  {
    name: 'content_aware_fill',
    description:
      'Rebuild an area from the rest of the photo, without any model: the current selection, or the rectangle given (document pixels). Best for texture such as walls, grass, sky or skin.',
    params: {
      x: { type: 'number', description: 'Left edge of the rectangle; omit to fill the selection.' },
      y: { type: 'number', description: 'Top edge of the rectangle.' },
      width: { type: 'number', description: 'Rectangle width.', minimum: 1 },
      height: { type: 'number', description: 'Rectangle height.', minimum: 1 },
    },
    run: async (args) => {
      if (args.x === undefined) {
        const filled = await contentAwareFill();
        return { messageKey: filled ? 'commands.healed' : 'router.done', handled: filled };
      }
      const [x, y] = [Number(args.x), Number(args.y ?? 0)];
      const [w, h] = [Number(args.width ?? 1), Number(args.height ?? 1)];
      const rect = [{ x, y }, { x: x + w, y }, { x: x + w, y: y + h }, { x, y: y + h }];
      const filled = await healActive(() => pathShape(rect), 'activity.contentAwareFill');
      return { messageKey: filled ? 'commands.healed' : 'router.done', handled: filled };
    },
  },
  {
    name: 'heal_spots',
    description:
      'Make small things disappear (dust, blemishes, a bird in the sky) by healing round spots, each rebuilt from what surrounds it. Coordinates in document pixels.',
    params: {
      spots: {
        type: 'array',
        items: { type: 'object' },
        description: 'Spots as {"x": number, "y": number, "radius": number}.',
        required: true,
      },
    },
    run: async (args) => {
      const spots = (args.spots as Array<{ x: number; y: number; radius?: number }>).filter(
        (s) => s && Number.isFinite(Number(s.x)) && Number.isFinite(Number(s.y))
      );
      if (spots.length === 0) return { messageKey: 'backend.retouchEmpty', handled: false };
      // One stroke per spot, drawn as one mask: the fill takes each spot from
      // its own neighbourhood anyway.
      const shapes = spots.map((s) =>
        strokeShape([{ x: Number(s.x), y: Number(s.y) }], 2 * Math.max(1, Number(s.radius ?? 8)))
      );
      const bounds = {
        x0: Math.min(...shapes.map((s) => s.bounds.x0)),
        y0: Math.min(...shapes.map((s) => s.bounds.y0)),
        x1: Math.max(...shapes.map((s) => s.bounds.x1)),
        y1: Math.max(...shapes.map((s) => s.bounds.y1)),
      };
      const draw = (ctx: CanvasRenderingContext2D) =>
        shapes.forEach((shape) => {
          ctx.save();
          shape.draw(ctx);
          ctx.restore();
        });
      const filled = await healActive(() => ({ bounds, draw }), 'activity.heal');
      return { messageKey: filled ? 'commands.healed' : 'router.done', handled: filled };
    },
  },
  {
    name: 'cut_out_selection',
    description: 'Cut out the selected object onto its own transparent layer.',
    params: {},
    run: async () => {
      await extractObject();
      return done();
    },
  },
  {
    name: 'remove_background',
    description: 'Remove the background of the photo, leaving the main subject on transparency.',
    params: {},
    run: async () => {
      await removeBackground();
      return done();
    },
  },
  {
    name: 'upscale',
    description: 'Increase the resolution of the active layer with the super-resolution model.',
    params: {},
    run: async () => {
      await upscale();
      return done();
    },
  },
  {
    name: 'stitch_panorama',
    description:
      'Stitch overlapping photos taken from one spot into a panorama and open it as a new document. Photos that do not overlap the rest are left out and listed.',
    params: {
      paths: { type: 'array', items: { type: 'string' }, description: 'Absolute paths of two or more photos.', required: true },
      projection: {
        type: 'string',
        description: 'Projection; auto picks flat, cylindrical or spherical from the field of view.',
        enum: ['auto', 'rectilinear', 'cylindrical', 'spherical'],
      },
      crop: { type: 'boolean', description: 'Crop to the largest clean rectangle (default true).' },
      max_megapixels: { type: 'number', description: 'Upper bound on the output size (default 60).', minimum: 1, maximum: 400 },
    },
    run: async (args) => {
      const paths = (args.paths as unknown[]).map(String);
      if (paths.length < 2) return { messageKey: 'backend.panoramaTooFew', handled: false };
      const files = await Promise.all(
        paths.map(async (path) => new File([(await readFile(path)) as BlobPart], baseName(path)))
      );
      const { info, dataUrl } = await stitchPanorama(
        files,
        {
          projection: (args.projection as PanoramaProjection | undefined) ?? 'auto',
          crop: args.crop !== false,
          maxMegapixels: Number(args.max_megapixels ?? 60),
        },
        () => undefined
      );
      useDocumentStore.getState().openImageAsDocument(t('panorama.documentName'), dataUrl, info.width, info.height);
      return {
        messageKey: 'commands.panorama',
        params: { used: info.used.length, width: info.width, height: info.height },
        handled: true,
        data: { ...info, used: info.used.map((i) => paths[i]), dropped: info.dropped.map((i) => paths[i]) },
      };
    },
  }
);

export const EXPORT_EXTENSIONS = Object.fromEntries(
  Object.entries(EXPORT_FORMATS).map(([format, spec]) => [format, spec.extension])
) as Record<ExportFormat, string>;
