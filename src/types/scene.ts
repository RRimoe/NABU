// src/types/scene.ts
// Mirrors the scene model serialised by src-tauri/src/ai/vision/scene.rs.
//
// The scene is the shared representation the vision engine builds of an image:
// what is in it, where, and how the image itself measures up. The language
// router receives a compact rendering of it, which is how a request like
// "the red car" resolves without the model ever seeing a pixel.

export interface SceneObject {
  /** Stable id inside one scene, what `select_object` refers to. */
  id: number;
  class: string;
  confidence: number;
  /** Pixel coordinates on the analysed layer: x0, y0, x1, y1. */
  bbox: [number, number, number, number];
  /** Fraction of the image the box covers, 0..1. */
  areaFraction: number;
  position: 'left' | 'center' | 'right';
  prominence: 'dominant' | 'secondary' | 'minor';
  /** Everyday colour name, present when one colour clearly dominates. */
  color?: string;
}

export interface SceneStats {
  width: number;
  height: number;
  meanLuma: number;
  clippedShadows: number;
  clippedHighlights: number;
  sharpness: number;
}

export interface SceneModel {
  modelId: string;
  objects: SceneObject[];
  stats: SceneStats;
}

/**
 * Renders the scene as the compact English block handed to the language
 * router. English on purpose: the scene is written for the model, which
 * handles the user's language itself.
 */
export function sceneSummary(scene: SceneModel): string {
  const lines = scene.objects.map((o) => {
    const traits: string[] = [o.position, o.prominence];
    if (o.color) traits.push(`color: ${o.color}`);
    return `#${o.id} ${o.class} — ${traits.join(', ')}`;
  });

  const { stats } = scene;
  const exposure =
    stats.meanLuma < 0.3 ? 'underexposed' : stats.meanLuma > 0.7 ? 'overexposed' : 'well exposed';
  const focus = stats.sharpness < 0.25 ? 'soft focus' : 'sharp';
  lines.push(`image: ${stats.width}x${stats.height}, ${exposure}, ${focus}`);

  return lines.join('\n');
}
