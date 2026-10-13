// src/components/TransformBar.tsx
import { useState } from 'react';
import { Blend, Check, FlipHorizontal, FlipVertical, RotateCcw, RotateCw } from 'lucide-react';
import { useActiveDocument, useDocumentFields } from '../store/documentStore';
import { canBlend } from '../ops/blend';
import { BlendPopover } from './BlendPopover';
import { useT } from '../i18n';

interface Props {
  layerId: string;
  onDone: () => void;
}

/** Quarter turns, the rotation people actually ask for by name. */
const QUARTER = Math.PI / 2;

/**
 * Readout and shortcuts for the layer being placed.
 *
 * It exists only while the transform tool is on, and shows the two numbers a
 * transform is judged by — size and angle — because a drag that reports nothing
 * has to be redone until it looks right by eye.
 */
export const TransformBar: React.FC<Props> = ({ layerId, onDone }) => {
  const doc = useActiveDocument();
  const { setLayerTransform, commitTransform, resetTransform } = useDocumentFields('setLayerTransform', 'commitTransform', 'resetTransform');
  const [blending, setBlending] = useState(false);
  const t = useT();

  const layer = doc?.layers[layerId];
  if (!doc || !layer) return null;

  // Only a layer with a photo underneath has anything to be matched against.
  const blendable = canBlend();

  const { transform } = layer;
  const width = Math.round((layer.width ?? doc.width) * (transform.scaleX || 1));
  const height = Math.round((layer.height ?? doc.height) * (transform.scaleY || 1));
  const degrees = Math.round(((transform.rotation || 0) * 180) / Math.PI);
  const untouched =
    transform.x === 0 &&
    transform.y === 0 &&
    (transform.scaleX || 1) === 1 &&
    (transform.scaleY || 1) === 1 &&
    !transform.rotation;

  const turn = (direction: number) => {
    setLayerTransform(layerId, { rotation: (transform.rotation || 0) + direction * QUARTER });
    commitTransform(layerId);
  };

  /** Mirroring is a negative scale: the renderer already handles it. */
  const flip = (axis: 'x' | 'y') => {
    setLayerTransform(
      layerId,
      axis === 'x'
        ? { scaleX: -(transform.scaleX || 1) }
        : { scaleY: -(transform.scaleY || 1) }
    );
    commitTransform(layerId);
  };

  return (
    <div
      className="transform-bar"
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
    >
      <span className="transform-readout">
        {Math.abs(width)} x {Math.abs(height)}
      </span>
      <span className="transform-readout dim">{degrees}&deg;</span>

      <span className="transform-sep" />

      <button onClick={() => turn(-1)} title={t('transform.turnLeft')}>
        <RotateCcw size={14} />
      </button>
      <button onClick={() => turn(1)} title={t('transform.turnRight')}>
        <RotateCw size={14} />
      </button>
      <button onClick={() => flip('x')} title={t('transform.flipH')}>
        <FlipHorizontal size={14} />
      </button>
      <button onClick={() => flip('y')} title={t('transform.flipV')}>
        <FlipVertical size={14} />
      </button>

      <span className="transform-sep" />

      {blendable && (
        <button
          className={`transform-blend ${blending ? 'on' : ''}`}
          onClick={() => setBlending((open) => !open)}
          title={t('blend.hint')}
        >
          <Blend size={14} />
          <span>{t('blend.action')}</span>
        </button>
      )}

      {blending && <BlendPopover onClose={() => setBlending(false)} />}

      <button onClick={() => resetTransform(layerId)} disabled={untouched}>
        {t('transform.reset')}
      </button>
      <button className="transform-done" onClick={onDone} title={t('transform.done.hint')}>
        <Check size={14} />
        <span>{t('transform.done')}</span>
      </button>
    </div>
  );
};
