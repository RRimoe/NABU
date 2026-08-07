// src/components/ResizePopover.tsx
import { useEffect, useState } from 'react';
import { Link2, Unlink2, Check, X } from 'lucide-react';
import { useActiveDocument } from '../store/documentStore';
import { resizeDocument, clampSide, MIN_SIDE, MAX_SIDE } from '../ops/resize';
import { useT } from '../i18n';

interface Props {
  onClose: () => void;
}

/** Quick factors, the ones people actually reach for. */
const PRESETS = [25, 50, 200];

/**
 * Resamples the document to different pixel dimensions.
 *
 * Opened from the size shown in the status bar, so the control lives where the
 * number it changes already is, and nothing permanent is added to the chrome.
 */
export const ResizePopover: React.FC<Props> = ({ onClose }) => {
  const doc = useActiveDocument();
  const t = useT();

  const [width, setWidth] = useState('');
  const [height, setHeight] = useState('');
  const [locked, setLocked] = useState(true);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!doc) return;
    setWidth(String(doc.width));
    setHeight(String(doc.height));
  }, [doc]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!doc) return null;

  const ratio = doc.width / doc.height;

  const changeWidth = (value: string) => {
    setWidth(value);
    const parsed = Number(value);
    if (locked && Number.isFinite(parsed) && parsed > 0) {
      setHeight(String(Math.round(parsed / ratio)));
    }
  };

  const changeHeight = (value: string) => {
    setHeight(value);
    const parsed = Number(value);
    if (locked && Number.isFinite(parsed) && parsed > 0) {
      setWidth(String(Math.round(parsed * ratio)));
    }
  };

  const applyPreset = (percent: number) => {
    setWidth(String(Math.round((doc.width * percent) / 100)));
    setHeight(String(Math.round((doc.height * percent) / 100)));
  };

  const apply = async () => {
    setBusy(true);
    await resizeDocument(Number(width), Number(height));
    setBusy(false);
    onClose();
  };

  const targetWidth = clampSide(Number(width) || doc.width);
  const targetHeight = clampSide(Number(height) || doc.height);
  const unchanged = targetWidth === doc.width && targetHeight === doc.height;
  // Enlarging by resampling cannot invent detail; saying so here is the only
  // thing standing between this and someone using it as "Enhance".
  const enlarging = targetWidth > doc.width;

  return (
    <div className="resize-popover" onMouseDown={(e) => e.stopPropagation()}>
      <div className="resize-head">
        <span>{t('resize.title')}</span>
        <button className="bubble-dismiss" onClick={onClose} title={t('bubble.dismiss')}>
          <X size={13} />
        </button>
      </div>

      <div className="resize-fields">
        <label>
          <span>{t('resize.width')}</span>
          <input
            type="number"
            min={MIN_SIDE}
            max={MAX_SIDE}
            value={width}
            onChange={(e) => changeWidth(e.target.value)}
          />
        </label>

        <button
          className={`resize-lock ${locked ? 'on' : ''}`}
          onClick={() => setLocked((v) => !v)}
          title={locked ? t('resize.unlock') : t('resize.lock')}
        >
          {locked ? <Link2 size={14} /> : <Unlink2 size={14} />}
        </button>

        <label>
          <span>{t('resize.height')}</span>
          <input
            type="number"
            min={MIN_SIDE}
            max={MAX_SIDE}
            value={height}
            onChange={(e) => changeHeight(e.target.value)}
          />
        </label>
      </div>

      <div className="resize-presets">
        {PRESETS.map((percent) => (
          <button key={percent} onClick={() => applyPreset(percent)}>
            {percent}%
          </button>
        ))}
      </div>

      {enlarging && <p className="resize-note">{t('resize.enlargeNote')}</p>}

      <button className="resize-apply" onClick={() => void apply()} disabled={busy || unchanged}>
        <Check size={14} />
        <span>
          {unchanged
            ? t('resize.unchanged')
            : t('resize.apply', { width: targetWidth, height: targetHeight })}
        </span>
      </button>
    </div>
  );
};
