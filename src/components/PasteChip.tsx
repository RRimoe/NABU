// src/components/PasteChip.tsx
import { useEffect, useState } from 'react';
import { ClipboardPaste, X } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { pasteClipboard } from '../ops/clipboard';
import { useT } from '../i18n';

/**
 * The offer to paste, where the paste would land.
 *
 * Copying and then arriving in another photo with nothing to show for it is a
 * dead end: the only way through was a keystroke, and a keystroke you have to
 * already know about is not an interface. So the thing you copied appears,
 * with its own thumbnail, in the photo it can go into — and it says so once,
 * then gets out of the way.
 *
 * Deliberately not shown in the photo it was copied from. There it would be an
 * offer to duplicate something in place, which is a different thing nobody
 * asked for.
 */
export const PasteChip: React.FC = () => {
  const clipboard = useDocumentStore((s) => s.clipboard);
  const activeId = useDocumentStore((s) => s.activeId);
  const isAIBusy = useDocumentStore((s) => s.isAIBusy);
  const t = useT();

  const [dismissed, setDismissed] = useState<string | null>(null);

  // A new copy is a new offer, wherever the last one was waved away.
  useEffect(() => {
    setDismissed(null);
  }, [clipboard?.imageDataUrl]);

  if (!clipboard || clipboard.sourceId === activeId) return null;
  if (dismissed === activeId) return null;

  return (
    <div className="paste-chip">
      <img src={clipboard.imageDataUrl} alt="" draggable={false} />

      <div className="paste-chip-text">
        <span className="paste-chip-title">{t('paste.ready')}</span>
        <span className="paste-chip-name">{clipboard.name}</span>
      </div>

      <button
        className="paste-chip-go"
        disabled={isAIBusy}
        onClick={() => {
          void pasteClipboard();
          setDismissed(activeId);
        }}
      >
        <ClipboardPaste size={14} />
        <span>{t('paste.action')}</span>
      </button>

      <button
        className="paste-chip-dismiss"
        onClick={() => setDismissed(activeId)}
        title={t('bubble.dismiss')}
      >
        <X size={13} />
      </button>
    </div>
  );
};
