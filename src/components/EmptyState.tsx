// src/components/EmptyState.tsx
import { ImagePlus, Images } from 'lucide-react';
import { Logo } from './Logo';
import { useT } from '../i18n';

interface Props {
  onOpen: () => void;
  onPanorama: () => void;
  dragging: boolean;
}

/**
 * Start screen: no document, no blank canvas.
 *
 * Until there is an image there is nothing to edit, so we do not show an empty
 * sheet — only the invitation to open one.
 */
export const EmptyState: React.FC<Props> = ({ onOpen, onPanorama, dragging }) => {
  const t = useT();

  return (
    <div className={`nabu-empty ${dragging ? 'dragging' : ''}`} onClick={onOpen}>
      <div className="empty-inner">
        {/* The only screen with room for the mark in full colour. Everywhere
            else it competes with the photograph, which always wins. */}
        <Logo size={96} tone="colour" className="empty-logo" />
        <p className="empty-title">{t('empty.title')}</p>
        <p className="empty-hint">
          <ImagePlus size={14} strokeWidth={1.6} />
          <span>{t('empty.hint')}</span>
        </p>
        <button
          className="empty-panorama"
          onClick={(e) => {
            e.stopPropagation();
            onPanorama();
          }}
        >
          <Images size={14} strokeWidth={1.6} />
          <span>{t('panorama.fromStart')}</span>
        </button>
      </div>
    </div>
  );
};
