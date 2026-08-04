// src/components/EmptyState.tsx
import { ImagePlus } from 'lucide-react';
import { useT } from '../i18n';

interface Props {
  onOpen: () => void;
  dragging: boolean;
}

/**
 * Start screen: no document, no blank canvas.
 *
 * Until there is an image there is nothing to edit, so we do not show an empty
 * sheet — only the invitation to open one.
 */
export const EmptyState: React.FC<Props> = ({ onOpen, dragging }) => {
  const t = useT();

  return (
    <div className={`nabu-empty ${dragging ? 'dragging' : ''}`} onClick={onOpen}>
      <div className="empty-inner">
        <ImagePlus size={30} strokeWidth={1.3} />
        <p className="empty-title">{t('empty.title')}</p>
        <p className="empty-hint">{t('empty.hint')}</p>
      </div>
    </div>
  );
};
