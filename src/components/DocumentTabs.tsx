// src/components/DocumentTabs.tsx
import { X } from 'lucide-react';
import { useDocumentStore } from '../store/documentStore';
import { useT } from '../i18n';

/**
 * List of open documents.
 *
 * It only appears once there is more than one: with a single file it would be
 * a row of interface that says nothing.
 */
export const DocumentTabs: React.FC = () => {
  const { sessions, order, activeId, setActiveDocument, closeDocument } = useDocumentStore();
  const t = useT();

  if (order.length < 2) return null;

  return (
    <div className="nabu-tabs">
      {order.map((id) => {
        const session = sessions[id];
        if (!session) return null;

        return (
          <div
            key={id}
            className={`nabu-tab ${id === activeId ? 'active' : ''}`}
            onClick={() => setActiveDocument(id)}
            title={`${session.document.title} - ${session.document.width}x${session.document.height}`}
          >
            <span className="nabu-tab-name">{session.document.title}</span>
            <button
              className="nabu-tab-close"
              onClick={(e) => {
                e.stopPropagation();
                closeDocument(id);
              }}
              title={t('tabs.close')}
            >
              <X size={12} />
            </button>
          </div>
        );
      })}
    </div>
  );
};
