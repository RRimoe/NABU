// src/components/LayerManager.tsx
import {
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Plus,
  Trash2,
  MoveUp,
  MoveDown,
  Sparkles,
  Image as ImageIcon,
} from 'lucide-react';
import { useDocumentStore, useActiveDocument } from '../store/documentStore';
import { BlendMode } from '../types/document';
import { useT } from '../i18n';

const BLEND_MODES: BlendMode[] = [
  'normal',
  'multiply',
  'screen',
  'overlay',
  'darken',
  'lighten',
  'color-dodge',
  'color-burn',
  'hard-light',
  'soft-light',
  'difference',
  'exclusion',
];

export const LayerManager: React.FC = () => {
  const document = useActiveDocument();
  const {
    setActiveLayer,
    toggleLayerVisibility,
    toggleLayerLock,
    setLayerBlendMode,
    setLayerOpacity,
    addLayer,
    removeLayer,
    reorderLayers,
  } = useDocumentStore();
  const t = useT();

  if (!document) return null;
  const activeLayer = document.activeLayerId ? document.layers[document.activeLayerId] : null;

  return (
    <div className="nabu-panel layer-panel">
      <div className="panel-header">
        <div className="panel-actions">
          <button className="icon-btn" onClick={() => addLayer('raster')} title={t('layers.new')}>
            <Plus size={16} />
          </button>
          <button
            className="icon-btn danger"
            onClick={() => document.activeLayerId && removeLayer(document.activeLayerId)}
            disabled={document.rootLayerIds.length <= 1}
            title={t('layers.delete')}
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {activeLayer && (
        <div className="layer-controls">
          <div className="control-row">
            <label className="control-label">{t('layers.opacity')}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={activeLayer.opacity}
              onChange={(e) => setLayerOpacity(activeLayer.id, parseFloat(e.target.value))}
              className="opacity-slider"
            />
            <span className="control-value">{Math.round(activeLayer.opacity * 100)}%</span>
          </div>

          <div className="control-row">
            <label className="control-label">{t('layers.blend')}</label>
            <select
              value={activeLayer.blendMode}
              onChange={(e) => setLayerBlendMode(activeLayer.id, e.target.value as BlendMode)}
              className="blend-select"
            >
              {BLEND_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {t(`blend.${mode}`)}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      <div className="layer-stack">
        {document.rootLayerIds.map((layerId, index) => {
          const layer = document.layers[layerId];
          if (!layer) return null;

          const isActive = document.activeLayerId === layerId;

          return (
            <div
              key={layerId}
              className={`layer-item ${isActive ? 'active' : ''} ${!layer.visible ? 'hidden-layer' : ''}`}
              onClick={() => setActiveLayer(layerId)}
            >
              <button
                className="layer-icon-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleLayerVisibility(layerId);
                }}
                title={layer.visible ? t('layers.hide') : t('layers.show')}
              >
                {layer.visible ? <Eye size={15} /> : <EyeOff size={15} />}
              </button>

              <button
                className="layer-icon-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleLayerLock(layerId);
                }}
                title={layer.locked ? t('layers.unlock') : t('layers.lock')}
              >
                {layer.locked ? <Lock size={15} /> : <Unlock size={15} />}
              </button>

              <div className="layer-thumbnail">
                {layer.imageDataUrl ? (
                  <img src={layer.imageDataUrl} alt="" />
                ) : (
                  <ImageIcon size={14} />
                )}
              </div>

              <div className="layer-info">
                <span className="layer-name">{layer.name}</span>
                {layer.aiMetadata && (
                  <span className="layer-ai-badge">
                    <Sparkles size={10} />
                  </span>
                )}
              </div>

              <div className="layer-reorder-btns">
                <button
                  className="reorder-btn"
                  disabled={index === 0}
                  onClick={(e) => {
                    e.stopPropagation();
                    reorderLayers(index, index - 1);
                  }}
                  title={t('layers.moveUp')}
                >
                  <MoveUp size={12} />
                </button>
                <button
                  className="reorder-btn"
                  disabled={index === document.rootLayerIds.length - 1}
                  onClick={(e) => {
                    e.stopPropagation();
                    reorderLayers(index, index + 1);
                  }}
                  title={t('layers.moveDown')}
                >
                  <MoveDown size={12} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
