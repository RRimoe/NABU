// src/components/LeftRail.tsx
import {
  Brush,
  Crop,
  Eraser,
  Hand,
  Move,
  Palette,
  SlidersHorizontal,
  SquareDashed,
} from 'lucide-react';
import { type DockPanel, useDocumentFields } from '../store/documentStore';
import type { ToolType } from '../types/document';
import { useT } from '../i18n';

interface Props {
  hasPhoto: boolean;
}

/**
 * The tools, and only the tools.
 *
 * Opening and exporting a photo are not editing, so they moved into the menu at
 * the top right; what is left is the five things you can be *doing* to the
 * picture in front of you. Each one is a mode, and exactly one is on at a time,
 * which is why they are a rail and not a scattering of buttons.
 *
 * The letters "AI" never appear: people want to edit a photo, not to be told
 * which technology edits it.
 */
export const LeftRail: React.FC<Props> = ({ hasPhoto }) => {
  const { activeTool, setActiveTool, dockPanel, setDockPanel } = useDocumentFields('activeTool', 'setActiveTool', 'dockPanel', 'setDockPanel');
  const t = useT();

  /** A rail entry that switches the canvas into a mode. */
  const tool = (id: ToolType, Icon: typeof Hand, key: string) => (
    <button
      className={`rail-item ${activeTool === id ? 'active' : ''}`}
      onClick={() => setActiveTool(activeTool === id && id !== 'select' ? 'select' : id)}
      title={t(`${key}.hint`)}
      disabled={!hasPhoto}
    >
      <Icon size={20} />
      <span>{t(key)}</span>
    </button>
  );

  /**
   * A rail entry that opens a section of the dock. Pressing it again goes back
   * to the suggestions, so the dock always has something in it.
   */
  const panel = (id: DockPanel, Icon: typeof Hand, key: string) => (
    <button
      className={`rail-item ${dockPanel === id ? 'active' : ''}`}
      onClick={() => setDockPanel(dockPanel === id ? 'suggestions' : id)}
      title={t(`${key}.hint`)}
      disabled={!hasPhoto}
    >
      <Icon size={20} />
      <span>{t(key)}</span>
    </button>
  );

  return (
    <nav className="nabu-rail">
      <div className="rail-items">
        {tool('select', SquareDashed, 'rail.select')}
        {tool('pan', Hand, 'rail.move')}
        {/* Its own entry now that the handles no longer appear on their own:
            a mode the canvas is in needs somewhere to be turned on. */}
        {tool('transform', Move, 'rail.transform')}
        {tool('crop', Crop, 'rail.crop')}
        {tool('brush', Brush, 'rail.brush')}
        {tool('eraser', Eraser, 'rail.eraser')}
        {panel('adjust', SlidersHorizontal, 'rail.adjust')}
        {panel('filters', Palette, 'rail.filters')}
      </div>
    </nav>
  );
};
