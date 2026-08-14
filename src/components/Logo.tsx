// src/components/Logo.tsx
import { useDocumentStore } from '../store/documentStore';
import colour from '../assets/logo-colour.png';
import black from '../assets/logo-black.png';
import white from '../assets/logo-white.png';

/**
 * Where the mark is going, which decides which of the three files is right.
 *
 * `surface` sits on the application's own background, so it takes the colour
 * that contrasts with the theme — white in the dark one, black in the light
 * one. `accent` sits on a filled accent shape, where the accent is *light* in
 * the dark theme and *dark* in the light one, so the mark has to invert the
 * other way. Getting these two the same way round is the classic way a logo
 * disappears into its own button.
 */
export type LogoTone = 'surface' | 'accent' | 'colour';

interface Props {
  size: number;
  tone?: LogoTone;
  className?: string;
}

/**
 * The dragon that writes the N.
 *
 * Three files rather than one recoloured by CSS: the coloured mark is a
 * gradient no filter would reproduce, and a `filter: invert()` on the mono one
 * would give a muddy grey rather than the flat silhouette it was drawn as.
 */
export const Logo: React.FC<Props> = ({ size, tone = 'surface', className }) => {
  const theme = useDocumentStore((s) => s.theme);

  const src =
    tone === 'colour'
      ? colour
      : tone === 'accent'
        ? theme === 'dark'
          ? black
          : white
        : theme === 'dark'
          ? white
          : black;

  return (
    <img
      src={src}
      className={className}
      alt=""
      draggable={false}
      // The mark is 840x890: constraining the height and letting the width
      // follow keeps it from being squashed wherever it is used.
      style={{ height: size, width: 'auto', display: 'block' }}
    />
  );
};
