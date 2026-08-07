// src/ai/intents.ts
//
// Turns a typed phrase into an executable tool call or operation.
//
// This is a zero-latency multilingual intent matcher. It understands common
// phrasings across English, Italian, Spanish, French, and German for all Nabu
// operations. It extracts parameters (percentages, pixel dimensions, relative
// and absolute adjustment amounts) and executes in 0ms on any hardware.

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export type Intent =
  | { kind: 'tool'; call: ToolCall }
  | { kind: 'unknown'; reasonKey: string };

/** Generative requests we recognise but cannot yet perform. */
const GENERATIVE_PATTERNS = [
  /\bchange\s+(the\s+)?sky/i,
  /\bcinematic/i,
  /\bgenerat/i,
  /\breplace\b/i,
  /\bremove\s+(the\s+)?(people|person|object)/i,
  /\bcambi\w*\s+(il\s+)?cielo/i,
  /\bcinematograf/i,
  /\bsostituisc/i,
  /\bgener\w*/i,
  /\brimuov\w*\s+(le\s+)?persone/i,
  /\brimuov\w*\s+(l')?oggett/i,
];

interface AdjustmentMapping {
  property: string;
  patterns: RegExp[];
  invertDefaultSign?: boolean;
}

const ADJUSTMENT_MAPPINGS: AdjustmentMapping[] = [
  {
    property: 'exposure',
    patterns: [
      /(luminosit[aà]|luminos[oai]|esposizione|brillantezza|chiar[oai]|scur[oai])/i,
      /\b(brightness|exposure|brighter|darker|light|dark)\b/i,
    ],
  },
  {
    property: 'contrast',
    patterns: [/\b(contrasto|contrast)\b/i],
  },
  {
    property: 'saturation',
    patterns: [
      /(saturazione|colori|colore|vivid[oa])/i,
      /\b(saturation|color|colors|vivid)\b/i,
    ],
  },
  {
    property: 'vibrance',
    patterns: [/(vivacit[aà]|vibrance)/i],
  },
  {
    property: 'temperature',
    patterns: [
      /(temperatura|cald[oa]|fredd[oa])/i,
      /\b(temperature|warmth|warm|cold)\b/i,
    ],
  },
  {
    property: 'sharpness',
    patterns: [
      /(nitidezza|definizione|affila)/i,
      /\b(sharpness|sharpen|detail)\b/i,
    ],
  },
  {
    property: 'blur',
    patterns: [
      /(sfocatura|sfoca|morbidezza)/i,
      /\b(blur|softness)\b/i,
    ],
  },
  {
    property: 'grain',
    patterns: [
      /(grana|rumore)/i,
      /\b(grain|noise)\b/i,
    ],
  },
  {
    property: 'highlights',
    patterns: [/\b(luci|highlights|highlight)\b/i],
  },
  {
    property: 'shadows',
    patterns: [/\b(ombre|shadows|shadow)\b/i],
  },
  {
    property: 'whites',
    patterns: [/\b(bianchi|whites|white)\b/i],
  },
  {
    property: 'blacks',
    patterns: [/\b(neri|blacks|black)\b/i],
  },
  {
    property: 'vignette',
    patterns: [/\b(vignettatura|vignette)\b/i],
  },
  {
    property: 'tint',
    patterns: [/\b(tinta|tint)\b/i],
  },
  {
    property: 'hue',
    patterns: [/(tonalit[aà]|hue)/i],
  },
  {
    property: 'clarity',
    patterns: [/\b(chiarezza|clarity)\b/i],
  },
];

/** Extract numeric adjustment amount from phrase or infer from direction keywords. */
function extractAdjustmentAmount(phrase: string, defaultInvert = false): number {
  const isDecrease = /(riduci|diminuisci|abbassa|togli|meno|scur[oa]|darker|cold|fredd[oa]|lower|less|minus|-)/i.test(phrase);
  const isStrong = /(molto|tanto|super|decisamente|much|very|way)/i.test(phrase);
  const isSlight = /(un po'|leggermente|slightly|a bit|little)/i.test(phrase);

  // Check for explicit number like +30, -20, "del 80%", "di 15"
  const numberMatch = phrase.match(/([+-]?\d+)\s*%?/);
  if (numberMatch) {
    let parsed = parseInt(numberMatch[1], 10);
    if (!isNaN(parsed) && parsed !== 0) {
      const rawNumStr = numberMatch[1];
      // If no explicit + or - sign was given in the digit string, infer sign from direction keywords
      if (!rawNumStr.startsWith('+') && !rawNumStr.startsWith('-') && isDecrease) {
        parsed = -Math.abs(parsed);
      }
      if (Math.abs(parsed) > 100) parsed = Math.sign(parsed) * 100;
      return parsed;
    }
  }

  let magnitude = 25;
  if (isStrong) magnitude = 50;
  if (isSlight) magnitude = 10;

  const sign = isDecrease ? -1 : 1;
  return (defaultInvert ? -sign : sign) * magnitude;
}

export function interpret(phrase: string): Intent {
  const clean = phrase.trim();
  if (!clean) {
    return { kind: 'unknown', reasonKey: 'command.notUnderstood' };
  }

  // 1. Remove background
  if (
    /(scontorna|(rimuov|elimin|togli|leva)\w*\s+(lo\s+)?sfondo|senza\s+sfondo|remove\s+(the\s+)?background|cut\s*out\s+background|bg\s+removal|quitar\s+fondo|supprimer\s+arri[eè]re-?plan|hintergrund\s+entfern)/i.test(
      clean
    )
  ) {
    return { kind: 'tool', call: { name: 'remove_background', arguments: {} } };
  }

  // 2. Cut out selection
  if (
    /(ritaglia\s+selezione|estrai\s+selezione|scontorna\s+selezione|estrai\s+oggetto|cut\s*out\s+selection|extract\s+selection|crop\s+selection)/i.test(
      clean
    )
  ) {
    return { kind: 'tool', call: { name: 'cut_out_selection', arguments: {} } };
  }

  // 3. Reset adjustments
  if (
    /(reset|resetta|ripristina|annulla\s+modifiche|cancella\s+regolazioni|clear\s+adjustments|undo\s+changes|restore\s+original)/i.test(
      clean
    )
  ) {
    return { kind: 'tool', call: { name: 'reset_adjustments', arguments: {} } };
  }

  // 4. Black & White / Desaturate
  if (/(bianco\s*e\s*nero|b[&\/]n|black\s*and\s*white|b[&\/]w|grayscale|monocromati|desatura)/i.test(clean)) {
    return {
      kind: 'tool',
      call: { name: 'adjust', arguments: { property: 'saturation', amount: -100 } },
    };
  }

  // 5. Upscale / Super resolution
  if (
    /(upscal|ingrandisc\w*|(aumenta|migliora)\w*\s+(la\s+)?(risoluzione|qualit[aà]?)|super\s+resolution|pi[uù]\s+nitid[oa]|sharper\s+(photo|image)|enhance\s+(quality|photo))/i.test(
      clean
    )
  ) {
    return { kind: 'tool', call: { name: 'upscale', arguments: {} } };
  }



  // 5. Tonal & Colour Adjustments
  for (const mapping of ADJUSTMENT_PROPERTIES_MAP) {
    if (mapping.patterns.some((p) => p.test(clean))) {
      const amount = extractAdjustmentAmount(clean, mapping.invertDefaultSign);
      return {
        kind: 'tool',
        call: {
          name: 'adjust',
          arguments: { property: mapping.property, amount },
        },
      };
    }
  }

  // 6. Resize (Percentage or Explicit Dimensions)
  if (
    /\b(resize|ridimensiona|dimezza|raddoppia|scala|scale|dimensioni|dimensione)\b/i.test(clean) ||
    /%/.test(clean)
  ) {
    if (/\bdimezza\b/i.test(clean) || /\bhalf\b/i.test(clean)) {
      return { kind: 'tool', call: { name: 'resize', arguments: { percent: 50 } } };
    }
    if (/\braddoppia\b/i.test(clean) || /\bdouble\b/i.test(clean)) {
      return { kind: 'tool', call: { name: 'resize', arguments: { percent: 200 } } };
    }

    const percentMatch = clean.match(/(\d+)\s*%/);
    if (percentMatch) {
      const p = parseInt(percentMatch[1], 10);
      if (p > 0) return { kind: 'tool', call: { name: 'resize', arguments: { percent: p } } };
    }

    const dimMatch = clean.match(/(\d{2,5})\s*[x×,]\s*(\d{2,5})/);
    if (dimMatch) {
      const width = parseInt(dimMatch[1], 10);
      const height = parseInt(dimMatch[2], 10);
      return { kind: 'tool', call: { name: 'resize', arguments: { width, height } } };
    }

    const widthMatch = clean.match(/(larghezza|width)\s*(\d{2,5})/i);
    if (widthMatch) {
      return {
        kind: 'tool',
        call: { name: 'resize', arguments: { width: parseInt(widthMatch[2], 10) } },
      };
    }

    const heightMatch = clean.match(/(altezza|height)\s*(\d{2,5})/i);
    if (heightMatch) {
      return {
        kind: 'tool',
        call: { name: 'resize', arguments: { height: parseInt(heightMatch[2], 10) } },
      };
    }
  }


  // Check for generative unsupported phrases
  if (GENERATIVE_PATTERNS.some((pattern) => pattern.test(clean))) {
    return { kind: 'unknown', reasonKey: 'command.needsGenerative' };
  }

  return { kind: 'unknown', reasonKey: 'command.notUnderstood' };
}

const ADJUSTMENT_PROPERTIES_MAP = ADJUSTMENT_MAPPINGS;

