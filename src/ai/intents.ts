// src/ai/intents.ts
//
// Turns a typed phrase into an executable operation.
//
// This is keyword recognition, not language understanding: it knows the common
// phrasings of the operations Nabu can actually perform, and says so for
// everything else. Patterns are grouped per language, so adding a language
// means adding its patterns here — the day a local language model exists, this
// becomes its fallback rather than the primary path.

export type Intent =
  | { kind: 'remove-background' }
  | { kind: 'upscale' }
  /** `reasonKey` is a translation key, not a message. */
  | { kind: 'unknown'; reasonKey: string };

interface Rule {
  kind: 'remove-background' | 'upscale';
  patterns: RegExp[];
}

const RULES: Rule[] = [
  {
    kind: 'remove-background',
    patterns: [
      // English
      /\bremove\s+(the\s+)?background/i,
      /\bcut\s*out\b/i,
      /\bbackground\b.*\bremov/i,
      // Italian
      /\bscontorn/i,
      /\b(rimuov|elimin|togli|leva)\w*\s+(lo\s+)?sfondo/i,
      /\bsfondo\b/i,
      // Spanish / French / German
      /\bquita\w*\s+(el\s+)?fondo/i,
      /\bsupprim\w*\s+(l\w*\s+)?arri[eè]re-?plan/i,
      /\bhintergrund\s+entfern/i,
    ],
  },
  {
    kind: 'upscale',
    patterns: [
      // English
      /\bupscal/i,
      /\bincrease\s+(the\s+)?resolution/i,
      /\benhance\b/i,
      /\bsharper\b/i,
      // Italian
      /\bingrandisc/i,
      /\brisoluzione\b/i,
      /\bmiglior\w*\s+(la\s+)?(risoluzione|qualit)/i,
      /\bmiglior\w*\s+(questa\s+)?(foto|immagine)/i,
      /\bpi[uù]\s+nitid/i,
      // Spanish / French / German
      /\bmejor\w*\s+(la\s+)?(resoluci|calidad)/i,
      /\baugment\w*\s+(la\s+)?r[eé]solution/i,
      /\baufl[oö]sung\s+erh[oö]h/i,
    ],
  },
];

/** Generative requests we recognise but cannot yet perform. */
const GENERATIVE = [
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

export function interpret(phrase: string): Intent {
  for (const rule of RULES) {
    if (rule.patterns.some((pattern) => pattern.test(phrase))) {
      return { kind: rule.kind };
    }
  }

  if (GENERATIVE.some((pattern) => pattern.test(phrase))) {
    return { kind: 'unknown', reasonKey: 'command.needsGenerative' };
  }

  return { kind: 'unknown', reasonKey: 'command.notUnderstood' };
}
