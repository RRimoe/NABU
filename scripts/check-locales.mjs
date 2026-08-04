// scripts/check-locales.mjs
//
// Compares every translation against the English base and reports missing,
// unknown or empty keys, plus placeholders that do not match.
//
// Run with `npm run check:locales`. A translation may be incomplete — the
// application falls back to English — so missing keys are a warning, while a
// broken placeholder is an error, because it would surface as literal `{name}`
// text in the interface.

import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const LOCALES = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'locales');
const BASE = 'en';

const placeholders = (text) =>
  new Set([...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]));

const load = (code) => JSON.parse(readFileSync(join(LOCALES, `${code}.json`), 'utf8'));

const base = load(BASE);
const baseKeys = Object.keys(base).filter((key) => key !== '$meta');

let errors = 0;
let warnings = 0;

const files = readdirSync(LOCALES)
  .filter((name) => name.endsWith('.json'))
  .map((name) => name.replace('.json', ''));

for (const code of files) {
  const strings = load(code);
  const keys = Object.keys(strings).filter((key) => key !== '$meta');
  const name = strings.$meta?.name ?? code;

  const missing = baseKeys.filter((key) => !(key in strings));
  const unknown = keys.filter((key) => !(key in base));
  const empty = keys.filter((key) => !String(strings[key]).trim());

  const mismatched = keys
    .filter((key) => key in base)
    .filter((key) => {
      const expected = placeholders(base[key]);
      const actual = placeholders(strings[key]);
      return (
        expected.size !== actual.size || [...expected].some((token) => !actual.has(token))
      );
    });

  const clean =
    !missing.length && !unknown.length && !empty.length && !mismatched.length;

  console.log(
    `\n${clean ? 'OK  ' : '--  '} ${code} (${name}) — ${keys.length}/${baseKeys.length} keys`
  );

  if (missing.length) {
    warnings += missing.length;
    console.log(`     missing (${missing.length}): ${missing.slice(0, 6).join(', ')}${missing.length > 6 ? ', …' : ''}`);
  }
  if (unknown.length) {
    warnings += unknown.length;
    console.log(`     unknown (${unknown.length}): ${unknown.join(', ')}`);
  }
  if (empty.length) {
    errors += empty.length;
    console.log(`     EMPTY (${empty.length}): ${empty.join(', ')}`);
  }
  if (mismatched.length) {
    errors += mismatched.length;
    console.log(`     PLACEHOLDER MISMATCH (${mismatched.length}): ${mismatched.join(', ')}`);
  }
}

console.log(
  `\n${errors ? 'FAILED' : 'PASSED'} — ${errors} error(s), ${warnings} warning(s)\n`
);
process.exit(errors ? 1 : 0);
