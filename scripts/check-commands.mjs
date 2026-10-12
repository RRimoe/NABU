// scripts/check-commands.mjs
//
// The local language model may only call tools the command registry runs.
// Its vocabulary is declared in Rust (src-tauri/src/ai/tools.rs) and the
// registry in TypeScript (src/commands/catalog.ts); this check fails when a
// tool or one of its parameters has no command behind it, which would
// otherwise surface only as a request that silently does nothing.

import { readFileSync } from 'node:fs';

const tools = readFileSync(new URL('../src-tauri/src/ai/tools.rs', import.meta.url), 'utf8');
const catalog = readFileSync(new URL('../src/commands/catalog.ts', import.meta.url), 'utf8');

// Handled by the router itself: it is how the model says "I can't".
const ROUTER_ONLY = new Set(['report_unsupported']);

const table = tools.slice(tools.indexOf('pub const TOOLS'), tools.indexOf('];', tools.indexOf('pub const TOOLS')));
const specs = table.split('ToolSpec {').slice(1).map((block) => {
  const [, name] = /name:\s*"(\w+)"/.exec(block);
  const params = [...block.matchAll(/ToolParam\s*\{\s*name:\s*"(\w+)"/g)].map((m) => m[1]);
  // The `amount(...)` helper declares a parameter called amount.
  if (/\bamount\(/.test(block)) params.push('amount');
  return { name, params };
});

let errors = 0;
for (const { name, params } of specs) {
  if (ROUTER_ONLY.has(name)) continue;
  const start = catalog.indexOf(`name: '${name}',`);
  if (start < 0) {
    console.error(`ERR  tool "${name}" has no command in src/commands/catalog.ts`);
    errors++;
    continue;
  }
  const next = catalog.indexOf("\n    name: '", start + 1);
  const block = catalog.slice(start, next < 0 ? undefined : next);
  for (const param of params) {
    if (!new RegExp(`\\b${param}: \\{`).test(block)) {
      console.error(`ERR  tool "${name}" passes "${param}", which the command does not declare`);
      errors++;
    }
  }
}

console.log(`${errors === 0 ? 'PASSED' : 'FAILED'} — ${specs.length} tools checked, ${errors} error(s)`);
process.exit(errors === 0 ? 0 : 1);
