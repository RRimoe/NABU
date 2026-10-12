// src/commands/bridge.ts
//
// The other end of the local bridge: the backend forwards calls from the
// command line and from MCP clients as events, the registry runs them, and the
// result goes back the same way.
//
// Calls run one at a time. Two agents editing the same photo at once would
// interleave their history steps, and a script that opens, edits and exports
// relies on nothing else changing the active document in between.

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { describeCommands, execute } from './registry';
import { desktopAvailable } from './files';
import './catalog';
import './batch';

interface BridgeCall {
  id: number;
  name: string;
  arguments?: Record<string, unknown>;
}

let tail: Promise<unknown> = Promise.resolve();

function serialised<T>(task: () => Promise<T>): Promise<T> {
  const run = tail.then(task, task);
  tail = run.catch(() => undefined);
  return run;
}

/** Starts answering bridge calls. Returns the function that stops it. */
export async function connectBridge(): Promise<() => void> {
  if (!desktopAvailable()) return () => undefined;

  const unlisten = await listen<BridgeCall>('nabu-command', (event) => {
    const { id, name } = event.payload;
    const args = event.payload.arguments ?? {};
    void serialised(() => execute(name, args)).then((result) =>
      invoke('command_result', { id, result }).catch(() => undefined)
    );
  });
  await invoke('bridge_ready', { commands: describeCommands() }).catch((err) => {
    console.warn('[Nabu] Command bridge unavailable:', err);
  });
  return unlisten;
}
