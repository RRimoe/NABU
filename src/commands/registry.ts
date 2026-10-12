// src/commands/registry.ts
//
// Everything Nabu can be asked to do, declared once.
//
// The command bar, the local language model, the command line and agents
// connected over MCP all end up here, so a feature added to the registry is
// available to every one of them at the same time and behaves the same way in
// each. Each command carries the description and parameter schema an agent
// reads, and the function that does the work on the open document.

import { t } from '../i18n';
import { captureReports } from '../ops/notify';
import { describeError } from '../i18n/backendError';

export type ParamType = 'number' | 'integer' | 'string' | 'boolean' | 'object' | 'array';

export interface ParamSpec {
  type: ParamType;
  description: string;
  required?: boolean;
  minimum?: number;
  maximum?: number;
  enum?: string[];
  /** For arrays: the type of each item. */
  items?: { type: ParamType };
}

/** What a command did, in a form both the interface and a script can use. */
export interface Outcome {
  /** Translation key describing what was done, or why nothing was. */
  messageKey: string;
  params?: Record<string, string | number>;
  handled: boolean;
  /** Structured result for scripts and agents. */
  data?: unknown;
  /** A picture to show an agent, already encoded. */
  image?: { mimeType: string; base64: string };
}

export interface Command {
  name: string;
  /** Written for whoever calls it: when to use it and what the numbers mean. */
  description: string;
  params: Record<string, ParamSpec>;
  /** True when the command only reads, so an agent can call it freely. */
  readOnly?: boolean;
  run: (args: Record<string, unknown>) => Promise<Outcome> | Outcome;
}

/** The answer sent back to a script or an agent. */
export interface CommandResult {
  ok: boolean;
  message: string;
  data?: unknown;
  image?: { mimeType: string; base64: string };
}

/** The shape handed out for discovery: JSON Schema, as MCP expects it. */
export interface CommandDescriptor {
  name: string;
  description: string;
  readOnly: boolean;
  inputSchema: {
    type: 'object';
    properties: Record<string, Omit<ParamSpec, 'required'>>;
    required: string[];
  };
}

const commands = new Map<string, Command>();

export function register(...list: Command[]) {
  for (const command of list) commands.set(command.name, command);
}

export const findCommand = (name: string) => commands.get(name);

export function describeCommands(): CommandDescriptor[] {
  return [...commands.values()].map((command) => ({
    name: command.name,
    description: command.description,
    readOnly: Boolean(command.readOnly),
    inputSchema: {
      type: 'object',
      properties: Object.fromEntries(
        Object.entries(command.params).map(([key, spec]) => {
          const { required: _required, ...schema } = spec;
          return [key, schema];
        })
      ),
      required: Object.entries(command.params)
        .filter(([, spec]) => spec.required)
        .map(([key]) => key),
    },
  }));
}

const typeOf = (value: unknown): ParamType | 'null' => {
  if (value === null || value === undefined) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
  return typeof value as ParamType;
};

/**
 * Turns numbers and booleans written as strings into what they say. Language
 * models and shell arguments both produce "20" where 20 was meant, and refusing
 * that would be pedantry rather than safety.
 */
export function coerceArgs(command: Command, args: Record<string, unknown>): Record<string, unknown> {
  const out = { ...args };
  for (const [key, spec] of Object.entries(command.params)) {
    const value = out[key];
    // 1920.0000001 from a model's arithmetic means 1920.
    if (spec.type === 'integer' && typeof value === 'number' && Number.isFinite(value)) {
      out[key] = Math.round(value);
      continue;
    }
    if (typeof value !== 'string') continue;
    if ((spec.type === 'number' || spec.type === 'integer') && value.trim() !== '' && Number.isFinite(Number(value))) {
      out[key] = spec.type === 'integer' ? Math.round(Number(value)) : Number(value);
    } else if (spec.type === 'boolean' && (value === 'true' || value === 'false')) {
      out[key] = value === 'true';
    }
  }
  return out;
}

/**
 * Checks the arguments against the schema. A wrong argument is answered with
 * what was wrong, because the caller is often a model that will correct itself
 * from the message.
 */
export function validate(command: Command, args: Record<string, unknown>): string | null {
  for (const [key, spec] of Object.entries(command.params)) {
    const value = args[key];
    const actual = typeOf(value);
    if (actual === 'null') {
      if (spec.required) return `Missing parameter "${key}".`;
      continue;
    }
    const fits =
      actual === spec.type ||
      (spec.type === 'number' && actual === 'integer') ||
      (spec.type === 'integer' && actual === 'integer');
    if (!fits) return `Parameter "${key}" must be ${spec.type}, got ${actual}.`;
    if (typeof value === 'number') {
      if (spec.minimum !== undefined && value < spec.minimum) {
        return `Parameter "${key}" must be at least ${spec.minimum}.`;
      }
      if (spec.maximum !== undefined && value > spec.maximum) {
        return `Parameter "${key}" must be at most ${spec.maximum}.`;
      }
    }
    if (spec.enum && !spec.enum.includes(String(value))) {
      return `Parameter "${key}" must be one of: ${spec.enum.join(', ')}.`;
    }
  }
  const unknown = Object.keys(args).filter((key) => !(key in command.params));
  if (unknown.length > 0) return `Unknown parameter "${unknown[0]}".`;
  return null;
}

/**
 * Runs a command for someone who is not looking at the screen: failures that
 * would have been a dialog come back in the result instead.
 */
export async function execute(name: string, args: Record<string, unknown> = {}): Promise<CommandResult> {
  const command = commands.get(name);
  if (!command) return { ok: false, message: `Unknown command "${name}".` };

  args = coerceArgs(command, args);
  const invalid = validate(command, args);
  if (invalid) return { ok: false, message: invalid };

  try {
    const { value, reports } = await captureReports(async () => command.run(args));
    const message = t(value.messageKey, value.params);
    if (reports.length > 0) {
      return { ok: false, message: reports.join('\n'), data: value.data };
    }
    return { ok: value.handled, message, data: value.data, image: value.image };
  } catch (err) {
    // Backend errors arrive as translation keys; describeError reads them.
    return { ok: false, message: describeError(err) };
  }
}

// Commands that need nothing but the registry itself.
register({
  name: 'list_commands',
  description: 'List every command Nabu understands, with its parameters.',
  params: {},
  readOnly: true,
  run: () => ({ messageKey: 'commands.listed', handled: true, data: describeCommands() }),
});
