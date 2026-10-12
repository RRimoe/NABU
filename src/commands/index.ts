// src/commands/index.ts
// Loads every command into the registry; import this, not the parts.
import './catalog';
import './batch';

export { execute, findCommand, describeCommands, validate, coerceArgs } from './registry';
export type { Outcome, CommandResult, CommandDescriptor } from './registry';
