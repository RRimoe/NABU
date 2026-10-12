// src/commands/batch.ts
//
// The same edit on many photos.
//
// A batch is a list of commands run on each photo in turn: open it, run the
// steps, export, close. Because the steps are ordinary registry commands,
// anything Nabu can do to one photo it can do to a folder of them, and the
// batch dialog, the command line and an agent all describe a batch the same way.

import { useDocumentStore } from '../store/documentStore';
import type { ExportFormat } from '../ops/export';
import { register, execute, type Outcome } from './registry';
import { openPath, exportActive, EXPORT_EXTENSIONS } from './catalog';
import { stem, joinPath, baseName } from './files';
import { describeError } from '../i18n/backendError';

export interface BatchStep {
  command: string;
  args?: Record<string, unknown>;
}

export interface BatchOptions {
  files: string[];
  steps: BatchStep[];
  outputDir: string;
  format: ExportFormat;
  quality: number;
  scale: number;
  /** Added to each file name, so originals are never overwritten by default. */
  suffix: string;
}

export interface BatchFileResult {
  file: string;
  ok: boolean;
  output?: string;
  error?: string;
}

export interface BatchProgress {
  index: number;
  total: number;
  file: string;
}

/** Commands that make no sense inside a batch step. */
const NOT_A_STEP = new Set(['batch', 'open_image', 'close_document', 'select_document', 'list_commands']);

export function validateSteps(steps: BatchStep[]): string | null {
  for (const step of steps) {
    if (!step || typeof step.command !== 'string') return 'Every step needs a "command".';
    if (NOT_A_STEP.has(step.command)) return `"${step.command}" cannot be a batch step.`;
  }
  return null;
}

/** Output path for one input: the folder, the name plus suffix, the format's extension. */
export function outputPath(file: string, options: Pick<BatchOptions, 'outputDir' | 'suffix' | 'format'>) {
  return joinPath(options.outputDir, `${stem(file)}${options.suffix}.${EXPORT_EXTENSIONS[options.format]}`);
}

export async function runBatch(
  options: BatchOptions,
  onProgress: (progress: BatchProgress) => void = () => undefined,
  isCancelled: () => boolean = () => false
): Promise<BatchFileResult[]> {
  const results: BatchFileResult[] = [];
  const store = useDocumentStore.getState;
  // The photo the user was looking at stays the one in front afterwards.
  const previous = store().activeId;

  for (const [index, file] of options.files.entries()) {
    if (isCancelled()) break;
    onProgress({ index, total: options.files.length, file });

    let opened: string | null = null;
    try {
      const open = await openPath(file);
      if (!open.handled) throw new Error(baseName(file));
      opened = store().activeId;

      for (const step of options.steps) {
        const result = await execute(step.command, step.args ?? {});
        if (!result.ok) throw new Error(`${step.command}: ${result.message}`);
      }

      const output = outputPath(file, options);
      await exportActive(output, { format: options.format, quality: options.quality, scale: options.scale });
      results.push({ file, ok: true, output });
    } catch (err) {
      results.push({ file, ok: false, error: describeError(err) });
    } finally {
      if (opened && store().sessions[opened]) store().closeDocument(opened);
    }
  }

  if (store().sessions[previous]) store().setActiveDocument(previous);
  onProgress({ index: options.files.length, total: options.files.length, file: '' });
  return results;
}

register({
  name: 'batch',
  description:
    'Apply the same steps to many photos and export each one. Every step is a command of this registry with its arguments, e.g. [{"command": "apply_look", "args": {"look": "portra"}}, {"command": "resize", "args": {"width": 2048}}]. Originals are never modified.',
  params: {
    files: { type: 'array', items: { type: 'string' }, description: 'Absolute paths of the photos.', required: true },
    steps: { type: 'array', items: { type: 'object' }, description: 'Commands to run on each photo, in order.', required: true },
    output_dir: { type: 'string', description: 'Absolute path of the folder to write into.', required: true },
    format: { type: 'string', description: 'Output format (default jpeg).', enum: ['jpeg', 'png', 'webp'] },
    quality: { type: 'number', description: 'JPEG/WebP quality, 0..1 (default 0.92).', minimum: 0.05, maximum: 1 },
    scale: { type: 'number', description: 'Output size relative to each photo, 0..1 (default 1).', minimum: 0.01, maximum: 1 },
    suffix: { type: 'string', description: 'Added to each output name (default "-nabu").' },
  },
  run: async (args): Promise<Outcome> => {
    const steps = args.steps as BatchStep[];
    const invalid = validateSteps(steps);
    if (invalid) throw new Error(invalid);
    const files = (args.files as unknown[]).map(String);
    const results = await runBatch({
      files,
      steps,
      outputDir: String(args.output_dir),
      format: (args.format as ExportFormat | undefined) ?? 'jpeg',
      quality: Number(args.quality ?? 0.92),
      scale: Number(args.scale ?? 1),
      suffix: typeof args.suffix === 'string' ? args.suffix : '-nabu',
    });
    const ok = results.filter((r) => r.ok).length;
    return {
      messageKey: 'commands.batchDone',
      params: { done: ok, total: files.length },
      handled: ok === files.length,
      data: results,
    };
  },
});
