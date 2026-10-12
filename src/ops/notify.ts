// src/ops/notify.ts
//
// How an operation tells someone it could not do what was asked.
//
// In the editor the answer is a dialog: the person who clicked is looking at
// the screen. When the same operation runs for a script or an agent through the
// command registry nobody is looking, and a modal dialog would freeze the call
// until someone dismissed it on a window that may be hidden. So the message is
// captured instead and becomes the command's error.

let sinks: string[][] = [];

/** Reports a failure: to the capture in progress, or to the person at the screen. */
export function report(message: string) {
  const sink = sinks[sinks.length - 1];
  if (sink) sink.push(message);
  else alert(message);
}

/** Runs `task` with reports collected rather than shown. */
export async function captureReports<T>(task: () => Promise<T>): Promise<{ value: T; reports: string[] }> {
  const sink: string[] = [];
  sinks = [...sinks, sink];
  try {
    const value = await task();
    return { value, reports: sink };
  } finally {
    sinks = sinks.filter((s) => s !== sink);
  }
}
