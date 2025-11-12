import { sprintf } from "sprintf-js";

export function timeclock<T>(label: string, fn: () => T): T {
  const start = performance.now();
  const result = fn();
  const end = performance.now();
  console.log(sprintf(`[${label}] time: %.3f ms`, end - start));
  return result;
}
