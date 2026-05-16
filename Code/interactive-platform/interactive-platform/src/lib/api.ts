import type { IntelligenceReport } from "./intelligence";

const defaultBase = "http://127.0.0.1:8787";

export function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? defaultBase;
}

export async function fetchLatestReport(): Promise<IntelligenceReport> {
  const res = await fetch(`${apiBase()}/api/report/latest`, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Latest report failed (${res.status})`);
  }
  return res.json() as Promise<IntelligenceReport>;
}

export async function runIntelligencePipeline(options: {
  includeBacktest: boolean;
  save: boolean;
}): Promise<IntelligenceReport> {
  const path = options.save ? "/api/report/run-and-save" : "/api/report/run";
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_backtest: options.includeBacktest }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Run failed (${res.status})`);
  }
  return res.json() as Promise<IntelligenceReport>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${apiBase()}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
