/** Ex-ante analytics from strategic diagnostics (μ, Σ) for arbitrary long-only weights. */

export type MixAnalytics = {
  expectedReturn: number;
  volatility: number;
  sharpe: number | null;
  /** Fractions summing to ~1 — share of variance / σ² */
  riskContributions: Record<string, number>;
};

export function weightsApproximatelyEqual(
  a: Record<string, number>,
  b: Record<string, number>,
  tol = 0.002,
): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    if (Math.abs((a[k] ?? 0) - (b[k] ?? 0)) > tol) return false;
  }
  return true;
}

/** For deduping client replay vs saved `daily_cumulative_returns` (float + rebalance noise). */
export function cumulativeSeriesApproxEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length || a.length === 0) return false;
  for (let i = 0; i < a.length; i++) {
    const u = a[i]!;
    const v = b[i]!;
    const tol = Math.max(5e-5, 1e-7 * (1 + Math.max(Math.abs(u), Math.abs(v))));
    if (Math.abs(u - v) > tol) return false;
  }
  return true;
}

/** Set one asset’s target weight; scale remaining weights proportionally so everything sums to 1. */
export function redistributeWeight(
  weights: Record<string, number>,
  asset: string,
  targetFraction: number,
  orderedAssets: string[],
): Record<string, number> {
  const t = Math.max(0, Math.min(1, targetFraction));
  const others = orderedAssets.filter((x) => x !== asset);
  if (others.length === 0) return { [asset]: 1 };

  const out: Record<string, number> = { ...weights };
  let sumOthers = others.reduce((s, x) => s + (out[x] ?? 0), 0);
  const rem = 1 - t;
  if (sumOthers < 1e-10) {
    const eq = rem / others.length;
    others.forEach((x) => {
      out[x] = eq;
    });
  } else {
    const scale = rem / sumOthers;
    others.forEach((x) => {
      out[x] = (out[x] ?? 0) * scale;
    });
  }
  out[asset] = t;
  return out;
}

export function analyzeMix(
  weights: Record<string, number>,
  orderedAssets: string[],
  annualMean: Record<string, number> | undefined,
  cov: number[][] | undefined,
  riskFreeAnnual: number | undefined,
): MixAnalytics | null {
  if (!annualMean || !cov?.length || !orderedAssets.length) return null;
  if (cov.length !== orderedAssets.length) return null;

  const w = orderedAssets.map((a) => Math.max(0, weights[a] ?? 0));
  const sum = w.reduce((s, x) => s + x, 0) || 1;
  const wn = w.map((x) => x / sum);

  let expRet = 0;
  for (let i = 0; i < orderedAssets.length; i++) {
    expRet += wn[i] * (annualMean[orderedAssets[i]] ?? 0);
  }

  let varP = 0;
  const n = orderedAssets.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      varP += wn[i] * (cov[i]?.[j] ?? 0) * wn[j];
    }
  }

  const vol = Math.sqrt(Math.max(0, varP));
  const sharpe =
    vol > 1e-8 && riskFreeAnnual !== undefined ? (expRet - riskFreeAnnual) / vol : null;

  const sigmaW = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      sigmaW[i] += (cov[i]?.[j] ?? 0) * wn[j];
    }
  }

  const riskContributions: Record<string, number> = {};
  if (vol > 1e-8) {
    const v2 = vol * vol;
    for (let i = 0; i < n; i++) {
      riskContributions[orderedAssets[i]] = (wn[i] * sigmaW[i]) / v2;
    }
  } else {
    orderedAssets.forEach((a) => {
      riskContributions[a] = 1 / n;
    });
  }

  return { expectedReturn: expRet, volatility: vol, sharpe, riskContributions };
}
