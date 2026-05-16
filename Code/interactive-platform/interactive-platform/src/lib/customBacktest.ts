/**
 * Replays the Egypt engine's drift + quarterly rebalance + transaction-cost logic
 * on published daily asset returns (see `daily_asset_returns` in the backtest JSON).
 */

export type CustomSimResult = {
  dailyReturns: number[];
  cumulative: number[];
  drawdown: number[];
};

export function simulateDriftRebalance(
  dailyDates: string[],
  assetReturns: Record<string, number[]>,
  assetOrder: string[],
  targetWeights: Record<string, number>,
  rebalanceDateStrs: string[],
  transactionCostBps: number,
): CustomSimResult | null {
  const n = dailyDates.length;
  if (n === 0 || assetOrder.length === 0) return null;

  for (const a of assetOrder) {
    const arr = assetReturns[a];
    if (!arr || arr.length !== n) return null;
  }

  const targets = assetOrder.map((a) => Math.max(0, targetWeights[a] ?? 0));
  const sumT = targets.reduce((s, x) => s + x, 0) || 1;
  const targetsNorm = targets.map((t) => t / sumT);
  let w = targetsNorm.slice();
  const costRate = transactionCostBps / 10_000;
  const rebSet = new Set(rebalanceDateStrs.map((d) => d.slice(0, 10)));

  const dailyReturns: number[] = [];
  for (let t = 0; t < n; t++) {
    const dayStr = dailyDates[t]!.slice(0, 10);
    const growth = assetOrder.map((a) => 1 + (assetReturns[a]![t] ?? 0));
    for (let j = 0; j < w.length; j++) w[j]! *= growth[j]!;
    const gross = w.reduce((s, x) => s + x, 0);
    if (gross <= 0) {
      w = targetsNorm.slice();
      dailyReturns.push(0);
      continue;
    }
    let rDay = gross - 1;
    for (let j = 0; j < w.length; j++) w[j]! /= gross;
    if (rebSet.has(dayStr)) {
      let turnover = 0;
      for (let j = 0; j < w.length; j++) turnover += Math.abs(w[j]! - targetsNorm[j]!);
      rDay -= costRate * turnover;
      w = targetsNorm.slice();
    }
    dailyReturns.push(rDay);
  }

  let equity = 1;
  const cumulative: number[] = [];
  for (const r of dailyReturns) {
    equity *= 1 + r;
    cumulative.push(equity - 1);
  }

  let peak = 1;
  const drawdown: number[] = [];
  for (const cum of cumulative) {
    const eq = 1 + cum;
    peak = Math.max(peak, eq);
    drawdown.push(eq / peak - 1);
  }

  return { dailyReturns, cumulative, drawdown };
}
