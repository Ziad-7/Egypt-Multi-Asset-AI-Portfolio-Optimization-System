/**
 * Client-side fusion preview matching `src/portfolio/layer_interaction.py`
 * for the current strategic weights + (replayed) tactical output.
 */

import type { LayerFusion } from "./intelligence";
import { strategicBiasFromWeights } from "./tacticalReplay";

type InflationRegime = "low" | "rising" | "high" | "elevated_falling";

type InflationContext = {
  regime?: string;
  yoy?: number;
  real_yield?: number;
};

const DECISION_MATRIX: Record<string, [string, number]> = {
  "bullish|1": ["High-conviction long allocation; deploy fully on schedule.", 0.92],
  "bullish|0": ["Hold strategic equity; await tactical confirmation.", 0.65],
  "bullish|-1": ["Strategic long with tactical hedge: trim equity, add Gold.", 0.62],
  "defensive|1": ["Counter-trend bounce; small risk-on overlay only.", 0.45],
  "defensive|0": ["Stay defensive; maintain T-Bills/Gold weight.", 0.7],
  "defensive|-1": ["Strong protection mode; maximize T-Bills and Gold.", 0.88],
  "neutral|1": ["Tactical long overlay on neutral macro; size modestly.", 0.6],
  "neutral|0": ["Wait for clearer setup; rebalance to profile target.", 0.55],
  "neutral|-1": ["Tactical de-risk on neutral macro; add T-Bills.", 0.65],
};

const REGIME_TILT_MULTIPLIERS: Record<InflationRegime, Record<string, number>> = {
  low: { EGX30: 1.0, EGX100: 1.0, Gold: 1.0, TBills: 1.0, EgyptiansRealEstateFund: 1.0 },
  rising: { EGX30: 1.1, EGX100: 1.1, Gold: 1.25, TBills: 0.75, EgyptiansRealEstateFund: 1.1 },
  high: { EGX30: 1.1, EGX100: 1.1, Gold: 1.5, TBills: 0.5, EgyptiansRealEstateFund: 1.2 },
  elevated_falling: { EGX30: 1.05, EGX100: 1.05, Gold: 1.1, TBills: 0.9, EgyptiansRealEstateFund: 1.05 },
};

function baseTilt(tacticalSignal: number, confidence: number, suggestedSize: number): Record<string, number> {
  if (tacticalSignal === 0) return {};
  const magnitude = Math.round(0.05 * confidence * suggestedSize * 10_000) / 10_000;
  if (magnitude < 0.005) return {};
  if (tacticalSignal === 1) {
    return { EGX30: magnitude, Gold: -magnitude * 0.5, TBills: -magnitude * 0.5 };
  }
  return { EGX30: -magnitude, Gold: magnitude * 0.5, TBills: magnitude * 0.5 };
}

function applyInflationModifier(base: Record<string, number>, inflation: InflationContext | null | undefined): Record<string, number> {
  if (!Object.keys(base).length || !inflation) return base;
  const regime = (inflation.regime ?? "low") as InflationRegime;
  const multipliers = REGIME_TILT_MULTIPLIERS[regime] ?? REGIME_TILT_MULTIPLIERS.low;
  let adjusted: Record<string, number> = {};
  for (const [asset, weight] of Object.entries(base)) {
    adjusted[asset] = weight * (multipliers[asset] ?? 1.0);
  }
  const realYield = inflation.real_yield ?? 0;
  if (realYield < 0 && (adjusted.TBills ?? 0) > 0) {
    adjusted = { ...adjusted, TBills: 0 };
  }
  let total = Object.values(adjusted).reduce((a, b) => a + b, 0);
  if (Math.abs(total) > 1e-9) {
    const denom = Object.values(adjusted).reduce((a, b) => a + Math.abs(b), 0) || 1;
    for (const k of Object.keys(adjusted)) {
      adjusted[k] -= total * (Math.abs(adjusted[k]) / denom);
    }
  }
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(adjusted)) {
    if (Math.abs(v) > 1e-4) out[k] = Math.round(v * 10_000) / 10_000;
  }
  return out;
}

export function fuseLayersPreview(args: {
  weights: Record<string, number>;
  strategicConfidence: number;
  tacticalSignal: number;
  tacticalConfidence: number;
  suggestedPositionSize: number;
  inflation: InflationContext | null | undefined;
}): LayerFusion {
  const bias = strategicBiasFromWeights(args.weights);
  const key = `${bias}|${args.tacticalSignal}`;
  const entry = DECISION_MATRIX[key] ?? ["Wait for clearer setup; rebalance to profile target.", 0.55];
  const [action, baseConf] = entry;
  const bt = baseTilt(args.tacticalSignal, args.tacticalConfidence, args.suggestedPositionSize);
  const suggested_tilt = applyInflationModifier(bt, args.inflation);

  const confidence = Math.round(
    Math.max(
      0.05,
      Math.min(
        0.99,
        baseConf * (0.55 + 0.45 * args.tacticalConfidence) * (0.65 + 0.35 * args.strategicConfidence),
      ),
    ) * 10_000,
  ) / 10_000;

  let note = "Layer 1 sets direction; Layer 2 governs timing, sizing and protection.";
  if (args.inflation?.regime != null) {
    const yoy = args.inflation.yoy ?? 0;
    const ry = args.inflation.real_yield ?? 0;
    note += ` Inflation regime=${args.inflation.regime} (YoY=${(yoy * 100).toFixed(2)}%, real_yield=${(ry * 100).toFixed(2)}%).`;
  }

  return {
    strategic_bias: bias,
    tactical_signal: args.tacticalSignal,
    confidence,
    action,
    note,
    suggested_tilt,
  };
}
