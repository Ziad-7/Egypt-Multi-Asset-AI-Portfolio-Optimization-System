/** Re-apply tactical strategic-alignment for alternate portfolio weights (matches Python `apply_strategic_alignment`). */

export type TacticalReplaySnapshot = {
  signal_after_regime?: number;
  confidence_after_regime?: number;
  quality_confidence_scale?: number;
};

function clip(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function strategicBiasFromWeights(weights: Record<string, number>): "bullish" | "defensive" | "neutral" {
  const egx = (weights.EGX30 ?? 0) + (weights.EGX100 ?? 0);
  const safe = (weights.Gold ?? 0) + (weights.TBills ?? 0);
  if (egx > safe + 0.05) return "bullish";
  if (safe > egx + 0.05) return "defensive";
  return "neutral";
}

function inRebalanceWindow(nextRebalanceDate: string, windowDays = 14): boolean {
  const t0 = new Date();
  t0.setHours(0, 0, 0, 0);
  const t1 = new Date(nextRebalanceDate.trim());
  t1.setHours(0, 0, 0, 0);
  const days = Math.round((t1.getTime() - t0.getTime()) / 86_400_000);
  return days >= 0 && days <= windowDays;
}

export function applyStrategicAlignment(
  signal: number,
  confidence: number,
  weights: Record<string, number> | null | undefined,
  nextRebalanceDate: string | null | undefined,
): { signal: number; confidence: number; rebalance_window_active: boolean } {
  let s = signal;
  let c = confidence;
  let rebalance_window_active = false;
  if (weights && nextRebalanceDate) {
    const bias = strategicBiasFromWeights(weights);
    rebalance_window_active = inRebalanceWindow(nextRebalanceDate);
    if (rebalance_window_active) {
      const aligned =
        (bias === "bullish" && s >= 0) || (bias === "defensive" && s <= 0) || bias === "neutral";
      c = clip(c * (aligned ? 1.1 : 0.85), 0.05, 0.98);
      if (!aligned && s === 1 && bias === "defensive") s = 0;
    }
  }
  return { signal: s, confidence: c, rebalance_window_active };
}

export function replayTacticalForWeights(
  snapshot: TacticalReplaySnapshot | undefined,
  weights: Record<string, number>,
  nextRebalanceDate: string | undefined,
): {
  signal: number;
  confidence: number;
  rebalance_window_active: boolean;
  usedReplay: boolean;
} | null {
  if (
    !snapshot ||
    typeof snapshot.signal_after_regime !== "number" ||
    typeof snapshot.confidence_after_regime !== "number"
  ) {
    return null;
  }
  const qscale = typeof snapshot.quality_confidence_scale === "number" ? snapshot.quality_confidence_scale : 1;
  const aligned = applyStrategicAlignment(
    snapshot.signal_after_regime,
    snapshot.confidence_after_regime,
    weights,
    nextRebalanceDate ?? undefined,
  );
  const confidence = clip(aligned.confidence * qscale, 0.05, 0.98);
  return {
    signal: aligned.signal,
    confidence,
    rebalance_window_active: aligned.rebalance_window_active,
    usedReplay: true,
  };
}
