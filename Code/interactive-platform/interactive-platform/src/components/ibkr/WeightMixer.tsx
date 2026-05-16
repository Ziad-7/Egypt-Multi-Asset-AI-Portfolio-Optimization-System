"use client";

import { ASSET_COLORS } from "@/lib/intelligence";
import { redistributeWeight } from "@/lib/portfolioAnalytics";

export function WeightMixer({
  orderedAssets,
  weights,
  onWeightsChange,
  onReset,
  profileName,
  isCustom,
}: {
  orderedAssets: string[];
  weights: Record<string, number>;
  onWeightsChange: (w: Record<string, number>) => void;
  onReset: () => void;
  profileName: string;
  isCustom: boolean;
}) {
  return (
    <div className="rounded-lg border border-[#d8dee6] bg-[#f7f9fc] p-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-[#5c6570]">Mix weights</p>
          <p className="text-[11px] text-[#7a8490]">
            Default matches <strong className="text-[#1a1a1a]">{profileName}</strong>. Drag to explore; donut, ex-ante
            stats, risk split, and frontier dot update live.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="shrink-0 rounded border border-[#d8dee6] bg-white px-3 py-1.5 text-xs font-medium text-[#0f4c9e] hover:bg-white"
        >
          Reset to {profileName}
        </button>
      </div>
      {isCustom ? (
        <p className="mt-2 rounded bg-[#fff8e6] px-2 py-1 text-[11px] text-[#7a5c00]">
          Custom mix: performance replays your weights on the report&apos;s <strong>daily asset returns</strong> (same
          rebalance + cost rules as the engine). <strong>Optimal</strong> stays the saved walk-forward line for
          comparison.
        </p>
      ) : null}
      <div className="mt-4 space-y-3">
        {orderedAssets.map((asset) => {
          const pct = (weights[asset] ?? 0) * 100;
          return (
            <div key={asset} className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
              <div className="flex w-full min-w-0 items-center gap-2 sm:w-44">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: ASSET_COLORS[asset] ?? "#64748b" }}
                />
                <span className="truncate text-xs font-medium text-[#1a1a1a]">{asset}</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                step={0.5}
                value={Math.round(pct * 10) / 10}
                onChange={(e) => {
                  const v = Number(e.target.value) / 100;
                  onWeightsChange(redistributeWeight(weights, asset, v, orderedAssets));
                }}
                className="h-2 flex-1 accent-[#0f4c9e]"
              />
              <span className="w-14 shrink-0 text-right text-xs tabular-nums text-[#3d454d]">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
