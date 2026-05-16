"use client";

import { ASSET_COLORS } from "@/lib/intelligence";

export function WeightsBars({
  weights,
  title,
  subtitle,
  variant = "light",
  showHeader = true,
}: {
  weights: Record<string, number>;
  title: string;
  subtitle?: string;
  variant?: "light" | "dark";
  showHeader?: boolean;
}) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 1e-6);

  const box =
    variant === "light"
      ? "rounded-lg border border-[#d8dee6] bg-white p-5 shadow-[0_1px_3px_rgba(0,0,0,0.06)]"
      : "rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)_inset]";
  const h3 = variant === "light" ? "text-sm font-semibold text-[#1a1a1a]" : "text-sm font-semibold text-white";
  const sub = variant === "light" ? "text-xs text-[#5c6570]" : "text-xs text-slate-400";
  const label = variant === "light" ? "text-xs text-[#3d454d]" : "text-xs text-slate-300";
  const val = variant === "light" ? "tabular-nums text-[#1a1a1a]" : "tabular-nums text-slate-200";
  const track = variant === "light" ? "bg-[#eef1f5]" : "bg-white/5";

  return (
    <div className={box}>
      {showHeader ? (
        <div className="mb-4">
          <h3 className={h3}>{title}</h3>
          {subtitle ? <p className={`mt-1 ${sub}`}>{subtitle}</p> : null}
        </div>
      ) : null}
      <div className="space-y-3">
        {entries.map(([asset, w]) => (
          <div key={asset}>
            <div className={`mb-1 flex justify-between gap-2 ${label}`}>
              <span className="min-w-0 truncate" title={asset}>
                {asset}
              </span>
              <span className={`shrink-0 ${val}`}>{(w * 100).toFixed(2)}%</span>
            </div>
            <div className={`h-2 overflow-hidden rounded-full ${track}`}>
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${(w / max) * 100}%`,
                  backgroundColor: ASSET_COLORS[asset] ?? "#94a3b8",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
