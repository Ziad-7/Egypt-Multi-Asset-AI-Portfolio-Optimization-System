"use client";

import type { StrategicProfile } from "@/lib/intelligence";

type AssetRow = { symbol: string; weightPct: number; annReturnPct: number };

function MoverTable({ title, rows }: { title: string; rows: AssetRow[] }) {
  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[#5c6570]">{title}</p>
      <table className="w-full table-fixed border-collapse text-sm">
        <thead>
          <tr className="border-b border-[#e8ecf1] text-left text-[11px] text-[#7a8490]">
            <th className="w-[44%] pb-2 pr-2 font-medium">Symbol</th>
            <th className="w-[28%] pb-2 pr-2 text-right font-medium">Weight %</th>
            <th className="pb-2 text-right font-medium">Ann. return</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol} className="border-b border-[#f0f2f5]">
              <td className="py-2 pr-2 align-top font-medium break-words text-[#1a1a1a]">{r.symbol}</td>
              <td className="py-2 pr-2 text-right tabular-nums text-[#3d454d]">{r.weightPct.toFixed(2)}</td>
              <td
                className={`py-2 text-right tabular-nums font-medium whitespace-nowrap ${
                  r.annReturnPct >= 0 ? "text-[#0d7a3e]" : "text-[#c62828]"
                }`}
              >
                {(r.annReturnPct * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AssetMovers({
  profile,
  assetSummary,
  weightsOverride,
}: {
  profile: StrategicProfile;
  assetSummary: Record<string, { annualized_mean?: number }> | undefined;
  /** When set (e.g. slider mix), weight column reflects this instead of the saved profile. */
  weightsOverride?: Record<string, number>;
}) {
  if (!assetSummary || !Object.keys(assetSummary).length) {
    return <p className="text-sm text-[#5c6570]">Asset statistics not available in this report.</p>;
  }

  const w = weightsOverride ?? profile.weights;
  const rows: AssetRow[] = Object.keys(assetSummary).map((symbol) => ({
    symbol,
    weightPct: (w[symbol] ?? 0) * 100,
    annReturnPct: assetSummary[symbol]?.annualized_mean ?? 0,
  }));

  const sorted = [...rows].sort((a, b) => b.annReturnPct - a.annReturnPct);
  const top = sorted.slice(0, Math.min(3, sorted.length));
  const bottom = sorted.slice(-Math.min(3, sorted.length)).reverse();

  return (
    <div className="grid gap-6 sm:grid-cols-2">
      <MoverTable title="Top performers (ann. return)" rows={top} />
      <MoverTable title="Bottom performers (ann. return)" rows={bottom} />
    </div>
  );
}
