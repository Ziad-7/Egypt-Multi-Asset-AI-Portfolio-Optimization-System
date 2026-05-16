"use client";

import type { BacktestPayload } from "@/lib/intelligence";
import { isBacktestPayload } from "@/lib/intelligence";

function evalWindowFootnote(backtest: BacktestPayload): string {
  const ew = backtest.evaluation_window;
  if (Array.isArray(ew) && ew.length === 2) return ` · Evaluation: ${ew[0]} — ${ew[1]}`;
  if (ew && typeof ew === "object" && "start" in ew && "end" in ew) {
    return ` · Evaluation: ${String((ew as { start?: string }).start)} — ${String((ew as { end?: string }).end)}`;
  }
  return "";
}

export function KeyPortfolioStats({
  backtest,
  profileLabel,
  customDailyCumulative,
  customDailyDates,
  customTitle = "Custom mix (replay)",
}: {
  backtest: unknown;
  profileLabel: string;
  /** When set, table uses daily cumulative series from client replay (same calendar as charts). */
  customDailyCumulative?: number[] | null;
  customDailyDates?: string[] | null;
  customTitle?: string;
}) {
  if (!isBacktestPayload(backtest)) {
    return <p className="text-sm text-[#5c6570]">Run a full backtest to populate period statistics.</p>;
  }

  if (customDailyCumulative?.length && customDailyDates?.length) {
    const cum = customDailyCumulative;
    const beginning = 0;
    const ending = cum[cum.length - 1] ?? 0;
    const change = ending - beginning;
    const rows = [
      { label: `${customTitle} — start`, value: beginning },
      { label: `${customTitle} — end`, value: ending },
      { label: "Change", value: change },
    ];
    return (
      <table className="w-full table-fixed border-collapse text-sm">
        <thead>
          <tr className="border-b border-[#e8ecf1] text-left text-[11px] text-[#7a8490]">
            <th className="w-[58%] pb-2 pr-3 font-medium">Metric</th>
            <th className="pb-2 text-right font-medium">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-[#f0f2f5]">
              <td className="py-2 pr-3 align-top text-[#3d454d] break-words">{r.label}</td>
              <td
                className={`py-2 text-right tabular-nums font-semibold whitespace-nowrap ${
                  r.label === "Change"
                    ? r.value >= 0
                      ? "text-[#0d7a3e]"
                      : "text-[#c62828]"
                    : "text-[#1a1a1a]"
                }`}
              >
                {(r.value * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={2} className="pt-3 text-left text-[11px] leading-relaxed break-words text-[#7a8490]">
              Daily window: {customDailyDates[0]?.slice(0, 10)} — {customDailyDates[customDailyDates.length - 1]?.slice(0, 10)}
              . Replay uses engine asset returns, rebalance calendar, and transaction cost from this report.
              {evalWindowFootnote(backtest)}
            </td>
          </tr>
        </tfoot>
      </table>
    );
  }

  const cum = backtest.cumulative_returns?.[profileLabel] ?? backtest.cumulative_returns?.Optimal;
  const dates = backtest.dates;
  if (!cum?.length || !dates?.length) {
    return <p className="text-sm text-[#5c6570]">No cumulative return series in backtest payload.</p>;
  }

  const beginning = cum[0] ?? 0;
  const ending = cum[cum.length - 1] ?? 0;
  const change = ending - beginning;

  const rows = [
    { label: "Beginning (cum. return)", value: beginning },
    { label: "Ending (cum. return)", value: ending },
    { label: "Change", value: change },
  ];

  return (
    <table className="w-full table-fixed border-collapse text-sm">
      <thead>
        <tr className="border-b border-[#e8ecf1] text-left text-[11px] text-[#7a8490]">
          <th className="w-[58%] pb-2 pr-3 font-medium">Metric</th>
          <th className="pb-2 text-right font-medium">Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} className="border-b border-[#f0f2f5]">
            <td className="py-2 pr-3 align-top text-[#3d454d] break-words">{r.label}</td>
            <td
              className={`py-2 text-right tabular-nums font-semibold whitespace-nowrap ${
                r.label === "Change"
                  ? r.value >= 0
                    ? "text-[#0d7a3e]"
                    : "text-[#c62828]"
                  : "text-[#1a1a1a]"
              }`}
            >
              {(r.value * 100).toFixed(2)}%
            </td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr>
          <td colSpan={2} className="pt-3 text-left text-[11px] leading-relaxed break-words text-[#7a8490]">
            Period: {dates[0]} — {dates[dates.length - 1]}
            {evalWindowFootnote(backtest)}
          </td>
        </tr>
      </tfoot>
    </table>
  );
}
