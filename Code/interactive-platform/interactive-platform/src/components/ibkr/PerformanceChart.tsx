"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Area,
  Line,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ASSET_COLORS, WEIGHT_REPLAY_SERIES_KEY } from "@/lib/intelligence";

/**
 * Series order: live weight replay first when present, then engine optimal, then benchmarks.
 * Raw `TBills` is omitted here on purpose — same as `report_plots._plot_backtest_curves`: it is
 * near–carry-linear and reads like a “wrong” line next to EGX_TBills_60_40; the policy blend is
 * `EGX_TBills_60_40`.
 */
const DISPLAY_ORDER = [WEIGHT_REPLAY_SERIES_KEY, "Optimal", "EGX30", "EGX_TBills_60_40", "SP500"] as const;

export function PerformanceChart({
  dates,
  cumulativeBySeries,
  subtitle,
}: {
  dates: string[];
  cumulativeBySeries: Record<string, number[]>;
  subtitle?: string;
}) {
  const { rows, keysInner, hasReplay, hasOptimal } = useMemo(() => {
    const keys = DISPLAY_ORDER.filter((k) => cumulativeBySeries[k]?.length);
    const n = Math.min(dates.length, ...keys.map((k) => cumulativeBySeries[k]!.length));
    const rowsInner = Array.from({ length: n }, (_, i) => {
      const row: Record<string, string | number> = {
        t: dates[i]?.slice(5) ?? String(i),
        full: dates[i] ?? "",
      };
      for (const k of keys) {
        const v = cumulativeBySeries[k]![i];
        row[k] = v != null ? v * 100 : 0;
      }
      return row;
    });
    return {
      rows: rowsInner,
      keysInner: keys,
      hasReplay: keys.includes(WEIGHT_REPLAY_SERIES_KEY),
      hasOptimal: keys.includes("Optimal"),
    };
  }, [dates, cumulativeBySeries]);

  if (!rows.length || !keysInner.length) {
    return (
      <div className="flex h-[340px] items-center justify-center text-sm text-[#5c6570]">
        No performance series for this report.
      </div>
    );
  }

  return (
    <div className="h-[360px] w-full">
      <p className="mb-1 text-xs text-[#5c6570]">{subtitle}</p>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 12, right: 16, left: 4, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8ecf1" vertical={false} />
          <XAxis dataKey="t" tick={{ fontSize: 11, fill: "#5c6570" }} axisLine={{ stroke: "#d8dee6" }} />
          <YAxis
            tick={{ fontSize: 11, fill: "#5c6570" }}
            axisLine={{ stroke: "#d8dee6" }}
            tickFormatter={(v) => `${v}%`}
            domain={["auto", "auto"]}
          />
          <Tooltip
            formatter={(v, name) => [`${Number(v ?? 0).toFixed(2)}%`, String(name)]}
            labelFormatter={(_, payload) => String((payload?.[0]?.payload as { full?: string })?.full ?? "")}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #d8dee6",
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />

          {hasReplay ? (
            <Area
              type="linear"
              dataKey={WEIGHT_REPLAY_SERIES_KEY}
              name="Replay (your weights)"
              stroke={ASSET_COLORS[WEIGHT_REPLAY_SERIES_KEY]}
              fill={ASSET_COLORS[WEIGHT_REPLAY_SERIES_KEY]}
              fillOpacity={0.14}
              strokeWidth={2.4}
            />
          ) : null}
          {!hasReplay && hasOptimal ? (
            <Area
              type="linear"
              dataKey="Optimal"
              name="Optimal (strategy)"
              stroke={ASSET_COLORS.Optimal}
              fill={ASSET_COLORS.Optimal}
              fillOpacity={0.12}
              strokeWidth={2}
            />
          ) : null}
          {hasReplay && hasOptimal ? (
            <Line
              type="linear"
              dataKey="Optimal"
              name="Optimal (engine walk-forward)"
              stroke={ASSET_COLORS.Optimal}
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              activeDot={{ r: 3 }}
            />
          ) : null}
          {keysInner
            .filter((k) => k !== WEIGHT_REPLAY_SERIES_KEY && k !== "Optimal")
            .map((k) => (
              <Line
                key={k}
                type="linear"
                dataKey={k}
                name={k}
                stroke={ASSET_COLORS[k] ?? "#64748b"}
                strokeWidth={1.8}
                dot={false}
                activeDot={{ r: 3 }}
              />
            ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
