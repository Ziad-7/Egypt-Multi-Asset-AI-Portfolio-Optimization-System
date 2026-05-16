"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ASSET_COLORS, WEIGHT_REPLAY_SERIES_KEY } from "@/lib/intelligence";

const DISPLAY_ORDER = [WEIGHT_REPLAY_SERIES_KEY, "Optimal", "EGX30", "EGX_TBills_60_40", "TBills", "SP500"] as const;

export function DrawdownChart({
  dates,
  drawdownsBySeries,
}: {
  dates: string[];
  drawdownsBySeries: Record<string, number[]>;
}) {
  const { rows, keys } = useMemo(() => {
    const keysInner = DISPLAY_ORDER.filter((k) => drawdownsBySeries[k]?.length);
    const n = Math.min(dates.length, ...keysInner.map((k) => drawdownsBySeries[k]!.length));
    const rowsInner = Array.from({ length: n }, (_, i) => {
      const row: Record<string, string | number> = { t: dates[i]?.slice(5) ?? String(i), full: dates[i] ?? "" };
      for (const k of keysInner) {
        const v = drawdownsBySeries[k]![i];
        row[k] = v != null ? v * 100 : 0;
      }
      return row;
    });
    return { rows: rowsInner, keys: keysInner };
  }, [dates, drawdownsBySeries]);

  if (!rows.length) {
    return (
      <div className="flex h-[260px] items-center justify-center text-sm text-[#5c6570]">No drawdown series.</div>
    );
  }

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8ecf1" vertical={false} />
          <XAxis dataKey="t" tick={{ fontSize: 11, fill: "#5c6570" }} axisLine={{ stroke: "#d8dee6" }} />
          <YAxis
            tick={{ fontSize: 11, fill: "#5c6570" }}
            axisLine={{ stroke: "#d8dee6" }}
            tickFormatter={(v) => `${v}%`}
            domain={["auto", 0]}
          />
          <Tooltip
            formatter={(v, name) => [`${Number(v ?? 0).toFixed(2)}%`, String(name)]}
            labelFormatter={(_, payload) => String((payload?.[0]?.payload as { full?: string })?.full ?? "")}
            contentStyle={{ borderRadius: 8, border: "1px solid #d8dee6", fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {keys.map((k) => (
            <Line
              key={k}
              type="linear"
              dataKey={k}
              name={k}
              stroke={ASSET_COLORS[k] ?? "#64748b"}
              strokeWidth={
                k === WEIGHT_REPLAY_SERIES_KEY ? 2.6 : k === "Optimal" ? 2 : 1.2
              }
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
