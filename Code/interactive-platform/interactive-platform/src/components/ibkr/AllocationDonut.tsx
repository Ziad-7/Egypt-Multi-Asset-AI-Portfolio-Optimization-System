"use client";

import { useMemo } from "react";
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ASSET_COLORS } from "@/lib/intelligence";

function formatPct(v: number) {
  return `${v.toFixed(2)}%`;
}

export function AllocationDonut({
  weights,
  profileName,
  asOfLabel,
}: {
  weights: Record<string, number>;
  profileName: string;
  asOfLabel: string;
}) {
  const data = useMemo(
    () =>
      Object.entries(weights)
        .filter(([, w]) => w > 0.0005)
        .map(([name, w]) => ({
          name,
          value: Math.round(w * 10000) / 100,
        }))
        .sort((a, b) => b.value - a.value),
    [weights],
  );

  const centerLine = profileName;
  const centerSub = data.length ? `${data.reduce((s, d) => s + d.value, 0).toFixed(1)}% allocated` : "—";

  return (
    <div className="relative h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="46%"
            innerRadius="58%"
            outerRadius="82%"
            paddingAngle={1}
            stroke="#fff"
            strokeWidth={2}
          >
            {data.map((d) => (
              <Cell key={d.name} fill={ASSET_COLORS[d.name] ?? "#64748b"} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v) => [formatPct(Number(v ?? 0)), "Weight"]}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #d8dee6",
              fontSize: 12,
            }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value) => <span className="text-xs text-[#3d454d]">{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute left-1/2 top-[42%] w-[140px] -translate-x-1/2 -translate-y-1/2 text-center">
        <p className="text-[11px] font-medium uppercase tracking-wide text-[#5c6570]">{asOfLabel}</p>
        <p className="mt-1 text-lg font-bold leading-tight text-[#0f4c9e]">{centerLine}</p>
        <p className="mt-0.5 text-xs tabular-nums text-[#1a1a1a]">{centerSub}</p>
      </div>
    </div>
  );
}
