"use client";

import { useMemo } from "react";
import { ASSET_COLORS } from "@/lib/intelligence";

type Series = Record<string, number[]>;

export function TimeSeriesChart({
  dates,
  series,
  title,
  subtitle,
}: {
  dates: string[];
  series: Series;
  title: string;
  subtitle?: string;
}) {
  const { paths, legend, meta } = useMemo(() => {
    const names = Object.keys(series);
    if (!names.length || !dates.length) {
      return { paths: [] as { name: string; d: string; color: string }[], legend: [] as string[], meta: { w: 900, h: 320, sliceLen: 0 } };
    }
    const maxLen = Math.max(dates.length, ...names.map((n) => series[n]?.length ?? 0));
    const sliceLen = Math.min(maxLen, dates.length);
    const allValues: number[] = [];
    names.forEach((n) => {
      const arr = series[n] ?? [];
      for (let i = 0; i < sliceLen; i++) allValues.push(arr[i] ?? 0);
    });
    const minY = allValues.length ? Math.min(...allValues, 0) : 0;
    const maxY = allValues.length ? Math.max(...allValues, 0) : 0;
    const padY = (maxY - minY) * 0.08 || 0.01;

    const w = 900;
    const h = 320;
    const pad = { l: 52, r: 20, t: 20, b: 48 };
    const innerW = w - pad.l - pad.r;
    const innerH = h - pad.t - pad.b;

    const sx = (i: number) => pad.l + (i / Math.max(sliceLen - 1, 1)) * innerW;
    const sy = (v: number) =>
      pad.t + (1 - (v - (minY - padY)) / (maxY + padY - (minY - padY) || 1)) * innerH;

    const pathsInner = names.map((name) => {
      const arr = series[name] ?? [];
      const d = Array.from({ length: sliceLen }, (_, i) => {
        const x = sx(i);
        const y = sy(arr[i] ?? 0);
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      }).join(" ");
      return { name, d, color: ASSET_COLORS[name] ?? "#94a3b8" };
    });

    return {
      paths: pathsInner,
      legend: names,
      meta: { w, h, sliceLen },
    };
  }, [dates, series]);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {subtitle ? <p className="mt-1 text-xs text-slate-400">{subtitle}</p> : null}
        </div>
        <div className="flex flex-wrap gap-3 text-[11px] text-slate-400">
          {legend.map((name) => (
            <span key={name} className="inline-flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: ASSET_COLORS[name] ?? "#94a3b8" }}
              />
              {name}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${meta.w} ${meta.h}`} className="h-auto w-full" role="img">
        <rect x="0" y="0" width={meta.w} height={meta.h} fill="rgba(15,23,42,0.35)" rx="12" />
        {!paths.length ? (
          <text x={meta.w / 2} y={meta.h / 2} textAnchor="middle" fill="#94a3b8" fontSize="13">
            No series to plot
          </text>
        ) : null}
        {paths.map((p) => (
          <path key={p.name} d={p.d} fill="none" stroke={p.color} strokeWidth={2} strokeLinejoin="round" />
        ))}
        <text
          x={meta.w / 2}
          y={meta.h - 12}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="11"
          fontFamily="var(--font-geist-sans), system-ui"
        >
          Time
        </text>
      </svg>
      <p className="mt-2 text-[11px] text-slate-500">
        Latest point:{" "}
        <span className="text-slate-300">
          {dates[meta.sliceLen - 1] ?? dates[dates.length - 1] ?? "—"}
        </span>
      </p>
    </div>
  );
}
