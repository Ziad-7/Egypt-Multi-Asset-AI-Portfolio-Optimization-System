"use client";

import { useId, useMemo } from "react";
import type { CloudBundle, RefPortfolio, StrategicDiagnostics } from "@/lib/intelligence";
import { sampleIndices } from "@/lib/intelligence";

type Point = { x: number; y: number; kind: "cloud" | "frontier" | "ref"; label?: string };

function bundleToPoints(bundle: CloudBundle | undefined, kind: Point["kind"]): Point[] {
  if (!bundle?.returns?.length || !bundle?.volatilities?.length) return [];
  const n = Math.min(bundle.returns.length, bundle.volatilities.length);
  const idx = sampleIndices(n, kind === "cloud" ? 900 : n);
  return idx.map((i) => ({
    x: bundle.volatilities[i] ?? 0,
    y: bundle.returns[i] ?? 0,
    kind,
  }));
}

function refPoints(refs: StrategicDiagnostics["reference_portfolios"]): Point[] {
  if (!refs) return [];
  return Object.entries(refs).map(([label, rp]) => ({
    x: (rp as RefPortfolio).volatility,
    y: (rp as RefPortfolio).expected_return,
    kind: "ref" as const,
    label,
  }));
}

function linearTicks(min: number, max: number, count: number): number[] {
  if (count < 2 || max <= min) return [min, max];
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

function formatPct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Robust percentile on sorted array, p in [0,100]. */
function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0] ?? 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const a = sorted[lo] ?? 0;
  const b = sorted[hi] ?? a;
  return a + (b - a) * (idx - lo);
}

/**
 * Tight plot bounds so the cloud + frontier fill the frame (avoid huge empty margins from outliers).
 * Percentile window on all scatter points, then expand to always include frontier, refs, and highlight.
 */
function computeTightView(
  cloud: Point[],
  frontier: Point[],
  refs: Point[],
  highlight: { volatility: number; expectedReturn: number } | undefined,
  padFraction = 0.05,
): { minX: number; maxX: number; minY: number; maxY: number } {
  const xs = cloud.map((p) => p.x);
  const ys = cloud.map((p) => p.y);
  const mustX: number[] = [...frontier.map((p) => p.x), ...refs.map((p) => p.x)];
  const mustY: number[] = [...frontier.map((p) => p.y), ...refs.map((p) => p.y)];
  if (highlight) {
    mustX.push(highlight.volatility);
    mustY.push(highlight.expectedReturn);
  }

  if (!xs.length && mustX.length) {
    const minX = Math.min(...mustX);
    const maxX = Math.max(...mustX);
    const minY = Math.min(...mustY);
    const maxY = Math.max(...mustY);
    const px = (maxX - minX) * padFraction || 0.01;
    const py = (maxY - minY) * padFraction || 0.01;
    return { minX: minX - px, maxX: maxX + px, minY: minY - py, maxY: maxY + py };
  }
  if (!xs.length) {
    return { minX: 0, maxX: 0.2, minY: 0, maxY: 0.2 };
  }

  const sx = [...xs].sort((a, b) => a - b);
  const sy = [...ys].sort((a, b) => a - b);
  let loX = percentile(sx, 2.5);
  let hiX = percentile(sx, 97.5);
  let loY = percentile(sy, 2.5);
  let hiY = percentile(sy, 97.5);

  if (mustX.length) {
    loX = Math.min(loX, ...mustX);
    hiX = Math.max(hiX, ...mustX);
  }
  if (mustY.length) {
    loY = Math.min(loY, ...mustY);
    hiY = Math.max(hiY, ...mustY);
  }

  const spanX = hiX - loX || 0.02;
  const spanY = hiY - loY || 0.02;
  const px = Math.max(spanX * padFraction, spanX * 0.02);
  const py = Math.max(spanY * padFraction, spanY * 0.02);

  return {
    minX: Math.max(0, loX - px),
    maxX: hiX + px,
    minY: Math.max(0, loY - py),
    maxY: hiY + py,
  };
}

/** Avoid overlapping callouts by nudging labels vertically. */
function layoutRefCallouts(
  refs: Point[],
  sx: (x: number) => number,
  sy: (y: number) => number,
  plotTop: number,
): { px: number; py: number; lx: number; ly: number; label: string }[] {
  const sorted = [...refs].sort((a, b) => sx(a.x) - sx(b.x) || sy(a.y) - sy(b.y));
  const out: { px: number; py: number; lx: number; ly: number; label: string }[] = [];
  const occupied: { lx: number; ly: number }[] = [];

  for (const p of sorted) {
    if (!p.label) continue;
    const px = sx(p.x);
    const py = sy(p.y);
    let lx = px + 14;
    let ly = py - 18;
    if (ly < plotTop + 4) ly = py + 22;
    for (let pass = 0; pass < 12; pass++) {
      const clash = occupied.some((o) => Math.abs(o.lx - lx) < 100 && Math.abs(o.ly - ly) < 15);
      if (!clash) break;
      ly += 16;
    }
    occupied.push({ lx, ly });
    out.push({ px, py, lx, ly, label: p.label.replace(/_/g, " ") });
  }
  return out;
}

export function FrontierChart({
  diagnostics,
  highlightMix,
}: {
  diagnostics: StrategicDiagnostics;
  highlightMix?: { volatility: number; expectedReturn: number };
}) {
  const uid = useId().replace(/:/g, "");

  const { points, view } = useMemo(() => {
    const cloud = bundleToPoints(diagnostics.random_cloud, "cloud");
    const lineBundle: CloudBundle | undefined = diagnostics.frontier_line
      ? {
          returns: diagnostics.frontier_line.returns ?? [],
          volatilities: diagnostics.frontier_line.volatilities ?? [],
        }
      : undefined;
    const frontier = bundleToPoints(lineBundle, "frontier");
    const refs = refPoints(diagnostics.reference_portfolios);
    const hasAny =
      cloud.length > 0 || frontier.length > 0 || refs.length > 0 || Boolean(highlightMix);
    if (!hasAny) {
      return {
        points: { cloud, frontier, refs },
        view: { minX: 0, maxX: 0.2, minY: 0, maxY: 0.2 },
      };
    }
    const view = computeTightView(cloud, frontier, refs, highlightMix, 0.055);
    return {
      points: { cloud, frontier, refs },
      view,
    };
  }, [diagnostics, highlightMix]);

  const W = 920;
  const H = 480;
  const pad = { l: 62, r: 36, t: 32, b: 58 };

  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - view.minX) / (view.maxX - view.minX || 1)) * plotW;
  const sy = (y: number) => pad.t + (1 - (y - view.minY) / (view.maxY - view.minY || 1)) * plotH;

  const frontierSorted = [...points.frontier].sort((a, b) => a.x - b.x);
  const linePath =
    frontierSorted.length > 1
      ? frontierSorted
          .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`)
          .join(" ")
      : "";

  const xTicks = useMemo(
    () => linearTicks(view.minX, view.maxX, 6),
    [view.minX, view.maxX],
  );
  const yTicks = useMemo(
    () => linearTicks(view.minY, view.maxY, 6),
    [view.minY, view.maxY],
  );

  const refLayouts = useMemo(
    () => layoutRefCallouts(points.refs, sx, sy, pad.t),
    [points.refs, view, pad.t, plotW, plotH],
  );

  const mixCx = highlightMix ? sx(highlightMix.volatility) : 0;
  const mixCy = highlightMix ? sy(highlightMix.expectedReturn) : 0;
  const mixLabel = useMemo(() => {
    if (!highlightMix) return null;
    let lx = mixCx + 16;
    let ly = mixCy - 20;
    if (ly < pad.t + 8) ly = mixCy + 26;
    if (lx > W - pad.r - 120) lx = mixCx - 120;
    return { lx, ly };
  }, [highlightMix, mixCx, mixCy, pad.t, W, pad.r]);

  return (
    <div className="rounded-lg border border-[#d0d7de] bg-white shadow-[0_1px_2px_rgba(15,23,42,0.06)]">
      <div className="border-b border-[#eef1f5] px-5 py-4">
        <h3 className="text-[15px] font-semibold tracking-tight text-[#0f172a]">Efficient frontier</h3>
        <p className="mt-1 max-w-2xl text-[13px] leading-snug text-[#64748b]">
          Axes auto-zoom to the bulk of the simulation (2.5–97.5% band) so the frontier and cloud fill the plot;
          reference points and your mix are always included.{" "}
          <span className="text-[#475569]">“Your mix”</span> follows interactive weights (annualized σ vs μ).
        </p>
      </div>
      <div className="px-4 pb-4 pt-3 sm:px-5">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto w-full min-h-[320px] max-h-[560px]"
          role="img"
          aria-label="Risk-return efficient frontier chart"
        >
          <defs>
            <filter id={`shadow-${uid}`} x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="1" stdDeviation="1.2" floodOpacity="0.12" />
            </filter>
            <clipPath id={`plot-clip-${uid}`}>
              <rect x={pad.l} y={pad.t} width={plotW} height={plotH} rx={4} />
            </clipPath>
          </defs>

          {/* Plot background */}
          <rect
            x={pad.l}
            y={pad.t}
            width={plotW}
            height={plotH}
            fill="#fafbfc"
            stroke="#e8ecf1"
            strokeWidth={1}
            rx={4}
          />

          {/* Grid + Y tick labels */}
          {yTicks.map((yt, i) => {
            const y = sy(yt);
            return (
              <g key={`gy-${i}`}>
                <line
                  x1={pad.l}
                  y1={y}
                  x2={pad.l + plotW}
                  y2={y}
                  stroke="#e8ecf1"
                  strokeWidth={1}
                  strokeDasharray={i === 0 ? undefined : "4 4"}
                />
                <text
                  x={pad.l - 8}
                  y={y + 4}
                  textAnchor="end"
                  fill="#64748b"
                  fontSize={11}
                  fontFamily="system-ui, sans-serif"
                >
                  {formatPct(yt, yt < 0.1 ? 1 : 0)}
                </text>
              </g>
            );
          })}

          {/* X grid + tick labels */}
          {xTicks.map((xt, i) => {
            const x = sx(xt);
            return (
              <g key={`gx-${i}`}>
                <line
                  x1={x}
                  y1={pad.t}
                  x2={x}
                  y2={pad.t + plotH}
                  stroke="#eef1f5"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                />
                <text
                  x={x}
                  y={pad.t + plotH + 18}
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize={11}
                  fontFamily="system-ui, sans-serif"
                >
                  {formatPct(xt, xt < 0.1 ? 1 : 0)}
                </text>
              </g>
            );
          })}

          <g clipPath={`url(#plot-clip-${uid})`}>
            {/* Monte Carlo cloud — larger dots so the mass reads clearly when axes are tight */}
            <g opacity={0.92}>
              {points.cloud.map((p, i) => (
                <circle
                  key={`c-${i}`}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={2.55}
                  fill="rgba(37, 99, 235, 0.38)"
                />
              ))}
            </g>

            {/* Frontier — bold stroke + light underlay for contrast on dense cloud */}
            {linePath ? (
              <>
                <path
                  d={linePath}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth={6}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  opacity={0.95}
                />
                <path
                  d={linePath}
                  fill="none"
                  stroke="#0b3d91"
                  strokeWidth={3.75}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  filter={`url(#shadow-${uid})`}
                />
              </>
            ) : null}
          </g>

          {/* Reference portfolios (labels may sit outside plot) */}
          {refLayouts.map(({ px, py, lx, ly, label }, i) => (
            <g key={`ref-${i}`}>
              <line
                x1={px}
                y1={py}
                x2={lx - 4}
                y2={ly + 5}
                stroke="#cbd5e1"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <circle cx={px} cy={py} r={7.5} fill="#ea580c" stroke="#fff" strokeWidth={2.5} />
              <rect
                x={lx - 4}
                y={ly - 11}
                width={Math.max(88, label.length * 6.2 + 14)}
                height={18}
                rx={4}
                fill="#fff"
                stroke="#e2e8f0"
                strokeWidth={1}
              />
              <text
                x={lx + 4}
                y={ly + 3}
                fill="#0f172a"
                fontSize={10}
                fontWeight={600}
                fontFamily="system-ui, sans-serif"
              >
                {label}
              </text>
            </g>
          ))}

          {/* Your mix (unclipped so callout stays readable) */}
          {highlightMix && mixLabel ? (
            <g>
              <line
                x1={mixCx}
                y1={mixCy}
                x2={mixLabel.lx + 36}
                y2={mixLabel.ly + 2}
                stroke="#c084fc"
                strokeWidth={1.75}
                strokeDasharray="4 3"
                opacity={0.85}
              />
              <circle cx={mixCx} cy={mixCy} r={11} fill="#9333ea" stroke="#fff" strokeWidth={3} />
              <circle cx={mixCx} cy={mixCy} r={15} fill="none" stroke="#9333ea" strokeWidth={1.5} opacity={0.45} />
              <rect
                x={mixLabel.lx - 2}
                y={mixLabel.ly - 12}
                width={78}
                height={20}
                rx={4}
                fill="#faf5ff"
                stroke="#d8b4fe"
                strokeWidth={1}
              />
              <text
                x={mixLabel.lx + 6}
                y={mixLabel.ly + 3}
                fill="#6b21a8"
                fontSize={11}
                fontWeight={700}
                fontFamily="system-ui, sans-serif"
              >
                Your mix
              </text>
            </g>
          ) : null}

          {/* Axis titles */}
          <text
            x={pad.l + plotW / 2}
            y={H - 8}
            textAnchor="middle"
            fill="#475569"
            fontSize={11}
            fontWeight={600}
            fontFamily="system-ui, sans-serif"
          >
            Volatility (annualized)
          </text>
          <text
            x={14}
            y={pad.t + plotH / 2}
            fill="#475569"
            fontSize={11}
            fontWeight={600}
            fontFamily="system-ui, sans-serif"
            transform={`rotate(-90 14 ${pad.t + plotH / 2})`}
            textAnchor="middle"
          >
            Expected return (annualized)
          </text>
        </svg>

        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-[#eef1f5] pt-3 text-[12px] text-[#64748b]">
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-blue-400/70" />
            Random portfolio cloud
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-1 w-8 rounded-sm bg-[#0b3d91]" />
            Efficient frontier
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-[#ea580c] ring-2 ring-white" />
            Reference portfolios
          </span>
          {highlightMix ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#9333ea] ring-2 ring-[#e9d5ff]" />
              Your mix (interactive weights)
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
