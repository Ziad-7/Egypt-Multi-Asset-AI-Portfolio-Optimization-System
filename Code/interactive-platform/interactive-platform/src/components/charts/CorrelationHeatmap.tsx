"use client";

import type { StrategicDiagnostics } from "@/lib/intelligence";

function colorForCorr(v: number): string {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const a = t;
    return `rgba(45, 212, 191, ${0.12 + a * 0.55})`;
  }
  const a = -t;
  return `rgba(248, 113, 113, ${0.12 + a * 0.55})`;
}

export function CorrelationHeatmap({ diagnostics }: { diagnostics: StrategicDiagnostics }) {
  const matrix = diagnostics.correlation_matrix;
  if (!matrix) {
    return (
      <div className="rounded-lg border border-dashed border-[#d8dee6] bg-[#f7f9fc] p-6 text-sm text-[#5c6570]">
        Correlation matrix not available in this payload.
      </div>
    );
  }
  const assets = Object.keys(matrix).sort();

  return (
    <div className="rounded-lg border border-[#d8dee6] bg-white p-5 shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
      <h3 className="text-sm font-semibold text-[#1a1a1a]">Asset correlation</h3>
      <p className="mt-1 text-xs text-[#5c6570]">Symmetric matrix from the latest strategic diagnostics.</p>
      <div className="mt-4 overflow-x-auto">
        <table className="min-w-max border-separate border-spacing-px text-[10px] sm:text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-white p-1 text-left text-[#7a8490]" />
              {assets.map((a) => {
                const short = a.replace("EgyptiansRealEstateFund", "RE Fund");
                return (
                  <th
                    key={a}
                    className="max-w-[4.25rem] p-1 text-center font-medium leading-tight break-words text-[#3d454d]"
                    title={a}
                  >
                    {short}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {assets.map((row) => (
              <tr key={row}>
                <td
                  className="sticky left-0 z-10 whitespace-nowrap bg-white p-1 text-[10px] font-medium text-[#3d454d] sm:text-xs"
                  title={row}
                >
                  {row.replace("EgyptiansRealEstateFund", "RE Fund")}
                </td>
                {assets.map((col) => {
                  const v = matrix[row]?.[col] ?? 0;
                  return (
                    <td
                      key={`${row}-${col}`}
                      className="min-w-[2.25rem] px-1 py-1.5 text-center tabular-nums leading-none text-slate-900"
                      style={{ backgroundColor: colorForCorr(v) }}
                      title={`${row} vs ${col}: ${v.toFixed(3)}`}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
