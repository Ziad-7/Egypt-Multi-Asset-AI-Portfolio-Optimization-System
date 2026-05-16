"use client";

import type { MixAnalytics } from "@/lib/portfolioAnalytics";

export function CustomMixStats({ mix, riskFreeAnnual }: { mix: MixAnalytics; riskFreeAnnual: number | undefined }) {
  return (
    <div className="space-y-3 text-sm">
      <p className="text-xs text-[#5c6570]">
        Ex-ante metrics from report <strong className="text-[#1a1a1a]">annual means</strong> and{" "}
        <strong className="text-[#1a1a1a]">covariance</strong> — same inputs the optimizer uses for the efficient
        frontier.
      </p>
      <table className="w-full table-fixed border-collapse">
        <tbody className="divide-y divide-[#f0f2f5]">
          <tr>
            <td className="w-[58%] py-2 pr-3 align-top text-[#5c6570] break-words">Expected return (ann.)</td>
            <td className="py-2 text-right font-semibold tabular-nums whitespace-nowrap text-[#1a1a1a]">
              {(mix.expectedReturn * 100).toFixed(2)}%
            </td>
          </tr>
          <tr>
            <td className="py-2 pr-3 align-top text-[#5c6570] break-words">Volatility (ann.)</td>
            <td className="py-2 text-right font-semibold tabular-nums whitespace-nowrap text-[#1a1a1a]">
              {(mix.volatility * 100).toFixed(2)}%
            </td>
          </tr>
          <tr>
            <td className="py-2 pr-3 align-top text-[#5c6570] break-words">Sharpe (vs report RF)</td>
            <td className="py-2 text-right font-semibold tabular-nums whitespace-nowrap text-[#1a1a1a]">
              {mix.sharpe != null ? mix.sharpe.toFixed(3) : "—"}
            </td>
          </tr>
          {riskFreeAnnual != null ? (
            <tr>
              <td className="py-2 text-left text-[11px] leading-relaxed break-words text-[#7a8490]" colSpan={2}>
                Risk-free used: {(riskFreeAnnual * 100).toFixed(2)}% (from report metadata)
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}
