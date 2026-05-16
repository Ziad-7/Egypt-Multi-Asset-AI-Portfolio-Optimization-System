"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { analyzeMix, cumulativeSeriesApproxEqual, weightsApproximatelyEqual } from "@/lib/portfolioAnalytics";
import { checkHealth, apiBase, fetchLatestReport, runIntelligencePipeline } from "@/lib/api";
import type { IntelligenceReport, StrategicProfile } from "@/lib/intelligence";
import { isBacktestPayload, WEIGHT_REPLAY_SERIES_KEY } from "@/lib/intelligence";
import { WeightsBars } from "@/components/charts/WeightsBars";
import { FrontierChart } from "@/components/charts/FrontierChart";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { IbkrShell } from "@/components/ibkr/IbkrShell";
import { EngineModulesStrip, type PlatformTabId } from "@/components/ibkr/EngineModulesStrip";
import { DashboardCard } from "@/components/ibkr/DashboardCard";
import { AllocationDonut } from "@/components/ibkr/AllocationDonut";
import { PerformanceChart } from "@/components/ibkr/PerformanceChart";
import { DrawdownChart } from "@/components/ibkr/DrawdownChart";
import { AssetMovers } from "@/components/ibkr/AssetMovers";
import { KeyPortfolioStats } from "@/components/ibkr/KeyPortfolioStats";
import { UniverseStrip } from "@/components/ibkr/UniverseStrip";
import { WeightMixer } from "@/components/ibkr/WeightMixer";
import { CustomMixStats } from "@/components/ibkr/CustomMixStats";
import { TacticalModelEvaluation } from "@/components/ibkr/TacticalModelEvaluation";
import { simulateDriftRebalance } from "@/lib/customBacktest";
import { fuseLayersPreview } from "@/lib/layerFusion";
import { replayTacticalForWeights, type TacticalReplaySnapshot } from "@/lib/tacticalReplay";

function signalLabel(signal: number): string {
  if (signal === 1) return "Risk-on / add equity exposure";
  if (signal === -1) return "Risk-off / raise protection";
  return "Neutral — align to strategic anchor";
}

export function PlatformDashboard() {
  const [report, setReport] = useState<IntelligenceReport | null>(null);
  const [loading, setLoading] = useState<"idle" | "latest" | "run">("idle");
  const [error, setError] = useState<string | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [includeBacktest, setIncludeBacktest] = useState(true);
  const [profileKey, setProfileKey] = useState<string | null>(null);
  const [tab, setTab] = useState<PlatformTabId>("dashboard");
  /** When null, slider mix matches the active profile exactly (see `displayWeights`). */
  const [manualWeights, setManualWeights] = useState<Record<string, number> | null>(null);

  const refreshHealth = useCallback(async () => {
    setApiOk(await checkHealth());
  }, []);

  const handleModuleSelect = useCallback((next: PlatformTabId, anchor?: string) => {
    setTab(next);
    if (anchor) {
      requestAnimationFrame(() => {
        document.getElementById(anchor)?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const profiles = report?.strategic_profiles ?? {};
  const selected =
    profileKey && profiles[profileKey]
      ? profileKey
      : Object.keys(profiles).includes("Optimal")
        ? "Optimal"
        : (report?.metadata?.selected_profile as string | undefined) ?? Object.keys(profiles)[0] ?? null;

  const activeProfile: StrategicProfile | null = selected ? profiles[selected] ?? null : null;

  useEffect(() => {
    setManualWeights(null);
  }, [selected]);

  const loadLatest = async () => {
    setError(null);
    setLoading("latest");
    try {
      const data = await fetchLatestReport();
      setReport(data);
      const keys = Object.keys(data.strategic_profiles);
      const pk = keys.includes("Optimal") ? "Optimal" : ((data.metadata?.selected_profile as string) ?? keys[0]);
      setProfileKey(pk ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load cached report");
    } finally {
      setLoading("idle");
    }
  };

  const runFull = async () => {
    setError(null);
    setLoading("run");
    try {
      const data = await runIntelligencePipeline({
        includeBacktest: includeBacktest,
        save: true,
      });
      setReport(data);
      const keys = Object.keys(data.strategic_profiles);
      const pk = keys.includes("Optimal") ? "Optimal" : ((data.metadata?.selected_profile as string) ?? keys[0]);
      setProfileKey(pk ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Pipeline run failed");
    } finally {
      setLoading("idle");
    }
  };

  const backtest = report?.backtest;
  const backtestOk = backtest && isBacktestPayload(backtest);

  const dailyDates = backtestOk ? backtest.daily_dates ?? [] : [];
  const dailyCum = backtestOk ? backtest.daily_cumulative_returns ?? {} : {};
  const dailyDd = backtestOk ? backtest.daily_drawdowns ?? {} : {};

  const persona = report?.persona_guidance;

  const assetSummary = useMemo(() => {
    const dq = report?.metadata?.data_quality as { asset_summary?: Record<string, { annualized_mean?: number }> } | undefined;
    return dq?.asset_summary;
  }, [report]);

  const stats = useMemo(() => {
    if (!report) return null;
    const m = report.metadata ?? {};
    return {
      universe: (m.universe as string[] | undefined) ?? [],
      rf: m.risk_free_rate_annual as number | undefined,
      inflation: m.inflation_context as Record<string, unknown> | undefined,
      panel: m.panel_window as string[] | undefined,
    };
  }, [report]);

  const orderedAssets = useMemo(() => {
    const fromDiag = report?.strategic_diagnostics?.assets;
    if (fromDiag?.length) return fromDiag;
    return stats?.universe ?? [];
  }, [report?.strategic_diagnostics?.assets, stats?.universe]);

  const displayWeights = useMemo(() => {
    if (!activeProfile) return {};
    if (manualWeights) return manualWeights;
    return activeProfile.weights;
  }, [activeProfile, manualWeights]);

  const liveTactical = useMemo(() => {
    if (!report || !activeProfile) return null;
    const snap = report.tactical_signal.replay_snapshot as TacticalReplaySnapshot | undefined;
    const replayed = replayTacticalForWeights(snap, displayWeights, activeProfile.next_rebalance_date);
    if (replayed) {
      return {
        signal: replayed.signal,
        confidence: replayed.confidence,
        rebalance_window_active: replayed.rebalance_window_active,
        suggested_position_size: report.tactical_signal.suggested_position_size,
        usedReplay: replayed.usedReplay,
      };
    }
    return {
      signal: report.tactical_signal.signal,
      confidence: report.tactical_signal.confidence,
      rebalance_window_active: report.tactical_signal.rebalance_window_active,
      suggested_position_size: report.tactical_signal.suggested_position_size,
      usedReplay: false,
    };
  }, [report, activeProfile, displayWeights]);

  const liveFusion = useMemo(() => {
    if (!report || !activeProfile || !liveTactical) return null;
    const inflation = report.metadata?.inflation_context as
      | { regime?: string; yoy?: number; real_yield?: number }
      | undefined;
    return fuseLayersPreview({
      weights: displayWeights,
      strategicConfidence: activeProfile.confidence,
      tacticalSignal: liveTactical.signal,
      tacticalConfidence: liveTactical.confidence,
      suggestedPositionSize: liveTactical.suggested_position_size,
      inflation: inflation ?? undefined,
    });
  }, [report, activeProfile, liveTactical, displayWeights]);

  const isCustomMix = useMemo(() => {
    if (!activeProfile) return false;
    return !weightsApproximatelyEqual(displayWeights, activeProfile.weights);
  }, [activeProfile, displayWeights]);

  const mixAnalytics = useMemo(() => {
    if (!report || !orderedAssets.length) return null;
    return analyzeMix(
      displayWeights,
      orderedAssets,
      report.strategic_diagnostics.annual_mean_returns,
      report.strategic_diagnostics.annual_covariance,
      stats?.rf,
    );
  }, [report, displayWeights, orderedAssets, stats?.rf]);

  /**
   * Drift + rebalance + txn-cost replay for **current** `displayWeights` whenever the report includes
   * `daily_asset_returns` (not only when weights differ from the profile).
   */
  const weightReplay = useMemo(() => {
    if (!backtestOk || !backtest.daily_asset_returns || !backtest.daily_asset_return_assets?.length) {
      return null;
    }
    return simulateDriftRebalance(
      dailyDates,
      backtest.daily_asset_returns,
      backtest.daily_asset_return_assets,
      displayWeights,
      backtest.rebalance_dates ?? [],
      backtest.transaction_cost_bps ?? 0,
    );
  }, [backtestOk, backtest, dailyDates, displayWeights]);

  const savedProfileCumulative = useMemo(() => {
    if (!backtestOk) return null;
    if (selected && dailyCum[selected]?.length) return dailyCum[selected]!;
    if (dailyCum.Optimal?.length) return dailyCum.Optimal;
    return null;
  }, [backtestOk, dailyCum, selected]);

  const savedProfileDrawdown = useMemo(() => {
    if (!backtestOk) return null;
    if (selected && dailyDd[selected]?.length) return dailyDd[selected]!;
    if (dailyDd.Optimal?.length) return dailyDd.Optimal;
    return null;
  }, [backtestOk, dailyDd, selected]);

  const displayDailyCum = useMemo(() => {
    if (!backtestOk) return {} as Record<string, number[]>;
    const base = { ...dailyCum };
    if (weightReplay?.cumulative?.length) {
      const ref = savedProfileCumulative;
      const sameAsSaved = ref && cumulativeSeriesApproxEqual(weightReplay.cumulative, ref);
      if (isCustomMix || !sameAsSaved) {
        base[WEIGHT_REPLAY_SERIES_KEY] = weightReplay.cumulative;
      }
    }
    return base;
  }, [backtestOk, dailyCum, weightReplay, savedProfileCumulative, isCustomMix]);

  const displayDailyDd = useMemo(() => {
    if (!backtestOk) return {} as Record<string, number[]>;
    const base = { ...dailyDd };
    if (weightReplay?.drawdown?.length) {
      const ref = savedProfileDrawdown;
      const sameAsSaved =
        ref &&
        ref.length === weightReplay.drawdown.length &&
        cumulativeSeriesApproxEqual(weightReplay.drawdown, ref);
      if (isCustomMix || !sameAsSaved) {
        base[WEIGHT_REPLAY_SERIES_KEY] = weightReplay.drawdown;
      }
    }
    return base;
  }, [backtestOk, dailyDd, weightReplay, savedProfileDrawdown, isCustomMix]);

  const cumStatsKey =
    backtestOk && selected && backtest.cumulative_returns?.[selected] ? selected : "Optimal";

  const perfSubtitle = useMemo(() => {
    if (!backtestOk || !dailyDates.length) return "Walk-forward daily cumulative returns vs benchmarks.";
    return `${dailyDates[0]?.slice(0, 10) ?? "—"} — ${dailyDates[dailyDates.length - 1]?.slice(0, 10) ?? "—"} · ${Object.keys(dailyCum).length} series`;
  }, [backtestOk, dailyDates, dailyCum]);

  const filterBar = (
    <div className="flex flex-col gap-3 lg:flex-row lg:flex-wrap lg:items-center lg:justify-between">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-[#5c6570]">Profile:</span>
        {report
          ? Object.keys(profiles).map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => setProfileKey(name)}
                className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                  name === selected
                    ? "border-[#0f4c9e] bg-[#0f4c9e] text-white"
                    : "border-[#d8dee6] bg-white text-[#3d454d] hover:border-[#0f4c9e]/50"
                }`}
              >
                {name}
              </button>
            ))
          : null}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs text-[#5c6570]">
        <span className="hidden sm:inline">Benchmarks in chart:</span>
        {Object.keys(dailyCum)
          .slice(0, 6)
          .map((k) => (
            <span key={k} className="rounded border border-[#d8dee6] bg-white px-2 py-0.5 font-medium text-[#3d454d]">
              {k}
            </span>
          ))}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-[#e8ecf1] pt-3 lg:border-0 lg:pt-0">
        <label className="flex cursor-pointer items-center gap-2 text-xs text-[#3d454d]">
          <input
            type="checkbox"
            className="accent-[#0f4c9e]"
            checked={includeBacktest}
            onChange={(e) => setIncludeBacktest(e.target.checked)}
          />
          Backtest in run
        </label>
        <button
          type="button"
          onClick={() => void loadLatest()}
          disabled={loading !== "idle"}
          className="rounded border border-[#d8dee6] bg-white px-3 py-1.5 text-xs font-medium text-[#1a1a1a] hover:bg-[#f7f9fc] disabled:opacity-50"
        >
          {loading === "latest" ? "Loading…" : "Load cached"}
        </button>
        <button
          type="button"
          onClick={() => void runFull()}
          disabled={loading !== "idle" || apiOk === false}
          className="rounded bg-[#0f4c9e] px-3 py-1.5 text-xs font-semibold text-white hover:bg-[#0d4287] disabled:opacity-50"
        >
          {loading === "run" ? "Running…" : "Run & save"}
        </button>
        <span className="text-[11px] text-[#7a8490]">{apiBase()}</span>
      </div>
    </div>
  );

  return (
    <IbkrShell
      activeTab={tab}
      onTabChange={setTab}
      profileCount={Object.keys(profiles).length}
      apiConnected={apiOk}
      filterBar={filterBar}
    >
      {error ? (
        <div className="mb-4 rounded-lg border border-[#f5c6c2] bg-[#fef2f2] px-4 py-3 text-sm text-[#9b1c1c]">{error}</div>
      ) : null}

      {!report ? (
        <div className="rounded-lg border border-dashed border-[#d8dee6] bg-white py-16 text-center text-sm text-[#5c6570]">
          Load a cached intelligence report or run the pipeline to populate the dashboard.
        </div>
      ) : null}

      {report ? (
        <div className="mb-5">
          <EngineModulesStrip activeTab={tab} onSelect={handleModuleSelect} disabled={false} />
        </div>
      ) : null}

      {report && tab === "dashboard" && activeProfile ? (
        <div className="space-y-4">
          <div id="module-strategic" className="scroll-mt-28">
            <WeightMixer
            orderedAssets={orderedAssets}
            weights={displayWeights}
            onWeightsChange={(w) => setManualWeights(w)}
            onReset={() => setManualWeights(null)}
            profileName={activeProfile.profile}
            isCustom={isCustomMix}
            />
          </div>

          <div id="module-data" className="scroll-mt-28 rounded-lg border border-[#d8dee6] bg-[#f7f9fc] px-4 py-3">
            <p className="text-xs font-medium text-[#5c6570]">Core investable universe</p>
            <div className="mt-2">
              <UniverseStrip assets={stats?.universe ?? []} />
            </div>
            {stats?.panel ? (
              <p className="mt-2 text-[11px] text-[#7a8490]">
                Data panel: {stats.panel[0]} — {stats.panel[1]}
                {stats.rf != null ? ` · Risk-free (ann.): ${(stats.rf * 100).toFixed(2)}%` : null}
              </p>
            ) : null}
          </div>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
            <div className="xl:col-span-7">
              <DashboardCard
                title="Allocation"
                subtitle={`As of ${activeProfile.next_rebalance_date} · ${activeProfile.profile}${isCustomMix ? " · custom mix" : ""}`}
              >
                <AllocationDonut
                  weights={displayWeights}
                  profileName={isCustomMix ? `${activeProfile.profile} (custom)` : activeProfile.profile}
                  asOfLabel={isCustomMix ? "Interactive mix" : "Strategic target"}
                />
              </DashboardCard>
            </div>
            <div className="flex flex-col gap-4 xl:col-span-5">
              <DashboardCard
                title="Key statistics"
                subtitle={
                  isCustomMix
                    ? "Ex-ante (μ & Σ) + realized replay (daily backtest calendar)"
                    : "Backtest window (cumulative return)"
                }
              >
                {isCustomMix && mixAnalytics ? (
                  <div className="mb-4 border-b border-[#eef1f5] pb-4">
                    <CustomMixStats mix={mixAnalytics} riskFreeAnnual={stats?.rf} />
                  </div>
                ) : isCustomMix ? (
                  <p className="mb-4 text-sm text-[#5c6570]">
                    No usable <code className="rounded bg-[#eef1f5] px-1">annual_covariance</code> for ex-ante slider
                    metrics.
                  </p>
                ) : null}
                {isCustomMix && weightReplay ? (
                  <KeyPortfolioStats
                    backtest={backtest}
                    profileLabel={cumStatsKey}
                    customDailyCumulative={weightReplay.cumulative}
                    customDailyDates={dailyDates}
                    customTitle="Your weights (replay)"
                  />
                ) : isCustomMix ? (
                  <p className="text-sm text-[#5c6570]">
                    Cumulative replay needs <code className="rounded bg-[#eef1f5] px-1">daily_asset_returns</code> in the
                    report. Run <strong>Run &amp; save</strong> on an updated engine (backtest with this field).
                  </p>
                ) : (
                  <KeyPortfolioStats backtest={backtest} profileLabel={cumStatsKey} />
                )}
              </DashboardCard>
              <DashboardCard
                title="Portfolio movers"
                subtitle="Weights follow your sliders; returns from data-quality diagnostics"
              >
                <AssetMovers
                  profile={activeProfile}
                  assetSummary={assetSummary}
                  weightsOverride={isCustomMix ? displayWeights : undefined}
                />
              </DashboardCard>
            </div>
          </div>

          <DashboardCard
            id="module-backtest"
            title="Performance"
            subtitle="Daily cumulative return (%) — strategy vs benchmarks"
          >
            {isCustomMix && weightReplay ? (
              <p className="mb-3 rounded-md border border-[#f3e8ff] bg-[#faf5ff] px-3 py-2 text-xs text-[#5b21b6]">
                <strong>Replay (your weights)</strong> updates as you move sliders (same <strong>daily asset returns</strong>,{" "}
                <strong>rebalance dates</strong>, and <strong>transaction cost</strong> as the Python engine). The{" "}
                <strong>Optimal</strong> line stays the saved <strong>walk-forward</strong> benchmark for comparison.
              </p>
            ) : isCustomMix ? (
              <p className="mb-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-[#7c5e10]">
                Update the backend report to include <code className="rounded bg-white/80 px-1">daily_asset_returns</code>{" "}
                (re-run pipeline) to see a <strong>real</strong> custom cumulative path here.
              </p>
            ) : null}
            {!backtestOk ? (
              <p className="text-sm text-[#5c6570]">
                No backtest in this file. Run the pipeline with backtests enabled or open a full report.
              </p>
            ) : (
              <PerformanceChart dates={dailyDates} cumulativeBySeries={displayDailyCum} subtitle={perfSubtitle} />
            )}
          </DashboardCard>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <DashboardCard title="Drawdowns" subtitle="Daily underwater paths (%)">
              {!backtestOk || !Object.keys(displayDailyDd).length ? (
                <p className="text-sm text-[#5c6570]">No drawdown series available.</p>
              ) : (
                <DrawdownChart dates={dailyDates} drawdownsBySeries={displayDailyDd} />
              )}
            </DashboardCard>
            <DashboardCard
              title="Risk contributions"
              subtitle={
                isCustomMix && mixAnalytics?.sharpe != null
                  ? `Custom mix · Sharpe (μ/Σ) ${mixAnalytics.sharpe.toFixed(3)} · σ ${(mixAnalytics.volatility * 100).toFixed(2)}%`
                  : `${activeProfile.profile} · Sharpe ${activeProfile.sharpe.toFixed(3)} · σ ${(activeProfile.expected_volatility * 100).toFixed(2)}%`
              }
            >
              <WeightsBars
                weights={
                  isCustomMix && mixAnalytics ? mixAnalytics.riskContributions : activeProfile.risk_contributions
                }
                title="Risk contributions"
                showHeader={false}
              />
            </DashboardCard>
          </div>
        </div>
      ) : null}

      {report && tab === "signals" && activeProfile && liveTactical && liveFusion ? (
        <div className="space-y-4">
          <div className="rounded-lg border border-[#c7e9d6] bg-[#f0fdf6] px-4 py-3 text-sm text-[#14532d]">
            <p className="font-semibold text-[#166534]">Live strategic + tactical</p>
            <p className="mt-1 text-xs leading-relaxed text-[#15803d]">
              Fusion and tactical <strong>alignment</strong> update from the weights below (profile pills + sliders).
              Classifier metrics on the <strong>Tactical ML</strong> tab are from the last pipeline run.
              {!liveTactical.usedReplay ? (
                <>
                  {" "}
                  <span className="font-medium text-[#a16207]">
                    Re-run <strong>Run &amp; save</strong> on an updated engine to enable per-mix tactical replay
                    (<code className="rounded bg-white/80 px-1">replay_snapshot</code>).
                  </span>
                </>
              ) : null}
            </p>
          </div>

          <DashboardCard
            title="Strategic mix for this view"
            subtitle="Same controls as Dashboard — layer fusion follows these weights"
          >
            <WeightMixer
              orderedAssets={orderedAssets}
              weights={displayWeights}
              onWeightsChange={(w) => setManualWeights(w)}
              onReset={() => setManualWeights(null)}
              profileName={activeProfile.profile}
              isCustom={isCustomMix}
            />
          </DashboardCard>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#d8dee6] bg-white px-4 py-3 shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-[#5c6570]">Classifier health (last run)</p>
              <p className="mt-1 text-sm text-[#3d454d]">
                Holdout metrics and confusion matrix are on the{" "}
                <button
                  type="button"
                  onClick={() => setTab("models")}
                  className="font-semibold text-[#0f4c9e] underline decoration-[#0f4c9e]/30 underline-offset-2 hover:decoration-[#0f4c9e]"
                >
                  Tactical ML
                </button>{" "}
                tab.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setTab("models")}
              className="rounded-md bg-[#0f4c9e] px-3 py-2 text-xs font-semibold text-white hover:bg-[#0d4287]"
            >
              Open Tactical ML
            </button>
          </div>

          <div className="rounded-lg border border-[#d8dee6] bg-white shadow-[0_1px_3px_rgba(0,0,0,0.06)]">
            <div className="border-b border-[#e8ecf1] bg-[#f7f9fc] px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-[#5c6570]">Layer comparison</p>
              <p className="mt-0.5 text-[11px] text-[#7a8490]">Strategic vs tactical — responsive layout avoids cramped columns.</p>
            </div>
            <div className="divide-y divide-[#eef1f5]">
              {[
                {
                  dim: "Horizon",
                  strat: "Quarterly anchor; mean–variance and Monte Carlo simulation.",
                  tact: "Short horizon; market regime detection and classifier ensemble.",
                },
                {
                  dim: "Outputs",
                  strat: "Target weights, efficient frontier, and risk contributions.",
                  tact: "Discrete signal (−1 / 0 / +1), confidence, sizing, and rationale lines.",
                },
              ].map((row) => (
                <div key={row.dim} className="grid gap-0 lg:grid-cols-3">
                  <div className="border-b border-[#eef1f5] px-4 py-3 lg:border-b-0 lg:border-r lg:border-[#eef1f5]">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-[#5c6570]">Dimension</p>
                    <p className="mt-1 text-sm font-medium text-[#1a1a1a]">{row.dim}</p>
                  </div>
                  <div className="border-b border-[#eef1f5] px-4 py-3 lg:border-b-0 lg:border-r lg:border-[#eef1f5]">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-[#0f4c9e]">Strategic layer</p>
                    <p className="mt-1 text-sm leading-relaxed break-words text-[#3d454d]">{row.strat}</p>
                  </div>
                  <div className="px-4 py-3">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-[#b45309]">Tactical layer</p>
                    <p className="mt-1 text-sm leading-relaxed break-words text-[#3d454d]">{row.tact}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div id="module-tactical" className="grid scroll-mt-28 gap-4 lg:grid-cols-3">
            <DashboardCard title="Tactical signal (for your mix)">
              <p className="text-4xl font-bold tabular-nums text-[#0f4c9e]">
                {liveTactical.signal > 0 ? "+1" : liveTactical.signal < 0 ? "−1" : "0"}
              </p>
              <p className="mt-2 text-sm text-[#3d454d]">{signalLabel(liveTactical.signal)}</p>
              <p className="mt-3 text-xs text-[#5c6570]">
                Regime: <strong className="text-[#1a1a1a]">{report.tactical_signal.regime}</strong>
              </p>
            </DashboardCard>
            <DashboardCard title="Confidence & risk" className="lg:col-span-2">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <p className="text-xs text-[#5c6570]">Model confidence (replayed)</p>
                  <p className="text-2xl font-semibold text-[#1a1a1a]">
                    {(liveTactical.confidence * 100).toFixed(1)}%
                  </p>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-[#eef1f5]">
                    <div
                      className="h-full rounded-full bg-[#0f4c9e]"
                      style={{ width: `${Math.min(100, liveTactical.confidence * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="space-y-1 text-sm text-[#3d454d]">
                  <p>
                    Risk-off alert: <strong>{report.tactical_signal.risk_off_alert ? "Yes" : "No"}</strong>
                  </p>
                  <p>
                    Rebalance window:{" "}
                    <strong>{liveTactical.rebalance_window_active ? "Active" : "Closed"}</strong>
                  </p>
                  <p>
                    Vol target / realized:{" "}
                    <span className="tabular-nums">
                      {(report.tactical_signal.target_volatility * 100).toFixed(2)}% /{" "}
                      {(report.tactical_signal.realized_volatility * 100).toFixed(2)}%
                    </span>
                  </p>
                </div>
              </div>
              <div className="mt-4 border-t border-[#e8ecf1] pt-4">
                <p className="text-xs font-semibold text-[#5c6570]">Engine rationale (unchanged)</p>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-[#3d454d]">
                  {report.tactical_signal.rationale.map((line, i) => (
                    <li key={i}>{line}</li>
                  ))}
                </ul>
              </div>
            </DashboardCard>
          </div>

          <DashboardCard
            id="module-fusion"
            title="Layer fusion (for your mix)"
            subtitle="Strategic bias from weights above + tactical + inflation"
          >
            <p className="text-base font-semibold text-[#1a1a1a]">{liveFusion.action}</p>
            <p className="mt-2 text-sm text-[#3d454d]">{liveFusion.note}</p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full bg-[#eef1f5] px-3 py-1 text-[#3d454d]">
                Bias: <strong className="text-[#1a1a1a]">{liveFusion.strategic_bias}</strong>
              </span>
              <span className="rounded-full bg-[#eef1f5] px-3 py-1 text-[#3d454d]">
                Confidence:{" "}
                <strong className="text-[#1a1a1a]">{(liveFusion.confidence * 100).toFixed(1)}%</strong>
              </span>
            </div>
          </DashboardCard>

          {persona ? (
            <div className="grid gap-4 md:grid-cols-2">
              {Object.entries(persona).map(([key, text]) => (
                <DashboardCard key={key} title={key.replace(/_/g, " ")}>
                  <p className="text-sm text-[#3d454d]">{text}</p>
                </DashboardCard>
              ))}
            </div>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <WeightsBars
              weights={displayWeights}
              title={`Weights — ${activeProfile.profile}${isCustomMix ? " (your mix)" : ""}`}
              subtitle={`${activeProfile.optimization_method} · next ${activeProfile.next_rebalance_date}`}
            />
            {Object.keys(liveFusion.suggested_tilt ?? {}).length ? (
              <WeightsBars
                weights={Object.fromEntries(
                  Object.entries(liveFusion.suggested_tilt ?? {}).map(([k, v]) => [k, Math.abs(v)]),
                )}
                title="Tilt magnitudes (abs.)"
                subtitle="Live fusion for current weights"
              />
            ) : (
              <DashboardCard title="Tactical tilt">
                <p className="text-sm text-[#5c6570]">No tilt — neutral signal or low conviction.</p>
              </DashboardCard>
            )}
          </div>
        </div>
      ) : null}

      {report && tab === "models" ? (
        <div id="module-ml" className="space-y-4 scroll-mt-28">
          <div className="rounded-lg border border-[#dbeafe] bg-[#f0f7ff] px-4 py-3 text-sm text-[#1e3a5f]">
            <p className="font-semibold text-[#0f4c9e]">Tactical ML diagnostics</p>
            <p className="mt-1 text-xs leading-relaxed">
              Holdout evaluation for the classifier ensemble (logistic, SVM, optional XGBoost). Values refresh when you
              run the Python pipeline. Use <strong>Signals &amp; fusion</strong> to see how these outputs combine with
              your strategic weights in real time.
            </p>
          </div>
          <DashboardCard title="Classifier ensemble — holdout evaluation" subtitle="Gates signal, confidence, and rationale">
            <TacticalModelEvaluation modelEvaluation={report.tactical_signal.model_evaluation} />
          </DashboardCard>
        </div>
      ) : null}

      {report && tab === "diagnostics" ? (
        <div className="space-y-4">
          {isCustomMix && mixAnalytics ? (
            <p className="rounded-lg border border-[#f3e8ff] bg-[#faf5ff] px-3 py-2 text-xs text-[#5b21b6]">
              The <strong>Your mix</strong> marker uses your current slider weights with the same μ and Σ as the
              frontier chart.
            </p>
          ) : null}
          <FrontierChart
            diagnostics={report.strategic_diagnostics}
            highlightMix={
              isCustomMix && mixAnalytics
                ? { volatility: mixAnalytics.volatility, expectedReturn: mixAnalytics.expectedReturn }
                : undefined
            }
          />
          <CorrelationHeatmap diagnostics={report.strategic_diagnostics} />
        </div>
      ) : null}
    </IbkrShell>
  );
}
