"""Reporting visualizations for the strategic + tactical engine.

Each chart uses the most statistically-appropriate primitive for the
metric it displays:

* Drawdowns -- line + ``fill_between`` underwater area.
* Cumulative returns -- percentage-return line chart.
* Asset correlations -- diverging palette centered exactly at 0.
* Strategic allocations -- 2x2 donut grid for point-in-time weights.
* Risk contributions -- horizontal stacked bars with a "= 100 %" tag.
* Efficient frontier -- scatter cloud + smooth interpolated frontier
  with reference portfolios as labelled stars.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.settings import OPTIMAL_PORTFOLIO_KEY

try:  # PCHIP delivers a smooth-without-overshoot frontier interpolation
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - scipy is in requirements
    PchipInterpolator = None


_THEME = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linestyle": ":",
    "grid.linewidth": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
}

_PROFILE_PALETTE = {
    "Optimal":      "#2ca02c",
    "Conservative": "#1f77b4",
    "Balanced":     "#2ca02c",
    "Growth":       "#ff7f0e",
    "Aggressive":   "#d62728",
}
_BENCHMARK_PALETTE = {
    "TBills":           "#8c564b",
    "EGX30":            "#1f77b4",
    "SP500":            "#9467bd",
}
_ASSET_PALETTE = {
    "EGX30":                   "#1f77b4",
    "EGX100":                  "#17becf",
    "Gold":                    "#ffbf00",
    "TBills":                  "#2ca02c",
    "EgyptiansRealEstateFund": "#d62728",
    "USDEGP":                  "#9467bd",
}


def build_visualizations(result: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_THEME):
        _plot_strategic_allocations(result["strategic_profiles"], output_dir / "strategic_allocations.png")
        _plot_tactical_overview(result["tactical_signal"], output_dir / "tactical_overview.png")
        if "backtest" in result and "cumulative_returns" in result["backtest"]:
            _plot_backtest_curves(result["backtest"], output_dir / "backtest_vs_benchmarks.png")
            _plot_drawdown_curves(result["backtest"], output_dir / "backtest_drawdowns.png")
        _plot_efficient_frontier(
            result["strategic_diagnostics"],
            result.get("strategic_profiles", {}),
            output_dir / "efficient_frontier.png",
        )
        _plot_correlation_heatmap(result["strategic_diagnostics"], output_dir / "asset_correlation_heatmap.png")
        _plot_risk_contributions(result["strategic_profiles"], output_dir / "risk_contributions.png")


def _plot_strategic_allocations(strategic_profiles: dict, out_path: Path) -> None:
    profiles = list(strategic_profiles.items())
    if not profiles:
        return
    n = len(profiles)
    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.5, 6.2))
        axes = [axes]
        suptitle = "Strategic Asset Allocation (Optimal Portfolio)"
    else:
        fig, ax_grid = plt.subplots(2, 2, figsize=(11, 9))
        axes = ax_grid.flat
        suptitle = "Strategic Asset Allocation Across Risk Profiles"
    for ax, (profile_name, profile) in zip(axes, profiles):
        weights = pd.Series(profile["weights"]).sort_values(ascending=False)
        weights = weights[weights > 0.001]
        colors = [_ASSET_PALETTE.get(a, "#888") for a in weights.index]
        wedges, _texts, autotexts = ax.pie(
            weights.values,
            labels=None,
            colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
            pctdistance=0.78,
            startangle=90,
            wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.5},
        )
        for txt in autotexts:
            txt.set_color("white")
            txt.set_fontweight("bold")
            txt.set_fontsize(10)
        ax.set_title(
            f"{profile_name} Portfolio\nExpected Volatility={profile['expected_volatility']:.1%}  Sharpe Ratio={profile['sharpe']:+.2f}",
            fontsize=11,
        )
        ax.text(0, 0, f"E[R]\n{profile['expected_return']:.1%}", ha="center", va="center", fontsize=11, fontweight="bold")
    # Shared legend below.
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=12, label=a)
               for a, c in _ASSET_PALETTE.items() if a in {asset for p in strategic_profiles.values() for asset in p["weights"]}]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tactical_overview(tactical_signal: dict, out_path: Path) -> None:
    signal_map = {-1: "Sell / Risk-Off", 0: "Neutral", 1: "Buy"}
    signal = tactical_signal["signal"]
    confidence = tactical_signal["confidence"]
    suggested = tactical_signal.get("suggested_position_size", 1.0)
    realized_vol = tactical_signal.get("realized_volatility", 0.0)
    target_vol = tactical_signal.get("target_volatility", 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(
        ["Signal", "Confidence", "Position Size", "Realized Vol", "Target Vol"],
        [signal, confidence, suggested, realized_vol, target_vol],
        color=["#e67e22", "#2e86de", "#16a085", "#b03a2e", "#8e44ad"],
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():+.2f}" if bar.get_width() < 0 else f"{bar.get_width():.2f}",
                va="center", fontsize=10)
    ax.set_xlim(min(-1.2, signal - 0.2), max(1.6, suggested + 0.4))
    ax.set_title(
        f"Tactical Signal Overview: {signal_map.get(signal, 'Unknown')} (Regime: {tactical_signal.get('regime', 'unknown')})"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_backtest_curves(backtest: dict, out_path: Path) -> None:
    # Prefer daily cumulative curves for a smooth "mountain" equity profile.
    if "daily_cumulative_returns" in backtest and "daily_dates" in backtest:
        dates = pd.to_datetime(backtest["daily_dates"])
        curves = pd.DataFrame(backtest["daily_cumulative_returns"], index=dates)
    else:
        dates = pd.to_datetime(backtest["dates"])
        curves = pd.DataFrame(backtest["cumulative_returns"], index=dates)

    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    primary = "Optimal" if "Optimal" in curves.columns else (
        "Balanced" if "Balanced" in curves.columns else (curves.columns[0] if len(curves.columns) else None)
    )
    for col in curves.columns:
        # Hide the near-linear TBills benchmark from this chart; it visually
        # dominates style without adding much informational value.
        if col == "TBills":
            continue
        is_profile = col in _PROFILE_PALETTE
        color = _PROFILE_PALETTE.get(col, _BENCHMARK_PALETTE.get(col, "#888"))
        if col == primary:
            ax.plot(curves.index, curves[col], label=col, color=color, linewidth=2.6, zorder=4)
            continue

        ls = "-" if is_profile else "--"
        lw = 1.4 if is_profile else 1.9
        alpha = 0.80 if is_profile else 0.95
        ax.plot(curves.index, curves[col], label=col, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=3)

    ax.axhline(0, color="#333", linewidth=0.8, alpha=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(f"Cumulative Backtest Performance vs S&P 500 (Primary Series: {primary})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="upper left", ncol=2)

    ew = backtest.get("evaluation_window")
    if ew and ew.get("start") and ew.get("end") and not curves.empty:
        x0 = pd.Timestamp(ew["start"])
        x1 = min(pd.Timestamp(ew["end"]), curves.index.max())
        ax.set_xlim(x0, x1)

    rebal = backtest.get("rebalance_dates", [])
    for d in rebal:
        ax.axvline(pd.Timestamp(d), color="#cccccc", linestyle=":", linewidth=0.75, zorder=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_drawdown_curves(backtest: dict, out_path: Path) -> None:
    if "daily_drawdowns" in backtest and "daily_dates" in backtest:
        dates = pd.to_datetime(backtest["daily_dates"])
        drawdown = pd.DataFrame(backtest["daily_drawdowns"], index=dates)
    else:
        dates = pd.to_datetime(backtest["dates"])
        equity = 1.0 + pd.DataFrame(backtest["cumulative_returns"], index=dates)
        drawdown = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for col in drawdown.columns:
        if col in _PROFILE_PALETTE:
            color = _PROFILE_PALETTE[col]
            alpha_fill, alpha_line = 0.18, 0.95
        else:
            color = _BENCHMARK_PALETTE.get(col, "#888")
            alpha_fill, alpha_line = 0.10, 0.90
        ax.fill_between(drawdown.index, drawdown[col].values, 0.0,
                        where=(drawdown[col].values < 0), color=color, alpha=alpha_fill, linewidth=0)
        ax.plot(drawdown.index, drawdown[col].values, label=col, color=color, linewidth=1.4, alpha=alpha_line)
        # Max-drawdown annotation
        if not drawdown[col].empty:
            min_idx = drawdown[col].idxmin()
            min_val = float(drawdown[col].min())
            if min_val < -0.005:
                ax.annotate(f"{col}: {min_val:.1%}",
                            xy=(min_idx, min_val), xytext=(8, -8), textcoords="offset points",
                            fontsize=8, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Backtest Drawdown Profile (Underwater Curves)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", ncol=2)
    ew = backtest.get("evaluation_window")
    if ew and ew.get("start") and ew.get("end") and not drawdown.empty:
        x0 = pd.Timestamp(ew["start"])
        x1 = min(pd.Timestamp(ew["end"]), drawdown.index.max())
        ax.set_xlim(x0, x1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_efficient_frontier(diagnostics: dict, strategic_profiles: dict, out_path: Path) -> None:
    cloud = diagnostics["random_cloud"]
    line = diagnostics["frontier_line"]
    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        cloud["volatilities"], cloud["returns"],
        c=cloud["sharpes"], cmap="viridis", s=10, alpha=0.35, label="Monte Carlo portfolios",
    )

    vols = np.array(line["volatilities"], dtype=float)
    rets = np.array(line["returns"], dtype=float)
    if PchipInterpolator is not None and len(vols) >= 4:
        order = np.argsort(vols)
        vols_s, rets_s = vols[order], rets[order]
        # Drop duplicate vols for monotone interpolation.
        uniq_mask = np.concatenate([[True], np.diff(vols_s) > 1e-9])
        vols_s, rets_s = vols_s[uniq_mask], rets_s[uniq_mask]
        if len(vols_s) >= 4:
            interp = PchipInterpolator(vols_s, rets_s, extrapolate=False)
            xs = np.linspace(vols_s.min(), vols_s.max(), 200)
            ax.plot(xs, interp(xs), color="crimson", linewidth=2.4, label="Efficient frontier")
        else:
            ax.plot(vols, rets, color="crimson", linewidth=2.0, label="Efficient frontier")
    else:
        ax.plot(vols, rets, color="crimson", linewidth=2.0, label="Efficient frontier")

    refs = diagnostics.get("reference_portfolios", {})
    palette = {"minimum_variance": "navy", "maximum_sharpe": "gold", "equal_risk_contribution": "darkgreen"}
    for key, port in refs.items():
        legend_label = "Optimal (tangency)" if key == "maximum_sharpe" else key.replace("_", " ").title()
        annotate_label = legend_label
        ax.scatter(
            [port["volatility"]], [port["expected_return"]],
            color=palette.get(key, "black"), marker="*", s=320,
            edgecolor="white", linewidth=1.2, zorder=5,
            label=legend_label,
        )
        ax.annotate(
            annotate_label,
            xy=(port["volatility"], port["expected_return"]),
            xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold",
        )

    # Optional: overlay non-tangency strategic profiles (none today besides Optimal).
    for profile_name, profile in strategic_profiles.items():
        if profile_name == OPTIMAL_PORTFOLIO_KEY:
            continue
        x = float(profile.get("expected_volatility", 0.0))
        y = float(profile.get("expected_return", 0.0))
        color = _PROFILE_PALETTE.get(profile_name, "black")
        ax.scatter([x], [y], marker="D", s=78, color=color, edgecolor="white", linewidth=1.0, zorder=6)
        ax.annotate(
            profile_name,
            xy=(x, y),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )

    ax.set_title("Efficient Frontier: Annualized Expected Return vs Volatility")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right")
    plt.colorbar(scatter, ax=ax, label="Sharpe ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_correlation_heatmap(diagnostics: dict, out_path: Path) -> None:
    corr = pd.DataFrame(diagnostics["correlation_matrix"]).dropna(how="all").dropna(axis=1, how="all")
    arr = corr.values
    fig, ax = plt.subplots(figsize=(8, 6.8))
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(arr, cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Asset Correlation Matrix (EM-Imputed Return Panel)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if abs(value) > 0.55 else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_risk_contributions(strategic_profiles: dict, out_path: Path) -> None:
    rc_df = pd.DataFrame({name: profile.get("risk_contributions", {}) for name, profile in strategic_profiles.items()}).T
    if rc_df.empty:
        return
    rc_df = rc_df.fillna(0.0)
    # Order: defensives left, risky right -- improves readability.
    asset_order = [a for a in ["TBills", "Gold", "EGX30", "EGX100", "EgyptiansRealEstateFund", "USDEGP"] if a in rc_df.columns]
    rc_df = rc_df[asset_order]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    pos_base = np.zeros(len(rc_df))
    neg_base = np.zeros(len(rc_df))
    profiles = list(rc_df.index)
    y_pos = np.arange(len(profiles))
    for asset in rc_df.columns:
        values = rc_df[asset].values
        left = np.where(values >= 0, pos_base, neg_base)
        bars = ax.barh(
            y_pos,
            values,
            left=left,
            label=asset,
            color=_ASSET_PALETTE.get(asset, "#888"),
            edgecolor="white",
            linewidth=1.2,
            height=0.62,
        )
        for i, (bar, v) in enumerate(zip(bars, values)):
            if abs(v) >= 0.05:
                txt_color = "white"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:+.0%}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=9,
                    fontweight="bold",
                )
        pos_base = np.where(values >= 0, pos_base + values, pos_base)
        neg_base = np.where(values < 0, neg_base + values, neg_base)

    totals = rc_df.sum(axis=1).values
    for i, total in enumerate(totals):
        ax.text(1.005, i, f"net = {total:.0%}", va="center", fontsize=10, fontweight="bold", color="#444")

    min_x = min(float(np.min(neg_base)), float(np.min(rc_df.values)), -0.02)
    max_x = max(float(np.max(pos_base)), float(np.max(rc_df.values)), 1.02)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(profiles)
    ax.set_xlim(min_x * 1.15, max_x * 1.15)
    ax.axvline(0.0, color="#333", linewidth=0.9)
    ax.axvline(1.0, color="#777", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Portfolio Risk Contribution by Asset (Variance Share, with Hedge Effects)")
    ax.set_xlabel("Component contribution to total variance")
    ax.legend(title="Asset", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
