"""Quarterly-rebalanced multi-asset backtest with transaction costs.

Improvements vs. the legacy implementation:

* Honors the README's quarterly rebalancing schedule -- the previous
  implementation held weights flat through the entire window.
* Applies a configurable round-trip transaction cost on every
  rebalance, charged proportionally to gross turnover.
* Anchors the benchmark to the *active* aligned panel window so
  cross-asset returns are real rather than NaN-filled with zeros.
* Adds an EGP-relative composite benchmark (60 % EGX30 / 40 %
  T-Bills) alongside the legacy S&P 500 cross-reference; an FX-adjusted
  S&P 500 series in EGP is the closer comparator for a domestic
  portfolio.
* Tries Yahoo Finance for the S&P 500 series; degrades gracefully when
  the network is unavailable, so the offline pipeline still produces
  a complete report.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.api.schemas import StrategicAllocationOutput
from src.config.settings import (
    BACKTEST_END_DATE,
    BACKTEST_START_DATE,
    BACKTEST_REBALANCE_FREQ,
    BACKTEST_TRANSACTION_COST_BPS,
    DEFAULT_RISK_FREE_RATE,
    EGX_EQUITY_CAP,
    OPTIMAL_PORTFOLIO_KEY,
    PROFILE_BOUNDS,
    STRATEGIC_PROFILES,
    TRADING_DAYS,
)
from src.strategic.allocation.engine import build_strategic_allocations
from src.strategic.risk_models.risk_engine import compute_portfolio_risk
from src.tactical.signals.engine import build_tactical_signal

logger = logging.getLogger(__name__)


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def _performance_metrics(daily_returns: pd.Series, risk_free_rate: float) -> Dict[str, float]:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    cumulative = (1.0 + daily_returns).cumprod()
    years = max(len(daily_returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    total_return = float(cumulative.iloc[-1] - 1.0)
    cagr = float(cumulative.iloc[-1] ** (1 / years) - 1)
    vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((daily_returns.mean() * TRADING_DAYS - risk_free_rate) / (vol + 1e-12))
    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "volatility": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(_max_drawdown(cumulative), 4),
    }


def _download_sp500_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    try:
        import yfinance as yf  # local import: keep optional

        data = yf.download(
            "^GSPC",
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty or "Close" not in data:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        returns = close.pct_change().dropna()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns.index = pd.to_datetime(returns.index).tz_localize(None)
        returns.name = "SP500"
        return returns
    except Exception as exc:  # pragma: no cover - network-dependent
        logger.warning("S&P 500 download skipped (%s); benchmark omitted.", exc)
        return None


def _rebalance_simulation(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    transaction_cost_bps: float,
) -> pd.Series:
    """Drift weights between rebalance dates and pay transaction cost on rebalance."""
    if asset_returns.empty:
        return pd.Series(dtype=float)
    cost_rate = transaction_cost_bps / 10_000.0
    dates = asset_returns.index
    n_assets = asset_returns.shape[1]
    targets = target_weights.reindex(asset_returns.columns).fillna(0.0).values

    portfolio_daily = []
    weights = targets.copy()
    rebalance_set = set(pd.DatetimeIndex(rebalance_dates).normalize())

    for date_, row in asset_returns.iterrows():
        # Apply daily returns to drift the weights.
        gross_growth = 1.0 + row.values
        weights = weights * gross_growth
        gross = weights.sum()
        if gross <= 0:
            weights = targets.copy()
            portfolio_daily.append(0.0)
            continue
        portfolio_daily.append(float(gross - 1.0))
        weights = weights / gross
        # Rebalance: align to targets and pay cost on turnover.
        if date_.normalize() in rebalance_set:
            turnover = float(np.sum(np.abs(weights - targets)))
            cost = cost_rate * turnover
            portfolio_daily[-1] = portfolio_daily[-1] - cost
            weights = targets.copy()
    return pd.Series(portfolio_daily, index=dates, name="portfolio_return")


def _quarterly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if len(index) == 0:
        return index
    quarter_ends = pd.date_range(start=index.min(), end=index.max(), freq="QE")
    seen = set()
    rebalance_dates = []
    for q_end in quarter_ends:
        eligible = index[index >= q_end]
        if eligible.empty:
            continue
        candidate = eligible.min()
        if candidate in seen:
            continue
        seen.add(candidate)
        rebalance_dates.append(candidate)
    return pd.DatetimeIndex(rebalance_dates)


def _enforce_bounds(weights: pd.Series, profile: str) -> pd.Series:
    # Tangency Optimal: keep optimizer weights (long-only renormalize only).
    if profile == OPTIMAL_PORTFOLIO_KEY:
        w = weights.copy().astype(float).clip(lower=0.0, upper=1.0)
        total = float(w.sum())
        if total <= 1e-12:
            return pd.Series({a: 1.0 / max(len(w), 1) for a in w.index}, dtype=float)
        return w / total

    bounds = PROFILE_BOUNDS.get(profile, {})
    w = weights.copy().astype(float)
    for asset in w.index:
        lb, ub = bounds.get(asset, (0.0, 1.0))
        w[asset] = float(np.clip(w[asset], lb, ub))
    for _ in range(40):
        total = float(w.sum())
        diff = 1.0 - total
        if abs(diff) < 1e-8:
            break
        if diff > 0:
            room = {a: max(0.0, bounds.get(a, (0.0, 1.0))[1] - w[a]) for a in w.index}
        else:
            room = {a: max(0.0, w[a] - bounds.get(a, (0.0, 1.0))[0]) for a in w.index}
        room_sum = float(sum(room.values()))
        if room_sum <= 1e-12:
            break
        for asset in w.index:
            if room[asset] > 0:
                w[asset] += diff * (room[asset] / room_sum)
        for asset in w.index:
            lb, ub = bounds.get(asset, (0.0, 1.0))
            w[asset] = float(np.clip(w[asset], lb, ub))
    total = float(w.sum())
    if total <= 0:
        w = pd.Series({a: 1.0 / len(w) for a in w.index}, dtype=float)
    else:
        w = w / total

    eq_cap = EGX_EQUITY_CAP.get(profile)
    if eq_cap is not None and {"EGX30", "EGX100"}.intersection(set(w.index)):
        eq_assets = [a for a in ("EGX30", "EGX100") if a in w.index]
        eq_weight = float(w[eq_assets].sum())
        if eq_weight > float(eq_cap) + 1e-8:
            excess = eq_weight - float(eq_cap)
            scale = float(eq_cap) / max(eq_weight, 1e-12)
            for asset in eq_assets:
                w[asset] *= scale
            if "TBills" in w.index:
                w["TBills"] += excess
            else:
                non_eq = [a for a in w.index if a not in eq_assets]
                if non_eq:
                    for asset in non_eq:
                        w[asset] += excess / len(non_eq)
            w = w / max(float(w.sum()), 1e-12)
    return w


def _apply_tactical_overlay(
    weights: pd.Series,
    tactical: Dict[str, object],
) -> pd.Series:
    signal = int(tactical.get("signal", 0))
    confidence = float(tactical.get("confidence", 0.0))
    size = float(tactical.get("suggested_position_size", 1.0))
    if signal == 0 or confidence < 0.45:
        return weights
    delta = 0.03 * confidence * min(size, 1.0) * signal
    w = weights.copy()
    if "EGX30" not in w.index:
        return w
    w["EGX30"] = max(0.0, w["EGX30"] + delta)
    offset = w["EGX30"] - weights["EGX30"]
    if abs(offset) > 0:
        sinks = [a for a in ("TBills", "Gold") if a in w.index]
        if sinks:
            sink_total = float(sum(max(weights[s], 1e-9) for s in sinks))
            for s in sinks:
                w[s] = max(0.0, w[s] - offset * (max(weights[s], 1e-9) / sink_total))
    w = w / max(float(w.sum()), 1e-12)
    return w


def _apply_risk_controls(
    weights: pd.Series,
    train_returns: pd.DataFrame,
    profile: str,
) -> tuple[pd.Series, Dict[str, float | bool]]:
    cfg = STRATEGIC_PROFILES.get(profile, {})
    target_vol = float(cfg.get("target_vol", 0.12))
    max_dd_tol = float(cfg.get("max_drawdown_tol", 0.20))

    w = weights.copy()
    train_port = train_returns.mul(w.reindex(train_returns.columns).fillna(0.0), axis=1).sum(axis=1)
    realized_vol = float(train_port.tail(min(126, len(train_port))).std() * np.sqrt(TRADING_DAYS))
    eq = (1.0 + train_port).cumprod()
    dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0) if not eq.empty else 0.0

    vol_breach = realized_vol > target_vol * 1.25
    dd_breach = dd < -abs(max_dd_tol)
    if vol_breach or dd_breach:
        risky = [a for a in w.index if a != "TBills"]
        risk_scale = 0.75 if vol_breach else 0.85
        if dd_breach:
            risk_scale *= 0.80
        shifted = float(w[risky].sum() * (1.0 - risk_scale))
        for a in risky:
            w[a] *= risk_scale
        if "TBills" in w.index:
            w["TBills"] += shifted

    w = _enforce_bounds(w, profile)
    return w, {
        "vol_breach": vol_breach,
        "drawdown_breach": dd_breach,
        "train_realized_vol": realized_vol,
        "train_drawdown": dd,
    }


def _stress_test(weights: pd.Series) -> Dict[str, Dict[str, float]]:
    scenarios = {
        "equity_selloff_20pct": {"EGX30": -0.20, "EGX100": -0.22, "Gold": 0.03, "TBills": 0.01, "EgyptiansRealEstateFund": -0.12},
        "devaluation_shock": {"EGX30": -0.10, "EGX100": -0.11, "Gold": 0.12, "TBills": -0.02, "EgyptiansRealEstateFund": -0.08},
        "rate_spike": {"EGX30": -0.08, "EGX100": -0.09, "Gold": -0.05, "TBills": -0.03, "EgyptiansRealEstateFund": -0.10},
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, shock in scenarios.items():
        pnl = float(sum(weights.get(asset, 0.0) * shock.get(asset, 0.0) for asset in weights.index))
        out[name] = {"stressed_pnl_1d": pnl, "stressed_drawdown_proxy": min(0.0, pnl)}
    return out


def run_backtest(
    asset_returns: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    transaction_cost_bps: float = BACKTEST_TRANSACTION_COST_BPS,
    include_sp500: bool = True,
    fx_levels: pd.Series | None = None,
) -> Dict[str, object]:
    end_cap = pd.Timestamp(BACKTEST_END_DATE)
    start_floor = pd.Timestamp(BACKTEST_START_DATE)
    # Panel through end_cap (includes history before start_floor for walk-forward training).
    returns_panel = asset_returns.dropna(how="any").copy()
    returns_panel = returns_panel.loc[returns_panel.index <= end_cap]
    if returns_panel.empty:
        returns_panel = asset_returns.dropna(how="any").copy()
        returns_panel = returns_panel.loc[returns_panel.index <= end_cap]
    if returns_panel.empty:
        return {"error": "Backtest window has no observations after panel alignment."}

    returns_eval = returns_panel.loc[returns_panel.index >= start_floor]
    if returns_eval.empty:
        return {"error": "Backtest evaluation window has no observations in the requested date range."}

    rebalance_dates = _quarterly_rebalance_dates(returns_eval.index)
    if len(rebalance_dates) < 1:
        return {"error": "Backtest window has no quarterly rebalance anchor in the evaluation range."}
    benchmarks: Dict[str, pd.Series] = {}
    if "EGX30" in returns_eval.columns:
        benchmarks["EGX30"] = returns_eval["EGX30"]
    if "TBills" in returns_eval.columns:
        benchmarks["TBills"] = returns_eval["TBills"]
    if {"EGX30", "TBills"}.issubset(set(returns_eval.columns)):
        benchmarks["EGX_TBills_60_40"] = 0.6 * returns_eval["EGX30"] + 0.4 * returns_eval["TBills"]

    sp500_returns = None
    if include_sp500:
        sp500_returns = _download_sp500_returns(returns_eval.index.min(), returns_eval.index.max())
        if sp500_returns is not None:
            sp500_aligned = sp500_returns.reindex(returns_eval.index).dropna()
            if not sp500_aligned.empty:
                benchmarks["SP500"] = sp500_aligned

    # USD-real return series for each EGP asset (deflate by USD/EGP).  Used
    # only for reporting -- a long EGP position takes an FX hit on devaluation.
    fx_returns_aligned: pd.Series | None = None
    if fx_levels is not None and not fx_levels.empty:
        fx_aligned = fx_levels.reindex(returns_eval.index).ffill()
        fx_returns_aligned = fx_aligned.pct_change().fillna(0.0)

    profile_names = list(STRATEGIC_PROFILES.keys())
    portfolios: Dict[str, dict] = {}
    walk_returns: Dict[str, list[float]] = {p: [] for p in profile_names}
    walk_dates: list[pd.Timestamp] = []
    last_weights: Dict[str, pd.Series | None] = {p: None for p in profile_names}
    leakage_checks: list[Dict[str, object]] = []
    controls_log: Dict[str, list[Dict[str, object]]] = {p: [] for p in profile_names}
    tactical_log: Dict[str, list[Dict[str, object]]] = {p: [] for p in profile_names}
    segment_log: Dict[str, list[Dict[str, object]]] = {p: [] for p in profile_names}

    # Days before the first quarterly rebalance were previously omitted (loop uses
    # strictly test dates > rebalance), leaving a flat 0% portfolio — fill with an
    # inaugural OOS segment: train on history before eval start, hold through first
    # rebalance date (inclusive), then continue quarterly as before.
    eval_start = returns_eval.index.min()
    first_rebalance = rebalance_dates[0]
    inaugural_segment = returns_eval.loc[(returns_eval.index >= eval_start) & (returns_eval.index <= first_rebalance)]
    if not inaugural_segment.empty:
        train_inaug = returns_panel.loc[returns_panel.index < eval_start]
        if not train_inaug.empty and len(train_inaug) >= 120:
            if train_inaug.index.max() >= inaugural_segment.index.min():
                raise RuntimeError(
                    f"Leakage detected (inaugural): train_end {train_inaug.index.max()} "
                    f"overlaps test_start {inaugural_segment.index.min()}"
                )
            leakage_checks.append(
                {
                    "rebalance_date": "inaugural",
                    "train_end": str(train_inaug.index.max().date()),
                    "test_start": str(inaugural_segment.index.min().date()),
                    "overlap": False,
                }
            )
            train_features_inaug = {
                asset: df[df.index < eval_start].copy()
                for asset, df in features.items()
                if isinstance(df, pd.DataFrame) and not df.empty
            }
            strat_inaug, _diag_inaug = build_strategic_allocations(
                train_inaug,
                train_features_inaug,
                risk_free_rate=risk_free_rate,
                imputed_fraction=None,
            )
            for profile_name in profile_names:
                profile = strat_inaug[profile_name]
                w = pd.Series(profile.weights, dtype=float).reindex(returns_eval.columns).fillna(0.0)
                tactical = build_tactical_signal(
                    train_features_inaug,
                    focus_asset="EGX30",
                    strategic_weights=profile.weights,
                    next_rebalance_date=str(eval_start.date()),
                )
                tactical_dict = {
                    "signal": tactical.signal,
                    "confidence": tactical.confidence,
                    "suggested_position_size": tactical.suggested_position_size,
                }
                w = _apply_tactical_overlay(w, tactical_dict)
                w, control_state = _apply_risk_controls(w, train_inaug, profile_name)
                prev = last_weights[profile_name]
                turnover = float(np.sum(np.abs((w - prev).values))) if prev is not None else float(np.sum(np.abs(w.values)))
                cost = transaction_cost_bps / 10_000.0 * turnover
                seg_ret = inaugural_segment.mul(w.reindex(inaugural_segment.columns).fillna(0.0), axis=1).sum(axis=1).copy()
                if not seg_ret.empty:
                    seg_ret.iloc[0] -= cost
                    pm = _performance_metrics(seg_ret, risk_free_rate)
                    segment_log[profile_name].append(
                        {
                            "rebalance_date": "inaugural",
                            "train_end": str(train_inaug.index.max().date()),
                            "oos_days": int(len(seg_ret)),
                            "segment_total_return": round(float((1.0 + seg_ret).prod() - 1.0), 6),
                            "segment_sharpe": float(pm["sharpe"]),
                            "segment_volatility": float(pm["volatility"]),
                        }
                    )
                walk_returns[profile_name].extend(seg_ret.tolist())
                controls_log[profile_name].append(
                    {
                        "rebalance_date": "inaugural",
                        "turnover": turnover,
                        "transaction_cost": cost,
                        **control_state,
                    }
                )
                tactical_log[profile_name].append({"rebalance_date": "inaugural", **tactical_dict})
                last_weights[profile_name] = w
            walk_dates.extend(inaugural_segment.index.tolist())

    for idx in range(len(rebalance_dates)):
        start = rebalance_dates[idx]
        end = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else returns_eval.index.max()
        segment = returns_eval.loc[(returns_eval.index > start) & (returns_eval.index <= end)]
        if segment.empty:
            continue
        train = returns_panel.loc[returns_panel.index < start]
        if train.empty or len(train) < 120:
            continue
        if train.index.max() >= segment.index.min():
            raise RuntimeError(
                f"Leakage detected: train_end {train.index.max()} overlaps test_start {segment.index.min()}"
            )
        leakage_checks.append(
            {
                "rebalance_date": str(start.date()),
                "train_end": str(train.index.max().date()),
                "test_start": str(segment.index.min().date()),
                "overlap": False,
            }
        )
        train_features = {
            asset: df[df.index < start].copy()
            for asset, df in features.items()
            if isinstance(df, pd.DataFrame) and not df.empty
        }
        strat_profiles, _diag = build_strategic_allocations(
            train,
            train_features,
            risk_free_rate=risk_free_rate,
            imputed_fraction=None,
        )
        for profile_name in profile_names:
            profile = strat_profiles[profile_name]
            w = pd.Series(profile.weights, dtype=float).reindex(returns_eval.columns).fillna(0.0)
            tactical = build_tactical_signal(
                train_features,
                focus_asset="EGX30",
                strategic_weights=profile.weights,
                next_rebalance_date=str(start.date()),
            )
            tactical_dict = {
                "signal": tactical.signal,
                "confidence": tactical.confidence,
                "suggested_position_size": tactical.suggested_position_size,
            }
            w = _apply_tactical_overlay(w, tactical_dict)
            w, control_state = _apply_risk_controls(w, train, profile_name)
            prev = last_weights[profile_name]
            turnover = float(np.sum(np.abs((w - prev).values))) if prev is not None else float(np.sum(np.abs(w.values)))
            cost = transaction_cost_bps / 10_000.0 * turnover
            seg_ret = segment.mul(w.reindex(segment.columns).fillna(0.0), axis=1).sum(axis=1).copy()
            if not seg_ret.empty:
                seg_ret.iloc[0] -= cost
                pm = _performance_metrics(seg_ret, risk_free_rate)
                segment_log[profile_name].append(
                    {
                        "rebalance_date": str(start.date()),
                        "train_end": str(train.index.max().date()),
                        "oos_days": int(len(seg_ret)),
                        "segment_total_return": round(float((1.0 + seg_ret).prod() - 1.0), 6),
                        "segment_sharpe": float(pm["sharpe"]),
                        "segment_volatility": float(pm["volatility"]),
                    }
                )
            walk_returns[profile_name].extend(seg_ret.tolist())
            controls_log[profile_name].append(
                {
                    "rebalance_date": str(start.date()),
                    "turnover": turnover,
                    "transaction_cost": cost,
                    **control_state,
                }
            )
            tactical_log[profile_name].append({"rebalance_date": str(start.date()), **tactical_dict})
            last_weights[profile_name] = w
        walk_dates.extend(segment.index.tolist())

    if not walk_dates:
        return {"error": "Walk-forward loop produced no test observations."}

    walk_index = pd.DatetimeIndex(walk_dates)
    walk_frame = pd.DataFrame(
        {p: pd.Series(vals, index=walk_index[: len(vals)]) for p, vals in walk_returns.items()}
    ).sort_index()
    walk_frame = walk_frame[~walk_frame.index.duplicated(keep="last")]

    # Full evaluation calendar: cumulatives start at 0% on the first backtest date.
    # Strategy days without walk-forward weights yet contribute 0 daily return (flat).
    full_ix = returns_eval.index
    cumulative_df = pd.DataFrame(index=full_ix)
    drawdown_df = pd.DataFrame(index=full_ix)
    for name, series in benchmarks.items():
        r = series.reindex(full_ix).fillna(0.0)
        equity = (1.0 + r).cumprod()
        cumulative_df[name] = equity - 1.0
        drawdown_df[name] = equity / equity.cummax() - 1.0

    local_benchmark = benchmarks.get("EGX_TBills_60_40")
    for profile_name in profile_names:
        profile_daily = walk_frame[profile_name].dropna()
        risk = compute_portfolio_risk(
            profile_daily,
            weights=(last_weights[profile_name].to_dict() if last_weights[profile_name] is not None else {}),
            risk_free_rate=risk_free_rate,
            benchmark_returns=local_benchmark.reindex(profile_daily.index) if local_benchmark is not None else None,
        )
        usd_real_metrics = None
        if fx_returns_aligned is not None:
            # USD-real = (1 + r_egp) / (1 + r_fx) - 1; positive FX return means EGP is depreciating.
            usd_real_daily = (1.0 + profile_daily) / (1.0 + fx_returns_aligned) - 1.0
            usd_real_metrics = _performance_metrics(usd_real_daily, risk_free_rate=0.04)
        portfolios[profile_name] = {
            "weights": (
                {k: round(float(v), 6) for k, v in last_weights[profile_name].to_dict().items()}
                if last_weights[profile_name] is not None else {}
            ),
            "metrics": _performance_metrics(profile_daily, risk_free_rate),
            "risk_metrics": risk.to_dict(),
            "usd_real_metrics": usd_real_metrics,
            "risk_controls": controls_log[profile_name],
            "tactical_history": tactical_log[profile_name],
            "stress_tests": _stress_test(last_weights[profile_name] if last_weights[profile_name] is not None else pd.Series(dtype=float)),
        }
        if local_benchmark is not None:
            active = profile_daily - local_benchmark.reindex(profile_daily.index).fillna(0.0)
            tracking_err = float(active.std() * np.sqrt(TRADING_DAYS))
            info_ratio = float((active.mean() * TRADING_DAYS) / (tracking_err + 1e-12))
            portfolios[profile_name]["active_vs_local_60_40"] = {
                "active_return_annual": float(active.mean() * TRADING_DAYS),
                "tracking_error_annual": tracking_err,
                "information_ratio": info_ratio,
            }
        r_pf = walk_frame[profile_name].reindex(full_ix).fillna(0.0)
        equity_pf = (1.0 + r_pf).cumprod()
        cumulative_df[profile_name] = equity_pf - 1.0
        drawdown_df[profile_name] = equity_pf / equity_pf.cummax() - 1.0

    daily_cumulative = {col: cumulative_df[col].round(6).tolist() for col in cumulative_df.columns}
    daily_dates = [str(d.date()) for d in cumulative_df.index]
    # Expose aligned per-asset daily returns so the frontend can replay
    # custom slider mixes with the same calendar and transaction-cost model.
    daily_asset_returns_df = returns_eval.reindex(full_ix).fillna(0.0)
    daily_asset_returns = {
        col: daily_asset_returns_df[col].round(6).tolist() for col in daily_asset_returns_df.columns
    }
    daily_asset_return_assets = list(daily_asset_returns_df.columns)
    monthly = cumulative_df.resample("ME").last().dropna(how="any")
    cumulative_returns = {col: monthly[col].round(6).tolist() for col in monthly.columns}
    daily_drawdowns = {col: drawdown_df[col].round(6).tolist() for col in drawdown_df.columns}

    benchmark_payload = {}
    for name, series in benchmarks.items():
        benchmark_payload[name] = {
            "metrics": _performance_metrics(series.reindex(returns_eval.index).fillna(0.0), risk_free_rate),
        }

    def _summarize_oos_segments(segments: list[Dict[str, object]]) -> Dict[str, object]:
        if not segments:
            return {"n_segments": 0}
        sharpes = [float(s["segment_sharpe"]) for s in segments]
        rets = [float(s["segment_total_return"]) for s in segments]
        return {
            "n_segments": len(segments),
            "mean_segment_sharpe": round(float(np.mean(sharpes)), 4),
            "std_segment_sharpe": round(float(np.std(sharpes)), 4) if len(sharpes) > 1 else 0.0,
            "mean_segment_total_return": round(float(np.mean(rets)), 6),
            "min_segment_sharpe": round(float(np.min(sharpes)), 4),
            "max_segment_sharpe": round(float(np.max(sharpes)), 4),
        }

    walk_forward_oos = {
        p: {"segments": segment_log[p], "summary": _summarize_oos_segments(segment_log[p])} for p in profile_names
    }

    return {
        "date_range": {"start": str(returns_eval.index.min().date()), "end": str(returns_eval.index.max().date())},
        "evaluation_window": {"start": str(start_floor.date()), "end": str(end_cap.date())},
        "rebalance_dates": [str(d.date()) for d in rebalance_dates],
        "transaction_cost_bps": transaction_cost_bps,
        "leakage_checks": leakage_checks,
        "benchmarks": benchmark_payload,
        "portfolios": portfolios,
        "cumulative_returns": cumulative_returns,
        "dates": [str(d.date()) for d in monthly.index],
        "daily_cumulative_returns": daily_cumulative,
        "daily_drawdowns": daily_drawdowns,
        "daily_dates": daily_dates,
        "daily_asset_returns": daily_asset_returns,
        "daily_asset_return_assets": daily_asset_return_assets,
        "walk_forward_oos": walk_forward_oos,
    }
