"""High-level orchestration for the multi-layer portfolio engine."""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict

import numpy as np

from src.config.risk_limits import institutional_risk_limits_metadata, methodology_documentation_paths
from src.config.settings import RANDOM_SEED, TRADING_DAYS
from src.governance.drift import assess_drift, fingerprint_recent_panel, load_baseline
from src.governance.metadata import collect_run_metadata
from src.data.loaders import load_market_panel
from src.data.macro_feeds import EgyptCPIFeed
from src.portfolio.layer_interaction import InflationContext, fuse_layers
from src.simulation.backtest import run_backtest
from src.strategic.allocation.engine import build_strategic_allocations
from src.tactical.signals.engine import build_tactical_signal

logger = logging.getLogger(__name__)


def _profile_selection_score(profile) -> float:
    """Risk-adjusted profile score used to choose the tactical anchor."""
    param = profile.monte_carlo.get("parametric", {}) if profile.monte_carlo else {}
    downside = float(param.get("downside_probability", 0.5))
    breach = float(param.get("target_breach_probability", 0.5))
    score = (
        float(profile.sharpe)
        + 0.20 * float(profile.confidence)
        - 0.65 * downside
        - 0.35 * breach
    )
    return float(score)


def _choose_anchor_profile(strategic_profiles: Dict[str, object]) -> tuple[str, object, Dict[str, float]]:
    scores = {name: _profile_selection_score(profile) for name, profile in strategic_profiles.items()}
    chosen_name = max(scores, key=scores.get)
    return chosen_name, strategic_profiles[chosen_name], scores


def _build_inflation_context(panel) -> InflationContext:
    cpi = EgyptCPIFeed()
    if panel.cpi_yoy.empty or panel.tbill_yield.empty:
        return InflationContext(regime="low", yoy=0.0, real_yield=0.0)
    today = panel.cpi_yoy.index.max()
    yoy = float(panel.cpi_yoy.loc[today])
    tbill = float(panel.tbill_yield.loc[today])
    return InflationContext(regime=cpi.classify_regime(today), yoy=yoy, real_yield=tbill - yoy)


def run_portfolio_intelligence(include_backtest: bool = True) -> Dict[str, object]:
    np.random.seed(RANDOM_SEED)

    panel = load_market_panel()
    risk_free_rate = float((1.0 + panel.risk_free_daily.mean()) ** TRADING_DAYS - 1.0)

    imputed_fraction = (panel.em_result.imputed_fraction if panel.em_result else None)

    strategic_profiles, strategic_diagnostics = build_strategic_allocations(
        panel.completed_returns,
        panel.features,
        risk_free_rate=risk_free_rate,
        imputed_fraction=imputed_fraction,
    )
    chosen_profile_name, strategic_anchor, profile_scores = _choose_anchor_profile(strategic_profiles)

    tactical = build_tactical_signal(
        panel.features,
        focus_asset="EGX30",
        strategic_weights=strategic_anchor.weights,
        next_rebalance_date=strategic_anchor.next_rebalance_date,
        panel_quality=panel.quality.to_dict(),
    )

    inflation_context = _build_inflation_context(panel)

    drift_fp = fingerprint_recent_panel(panel.completed_returns, tail=504)
    drift_assessment = assess_drift(drift_fp, load_baseline())

    interactions = {
        profile: asdict(fuse_layers(strat, tactical, inflation=inflation_context))
        for profile, strat in strategic_profiles.items()
    }

    backtest_payload: Dict[str, object] = {}
    if include_backtest:
        try:
            backtest_payload = run_backtest(
                panel.completed_returns[list(strategic_anchor.weights.keys())],
                panel.features,
                risk_free_rate=risk_free_rate,
                fx_levels=panel.fx_levels,
            )
        except Exception as exc:
            logger.exception("Backtest failed; continuing without it.")
            backtest_payload = {"error": str(exc)}

    report: Dict[str, object] = {
        "metadata": {
            "risk_free_rate_annual": round(risk_free_rate, 6),
            "trading_days": TRADING_DAYS,
            "universe": panel.universe,
            "selected_profile": chosen_profile_name,
            "profile_selection_scores": {k: round(v, 6) for k, v in profile_scores.items()},
            "panel_window": panel.quality.to_dict()["panel_window"],
            "aligned_panel_window": panel.quality.to_dict()["aligned_panel_window"],
            "data_quality": panel.quality.to_dict(),
            "inflation_context": inflation_context.to_dict(),
            "run_metadata": collect_run_metadata(),
            "institutional_risk_limits": institutional_risk_limits_metadata(),
            "drift_assessment": drift_assessment,
            "documentation": {label: path for label, path in methodology_documentation_paths()},
            "model_risk_controls": {
                "strategic": (
                    "Constrained frontier constructions; MC-scored Optimal pick; "
                    "constraint_diagnostics on each allocation."
                ),
                "tactical": (
                    "Time-ordered CV, class-imbalance gate, fold F1 stability and calibration dampeners, "
                    "ensemble disagreement penalty, panel-QA confidence scaling (no raw data mutation)."
                ),
                "backtest": "Walk-forward segments with persisted leakage checks.",
            },
            "fx_latest": (
                {"date": str(panel.fx_levels.index.max().date()),
                 "usdegp": round(float(panel.fx_levels.iloc[-1]), 4)}
                if not panel.fx_levels.empty else None
            ),
        },
        "strategic_profiles": {k: asdict(v) for k, v in strategic_profiles.items()},
        "tactical_signal": asdict(tactical),
        "layer_interaction": interactions,
        "strategic_diagnostics": strategic_diagnostics,
        "backtest": backtest_payload,
        "persona_guidance": {
            "strategic_investor": (
                "Use the strategic Optimal allocation as a quarterly anchor. "
                "Inflation regime adjusts the tactical tilt automatically."
            ),
            "tactical_trader": (
                "Read tactical_signal.signal alongside suggested_position_size; "
                "apply suggested_tilt only after confirming the rebalance window is open."
            ),
        },
    }
    return report
