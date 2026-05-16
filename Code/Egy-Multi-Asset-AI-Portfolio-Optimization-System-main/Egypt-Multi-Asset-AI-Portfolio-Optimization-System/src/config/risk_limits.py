"""Formal institutional risk limits referenced by reports and runbooks.

These are declarative policy constants.  Enforcement is implemented in
``strategic.allocation.engine`` (bounds + equity cap), ``simulation.backtest``
(``_apply_risk_controls``, stress scenarios), and Monte Carlo drawdown targets.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Long-only; no borrowing.
MAX_PORTFOLIO_LEVERAGE: float = 1.0

# Walk-forward training window: realized vol above target * factor triggers de-risking.
TRAIN_VOL_BREACH_FACTOR: float = 1.25

# Drawdown from training-window equity curve vs stated profile tolerance (backtest).
TRAIN_DRAWDOWN_BREACH_USES_PROFILE_TOL: bool = True

# Stress scenarios evaluated on each backtest endpoint (1-day proxy shocks).
STRESS_SCENARIOS_DOCUMENTED: Tuple[str, ...] = (
    "equity_selloff_20pct",
    "devaluation_shock",
    "rate_spike",
)

# Concentration: enforced per profile via PROFILE_BOUNDS and EGX_EQUITY_CAP in settings.
ENFORCED_CONCENTRATION_KEYS: Tuple[str, ...] = ("PROFILE_BOUNDS", "EGX_EQUITY_CAP")


def institutional_risk_limits_metadata() -> Dict[str, object]:
    """Serializable block for ``metadata.institutional_risk_limits`` in JSON reports."""
    return {
        "max_portfolio_leverage": MAX_PORTFOLIO_LEVERAGE,
        "train_vol_breach_factor": TRAIN_VOL_BREACH_FACTOR,
        "train_drawdown_uses_profile_tolerance": TRAIN_DRAWDOWN_BREACH_USES_PROFILE_TOL,
        "stress_scenarios": list(STRESS_SCENARIOS_DOCUMENTED),
        "concentration_enforcement": {
            "settings_keys": list(ENFORCED_CONCENTRATION_KEYS),
            "description": "Per-asset floors/ceilings and joint EGX30+EGX100 cap in optimizer and diagnostics.",
        },
        "operational_note": (
            "Pre-trade enterprise limits (custodian, mandate, liquidity) are out of scope for this codebase; "
            "document in external policy and map to PROFILE_BOUNDS as needed. "
            "The Optimal (tangency) strategic profile intentionally may not satisfy PROFILE_BOUNDS."
        ),
    }


def methodology_documentation_paths() -> List[Tuple[str, str]]:
    """Relative paths from project root; used for report metadata and audits."""
    return [
        ("methodology", "docs/METHODOLOGY.md"),
        ("runbook", "docs/RUNBOOK.md"),
        ("compliance", "docs/COMPLIANCE.md"),
    ]
