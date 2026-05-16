"""Independent model validation pass (loads data; separate from pytest).

Use after material model changes or before publishing a report.  Complements
unit tests with end-to-end constraint and finite-metric checks.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.settings import OPTIMAL_PORTFOLIO_KEY, TRADING_DAYS
from src.data.loaders import load_market_panel
from src.governance.drift import assess_drift, fingerprint_recent_panel, load_baseline
from src.governance.independent_checks import run_documentation_and_config_audit
from src.strategic.allocation.engine import build_strategic_allocations


def main() -> int:
    run_documentation_and_config_audit()
    panel = load_market_panel()
    risk_free = float((1.0 + panel.risk_free_daily.mean()) ** TRADING_DAYS - 1.0)
    imputed = panel.em_result.imputed_fraction if panel.em_result else None

    outputs, _diag = build_strategic_allocations(
        panel.completed_returns,
        panel.features,
        risk_free_rate=risk_free,
        imputed_fraction=imputed,
    )
    opt = outputs[OPTIMAL_PORTFOLIO_KEY]
    if opt.optimization_method != "tangency_max_sharpe":
        print(f"FAIL: expected tangency_max_sharpe, got {opt.optimization_method}")
        return 1
    if not math.isfinite(opt.sharpe) or not math.isfinite(opt.expected_volatility):
        print("FAIL: non-finite strategic metrics")
        return 1

    fp = fingerprint_recent_panel(panel.completed_returns, tail=504)
    drift = assess_drift(fp, load_baseline())
    if drift.get("status") == "invalid_baseline":
        print("WARN: drift baseline invalid JSON")
    elif drift.get("status") == "alert":
        print("WARN: drift alert:", drift.get("violations"))

    print("Independent validation OK (tangency Optimal + finite metrics; drift status=%s)" % drift.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
