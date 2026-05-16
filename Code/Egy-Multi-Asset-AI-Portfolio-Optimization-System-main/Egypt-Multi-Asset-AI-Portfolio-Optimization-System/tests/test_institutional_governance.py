from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.risk_limits import institutional_risk_limits_metadata
from src.governance.drift import assess_drift, fingerprint_panel
from src.governance.independent_checks import run_documentation_and_config_audit


def test_documentation_and_risk_limits_audit():
    run_documentation_and_config_audit()
    meta = institutional_risk_limits_metadata()
    assert meta.get("max_portfolio_leverage") == 1.0
    assert meta.get("stress_scenarios")


def test_drift_assess_without_baseline():
    r = pd.DataFrame(
        {
            "EGX30": np.random.default_rng(1).normal(0.0003, 0.015, 120),
            "Gold": np.random.default_rng(2).normal(0.0002, 0.012, 120),
        }
    )
    fp = fingerprint_panel(r)
    out = assess_drift(fp, None)
    assert out["status"] == "no_baseline"


def test_drift_detects_large_mean_shift():
    base = fingerprint_panel(pd.DataFrame({"EGX30": np.full(100, 0.0005), "Gold": np.full(100, 0.0001)}))
    cur = fingerprint_panel(pd.DataFrame({"EGX30": np.full(100, 0.002), "Gold": np.full(100, 0.0001)}))
    baseline = {"fingerprint": base}
    out = assess_drift(cur, baseline)
    assert out["status"] in ("alert", "warning")
    assert out["violations"]
