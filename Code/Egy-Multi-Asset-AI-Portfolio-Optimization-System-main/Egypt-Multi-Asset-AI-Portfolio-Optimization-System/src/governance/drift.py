"""Distribution drift monitoring vs an optional frozen fingerprint baseline.

Does not modify data.  Baseline is produced offline via
``scripts/snapshot_drift_baseline.py`` and stored as JSON under ``config/``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config.settings import PROJECT_ROOT, TRADING_DAYS

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_PATH = PROJECT_ROOT / "config" / "drift_baseline.json"
REL_MEAN_ALERT = 0.50
REL_VOL_ALERT = 0.45
CORR_SHIFT_ALERT = 0.22


def fingerprint_panel(returns: pd.DataFrame) -> Dict[str, object]:
    """Annualized moments and EGX30 correlations from a return panel (any alignment)."""
    r = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    ann_mean = (r.mean() * TRADING_DAYS).round(6)
    ann_vol = (r.std() * np.sqrt(TRADING_DAYS)).round(6)
    per_asset = {
        str(col): {"mean_ann": float(ann_mean[col]), "vol_ann": float(ann_vol[col])}
        for col in r.columns
        if np.isfinite(ann_mean[col]) and np.isfinite(ann_vol[col])
    }
    corr_vs: Dict[str, float] = {}
    if "EGX30" in r.columns:
        ref = r["EGX30"]
        for col in r.columns:
            if col == "EGX30":
                continue
            pair = r[[col, "EGX30"]].dropna(how="any")
            if len(pair) < 40:
                continue
            corr_vs[str(col)] = round(float(pair[col].corr(pair["EGX30"])), 6)
    return {
        "schema_version": 1,
        "per_asset": per_asset,
        "corr_vs_egx30": corr_vs,
    }


def load_baseline(path: Path | None = None) -> Dict[str, object] | None:
    p = path or DEFAULT_BASELINE_PATH
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Drift baseline unreadable (%s): %s", p, exc)
        return None
    fp = raw.get("fingerprint")
    if not isinstance(fp, dict):
        return None
    return raw


def assess_drift(
    current_fp: Dict[str, object],
    baseline: Dict[str, object] | None,
) -> Dict[str, object]:
    if not baseline:
        return {
            "status": "no_baseline",
            "message": "No config/drift_baseline.json — run scripts/snapshot_drift_baseline.py and commit for drift monitoring.",
            "violations": [],
        }
    base_fp = baseline.get("fingerprint")
    if not isinstance(base_fp, dict):
        return {"status": "invalid_baseline", "violations": ["baseline.fingerprint missing"]}

    violations: List[Dict[str, object]] = []
    cur_pa = current_fp.get("per_asset") or {}
    base_pa = base_fp.get("per_asset") or {}
    if isinstance(cur_pa, dict) and isinstance(base_pa, dict):
        for asset, bstats in base_pa.items():
            if not isinstance(bstats, dict) or asset not in cur_pa:
                continue
            cstats = cur_pa[asset]
            if not isinstance(cstats, dict):
                continue
            bm, bv = bstats.get("mean_ann"), bstats.get("vol_ann")
            cm, cv = cstats.get("mean_ann"), cstats.get("vol_ann")
            if bm not in (None, 0) and cm is not None and np.isfinite(bm) and np.isfinite(cm):
                rel = abs(cm - bm) / max(abs(float(bm)), 1e-6)
                if rel > REL_MEAN_ALERT:
                    violations.append(
                        {"type": "mean_shift", "asset": asset, "relative_delta": round(rel, 4), "threshold": REL_MEAN_ALERT}
                    )
            if bv not in (None, 0) and cv is not None and np.isfinite(bv) and np.isfinite(cv):
                relv = abs(cv - bv) / max(abs(float(bv)), 1e-6)
                if relv > REL_VOL_ALERT:
                    violations.append(
                        {"type": "vol_shift", "asset": asset, "relative_delta": round(relv, 4), "threshold": REL_VOL_ALERT}
                    )

    cur_c = current_fp.get("corr_vs_egx30") or {}
    base_c = base_fp.get("corr_vs_egx30") or {}
    if isinstance(cur_c, dict) and isinstance(base_c, dict):
        for asset, bc in base_c.items():
            if asset not in cur_c or bc is None:
                continue
            cc = cur_c[asset]
            try:
                shift = abs(float(cc) - float(bc))
            except (TypeError, ValueError):
                continue
            if shift > CORR_SHIFT_ALERT:
                violations.append(
                    {
                        "type": "correlation_shift_vs_egx30",
                        "asset": asset,
                        "abs_delta": round(shift, 4),
                        "threshold": CORR_SHIFT_ALERT,
                    }
                )

    status = "ok"
    if violations:
        status = "alert" if any(v.get("type") in ("vol_shift", "mean_shift") for v in violations) else "warning"
    return {
        "status": status,
        "violations": violations,
        "baseline_generated_at": baseline.get("generated_at_utc"),
        "thresholds": {
            "relative_mean_ann": REL_MEAN_ALERT,
            "relative_vol_ann": REL_VOL_ALERT,
            "corr_vs_egx30_abs_delta": CORR_SHIFT_ALERT,
        },
    }


def fingerprint_recent_panel(completed_returns: pd.DataFrame, tail: int = 504) -> Dict[str, object]:
    """Use the last ``tail`` rows of the completed return panel (typ. ~2 trading years)."""
    window = completed_returns.tail(int(tail))
    return fingerprint_panel(window)
