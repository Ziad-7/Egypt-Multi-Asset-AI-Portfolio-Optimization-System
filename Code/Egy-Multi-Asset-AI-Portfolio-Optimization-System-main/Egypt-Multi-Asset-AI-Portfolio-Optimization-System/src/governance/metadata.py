"""Reproducibility and audit metadata attached to each intelligence run."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict

from src.config.settings import PROJECT_ROOT, RANDOM_SEED, TRADING_DAYS

ENGINE_SEMANTIC_VERSION = "0.3.0"


def collect_run_metadata() -> Dict[str, object]:
    meta: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed_used": RANDOM_SEED,
        "trading_days_per_year": TRADING_DAYS,
        "engine_semantic_version": ENGINE_SEMANTIC_VERSION,
    }
    try:
        import numpy as np

        meta["numpy_version"] = np.__version__
    except Exception:
        meta["numpy_version"] = None
    try:
        import sklearn

        meta["sklearn_version"] = sklearn.__version__
    except Exception:
        meta["sklearn_version"] = None
    try:
        meta["git_commit_short"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        meta["git_commit_short"] = None
    return meta
