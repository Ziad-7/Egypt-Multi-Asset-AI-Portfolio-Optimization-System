from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _post_pytest_checks() -> None:
    from src.governance.independent_checks import run_documentation_and_config_audit
    from src.governance.metadata import collect_run_metadata
    from src.config.settings import OPTIMAL_PORTFOLIO_KEY, PROFILE_BOUNDS, STRATEGIC_PROFILES

    run_documentation_and_config_audit()
    meta = collect_run_metadata()
    for key in ("generated_at_utc", "engine_semantic_version", "random_seed_used"):
        if key not in meta:
            raise RuntimeError(f"run_metadata missing {key}")
    if OPTIMAL_PORTFOLIO_KEY not in STRATEGIC_PROFILES:
        raise RuntimeError("STRATEGIC_PROFILES must include OPTIMAL_PORTFOLIO_KEY")
    if OPTIMAL_PORTFOLIO_KEY not in PROFILE_BOUNDS:
        raise RuntimeError("PROFILE_BOUNDS must include optimal profile")


def main() -> int:
    print("Running institutional quant validation suite...")
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print("Validation failed.")
        return proc.returncode
    try:
        _post_pytest_checks()
    except Exception as exc:
        print(f"Post-pytest validation failed: {exc}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
