"""Lightweight audits that do not require pytest (documentation and config integrity)."""
from __future__ import annotations

from pathlib import Path

from src.config.risk_limits import institutional_risk_limits_metadata, methodology_documentation_paths
from src.config.settings import PROJECT_ROOT


def run_documentation_and_config_audit() -> None:
    missing = []
    for _label, rel in methodology_documentation_paths():
        if not (PROJECT_ROOT / rel).is_file():
            missing.append(rel)
    if missing:
        raise RuntimeError(f"Required documentation missing: {missing}")
    meta = institutional_risk_limits_metadata()
    if not meta.get("stress_scenarios"):
        raise RuntimeError("institutional_risk_limits_metadata empty")


def project_paths_exist(rel_paths: list[str]) -> bool:
    return all((PROJECT_ROOT / p).is_file() for p in rel_paths)
