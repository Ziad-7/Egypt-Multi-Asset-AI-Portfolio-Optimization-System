"""Command-line entry point for the multi-layer portfolio engine."""
from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

from src.api.service import run_portfolio_intelligence
from src.config.settings import OPTIMAL_PORTFOLIO_KEY

warnings.filterwarnings("ignore")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-layer portfolio intelligence engine")
    parser.add_argument("--out", type=Path, default=Path("outputs/intelligence_report.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG visualization generation.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest (also avoids Yahoo Finance).")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)

    result = run_portfolio_intelligence(include_backtest=not args.no_backtest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(result), indent=2))

    if not args.no_plots:
        from src.visualization.report_plots import build_visualizations

        build_visualizations(result, args.plots_dir)

    print(f"Saved intelligence report to {args.out}")
    selected = result.get("metadata", {}).get("selected_profile", OPTIMAL_PORTFOLIO_KEY)
    print(json.dumps(_json_safe(result["layer_interaction"].get(selected, {})), indent=2))


if __name__ == "__main__":
    main()
