# Egypt Multi-Asset AI Portfolio Optimization System

Formal, research-oriented portfolio intelligence engine for the Egyptian market, designed to produce auditable strategic allocations, tactical overlays, risk diagnostics, and benchmarked backtests.

## Overview

The system combines two coordinated layers:

- **Strategic Layer (`src/strategic/`)**  
  Builds capital-market assumptions, computes constrained portfolio weights across investor profiles, evaluates the efficient frontier, and runs Monte Carlo risk simulations.
- **Tactical Layer (`src/tactical/`)**  
  Produces short-horizon buy/hold/sell signals using market regime detection, momentum, and classifier ensembles with explicit confidence and holdout evaluation diagnostics.

Final decisions are fused in `src/portfolio/layer_interaction.py` and published through `src/api/service.py`.

## Investable Universe

The strategic optimizer and Monte Carlo simulation include all five core assets:

- `TBills`
- `EGX30`
- `EGX100`
- `EgyptiansRealEstateFund`
- `Gold`

Profile constraints enforce strictly positive lower bounds for each asset so no core asset is assigned zero strategic weight.

## Core Capabilities

- Data loading, quality checks, and EM-based panel completion for staggered asset histories.
- Strategic optimization: one **Optimal** portfolio — **tangency (long-only maximum Sharpe)** on the mean–variance frontier, identical in construction to the `maximum_sharpe` reference point in diagnostics (mandate `PROFILE_BOUNDS` are advisory for this profile only).
- Efficient frontier diagnostics and random portfolio cloud generation.
- Parametric (Student-t) and block-bootstrap Monte Carlo simulation.
- Tactical signal generation with regime awareness and volatility-targeted position sizing.
- Classification evaluation reporting with:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix
  - Per-class metrics
  - Model-risk gates: class-imbalance fallback, fold F1 stability, calibration and ensemble-disagreement dampeners; optional panel-QA confidence scaling (does not alter raw inputs)
- Intelligence report `metadata.run_metadata` (UTC timestamp, dependency versions, random seed, optional git SHA) and `model_risk_controls` summary
- Quarterly-rebalanced backtesting with transaction costs.
- Walk-forward backtesting with strict anti-leakage chronology:
  - evaluation panel clipped to **`BACKTEST_START_DATE` … `BACKTEST_END_DATE`** in `src/config/settings.py` (default **2026-01-01** through **2026-05-31**; training uses prior history for walk-forward)
  - training window: `[start, rebalance_date - 1]`
  - test window: `(rebalance_date, next_rebalance_date]`
  - explicit leakage checks persisted in output JSON
- Benchmark comparison versus:
  - `EGX30`
  - `TBills`
  - `EGX_TBills_60_40` (local policy benchmark)
  - `SP500`
- Automated report and visualization generation.

## Institutional Methodology Controls

- **Signed methodology & operations** (see `docs/METHODOLOGY.md`, `docs/RUNBOOK.md`, `docs/COMPLIANCE.md`)
- **Formal risk limits** — Declared in `src/config/risk_limits.py` and echoed under `metadata.institutional_risk_limits` in each JSON report.
- **Distribution drift monitoring** — Optional `config/drift_baseline.json` (create with `python3 scripts/snapshot_drift_baseline.py`). Each run publishes `metadata.drift_assessment` vs that baseline (no data files are modified).
- **Walk-forward OOS segments** — Backtest JSON includes `walk_forward_oos` with per-rebalance segment Sharpe and returns plus a summary (longer out-of-sample evidence without changing feeds).
- **Independent validation** — `python3 scripts/independent_validation.py` end-to-end constraint and finiteness checks (use before external distribution).
- **No-leakage design**
  - Strategic optimization and tactical inference are recomputed at each rebalance date using only information available at that point in time.
  - Backtest payload publishes `leakage_checks` with `train_end < test_start` assertions.
- **Constraint policy**
  - Profile-level asset bounds (`PROFILE_BOUNDS`) are enforced in the live optimizer path.
  - Joint equity cap (`EGX_EQUITY_CAP`) for `EGX30 + EGX100` is enforced for every profile.
  - Constraint diagnostics are embedded in each profile's risk metrics.
- **Risk-control policy**
  - Pre-trade guardrails reduce risk when rolling realized volatility breaches target-volatility tolerance or trailing drawdown breaches profile tolerance.
  - Stress scenarios (equity selloff, devaluation shock, rate spike) are reported per profile.
- **Data provenance**
  - Report includes per-asset provenance (`raw`, `synthetic_feed`, `mostly_imputed`), imputation fractions, and quality flags (stale series, zero-return concentration, gap metrics).

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Run the full pipeline (report + plots):

```bash
python3 -m src.main --out outputs/intelligence_report.json --plots-dir outputs/plots
```

Run without plot generation and without benchmark download:

```bash
python3 -m src.main --no-plots --no-backtest
```

Run institutional validation checks:

```bash
python3 scripts/validate_quant.py
python3 scripts/independent_validation.py
```

## Output Artifacts

- `outputs/intelligence_report.json` (includes `run_metadata`, `institutional_risk_limits`, `drift_assessment`, `documentation` paths, and backtest `walk_forward_oos` when the backtest runs)
- `outputs/plots/strategic_allocations.png`
- `outputs/plots/risk_contributions.png`
- `outputs/plots/efficient_frontier.png`
- `outputs/plots/asset_correlation_heatmap.png`
- `outputs/plots/tactical_overview.png`
- `outputs/plots/backtest_vs_benchmarks.png`
- `outputs/plots/backtest_drawdowns.png`

## Repository Structure

- `src/config/`: global settings and portfolio constraints
- `src/data/`: loaders, transformations, macro feeds, imputation
- `src/strategic/`: forecasts, optimization, risk models, Monte Carlo
- `src/tactical/`: classifiers, momentum, regime detection, signal engine
- `src/simulation/`: backtesting engine
- `src/visualization/`: report plots
- `docs/`: technical audit and documentation

## Technical Audit

The audit and remediation log is available at [`docs/TECHNICAL_AUDIT.md`](docs/TECHNICAL_AUDIT.md).
