# Methodology: Egypt Multi-Asset AI Portfolio Optimization System

**Document type:** Technical methodology (research / systems).  
**Version:** Aligns with `ENGINE_SEMANTIC_VERSION` in `src/governance/metadata.py`.  
**Approval:** Maintain a change log in version control; formal sign-off is an organizational process outside this repository.

## 1. Purpose

Produce auditable **strategic** weights (long-term allocation), **tactical** overlays (short-horizon signal), **risk diagnostics**, and **walk-forward backtests** for an Egyptian multi-asset universe, with explicit limitations where histories are staggered or imputed.

## 2. Scope and limitations

- **In scope:** Five core EGP assets (EGX30, EGX100, Gold, T-Bills, Egyptian real-estate fund), optional USD/EGP hedge sleeve flag, quarterly strategic rebalance assumption, transaction cost model in backtest.
- **Out of scope:** Custodian mandates, liquidity gates, tax, Islamic structuring, and live order execution.
- **Data:** Raw vendor files under `data/final_data/` are not rewritten by this engine. The panel may be **EM-completed** for staggered inception dates; imputation fractions and provenance are surfaced in `metadata.data_quality`. Users must treat imputed regions as **model-derived**, not exchange-traded ground truth.

## 3. Layer architecture

1. **Data:** Load, winsorize where configured, build `completed_returns` and per-asset feature frames (returns, lags, technicals, labels).
2. **Strategic:** Forecast expected returns, build covariance. **Optimal** is the **tangency portfolio** (long-only maximum Sharpe, no per-asset mandate bounds), matching `reference_portfolios.maximum_sharpe`. Monte Carlo and other reference points (min variance, ERC) support diagnostics. Mandate-style bounds in settings remain for reporting comparison only for Optimal.
3. **Tactical:** Regime detector, momentum, classifier ensemble with time-ordered folds, confidence dampeners (imbalance, fold stability, calibration, disagreement), optional panel-QA scaling.
4. **Fusion:** Inflation-aware tilt matrix in `layer_interaction.py`.
5. **Backtest:** Walk-forward: train strictly before each rebalance date; apply tactical overlay and risk controls; log leakage checks and segment-level OOS metrics.

## 4. Anti-leakage policy

- **Strategic / tactical training:** Uses only rows with index **strictly before** the rebalance date for that segment.
- **Backtest:** Asserts `train_end < test_start` per segment; violations raise.
- **Reporting:** Headline performance from the walk-forward path, not from in-sample optimization on the full evaluation window.

## 5. Risk limits (declarative)

Formal constants and narrative: `src/config/risk_limits.py`.  
Enforcement: optimizer bounds + EGX equity cap; backtest `_apply_risk_controls` and stress scenarios; Monte Carlo target drawdown tied to profile tolerance.

## 6. Validation

- **Unit / integration:** `pytest`, `scripts/validate_quant.py` (includes documentation presence checks).
- **Independent pass:** `scripts/independent_validation.py` (constraints + finite metrics + drift vs baseline if present).
- **Drift:** Optional `config/drift_baseline.json` from `scripts/snapshot_drift_baseline.py`; compare last-504-day fingerprint to baseline in each intelligence run.

## 7. Reproducibility

- Global RNG seed: `RANDOM_SEED` in `src/config/settings.py`.
- Each JSON report includes `metadata.run_metadata` (timestamp, versions, seed, optional git SHA).

## 8. Change control

Material changes to optimization objectives, imputation defaults, or backtest chronology require: updated tests, refreshed drift baseline if fingerprints are part of governance, and an entry in git history describing the rationale.
