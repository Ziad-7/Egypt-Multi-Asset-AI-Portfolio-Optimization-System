# Operational runbook

## Standard production of a report

1. Ensure `data/final_data/` contains the expected CSV inputs (see `README.md`).
2. From the project root:
   ```bash
   python3 -m src.main --out outputs/intelligence_report.json --plots-dir outputs/plots
   ```
3. Archive `outputs/intelligence_report.json`, plots, and note `metadata.run_metadata.git_commit_short` (if non-null).

## Validation gates

```bash
python3 -m pytest -q
python3 scripts/validate_quant.py
```

Before external distribution, run:

```bash
python3 scripts/independent_validation.py
```

Exit code `0` required.

## Drift monitoring

1. After a **blessed** data + code state, snapshot the fingerprint:
   ```bash
   python3 scripts/snapshot_drift_baseline.py
   ```
2. Commit `config/drift_baseline.json` (or store in secure artifact store per policy).
3. Subsequent runs embed `metadata.drift_assessment`; treat `status: alert` as a review trigger (not an automatic halt unless policy says so).

## Interpreting tactical confidence

- Low confidence after dampeners may indicate class imbalance, unstable folds, miscalibration, ensemble disagreement, or panel QA flags—not necessarily a “broken” model.
- Prefer reading `tactical_signal.model_evaluation` alongside `signal` and `suggested_position_size`.

## Escalation

| Symptom | Action |
|--------|--------|
| `drift_assessment.status == alert` | Review data pipeline dates, imputation warnings, and recent macro shocks; consider re-snapshot baseline after approved change. |
| Backtest `error` key present | Check date window, minimum history, network (S&P optional); file issue with traceback. |
| Constraint `RuntimeError` from strategic engine | Bounds or equity cap infeasible for current CMA; adjust `PROFILE_BOUNDS` / cap in settings with governance approval. |

## What this system does not do

No live trading, no order routing, no suitability or Know-Your-Client—see `docs/COMPLIANCE.md`.
