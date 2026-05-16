from __future__ import annotations

import pandas as pd


def latest_technical_snapshot(df: pd.DataFrame) -> dict:
    row = df.dropna().iloc[-1]
    snapshot = {}
    for col in ["rsi", "macd_hist", "dist_to_ma5", "bb_pb", "bb_bandwidth", "rolling_volatility"]:
        if col in row.index:
            snapshot[col] = float(row[col])
    return snapshot
