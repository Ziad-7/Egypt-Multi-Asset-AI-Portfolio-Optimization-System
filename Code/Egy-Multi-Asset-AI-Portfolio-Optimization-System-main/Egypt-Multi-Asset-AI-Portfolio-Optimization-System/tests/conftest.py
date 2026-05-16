from __future__ import annotations

import pytest

from src.config.settings import TRADING_DAYS
from src.data.loaders import load_market_panel


@pytest.fixture(scope="session")
def market_panel():
    return load_market_panel()


@pytest.fixture(scope="session")
def risk_free_rate(market_panel) -> float:
    return float((1.0 + market_panel.risk_free_daily.mean()) ** TRADING_DAYS - 1.0)
