from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StrategicAllocationOutput:
    profile: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe: float
    confidence: float
    rebalancing_note: str
    rebalance_frequency: str
    next_rebalance_date: str
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    diversification_ratio: float = 0.0
    risk_metrics: Dict[str, object] = field(default_factory=dict)
    monte_carlo: Dict[str, object] = field(default_factory=dict)
    optimization_method: str = "target_volatility"


@dataclass
class TacticalSignalOutput:
    signal: int
    confidence: float
    regime: str
    risk_off_alert: bool
    rebalance_window_active: bool
    rationale: List[str]
    suggested_position_size: float = 1.0
    target_volatility: float = 0.0
    realized_volatility: float = 0.0
    model_evaluation: Dict[str, object] = field(default_factory=dict)


@dataclass
class IntegratedDecisionOutput:
    strategic_bias: str
    tactical_signal: int
    confidence: float
    action: str
    note: str
    suggested_tilt: Dict[str, float] = field(default_factory=dict)
