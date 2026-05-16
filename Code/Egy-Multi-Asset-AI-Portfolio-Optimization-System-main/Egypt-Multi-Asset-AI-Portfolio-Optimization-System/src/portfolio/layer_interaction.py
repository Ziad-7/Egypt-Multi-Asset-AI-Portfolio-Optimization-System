"""Strategic + tactical layer fusion.

Adds an inflation-aware tilt modifier so the suggested overlay reflects
the Egyptian real-yield environment, not just price momentum.  The
fusion still maintains a complete 9-cell decision matrix; the inflation
context only modifies *how aggressively* the per-asset tilt expresses a
signal, never the directional decision itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from src.api.schemas import IntegratedDecisionOutput, StrategicAllocationOutput, TacticalSignalOutput


InflationRegime = Literal["low", "rising", "high", "elevated_falling"]


@dataclass
class InflationContext:
    """Compact carrier for the macro inflation state needed by fusion."""

    regime: InflationRegime
    yoy: float
    real_yield: float

    def to_dict(self) -> Dict[str, object]:
        return {"regime": self.regime, "yoy": self.yoy, "real_yield": self.real_yield}


_DECISION_MATRIX: Dict[tuple, tuple] = {
    ("bullish", 1):  ("High-conviction long allocation; deploy fully on schedule.",  0.92),
    ("bullish", 0):  ("Hold strategic equity; await tactical confirmation.",         0.65),
    ("bullish", -1): ("Strategic long with tactical hedge: trim equity, add Gold.",  0.62),
    ("defensive", 1):  ("Counter-trend bounce; small risk-on overlay only.",         0.45),
    ("defensive", 0):  ("Stay defensive; maintain T-Bills/Gold weight.",             0.70),
    ("defensive", -1): ("Strong protection mode; maximize T-Bills and Gold.",        0.88),
    ("neutral", 1):  ("Tactical long overlay on neutral macro; size modestly.",      0.60),
    ("neutral", 0):  ("Wait for clearer setup; rebalance to profile target.",        0.55),
    ("neutral", -1): ("Tactical de-risk on neutral macro; add T-Bills.",             0.65),
}


# Per-regime multipliers applied component-wise to the suggested tilt.  Values
# are interpreted as multiplicative scalars (e.g. 1.5 boosts the magnitude of
# a Gold tilt by 50 %).  Sign flips are explicit -- a real-yield-negative
# environment forces T-Bill tilts to be non-positive regardless of signal.
_REGIME_TILT_MULTIPLIERS: Dict[InflationRegime, Dict[str, float]] = {
    "low":               {"EGX30": 1.00, "EGX100": 1.00, "Gold": 1.00, "TBills": 1.00, "EgyptiansRealEstateFund": 1.00},
    "rising":            {"EGX30": 1.10, "EGX100": 1.10, "Gold": 1.25, "TBills": 0.75, "EgyptiansRealEstateFund": 1.10},
    "high":              {"EGX30": 1.10, "EGX100": 1.10, "Gold": 1.50, "TBills": 0.50, "EgyptiansRealEstateFund": 1.20},
    "elevated_falling":  {"EGX30": 1.05, "EGX100": 1.05, "Gold": 1.10, "TBills": 0.90, "EgyptiansRealEstateFund": 1.05},
}


def _strategic_bias(strategic: StrategicAllocationOutput) -> str:
    egx_weight = strategic.weights.get("EGX30", 0.0) + strategic.weights.get("EGX100", 0.0)
    safe_weight = strategic.weights.get("Gold", 0.0) + strategic.weights.get("TBills", 0.0)
    if egx_weight > safe_weight + 0.05:
        return "bullish"
    if safe_weight > egx_weight + 0.05:
        return "defensive"
    return "neutral"


def _base_tilt(strategic: StrategicAllocationOutput, tactical: TacticalSignalOutput) -> Dict[str, float]:
    if tactical.signal == 0:
        return {}
    magnitude = round(0.05 * tactical.confidence * tactical.suggested_position_size, 4)
    if magnitude < 0.005:
        return {}

    if tactical.signal == 1:
        return {
            "EGX30": magnitude,
            "Gold": -magnitude * 0.5,
            "TBills": -magnitude * 0.5,
        }
    return {
        "EGX30": -magnitude,
        "Gold": magnitude * 0.5,
        "TBills": magnitude * 0.5,
    }


def _apply_inflation_modifier(
    base_tilt: Dict[str, float],
    inflation: InflationContext | None,
) -> Dict[str, float]:
    if not base_tilt or inflation is None:
        return base_tilt
    multipliers = _REGIME_TILT_MULTIPLIERS.get(inflation.regime, _REGIME_TILT_MULTIPLIERS["low"])
    adjusted = {asset: float(weight) * multipliers.get(asset, 1.0) for asset, weight in base_tilt.items()}

    # Real-yield safety rule: if real yield is negative, never *add* T-Bill weight
    # via the tilt -- the real return is structurally negative.
    if inflation.real_yield < 0 and adjusted.get("TBills", 0.0) > 0:
        adjusted["TBills"] = 0.0

    # Re-balance the tilt so it remains net-zero (preserving the budget).
    total = sum(adjusted.values())
    if abs(total) > 1e-9:
        # Spread the residual proportionally onto the magnitude-weighted basket.
        denom = sum(abs(v) for v in adjusted.values()) or 1.0
        for asset in adjusted:
            adjusted[asset] -= total * abs(adjusted[asset]) / denom
    return {k: round(v, 4) for k, v in adjusted.items() if abs(v) > 1e-4}


def fuse_layers(
    strategic: StrategicAllocationOutput,
    tactical: TacticalSignalOutput,
    inflation: InflationContext | None = None,
) -> IntegratedDecisionOutput:
    bias = _strategic_bias(strategic)
    pair = (bias, tactical.signal)
    action, base_conf = _DECISION_MATRIX[pair]

    base_tilt = _base_tilt(strategic, tactical)
    suggested_tilt = _apply_inflation_modifier(base_tilt, inflation)

    confidence = round(
        max(0.05, min(0.99, base_conf * (0.55 + 0.45 * tactical.confidence) * (0.65 + 0.35 * strategic.confidence))),
        4,
    )

    note = "Layer 1 sets direction; Layer 2 governs timing, sizing and protection."
    if inflation is not None:
        note += (
            f" Inflation regime={inflation.regime} (YoY={inflation.yoy:.2%},"
            f" real_yield={inflation.real_yield:.2%})."
        )

    return IntegratedDecisionOutput(
        strategic_bias=bias,
        tactical_signal=tactical.signal,
        confidence=confidence,
        action=action,
        note=note,
        suggested_tilt=suggested_tilt,
    )
