"""Turn one closed bar into BUY, SELL, or HOLD.

The model does not get to override the setup. No H1 agreement, no pullback,
or a failed break means HOLD even if the probability is high. Both directions
clearing 0.70 on the same bar is a conflict and also a HOLD.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from exness_mt5_agent.config import StrategyParams
from exness_mt5_agent.features import FEATURE_COLUMNS
from exness_mt5_agent.modeling import predict_proba


@dataclass(frozen=True)
class Signal:
    action: str
    probability: float | None
    reason: str
    atr: float | None
    bar_time: str


def _ready(row: pd.Series) -> str | None:
    if any(pd.isna(row.get(col)) for col in FEATURE_COLUMNS):
        return "indicators are still warming up"
    if not np_finite(row.get("atr")) or float(row["atr"]) <= 0:
        return "ATR is not usable"
    return None


def np_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def decide(
    row: pd.Series,
    long_model,
    short_model,
    params: StrategyParams,
) -> Signal:
    bar_time = str(row.get("close_time", row.get("time")))
    blocked = _ready(row)
    if blocked:
        return Signal("HOLD", None, blocked, None, bar_time)
    atr = float(row["atr"])
    long_setup = bool(row.get("long_setup", False))
    short_setup = bool(row.get("short_setup", False))
    if not long_setup and not short_setup:
        return Signal("HOLD", None, "no pullback setup with H1 agreement", atr, bar_time)

    frame = pd.DataFrame([row])
    long_p = predict_proba(long_model, frame) if long_setup and long_model is not None else None
    short_p = predict_proba(short_model, frame) if short_setup and short_model is not None else None
    long_ok = long_p is not None and long_p >= params.proba_threshold
    short_ok = short_p is not None and short_p >= params.proba_threshold
    if long_ok and short_ok:
        return Signal("HOLD", max(long_p, short_p), "long and short both cleared the threshold", atr, bar_time)
    if long_ok:
        return Signal(
            "BUY",
            long_p,
            f"H1 uptrend pullback and long probability cleared {params.proba_threshold:.2f}",
            atr,
            bar_time,
        )
    if short_ok:
        return Signal(
            "SELL",
            short_p,
            f"H1 downtrend pullback and short probability cleared {params.proba_threshold:.2f}",
            atr,
            bar_time,
        )
    if long_setup and long_model is None:
        return Signal("HOLD", None, "long setup has no trained model", atr, bar_time)
    if short_setup and short_model is None:
        return Signal("HOLD", None, "short setup has no trained model", atr, bar_time)
    shown = long_p if long_p is not None else short_p
    if shown is None:
        return Signal("HOLD", None, "setup has no probability", atr, bar_time)
    return Signal(
        "HOLD",
        shown,
        f"setup probability {shown:.3f} is below {params.proba_threshold:.2f}",
        atr,
        bar_time,
    )
