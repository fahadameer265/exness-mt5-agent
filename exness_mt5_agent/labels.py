"""Training labels. See path.scan_trade for the fill and the same-bar rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exness_mt5_agent.config import StrategyParams
from exness_mt5_agent.path import scan_trade


def _arrays(df: pd.DataFrame, point: float) -> tuple[np.ndarray, ...]:
    if "spread" in df.columns:
        spread_price = df["spread"].to_numpy(dtype=float) * point
    else:
        spread_price = np.zeros(len(df), dtype=float)
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        spread_price,
    )


def label_side(
    df: pd.DataFrame, side: int, params: StrategyParams, point: float, digits: int
) -> np.ndarray:
    """1 when the scanned path hits the target, 0 when it does not, NaN if unknown."""

    if side not in (1, -1):
        raise ValueError("side must be +1 or -1.")
    open_, high, low, close, spread_price = _arrays(df, point)
    atr = df["atr"].to_numpy(dtype=float)
    labels = np.full(len(df), np.nan)
    order_side = "BUY" if side == 1 else "SELL"
    for i in range(len(df) - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        scanned = scan_trade(
            side=order_side,
            entry_index=i + 1,
            open_=open_,
            high=high,
            low=low,
            close=close,
            spread_price=spread_price,
            atr=float(atr[i]),
            sl_atr=params.sl_atr,
            tp_atr=params.tp_atr,
            horizon=params.horizon,
            digits=digits,
            require_full_horizon=True,
        )
        if scanned is None:
            continue
        labels[i] = scanned.win
    return labels


def attach_labels(
    df: pd.DataFrame, params: StrategyParams, point: float, digits: int
) -> pd.DataFrame:
    out = df.copy()
    out["y_long"] = label_side(out, 1, params, point, digits)
    out["y_short"] = label_side(out, -1, params, point, digits)
    return out
