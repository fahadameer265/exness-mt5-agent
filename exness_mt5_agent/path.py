"""Shared stop/target path scan for labels and the backtest.

Both layers call this so a "win" in training is the same event the backtest
counts as a win. Bars are bid prices, which is how MT5 builds candles.
A long is filled at the ask (bid plus that bar's spread). A short is filled
at the bid and covered at the ask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exness_mt5_agent.risk import protective_prices, stops_are_valid


@dataclass(frozen=True)
class PathResult:
    win: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop: float
    target: float
    reason: str


def scan_trade(
    *,
    side: str,
    entry_index: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread_price: np.ndarray,
    atr: float,
    sl_atr: float,
    tp_atr: float,
    horizon: int,
    digits: int,
    require_full_horizon: bool,
) -> PathResult | None:
    """Walk forward from the entry bar.

    `horizon` counts the entry bar as the first candle. Incomplete windows
    return None when `require_full_horizon` is set (training). The backtest
    allows a shortened window and records a timeout.
    """

    n = len(close)
    if entry_index < 0 or entry_index >= n:
        return None
    last = entry_index + horizon - 1
    if last >= n:
        if require_full_horizon:
            return None
        last = n - 1
    if last < entry_index:
        return None
    if not np.isfinite(atr) or atr <= 0:
        return None

    spread_entry = float(spread_price[entry_index])
    bid_open = float(open_[entry_index])
    if side == "BUY":
        entry = bid_open + spread_entry
    elif side == "SELL":
        entry = bid_open
    else:
        raise ValueError("side must be BUY or SELL.")
    if not np.isfinite(entry):
        return None
    stop, target = protective_prices(side, entry, float(atr), sl_atr, tp_atr, digits)
    if not stops_are_valid(side, entry, stop, target, min_distance=0):
        return None

    for j in range(entry_index, last + 1):
        bar_spread = float(spread_price[j])
        bar_open = float(open_[j])
        bar_high = float(high[j])
        bar_low = float(low[j])
        if side == "BUY":
            gapped = j > entry_index and bar_open <= stop
            if gapped:
                return _result(0, j, entry, bar_open, stop, target, "gap_stop")
            hit_stop = bar_low <= stop
            hit_target = bar_high >= target
            if hit_stop:
                return _result(0, j, entry, stop, stop, target, "stop")
            if hit_target:
                return _result(1, j, entry, target, stop, target, "target")
        else:
            gapped = j > entry_index and bar_open + bar_spread >= stop
            if gapped:
                return _result(0, j, entry, bar_open + bar_spread, stop, target, "gap_stop")
            hit_stop = bar_high + bar_spread >= stop
            hit_target = bar_low + bar_spread <= target
            if hit_stop:
                return _result(0, j, entry, stop, stop, target, "stop")
            if hit_target:
                return _result(1, j, entry, target, stop, target, "target")

    exit_bid = float(close[last])
    if side == "BUY":
        exit_price = exit_bid
    else:
        exit_price = exit_bid + float(spread_price[last])
    return _result(0, last, entry, exit_price, stop, target, "timeout")


def _result(win, exit_index, entry, exit_price, stop, target, reason) -> PathResult:
    return PathResult(
        win=win,
        exit_index=exit_index,
        entry_price=float(entry),
        exit_price=float(exit_price),
        stop=float(stop),
        target=float(target),
        reason=reason,
    )
