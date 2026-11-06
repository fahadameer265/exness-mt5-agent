"""Bar-by-bar backtest aligned with the live gates and the training label."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from exness_mt5_agent.config import StrategyParams
from exness_mt5_agent.path import scan_trade
from exness_mt5_agent.risk import lots_for_risk, next_trade_fits


@dataclass(frozen=True)
class SymbolSpec:
    point: float
    digits: int
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    max_spread_points: int


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    summary: dict[str, float | int | None]


def run_backtest(
    df: pd.DataFrame,
    proba_long: np.ndarray,
    proba_short: np.ndarray,
    params: StrategyParams,
    spec: SymbolSpec,
    *,
    commission_per_lot: float = 0.0,
    starting_equity: float = 10_000.0,
    news_block: callable | None = None,
) -> BacktestResult:
    """One position at a time. New trades stop after a 3% down day.

    `news_block`, when provided, receives the decision bar's close time and
    returns True when that bar must be skipped.
    """

    frame = df.reset_index(drop=True)
    if len(proba_long) != len(frame) or len(proba_short) != len(frame):
        raise ValueError("Probability arrays must match the feature frame.")
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    if "spread" in frame.columns:
        spread_points = frame["spread"].to_numpy(dtype=float)
    else:
        spread_points = np.zeros(len(frame))
    spread_price = spread_points * spec.point
    atr = frame["atr"].to_numpy(dtype=float)
    long_setup = frame["long_setup"].fillna(False).to_numpy(dtype=bool)
    short_setup = frame["short_setup"].fillna(False).to_numpy(dtype=bool)
    close_time = pd.to_datetime(frame["close_time"])

    equity = float(starting_equity)
    curve = [equity]
    day_start: dict = {}
    day_pnl: dict = {}
    trades: list[dict] = []
    skipped = 0
    i = 0
    while i < len(frame) - 1:
        bar_day = close_time.iloc[i].date()
        day_start.setdefault(bar_day, equity)
        equity_now = day_start[bar_day] + day_pnl.get(bar_day, 0.0)
        if not next_trade_fits(day_start[bar_day], equity_now, params.risk_pct, params.max_daily_loss):
            i += 1
            continue
        if news_block is not None and news_block(_as_naive(close_time.iloc[i])):
            skipped += 1
            i += 1
            continue
        side = _side(i, long_setup, short_setup, proba_long, proba_short, params)
        if side is None:
            i += 1
            continue
        if spread_points[i + 1] > spec.max_spread_points:
            skipped += 1
            i += 1
            continue
        scanned = scan_trade(
            side=side,
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
            digits=spec.digits,
            require_full_horizon=False,
        )
        if scanned is None:
            skipped += 1
            i += 1
            continue
        loss_per_lot = abs(scanned.entry_price - scanned.stop) * spec.contract_size
        lots = lots_for_risk(
            equity=equity,
            risk_pct=params.risk_pct,
            loss_per_lot=loss_per_lot,
            volume_min=spec.volume_min,
            volume_max=spec.volume_max,
            volume_step=spec.volume_step,
            max_lots=100.0,
            commission_per_lot=commission_per_lot,
        )
        if lots is None:
            skipped += 1
            i += 1
            continue
        if side == "BUY":
            pnl = (scanned.exit_price - scanned.entry_price) * spec.contract_size * lots
        else:
            pnl = (scanned.entry_price - scanned.exit_price) * spec.contract_size * lots
        pnl -= commission_per_lot * lots
        exit_day = close_time.iloc[scanned.exit_index].date()
        day_start.setdefault(exit_day, equity)
        day_pnl[exit_day] = day_pnl.get(exit_day, 0.0) + pnl
        equity += pnl
        curve.append(equity)
        trades.append(
            {
                "decision_time": close_time.iloc[i],
                "exit_time": close_time.iloc[scanned.exit_index],
                "side": side,
                "lots": lots,
                "entry": scanned.entry_price,
                "stop": scanned.stop,
                "target": scanned.target,
                "exit": scanned.exit_price,
                "reason": scanned.reason,
                "probability": float(proba_long[i] if side == "BUY" else proba_short[i]),
                "pnl": pnl,
                "equity": equity,
            }
        )
        i = scanned.exit_index + 1

    trade_frame = pd.DataFrame(trades)
    summary = _summarize(trade_frame, curve, starting_equity, skipped)
    return BacktestResult(trades=trade_frame, summary=summary)


def _side(i, long_setup, short_setup, proba_long, proba_short, params) -> str | None:
    long_ok = bool(long_setup[i]) and _clears(proba_long[i], params.proba_threshold)
    short_ok = bool(short_setup[i]) and _clears(proba_short[i], params.proba_threshold)
    if long_ok and short_ok:
        return None
    if long_ok:
        return "BUY"
    if short_ok:
        return "SELL"
    return None


def _clears(probability, threshold: float) -> bool:
    value = float(probability)
    return np.isfinite(value) and value >= threshold


def _as_naive(stamp) -> datetime:
    value = pd.Timestamp(stamp).to_pydatetime()
    if value.tzinfo is not None:
        value = value.replace(tzinfo=None)
    return value


def _summarize(trades: pd.DataFrame, curve: list[float], starting_equity: float, skipped: int) -> dict:
    equity = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, np.nan)
    if trades.empty:
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        pnl = 0.0
    else:
        pnl_series = trades["pnl"]
        wins = int((pnl_series > 0).sum())
        gross_profit = float(pnl_series.clip(lower=0).sum())
        gross_loss = float((-pnl_series.clip(upper=0)).sum())
        pnl = float(pnl_series.sum())
    count = int(len(trades))
    return {
        "trades": count,
        "wins": wins,
        "win_rate": (wins / count) if count else None,
        "net_pnl": pnl,
        "ending_equity": float(equity[-1]),
        "max_drawdown": float(np.nanmax(drawdown)) if len(drawdown) else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy": (pnl / count) if count else None,
        "skipped": skipped,
        "starting_equity": starting_equity,
    }
