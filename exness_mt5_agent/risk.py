"""Position size, rounded stops, and the daily loss halt.

Lot size is floored to the broker step so the cash risk of the rounded stop
stays at or under the budget. If the minimum lot would risk more than that
budget, the trade is skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("volume step must be positive")
    steps = math.floor((value / step) + 1e-8)
    return round(steps * step, 8)


def round_to_digits(price: float, digits: int) -> float:
    return round(float(price), int(digits))


def protective_prices(
    side: str, entry: float, atr: float, sl_atr: float, tp_atr: float, digits: int
) -> tuple[float, float]:
    """Stop and target rounded to the symbol's digits."""

    if atr <= 0:
        raise ValueError("ATR must be positive.")
    if side == "BUY":
        stop = entry - sl_atr * atr
        target = entry + tp_atr * atr
    elif side == "SELL":
        stop = entry + sl_atr * atr
        target = entry - tp_atr * atr
    else:
        raise ValueError("side must be BUY or SELL.")
    return round_to_digits(stop, digits), round_to_digits(target, digits)


def stops_are_valid(
    side: str, price: float, stop: float, target: float, min_distance: float
) -> bool:
    if min_distance < 0:
        raise ValueError("min_distance cannot be negative.")
    if abs(price - stop) < min_distance or abs(target - price) < min_distance:
        return False
    if side == "BUY":
        return stop < price < target
    if side == "SELL":
        return target < price < stop
    raise ValueError("side must be BUY or SELL.")


def lots_for_risk(
    equity: float,
    risk_pct: float,
    loss_per_lot: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
    max_lots: float,
    commission_per_lot: float = 0.0,
) -> float | None:
    """Largest step-aligned volume whose stop plus commission stays within budget.

    Returns None when even the minimum volume would exceed the risk budget or
    the broker limits. The safety cap `max_lots` refuses a size instead of
    silently trading a different risk.
    """

    if equity <= 0 or risk_pct <= 0:
        return None
    cost_per_lot = loss_per_lot + max(0.0, commission_per_lot)
    if cost_per_lot <= 0 or volume_step <= 0 or volume_min <= 0:
        return None
    budget = equity * risk_pct
    raw = budget / cost_per_lot
    if raw > max_lots:
        return None
    sized = floor_to_step(min(raw, volume_max), volume_step)
    if sized < volume_min or sized > volume_max or sized > max_lots:
        return None
    if sized * cost_per_lot > budget + 1e-6:
        return None
    return sized


def daily_loss_reached(start_equity: float, equity: float, limit: float) -> bool:
    if start_equity <= 0:
        return True
    return (start_equity - equity) / start_equity >= limit


def next_trade_fits(start_equity: float, equity: float, risk_pct: float, limit: float) -> bool:
    """True when a full 1% loss from here would still finish the day inside the limit.

    Risk is taken on current equity, then measured against the morning equity.
    Three stopped trades of 1% stay near a 3% day. The fourth does not fit.
    A gap through the stop can still finish past the limit.
    """

    if daily_loss_reached(start_equity, equity, limit):
        return False
    equity_after_planned_loss = equity * (1.0 - risk_pct)
    return not daily_loss_reached(start_equity, equity_after_planned_loss, limit)


@dataclass
class DailyRiskState:
    day: str | None = None
    start_equity: float | None = None
    halted: bool = False
    last_bar: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.last_bar is None:
            self.last_bar = {}


class DailyRiskGuard:
    """Persisted circuit breaker. A restart must not clear today's halt."""

    def __init__(self, path: Path, max_loss_frac: float, risk_pct: float = 0.01) -> None:
        self.path = path
        self.max_loss_frac = max_loss_frac
        self.risk_pct = risk_pct
        self.state = self._load()

    def _load(self) -> DailyRiskState:
        if not self.path.exists():
            return DailyRiskState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Risk state file is unreadable ({exc}). Trading is blocked until you inspect {self.path}."
            ) from exc
        return DailyRiskState(
            day=raw.get("day"),
            start_equity=raw.get("start_equity"),
            halted=bool(raw.get("halted", False)),
            last_bar=dict(raw.get("last_bar") or {}),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "day": self.state.day,
            "start_equity": self.state.start_equity,
            "halted": self.state.halted,
            "last_bar": self.state.last_bar,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def allow_new_trade(self, equity: float, now_server: datetime) -> tuple[bool, str]:
        day = now_server.strftime("%Y-%m-%d")
        if self.state.day != day:
            self.state.day = day
            self.state.start_equity = float(equity)
            self.state.halted = False
            self.save()
        if self.state.start_equity is None:
            self.state.start_equity = float(equity)
            self.save()
        if self.state.halted:
            return False, f"daily halt already latched for {day}"
        if daily_loss_reached(self.state.start_equity, equity, self.max_loss_frac):
            self.state.halted = True
            self.save()
            return False, (
                f"daily loss reached {self.max_loss_frac:.0%} "
                f"from {self.state.start_equity:.2f} to {equity:.2f}"
            )
        if not next_trade_fits(self.state.start_equity, equity, self.risk_pct, self.max_loss_frac):
            return False, (
                f"the next {self.risk_pct:.0%} trade would push the day through "
                f"{self.max_loss_frac:.0%}"
            )
        return True, "daily loss inside the limit"

    def already_processed(self, symbol: str, bar_time: str) -> bool:
        return (self.state.last_bar or {}).get(symbol) == bar_time

    def mark_processed(self, symbol: str, bar_time: str) -> None:
        if self.state.last_bar is None:
            self.state.last_bar = {}
        self.state.last_bar[symbol] = bar_time
        self.save()
