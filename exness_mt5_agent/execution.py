"""Order construction and sending.

Sizing uses MetaTrader's own profit calculator on the rounded stop, so a
gold contract and a euro contract share one risk rule. The position is
refused when the minimum lot, the stop distance, the spread, or free margin
would break that rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from exness_mt5_agent.config import Settings
from exness_mt5_agent.mt5_client import require_mt5
from exness_mt5_agent.risk import lots_for_risk, protective_prices, stops_are_valid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str
    lots: float
    price: float
    stop: float
    target: float
    risk_cash: float


@dataclass(frozen=True)
class OrderOutcome:
    sent: bool
    dry_run: bool
    retcode: int | None
    comment: str
    plan: OrderPlan | None


def filling_mode(symbol: str):
    lib = require_mt5()
    info = lib.symbol_info(symbol)
    mode = int(info.filling_mode)
    # Exness deals are usually IOC. Prefer that bit when the symbol allows it.
    if mode & 2:
        return lib.ORDER_FILLING_IOC
    if mode & 1:
        return lib.ORDER_FILLING_FOK
    return lib.ORDER_FILLING_RETURN


def plan_order(symbol: str, side: str, atr: float, settings: Settings) -> OrderPlan | None:
    lib = require_mt5()
    info = lib.symbol_info(symbol)
    tick = lib.symbol_info_tick(symbol)
    account = lib.account_info()
    if info is None or tick is None or account is None:
        logger.error("Missing symbol, tick, or account info for %s", symbol)
        return None
    if int(info.spread) > settings.max_spread_for(symbol):
        logger.info(
            "%s spread %s points is above the %s limit",
            symbol,
            info.spread,
            settings.max_spread_for(symbol),
        )
        return None
    price = float(tick.ask if side == "BUY" else tick.bid)
    stop, target = protective_prices(
        side, price, atr, settings.strategy.sl_atr, settings.strategy.tp_atr, int(info.digits)
    )
    min_distance = int(info.trade_stops_level) * float(info.point)
    if not stops_are_valid(side, price, stop, target, min_distance):
        logger.info("%s stop/target fails the broker stop level or the price order", symbol)
        return None
    order_type = lib.ORDER_TYPE_BUY if side == "BUY" else lib.ORDER_TYPE_SELL
    profit = lib.order_calc_profit(order_type, symbol, 1.0, price, stop)
    if profit is None or profit >= 0:
        logger.info("%s profit calculator rejected the stop (%s)", symbol, profit)
        return None
    loss_per_lot = abs(float(profit))
    lots = lots_for_risk(
        equity=float(account.equity),
        risk_pct=settings.strategy.risk_pct,
        loss_per_lot=loss_per_lot,
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        max_lots=settings.max_lots,
        commission_per_lot=settings.commission_per_lot,
    )
    if lots is None:
        logger.info("%s lot size is below the minimum or above the safety cap", symbol)
        return None
    margin = lib.order_calc_margin(order_type, symbol, lots, price)
    if margin is None or float(margin) > float(account.margin_free):
        logger.info("%s free margin cannot cover %.2f", symbol, 0.0 if margin is None else margin)
        return None
    return OrderPlan(
        symbol=symbol,
        side=side,
        lots=lots,
        price=price,
        stop=stop,
        target=target,
        risk_cash=loss_per_lot * lots + settings.commission_per_lot * lots,
    )


def send_order(plan: OrderPlan, settings: Settings, atr: float) -> OrderOutcome:
    """Send a market deal, or log the plan when LIVE_TRADING is off."""

    if not settings.live_trading:
        logger.info(
            "DRY RUN %s %s lots=%.2f price=%s sl=%s tp=%s risk=%.2f",
            plan.side,
            plan.symbol,
            plan.lots,
            plan.price,
            plan.stop,
            plan.target,
            plan.risk_cash,
        )
        return OrderOutcome(False, True, None, "dry run", plan)

    lib = require_mt5()
    terminal = lib.terminal_info()
    account = lib.account_info()
    if terminal is None or not terminal.trade_allowed or account is None or not account.trade_allowed:
        return OrderOutcome(False, False, None, "terminal or account is not allowed to trade", plan)

    outcome = _send_once(plan, settings)
    if outcome.retcode == getattr(lib, "TRADE_RETCODE_REQUOTE", 10004):
        refreshed = plan_order(plan.symbol, plan.side, atr, settings)
        if refreshed is None:
            return outcome
        outcome = _send_once(refreshed, settings)
    return outcome


def _send_once(plan: OrderPlan, settings: Settings) -> OrderOutcome:
    lib = require_mt5()
    order_type = lib.ORDER_TYPE_BUY if plan.side == "BUY" else lib.ORDER_TYPE_SELL
    request = {
        "action": lib.TRADE_ACTION_DEAL,
        "symbol": plan.symbol,
        "volume": plan.lots,
        "type": order_type,
        "price": plan.price,
        "sl": plan.stop,
        "tp": plan.target,
        "deviation": settings.deviation_points,
        "magic": settings.magic,
        "comment": "m15-h1-agent",
        "type_time": lib.ORDER_TIME_GTC,
        "type_filling": filling_mode(plan.symbol),
    }
    result = lib.order_send(request)
    if result is None:
        code, message = lib.last_error()
        logger.error("order_send returned nothing (%s): %s", code, message)
        return OrderOutcome(False, False, code, message, plan)
    done = {
        getattr(lib, "TRADE_RETCODE_DONE", 10009),
        getattr(lib, "TRADE_RETCODE_DONE_PARTIAL", 10010),
    }
    if result.retcode not in done:
        logger.error(
            "order rejected retcode=%s comment=%s request=%s",
            result.retcode,
            result.comment,
            request,
        )
        return OrderOutcome(False, False, int(result.retcode), str(result.comment), plan)
    logger.info(
        "filled %s %s lots=%.2f price=%s sl=%s tp=%s deal=%s",
        plan.side,
        plan.symbol,
        plan.lots,
        result.price,
        plan.stop,
        plan.target,
        result.deal,
    )
    return OrderOutcome(True, False, int(result.retcode), str(result.comment), plan)


def has_open_position(symbol: str, magic: int) -> bool:
    lib = require_mt5()
    positions = lib.positions_get(symbol=symbol) or []
    return any(int(position.magic) == int(magic) for position in positions)
