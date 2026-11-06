"""MetaTrader 5 connection, symbol lookup, and historical bars.

The Python package drives the Exness terminal that is already installed on
this machine. It is not a standalone broker API. Log into the demo terminal
first, and leave MT5_LOGIN blank unless you intentionally want the script to
switch accounts.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from exness_mt5_agent.config import Settings

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - exercised only on machines without the package
    mt5 = None


def require_mt5():
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 is not installed for this Python. Use a 64-bit Python "
            "supported by the MetaTrader5 package, then pip install -r requirements.txt."
        )
    return mt5


def connect(settings: Settings):
    """Attach to the terminal and, when credentials are present, log in."""

    lib = require_mt5()
    kwargs = {}
    if settings.path:
        kwargs["path"] = settings.path
    if settings.login:
        if not settings.password or not settings.server:
            raise ValueError("MT5_LOGIN is set, so MT5_PASSWORD and MT5_SERVER are required.")
        kwargs["login"] = int(settings.login)
        kwargs["password"] = settings.password
        kwargs["server"] = settings.server
    if not lib.initialize(**kwargs):
        code, message = lib.last_error()
        raise ConnectionError(f"MT5 initialize failed ({code}): {message}")
    return lib


def shutdown() -> None:
    if mt5 is not None:
        mt5.shutdown()


def account_snapshot() -> dict:
    lib = require_mt5()
    account = lib.account_info()
    terminal = lib.terminal_info()
    if account is None or terminal is None:
        code, message = lib.last_error()
        raise ConnectionError(f"MT5 returned no account info ({code}): {message}")
    real_mode = getattr(lib, "ACCOUNT_TRADE_MODE_REAL", 2)
    return {
        "login": account.login,
        "server": account.server,
        "name": account.name,
        "currency": account.currency,
        "equity": float(account.equity),
        "balance": float(account.balance),
        "margin_free": float(account.margin_free),
        "leverage": int(account.leverage),
        "trade_mode": int(account.trade_mode),
        "is_real": int(account.trade_mode) == int(real_mode),
        "account_trade_allowed": bool(account.trade_allowed),
        "terminal_trade_allowed": bool(terminal.trade_allowed),
        "company": terminal.company,
    }


def assert_trade_account(settings: Settings) -> dict:
    snapshot = account_snapshot()
    if snapshot["is_real"] and not settings.allow_real:
        raise RuntimeError(
            f"Connected to real account {snapshot['login']} on {snapshot['server']}. "
            "Use an Exness demo, or set ALLOW_REAL_ACCOUNT=true only when you mean to."
        )
    return snapshot


def resolve_symbol(requested: str) -> str:
    """Pick the broker's name for a logical symbol such as EURUSD or XAUUSD."""

    lib = require_mt5()
    direct = lib.symbol_info(requested)
    if direct is not None:
        if not lib.symbol_select(direct.name, True):
            raise LookupError(f"Could not select {direct.name} in Market Watch.")
        return direct.name
    root = requested.upper().split(".")[0]
    matches = []
    for item in lib.symbols_get() or []:
        if item.name.upper().startswith(root):
            matches.append(item.name)
    if not matches:
        raise LookupError(
            f"No MT5 symbol starts with {root}. Open Market Watch in the Exness terminal "
            "and put the exact name in SYMBOLS."
        )
    matches.sort(key=lambda name: (len(name), name))
    chosen = matches[0]
    if not lib.symbol_select(chosen, True):
        raise LookupError(f"Could not select {chosen} in Market Watch.")
    return chosen


def copy_rates(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    lib = require_mt5()
    tf = getattr(lib, timeframe)
    rates = lib.copy_rates_from_pos(symbol, tf, 0, int(bars))
    if rates is None or len(rates) == 0:
        code, message = lib.last_error()
        raise RuntimeError(f"No {timeframe} rates for {symbol} ({code}): {message}")
    frame = pd.DataFrame(rates)
    frame["time"] = pd.to_datetime(frame["time"], unit="s")
    return frame


def symbol_spec(symbol: str) -> dict:
    lib = require_mt5()
    info = lib.symbol_info(symbol)
    if info is None:
        raise LookupError(f"symbol_info returned nothing for {symbol}.")
    return {
        "name": info.name,
        "point": float(info.point),
        "digits": int(info.digits),
        "spread_points": int(info.spread),
        "contract_size": float(info.trade_contract_size),
        "volume_min": float(info.volume_min),
        "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step),
        "stops_level_points": int(info.trade_stops_level),
        "filling_mode": int(info.filling_mode),
        "tick_time_utc": datetime.fromtimestamp(lib.symbol_info_tick(symbol).time, tz=timezone.utc),
    }


def server_now_guess(symbol: str) -> datetime:
    """Naive broker wall time of the latest tick, using the bar-time convention.

    Tick timestamps from the API are real Unix time. Bar timestamps are the
    server clock packed as if it were UTC. The check command prints both so
    you can set BROKER_UTC_OFFSET from the difference you observe.
    """

    lib = require_mt5()
    tick = lib.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for {symbol}.")
    return datetime.fromtimestamp(tick.time, tz=timezone.utc).replace(tzinfo=None)
