"""Environment-backed settings.

Numbers that change behaviour live here so the trading loop does not hide them.
Defaults match the requested risk spec: 1% per trade, 1.5/3.0 ATR stops, a 3%
daily halt, and a 0.70 probability cutoff.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


@dataclass(frozen=True)
class StrategyParams:
    """Label, setup, and risk constants shared by train, backtest, and live."""

    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma: int = 20
    donchian: int = 20
    fail_lookback: int = 3
    min_trend_age_m15: int = 3
    min_trend_age_h1: int = 2
    vol_spike_mult: float = 1.2
    rsi_long_rank_low: float = 0.20
    rsi_long_rank_high: float = 0.55
    rsi_short_rank_low: float = 0.45
    rsi_short_rank_high: float = 0.80
    rsi_long_hard_cap: float = 65.0
    rsi_short_hard_floor: float = 35.0
    rsi_rank_window: int = 100
    horizon: int = 12
    sl_atr: float = 1.5
    tp_atr: float = 3.0
    proba_threshold: float = 0.70
    risk_pct: float = 0.01
    max_daily_loss: float = 0.03
    news_blackout_minutes: int = 30
    n_splits: int = 5
    embargo_bars: int = 12

    def __post_init__(self) -> None:
        if self.tp_atr < 2.0 * self.sl_atr:
            raise ValueError("Take profit must be at least twice the stop (1:2).")
        if not 0 < self.risk_pct <= 0.02:
            raise ValueError("Risk per trade must stay within (0, 2%].")
        if not 0 < self.max_daily_loss <= 0.10:
            raise ValueError("Daily loss halt must stay within (0, 10%].")
        if not 0.5 <= self.proba_threshold < 1:
            raise ValueError("Probability threshold must sit in [0.50, 1).")


@dataclass(frozen=True)
class Settings:
    login: int | None
    password: str
    server: str
    path: str
    symbols: tuple[str, ...]
    live_trading: bool
    allow_real: bool
    broker_utc_offset: float
    news_calendar: Path
    news_filter_enabled: bool
    strategy: StrategyParams
    commission_per_lot: float
    deviation_points: int
    max_spread_points: dict[str, int]
    model_type: str
    history_bars: int
    magic: int
    max_lots: float
    root: Path = ROOT

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def state_path(self) -> Path:
        return self.root / "data" / "runtime_state.json"

    def model_path(self, symbol: str) -> Path:
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol.upper())
        return self.model_dir / f"{safe}.joblib"

    def max_spread_for(self, symbol: str) -> int:
        upper = symbol.upper()
        if "XAU" in upper or "GOLD" in upper:
            return self.max_spread_points["XAUUSD"]
        return self.max_spread_points["EURUSD"]

    @classmethod
    def from_env(cls) -> Settings:
        _load_env()
        raw_symbols = os.getenv("SYMBOLS", "EURUSD,XAUUSD")
        symbols = tuple(part.strip() for part in raw_symbols.split(",") if part.strip())
        if not symbols:
            raise ValueError("SYMBOLS is empty.")
        model_type = os.getenv("MODEL_TYPE", "xgboost").strip().lower()
        if model_type not in {"xgboost", "random_forest"}:
            raise ValueError("MODEL_TYPE must be xgboost or random_forest.")
        strategy = StrategyParams(
            horizon=_int("HORIZON_BARS", 12),
            sl_atr=_float("SL_ATR", 1.5),
            tp_atr=_float("TP_ATR", 3.0),
            proba_threshold=_float("PROBA_THRESHOLD", 0.70),
            risk_pct=_float("RISK_PCT", 0.01),
            max_daily_loss=_float("MAX_DAILY_LOSS", 0.03),
            embargo_bars=_int("HORIZON_BARS", 12),
        )
        return cls(
            login=_optional_int("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD", ""),
            server=os.getenv("MT5_SERVER", "").strip(),
            path=os.getenv("MT5_PATH", "").strip(),
            symbols=symbols,
            live_trading=_bool("LIVE_TRADING", False),
            allow_real=_bool("ALLOW_REAL_ACCOUNT", False),
            broker_utc_offset=_float("BROKER_UTC_OFFSET", 2),
            news_calendar=ROOT / os.getenv("NEWS_CALENDAR", "data/news_events.csv"),
            news_filter_enabled=_bool("NEWS_FILTER_ENABLED", True),
            strategy=strategy,
            commission_per_lot=_float("COMMISSION_PER_LOT", 0.0),
            deviation_points=_int("DEVIATION_POINTS", 50),
            max_spread_points={
                "EURUSD": _int("MAX_SPREAD_POINTS_EURUSD", 30),
                "XAUUSD": _int("MAX_SPREAD_POINTS_XAUUSD", 80),
            },
            model_type=model_type,
            history_bars=_int("HISTORY_BARS", 20000),
            magic=_int("MAGIC", 260106),
            max_lots=_float("MAX_LOTS", 20.0),
        )
