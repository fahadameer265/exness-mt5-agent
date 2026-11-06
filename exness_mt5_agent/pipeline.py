"""Train, backtest, and live loops.

These functions are the operators. Indicators, labels, sizing, and order
construction stay in their own modules so a backtest cannot quietly drift
away from the order the terminal would send.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exness_mt5_agent.backtest import SymbolSpec, run_backtest
from exness_mt5_agent.config import Settings
from exness_mt5_agent.execution import has_open_position, plan_order, send_order
from exness_mt5_agent.features import FEATURE_COLUMNS, build_feature_frame, drop_forming_bar
from exness_mt5_agent.labels import attach_labels
from exness_mt5_agent.modeling import TrainResult, train_direction
from exness_mt5_agent.mt5_client import (
    account_snapshot,
    assert_trade_account,
    connect,
    copy_rates,
    resolve_symbol,
    shutdown,
    symbol_spec,
)
from exness_mt5_agent.news import in_news_blackout, load_calendar
from exness_mt5_agent.risk import DailyRiskGuard
from exness_mt5_agent.signals import decide

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_dir / "agent.log", encoding="utf-8"),
        ],
    )


def run_check(settings: Settings) -> None:
    connect(settings)
    try:
        account = account_snapshot()
        logger.info(
            "account=%s server=%s currency=%s equity=%.2f leverage=%s real=%s "
            "account_trade_allowed=%s terminal_trade_allowed=%s",
            account["login"],
            account["server"],
            account["currency"],
            account["equity"],
            account["leverage"],
            account["is_real"],
            account["account_trade_allowed"],
            account["terminal_trade_allowed"],
        )
        if account["is_real"] and not settings.allow_real:
            logger.warning("Real account detected. Live orders stay blocked until ALLOW_REAL_ACCOUNT=true.")
        for requested in settings.symbols:
            resolved = resolve_symbol(requested)
            spec = symbol_spec(resolved)
            logger.info(
                "%s -> %s digits=%s point=%s spread=%s contract=%s volume_min=%s stops_level=%s",
                requested,
                resolved,
                spec["digits"],
                spec["point"],
                spec["spread_points"],
                spec["contract_size"],
                spec["volume_min"],
                spec["stops_level_points"],
            )
            rates = copy_rates(resolved, "TIMEFRAME_M15", 2)
            logger.info("latest M15 open timestamp (broker clock): %s", rates["time"].iloc[-1])
            logger.info("latest tick UTC: %s", spec["tick_time_utc"].isoformat())
        calendar = load_calendar(settings.news_calendar)
        logger.info("news calendar: %s", calendar.reason)
        logger.info(
            "BROKER_UTC_OFFSET=%s. Compare the broker clock in the MT5 toolbar with UTC "
            "and correct this value before relying on the news blackout.",
            settings.broker_utc_offset,
        )
        logger.info("LIVE_TRADING=%s", settings.live_trading)
    finally:
        shutdown()


def train_symbols(settings: Settings, symbols: list[str], bars: int, min_setups: int) -> None:
    connect(settings)
    try:
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        failures = []
        for requested in symbols:
            try:
                _train_one(settings, requested, bars, min_setups)
            except Exception as exc:
                logger.exception("training failed for %s", requested)
                failures.append(f"{requested}: {exc}")
        if failures:
            raise RuntimeError(" ; ".join(failures))
    finally:
        shutdown()


def _train_one(settings: Settings, requested: str, bars: int, min_setups: int) -> None:
    resolved = resolve_symbol(requested)
    spec = symbol_spec(resolved)
    logger.info("training %s on %s M15 bars", resolved, bars)
    m15 = drop_forming_bar(copy_rates(resolved, "TIMEFRAME_M15", bars))
    h1_bars = max(1000, bars // 4 + 400)
    h1 = drop_forming_bar(copy_rates(resolved, "TIMEFRAME_H1", h1_bars))
    frame = attach_labels(
        build_feature_frame(m15, h1, settings.strategy),
        settings.strategy,
        point=spec["point"],
        digits=spec["digits"],
    )
    payload = {
        "version": 1,
        "symbol": resolved,
        "feature_columns": list(FEATURE_COLUMNS),
        "label": {
            "sl_atr": settings.strategy.sl_atr,
            "tp_atr": settings.strategy.tp_atr,
            "horizon": settings.strategy.horizon,
        },
        "long_model": None,
        "short_model": None,
        "report": {},
    }
    for side, setup_col, label_col, key in (
        ("long", "long_setup", "y_long", "long_model"),
        ("short", "short_setup", "y_short", "short_model"),
    ):
        setups = frame.loc[frame[setup_col] & frame[label_col].notna()].copy()
        logger.info("%s %s setups with labels: %s", resolved, side, len(setups))
        try:
            result = train_direction(
                setups,
                label_col,
                settings.strategy,
                settings.model_type,
                min_rows=min_setups,
            )
        except ValueError as exc:
            logger.warning("%s %s model skipped: %s", resolved, side, exc)
            continue
        payload[key] = result.model
        payload["report"][side] = _report(result)
        _log_report(resolved, side, result)
    if payload["long_model"] is None and payload["short_model"] is None:
        raise RuntimeError(
            f"{resolved} produced no model. Download more M15 history in MT5 "
            "(Tools → Options → Charts → Max bars in chart) or review the setup filters."
        )
    path = settings.model_path(resolved)
    joblib.dump(payload, path)
    logger.info("saved %s", path)


def _report(result: TrainResult) -> dict:
    return {
        "rows": result.rows,
        "positives": result.positives,
        "holdout_rows": result.holdout_rows,
        "holdout_base_rate": result.holdout_base_rate,
        "holdout_precision": result.holdout_precision,
        "holdout_signals": result.holdout_signals,
        "folds": [score.__dict__ for score in result.folds],
    }


def _log_report(symbol: str, side: str, result: TrainResult) -> None:
    for score in result.folds:
        precision = "n/a" if score.precision is None else f"{score.precision:.3f}"
        logger.info(
            "%s %s fold %s train=%s test=%s base=%.3f precision@threshold=%s signals=%s",
            symbol,
            side,
            score.fold,
            score.train_rows,
            score.test_rows,
            score.base_rate,
            precision,
            score.signals,
        )
    precision = "n/a" if result.holdout_precision is None else f"{result.holdout_precision:.3f}"
    logger.info(
        "%s %s HOLDOUT rows=%s base=%.3f precision@threshold=%s signals=%s",
        symbol,
        side,
        result.holdout_rows,
        result.holdout_base_rate,
        precision,
        result.holdout_signals,
    )
    if result.holdout_signals == 0:
        logger.warning(
            "%s %s holdout never reached the probability threshold. "
            "That is a valid result. Do not lower the threshold to manufacture trades.",
            symbol,
            side,
        )


def load_payload(path: Path, settings: Settings) -> dict:
    payload = joblib.load(path)
    if list(payload.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise RuntimeError(f"{path} was trained with different features. Retrain before trading.")
    label = payload.get("label", {})
    current = {
        "sl_atr": settings.strategy.sl_atr,
        "tp_atr": settings.strategy.tp_atr,
        "horizon": settings.strategy.horizon,
    }
    if label != current:
        raise RuntimeError(f"{path} label geometry {label} does not match current {current}. Retrain.")
    return payload


def run_backtest_symbol(
    settings: Settings,
    requested: str,
    bars: int,
    starting_equity: float,
    rules_only: bool,
) -> None:
    connect(settings)
    try:
        resolved = resolve_symbol(requested)
        spec_raw = symbol_spec(resolved)
        payload = None if rules_only else load_payload(settings.model_path(resolved), settings)
        m15 = drop_forming_bar(copy_rates(resolved, "TIMEFRAME_M15", bars))
        h1 = drop_forming_bar(copy_rates(resolved, "TIMEFRAME_H1", max(1000, bars // 4 + 400)))
        frame = build_feature_frame(m15, h1, settings.strategy)
        long_p, short_p = _probabilities(frame, payload, rules_only)
        calendar = load_calendar(settings.news_calendar)
        news_block = None
        if settings.news_filter_enabled and calendar.ok:
            logger.info("backtest news filter on: %s", calendar.reason)

            def news_block(moment, _symbol=resolved, _calendar=calendar.events):
                return in_news_blackout(
                    moment,
                    _symbol,
                    _calendar,
                    settings.broker_utc_offset,
                    settings.strategy.news_blackout_minutes,
                )
        elif settings.news_filter_enabled:
            logger.warning("backtest news filter off: %s", calendar.reason)
        result = run_backtest(
            frame,
            long_p,
            short_p,
            settings.strategy,
            SymbolSpec(
                point=spec_raw["point"],
                digits=spec_raw["digits"],
                contract_size=spec_raw["contract_size"],
                volume_min=spec_raw["volume_min"],
                volume_max=spec_raw["volume_max"],
                volume_step=spec_raw["volume_step"],
                max_spread_points=settings.max_spread_for(resolved),
            ),
            commission_per_lot=settings.commission_per_lot,
            starting_equity=starting_equity,
            news_block=news_block,
        )
        out = settings.root / "data" / f"{resolved}_backtest_trades.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.trades.to_csv(out, index=False)
        logger.info("backtest %s %s", resolved, result.summary)
        logger.info("trade log: %s", out)
    finally:
        shutdown()


def _probabilities(frame: pd.DataFrame, payload: dict | None, rules_only: bool):
    ones = np.ones(len(frame), dtype=float)
    if rules_only:
        return ones, ones
    assert payload is not None
    ready = frame.loc[:, FEATURE_COLUMNS].notna().all(axis=1).to_numpy()
    long_p = np.zeros(len(frame), dtype=float)
    short_p = np.zeros(len(frame), dtype=float)
    features = frame.loc[ready, FEATURE_COLUMNS].astype(float)
    if payload["long_model"] is not None and ready.any():
        long_p[ready] = payload["long_model"].predict_proba(features)[:, 1]
    if payload["short_model"] is not None and ready.any():
        short_p[ready] = payload["short_model"].predict_proba(features)[:, 1]
    return long_p, short_p


def run_live(settings: Settings) -> None:
    connect(settings)
    try:
        account = assert_trade_account(settings)
        logger.info(
            "live loop equity=%.2f server=%s LIVE_TRADING=%s",
            account["equity"],
            account["server"],
            settings.live_trading,
        )
        if not settings.live_trading:
            logger.info("Dry run: signals are logged and no orders are sent.")
        models = {}
        for requested in settings.symbols:
            resolved = resolve_symbol(requested)
            models[resolved] = load_payload(settings.model_path(resolved), settings)
            logger.info("loaded model for %s", resolved)
        guard = DailyRiskGuard(
            settings.state_path,
            settings.strategy.max_daily_loss,
            settings.strategy.risk_pct,
        )
        while True:
            account = assert_trade_account(settings)
            server_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                hours=settings.broker_utc_offset
            )
            allowed, why = guard.allow_new_trade(float(account["equity"]), server_now)
            calendar = load_calendar(settings.news_calendar)
            for symbol, payload in models.items():
                try:
                    _process_bar(settings, symbol, payload, guard, allowed, why, calendar)
                except Exception:
                    logger.exception("bar processing failed for %s", symbol)
            time.sleep(20)
    finally:
        shutdown()


def _process_bar(settings, symbol, payload, guard, allowed, why, calendar) -> None:
    m15 = drop_forming_bar(copy_rates(symbol, "TIMEFRAME_M15", 3000))
    h1 = drop_forming_bar(copy_rates(symbol, "TIMEFRAME_H1", 800))
    frame = build_feature_frame(m15, h1, settings.strategy)
    row = frame.iloc[-1]
    bar_time = pd.Timestamp(row["close_time"]).isoformat()
    if guard.already_processed(symbol, bar_time):
        return
    if not allowed:
        logger.warning("%s HOLD %s", symbol, why)
        guard.mark_processed(symbol, bar_time)
        return
    if settings.news_filter_enabled:
        if not calendar.ok:
            logger.error("%s HOLD %s", symbol, calendar.reason)
            guard.mark_processed(symbol, bar_time)
            return
        moment = pd.Timestamp(row["close_time"]).to_pydatetime().replace(tzinfo=None)
        if in_news_blackout(
            moment,
            symbol,
            calendar.events,
            settings.broker_utc_offset,
            settings.strategy.news_blackout_minutes,
        ):
            logger.info("%s HOLD high-impact news blackout", symbol)
            guard.mark_processed(symbol, bar_time)
            return
    signal = decide(row, payload["long_model"], payload["short_model"], settings.strategy)
    probability = "n/a" if signal.probability is None else f"{signal.probability:.3f}"
    logger.info("%s %s probability=%s bar=%s (%s)", symbol, signal.action, probability, bar_time, signal.reason)
    guard.mark_processed(symbol, bar_time)
    if signal.action == "HOLD" or signal.atr is None:
        return
    if has_open_position(symbol, settings.magic):
        logger.info("%s HOLD an agent position is already open", symbol)
        return
    plan = plan_order(symbol, signal.action, signal.atr, settings)
    if plan is None:
        return
    send_order(plan, settings, signal.atr)
