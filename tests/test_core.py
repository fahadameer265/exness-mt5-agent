"""Core behaviour that must stay true without a terminal connected."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from exness_mt5_agent.backtest import SymbolSpec, run_backtest
from exness_mt5_agent.config import StrategyParams
from exness_mt5_agent.features import (
    FEATURE_COLUMNS,
    add_price_features,
    apply_setup_masks,
    merge_higher_timeframe,
    trend_age,
)
from exness_mt5_agent.labels import label_side
from exness_mt5_agent.news import in_news_blackout, load_calendar, symbol_currencies
from exness_mt5_agent.path import scan_trade
from exness_mt5_agent.risk import (
    DailyRiskGuard,
    daily_loss_reached,
    lots_for_risk,
    stops_are_valid,
)
from exness_mt5_agent.signals import decide
from exness_mt5_agent.validation import precision_at_threshold, purged_folds


class FakeModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame):
        row = np.array([1.0 - self.probability, self.probability])
        return np.tile(row, (len(frame), 1))


def _bars(n=30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "open": np.full(n, 9.5),
            "high": np.full(n, 10.0),
            "low": np.full(n, 9.0),
            "close": np.full(n, 9.5),
            "tick_volume": np.full(n, 100.0),
        }
    )


def _bullish_setup(**overrides) -> pd.DataFrame:
    row = {
        "h1_trend": 1,
        "h1_trend_age": 5,
        "ema_trend": 1,
        "trend_age": 5,
        "close": 1.10,
        "open": 1.09,
        "high": 1.11,
        "low": 1.00,
        "ema_fast": 1.05,
        "ema_slow": 1.00,
        "broke_up": False,
        "failed_up_break": False,
        "broke_down": False,
        "failed_down_break": False,
        "rsi_rank": 0.40,
        "rsi": 50.0,
        "vol_ratio": 1.5,
        "atr": 0.01,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _signal_row(**overrides) -> pd.Series:
    data = {column: 0.2 for column in FEATURE_COLUMNS}
    data.update({"atr": 1.0, "long_setup": False, "short_setup": False, "close_time": "2026-01-01T10:15:00"})
    data.update(overrides)
    return pd.Series(data)


def _flat_market(n=16) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-05", periods=n, freq="15min"),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.2),
            "atr": np.full(n, 1.0),
            "spread": np.zeros(n),
        }
    )


class FeatureTests(unittest.TestCase):
    def test_changing_the_latest_close_does_not_change_the_prior_ema(self):
        base = _bars()
        changed = base.copy()
        changed.loc[changed.index[-1], "close"] = 999.0
        params = StrategyParams()
        original = add_price_features(base, params, 15)
        altered = add_price_features(changed, params, 15)
        self.assertEqual(original["ema_fast"].iloc[-2], altered["ema_fast"].iloc[-2])

    def test_trend_age_resets_when_the_emas_cross(self):
        fast = pd.Series([1, 2, 3, 2, 1, 2, 3], dtype=float)
        slow = pd.Series([2, 2, 2, 2, 2, 2, 2], dtype=float)
        self.assertEqual(list(trend_age(fast, slow)), [1, 2, 1, 1, 2, 3, 1])

    def test_failed_breakout_is_flagged_after_price_falls_back(self):
        frame = _bars(30)
        frame.loc[22, ["open", "high", "close"]] = [10.2, 11.0, 11.0]
        frame.loc[23, ["open", "high", "low", "close"]] = [10.0, 10.0, 8.5, 9.0]
        features = add_price_features(frame, StrategyParams(), 15)
        self.assertTrue(bool(features.loc[22, "broke_up"]))
        self.assertTrue(bool(features.loc[23, "failed_up_break"]))

    def test_h1_bar_is_used_only_after_it_closes(self):
        higher = pd.DataFrame(
            {
                "close_time": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00"]),
                "ema_spread_atr": [1.0, 9.0],
                "rsi": [55.0, 10.0],
                "ema_trend": [1.0, -1.0],
                "trend_age": [5.0, 1.0],
            }
        )
        lower = pd.DataFrame({"close_time": pd.to_datetime(["2026-01-01 10:15"]), "close": [1.0]})
        merged = merge_higher_timeframe(lower, higher)
        self.assertEqual(merged.loc[0, "h1_ema_spread_atr"], 1.0)
        self.assertEqual(merged.loc[0, "h1_trend"], 1.0)

    def test_breakout_and_failed_break_cannot_be_a_long_setup(self):
        params = StrategyParams()
        clean = apply_setup_masks(_bullish_setup(), params)
        chasing = apply_setup_masks(_bullish_setup(broke_up=True), params)
        failed = apply_setup_masks(_bullish_setup(failed_up_break=True), params)
        self.assertTrue(bool(clean.loc[0, "long_setup"]))
        self.assertFalse(bool(chasing.loc[0, "long_setup"]))
        self.assertFalse(bool(failed.loc[0, "long_setup"]))


class LabelTests(unittest.TestCase):
    def test_target_stop_and_timeout(self):
        params = StrategyParams()
        winner = _flat_market()
        winner.loc[6, "high"] = 104.0
        self.assertEqual(label_side(winner, 1, params, point=0.00001, digits=5)[0], 1)

        both = _flat_market()
        both.loc[1, ["high", "low"]] = [104.0, 98.0]
        self.assertEqual(label_side(both, 1, params, point=0.00001, digits=5)[0], 0)

        quiet = _flat_market()
        self.assertEqual(label_side(quiet, 1, params, point=0.00001, digits=5)[0], 0)

        short = _flat_market()
        short["high"] = 101.0
        short.loc[4, "low"] = 96.0
        self.assertEqual(label_side(short, -1, params, point=0.00001, digits=5)[0], 1)

    def test_a_gap_through_the_stop_fills_at_the_open(self):
        scanned = scan_trade(
            side="BUY",
            entry_index=1,
            open_=np.array([100.0, 100.0, 90.0]),
            high=np.array([101.0, 101.0, 91.0]),
            low=np.array([99.0, 99.0, 89.0]),
            close=np.array([100.0, 100.0, 90.0]),
            spread_price=np.zeros(3),
            atr=1.0,
            sl_atr=1.5,
            tp_atr=3.0,
            horizon=2,
            digits=5,
            require_full_horizon=True,
        )
        self.assertIsNotNone(scanned)
        self.assertEqual(scanned.reason, "gap_stop")
        self.assertEqual(scanned.exit_price, 90.0)
        self.assertEqual(scanned.win, 0)


class RiskTests(unittest.TestCase):
    def test_lot_size_floors_and_refuses_an_oversized_minimum(self):
        self.assertAlmostEqual(lots_for_risk(10_000, 0.01, 250, 0.01, 100, 0.01, 20), 0.4)
        self.assertIsNone(lots_for_risk(10_000, 0.01, 250, 0.5, 100, 0.01, 20))
        self.assertAlmostEqual(lots_for_risk(10_000, 0.01, 1.5, 0.01, 100, 0.01, 100), 66.66)
        self.assertIsNone(lots_for_risk(10_000, 0.01, 1.5, 0.01, 100, 0.01, 20))
        self.assertIsNone(lots_for_risk(10_000_000, 0.01, 1.0, 0.01, 500, 0.01, 20))

    def test_stop_must_sit_on_the_correct_side_of_price(self):
        self.assertTrue(stops_are_valid("BUY", 1.10, 1.00, 1.30, 0.05))
        self.assertFalse(stops_are_valid("BUY", 1.10, 1.00, 1.30, 0.20))
        self.assertFalse(stops_are_valid("SELL", 1.10, 1.00, 0.90, 0.0))

    def test_daily_halt_persists_across_restarts(self):
        path = Path(self.id().replace(".", "_") + ".json")
        try:
            guard = DailyRiskGuard(path, 0.03)
            allowed, _ = guard.allow_new_trade(10_000, datetime(2026, 11, 6, 10, 0))
            self.assertTrue(allowed)
            allowed, _ = guard.allow_new_trade(9_700, datetime(2026, 11, 6, 12, 0))
            self.assertFalse(allowed)
            reloaded = DailyRiskGuard(path, 0.03)
            allowed, _ = reloaded.allow_new_trade(9_700, datetime(2026, 11, 6, 13, 0))
            self.assertFalse(allowed)
            allowed, _ = reloaded.allow_new_trade(9_700, datetime(2026, 11, 7, 0, 5))
            self.assertTrue(allowed)
            self.assertTrue(daily_loss_reached(10_000, 9_700, 0.03))
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".json.tmp").unlink(missing_ok=True)

    def test_corrupt_state_blocks_trading(self):
        path = Path(self.id().replace(".", "_") + "_bad.json")
        path.write_text("{", encoding="utf-8")
        try:
            with self.assertRaises(RuntimeError):
                DailyRiskGuard(path, 0.03)
        finally:
            path.unlink(missing_ok=True)


class NewsTests(unittest.TestCase):
    def test_blackout_is_thirty_minutes_either_side_in_server_time(self):
        path = Path(self.id().replace(".", "_") + ".csv")
        path.write_text(
            "time_utc,currency,impact,title\n2026-11-06T13:30:00Z,USD,high,Payrolls\n",
            encoding="utf-8",
        )
        try:
            status = load_calendar(path, now_utc=datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc))
            self.assertTrue(status.ok)
            self.assertTrue(
                in_news_blackout(datetime(2026, 11, 6, 15, 0), "EURUSDm", status.events, 2, 30)
            )
            self.assertFalse(
                in_news_blackout(datetime(2026, 11, 6, 14, 59), "EURUSD", status.events, 2, 30)
            )
            self.assertIn("USD", symbol_currencies("XAUUSDm"))
        finally:
            path.unlink(missing_ok=True)

    def test_stale_or_missing_calendar_is_not_tradable(self):
        missing = load_calendar(Path("does_not_exist.csv"))
        self.assertFalse(missing.ok)
        stale = Path(self.id().replace(".", "_") + ".csv")
        stale.write_text(
            "time_utc,currency,impact,title\n2020-01-01T00:00:00Z,USD,high,Old\n",
            encoding="utf-8",
        )
        try:
            self.assertFalse(load_calendar(stale).ok)
        finally:
            stale.unlink(missing_ok=True)


class ModelGateTests(unittest.TestCase):
    def test_probability_cutoff_and_conflict(self):
        params = StrategyParams()
        buy = decide(_signal_row(long_setup=True), FakeModel(0.71), None, params)
        quiet = decide(_signal_row(long_setup=True), FakeModel(0.69), None, params)
        conflict = decide(_signal_row(long_setup=True, short_setup=True), FakeModel(0.9), FakeModel(0.9), params)
        hold = decide(_signal_row(), FakeModel(0.99), FakeModel(0.99), params)
        self.assertEqual(buy.action, "BUY")
        self.assertEqual(quiet.action, "HOLD")
        self.assertEqual(conflict.action, "HOLD")
        self.assertEqual(hold.action, "HOLD")

    def test_precision_is_silent_when_nothing_clears_the_threshold(self):
        precision, count = precision_at_threshold(np.array([0, 1]), np.array([0.1, 0.2]), 0.70)
        self.assertIsNone(precision)
        self.assertEqual(count, 0)

    def test_embargo_keeps_training_labels_out_of_the_test_window(self):
        bar_index = np.arange(120)
        folds = list(purged_folds(bar_index, n_splits=5, embargo=12))
        self.assertGreaterEqual(len(folds), 4)
        for train_pos, test_pos in folds:
            self.assertLess(bar_index[train_pos].max(), bar_index[test_pos].min() - 12)


class BacktestTests(unittest.TestCase):
    def _spec(self) -> SymbolSpec:
        return SymbolSpec(0.01, 2, 1.0, 0.01, 100.0, 0.01, 50)

    def _frame(self, n=8) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "open": np.full(n, 100.0),
                "high": np.full(n, 100.4),
                "low": np.full(n, 99.6),
                "close": np.full(n, 100.1),
                "atr": np.full(n, 1.0),
                "spread": np.zeros(n),
                "long_setup": np.zeros(n, dtype=bool),
                "short_setup": np.zeros(n, dtype=bool),
                "close_time": pd.date_range("2026-01-05", periods=n, freq="15min"),
            }
        )
        return frame

    def test_target_pays_about_two_r_after_step_rounding(self):
        frame = self._frame()
        frame.loc[0, "long_setup"] = True
        frame.loc[4, ["high", "low"]] = [104.0, 100.0]
        result = run_backtest(
            frame,
            np.array([0.9, 0, 0, 0, 0, 0, 0, 0]),
            np.zeros(8),
            StrategyParams(),
            self._spec(),
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.loc[0, "reason"], "target")
        self.assertAlmostEqual(result.trades.loc[0, "pnl"], 199.98, places=2)
        self.assertLessEqual(result.trades.loc[0, "lots"] * 1.5, 100.0 + 1e-6)

    def test_three_percent_day_stops_the_next_entry(self):
        frame = self._frame(10)
        frame.loc[[0, 2, 4, 6], "long_setup"] = True
        frame.loc[[1, 3, 5, 7], "low"] = 90.0
        probabilities = np.zeros(10)
        probabilities[[0, 2, 4, 6]] = 0.9
        result = run_backtest(
            frame,
            probabilities,
            np.zeros(10),
            StrategyParams(),
            self._spec(),
        )
        self.assertEqual(len(result.trades), 3)
        self.assertGreater(result.summary["ending_equity"], 10_000 * 0.97)


if __name__ == "__main__":
    unittest.main()
