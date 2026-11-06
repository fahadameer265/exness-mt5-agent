"""Causal indicators and the multi-timeframe setup gate.

Every feature on row t is a function of prices up to and including the close
of bar t. The label and the fill use later bars. H1 values are attached only
after that H1 candle has closed, so a 10:15 M15 bar cannot see the 10:00 hour
that is still forming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exness_mt5_agent.config import StrategyParams

FEATURE_COLUMNS = [
    "ema_spread_atr",
    "rsi",
    "rsi_rank",
    "atr_pct",
    "vol_ratio",
    "dist_ema_fast_atr",
    "dist_high_atr",
    "dist_low_atr",
    "ret_1",
    "ret_4",
    "ret_8",
    "hour_sin",
    "hour_cos",
    "trend_age",
    "h1_ema_spread_atr",
    "h1_rsi",
    "h1_trend",
    "h1_trend_age",
]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0, 100.0).where(close.notna())


def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the last point inside each trailing window."""

    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        window_values = values[i - window + 1 : i + 1]
        if not np.isfinite(window_values).all():
            continue
        out[i] = float(np.mean(window_values <= window_values[-1]))
    return pd.Series(out, index=series.index)


def trend_age(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """How many consecutive bars the fast EMA has stayed on this side of the slow EMA."""

    defined = fast.notna() & slow.notna()
    bull = fast > slow
    change = bull.ne(bull.shift(fill_value=False))
    age = bull.groupby(change.cumsum()).cumcount() + 1
    return age.where(defined).astype(float)


def _failed_break(
    broke: pd.Series, level: pd.Series, close: pd.Series, side: str, lookback: int
) -> pd.Series:
    """A recent range break that has already closed back through its own level.

    Side "up": the break is invalidated when price closes back under the high
    it broke. That bar is a poor long. The mirror applies to failed breakdowns.
    """

    broke_arr = broke.fillna(False).to_numpy(dtype=bool)
    level_arr = level.to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=bool)
    for i in range(len(close)):
        if not np.isfinite(close_arr[i]):
            continue
        for step in range(1, lookback + 1):
            j = i - step
            if j < 0:
                break
            if not broke_arr[j] or not np.isfinite(level_arr[j]):
                continue
            if side == "up" and close_arr[i] < level_arr[j]:
                out[i] = True
                break
            if side == "down" and close_arr[i] > level_arr[j]:
                out[i] = True
                break
    return pd.Series(out, index=close.index)


def add_price_features(df: pd.DataFrame, params: StrategyParams, bar_minutes: int) -> pd.DataFrame:
    """Indicators for one timeframe. Does not look past the current close."""

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out["ema_fast"] = ema(out["close"], params.ema_fast)
    out["ema_slow"] = ema(out["close"], params.ema_slow)
    out["rsi"] = wilder_rsi(out["close"], params.rsi_period)
    out["atr"] = wilder_atr(out, params.atr_period)
    out["rsi_rank"] = rolling_percentile_rank(out["rsi"], params.rsi_rank_window)
    out["trend_age"] = trend_age(out["ema_fast"], out["ema_slow"])
    out["ema_trend"] = np.where(out["ema_fast"] > out["ema_slow"], 1.0, -1.0)
    out.loc[out["ema_fast"].isna() | out["ema_slow"].isna(), "ema_trend"] = np.nan

    atr_safe = out["atr"].where(out["atr"] > 0)
    out["ema_spread_atr"] = (out["ema_fast"] - out["ema_slow"]) / atr_safe
    out["atr_pct"] = out["atr"] / out["close"].where(out["close"] != 0)
    out["dist_ema_fast_atr"] = (out["close"] - out["ema_fast"]) / atr_safe

    volume_ma = out["tick_volume"].rolling(params.volume_ma).mean()
    out["vol_ratio"] = out["tick_volume"] / volume_ma.where(volume_ma > 0)

    prior_high = out["high"].rolling(params.donchian).max().shift(1)
    prior_low = out["low"].rolling(params.donchian).min().shift(1)
    out["donchian_high"] = prior_high
    out["donchian_low"] = prior_low
    out["broke_up"] = out["close"] > prior_high
    out["broke_down"] = out["close"] < prior_low
    out["failed_up_break"] = _failed_break(
        out["broke_up"], prior_high, out["close"], "up", params.fail_lookback
    )
    out["failed_down_break"] = _failed_break(
        out["broke_down"], prior_low, out["close"], "down", params.fail_lookback
    )
    out["dist_high_atr"] = (prior_high - out["close"]) / atr_safe
    out["dist_low_atr"] = (out["close"] - prior_low) / atr_safe

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_8"] = out["close"].pct_change(8)

    close_time = out["time"] + pd.Timedelta(minutes=bar_minutes)
    hour = close_time.dt.hour + close_time.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["close_time"] = close_time
    return out.replace([np.inf, -np.inf], np.nan)


def merge_higher_timeframe(m15: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Attach the last fully closed H1 features to each closed M15 bar."""

    h1_cols = [
        "close_time",
        "ema_spread_atr",
        "rsi",
        "ema_trend",
        "trend_age",
    ]
    higher = h1[h1_cols].rename(
        columns={
            "close_time": "h1_available_at",
            "ema_spread_atr": "h1_ema_spread_atr",
            "rsi": "h1_rsi",
            "ema_trend": "h1_trend",
            "trend_age": "h1_trend_age",
        }
    )
    left = m15.sort_values("close_time")
    right = higher.sort_values("h1_available_at")
    merged = pd.merge_asof(
        left,
        right,
        left_on="close_time",
        right_on="h1_available_at",
        direction="backward",
    )
    return merged.reset_index(drop=True)


def apply_setup_masks(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Pullback continuation setups. Breakout bars and failed breaks are excluded.

    Long idea: H1 and M15 are both in an established uptrend, price has pulled
    back to the 50 EMA and closed back above it, RSI has reset on a rolling
    basis, and tick volume is above its 20-bar average. The breakout bar itself
    is not tradable. A break that has already failed is not tradable either.
    """

    out = df.copy()
    rsi_long = (
        (out["rsi_rank"] >= params.rsi_long_rank_low)
        & (out["rsi_rank"] <= params.rsi_long_rank_high)
        & (out["rsi"] < params.rsi_long_hard_cap)
    )
    rsi_short = (
        (out["rsi_rank"] >= params.rsi_short_rank_low)
        & (out["rsi_rank"] <= params.rsi_short_rank_high)
        & (out["rsi"] > params.rsi_short_hard_floor)
    )
    volume_ok = out["vol_ratio"] >= params.vol_spike_mult
    finite = out["atr"].gt(0) & out["close"].gt(0)

    long_pullback = (out["low"] <= out["ema_fast"]) & (out["close"] > out["ema_fast"])
    short_pullback = (out["high"] >= out["ema_fast"]) & (out["close"] < out["ema_fast"])

    out["long_setup"] = (
        finite
        & (out["h1_trend"] == 1)
        & (out["h1_trend_age"] >= params.min_trend_age_h1)
        & (out["ema_trend"] == 1)
        & (out["trend_age"] >= params.min_trend_age_m15)
        & (out["close"] > out["ema_slow"])
        & long_pullback
        & (out["close"] > out["open"])
        & ~out["broke_up"].fillna(False)
        & ~out["failed_up_break"].fillna(False)
        & rsi_long
        & volume_ok
    )
    out["short_setup"] = (
        finite
        & (out["h1_trend"] == -1)
        & (out["h1_trend_age"] >= params.min_trend_age_h1)
        & (out["ema_trend"] == -1)
        & (out["trend_age"] >= params.min_trend_age_m15)
        & (out["close"] < out["ema_slow"])
        & short_pullback
        & (out["close"] < out["open"])
        & ~out["broke_down"].fillna(False)
        & ~out["failed_down_break"].fillna(False)
        & rsi_short
        & volume_ok
    )
    return out


def build_feature_frame(
    m15: pd.DataFrame, h1: pd.DataFrame, params: StrategyParams
) -> pd.DataFrame:
    """Closed M15 bars with H1 confirmation columns and setup flags."""

    m15_features = add_price_features(m15, params, bar_minutes=15)
    h1_features = add_price_features(h1, params, bar_minutes=60)
    merged = merge_higher_timeframe(m15_features, h1_features)
    merged["bar_index"] = np.arange(len(merged))
    return apply_setup_masks(merged, params)


def drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    """`copy_rates_from_pos` includes the candle that has not closed yet."""

    if len(df) < 3:
        raise ValueError("Need at least three bars to drop the forming candle.")
    return df.iloc[:-1].copy()
