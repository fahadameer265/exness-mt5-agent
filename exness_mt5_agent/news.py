"""High-impact news blackout.

The calendar is a CSV you maintain. The live loop does not scrape a news site.
When the filter is on and the file is missing or older than seven days, new
trades are blocked. A quiet parser failure must not become a trading session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HIGH_IMPACT = {"high", "h", "3", "red"}
STALE_AFTER = timedelta(days=7)


@dataclass(frozen=True)
class CalendarStatus:
    ok: bool
    reason: str
    events: pd.DataFrame


def symbol_currencies(symbol: str) -> set[str]:
    name = symbol.upper().split(".")[0]
    for suffix in ("MICRO", "PRO", "M", "C", "R"):
        if name.endswith(suffix) and len(name) - len(suffix) >= 6:
            name = name[: -len(suffix)]
            break
    if "XAU" in name or "GOLD" in name:
        return {"USD", "XAU"}
    letters = "".join(ch for ch in name if ch.isalpha())
    if len(letters) >= 6:
        return {letters[:3], letters[3:6]}
    return {"USD"}


def _impact_is_high(value: object) -> bool:
    return str(value).strip().lower() in HIGH_IMPACT


def load_calendar(path: Path, now_utc: datetime | None = None) -> CalendarStatus:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if not path.exists():
        return CalendarStatus(False, f"news calendar not found: {path}", pd.DataFrame())
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        return CalendarStatus(False, f"news calendar unreadable: {exc}", pd.DataFrame())
    required = {"time_utc", "currency", "impact"}
    missing = required.difference(frame.columns)
    if missing:
        return CalendarStatus(False, f"news calendar missing columns: {sorted(missing)}", pd.DataFrame())
    frame = frame.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce")
    frame["currency"] = frame["currency"].astype(str).str.upper().str.strip()
    frame = frame.dropna(subset=["time_utc"])
    frame = frame.loc[frame["impact"].map(_impact_is_high)].sort_values("time_utc")
    if frame.empty:
        return CalendarStatus(False, "news calendar has no high-impact rows", frame)
    newest = frame["time_utc"].max().to_pydatetime()
    if newest < now - STALE_AFTER:
        return CalendarStatus(
            False,
            f"news calendar is stale; newest high-impact event is {newest.isoformat()}",
            frame,
        )
    return CalendarStatus(True, f"{len(frame)} high-impact events loaded", frame.reset_index(drop=True))


def server_time_to_utc(server_naive: datetime, broker_utc_offset_hours: float) -> datetime:
    """Convert a naive Exness server timestamp to true UTC."""

    if server_naive.tzinfo is not None:
        raise ValueError("server time must be naive broker time")
    return (server_naive - timedelta(hours=broker_utc_offset_hours)).replace(tzinfo=timezone.utc)


def in_news_blackout(
    server_naive: datetime,
    symbol: str,
    calendar: pd.DataFrame,
    broker_utc_offset_hours: float,
    minutes: int = 30,
) -> bool:
    """True when a relevant high-impact event is within `minutes` of this time."""

    if calendar.empty:
        return False
    moment = server_time_to_utc(server_naive, broker_utc_offset_hours)
    currencies = symbol_currencies(symbol)
    relevant = calendar.loc[calendar["currency"].isin(currencies), "time_utc"]
    if relevant.empty:
        return False
    delta = (relevant - pd.Timestamp(moment)).abs()
    return bool((delta <= pd.Timedelta(minutes=minutes)).any())
