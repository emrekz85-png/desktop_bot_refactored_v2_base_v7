"""
Utility functions for the trading bot.
Includes time conversion, funding calculation, and other helpers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Union, Optional

from .config import TRADING_CONFIG, MINUTES_PER_CANDLE


def utcnow() -> datetime:
    """
    Get current UTC time as a naive datetime.

    This replaces datetime.utcnow() which is deprecated in Python 3.12+.
    Returns a naive datetime (no timezone info) for backward compatibility
    with existing code that expects naive datetimes.

    Returns:
        datetime: Current UTC time without timezone info
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_datetime(dt: Union[datetime, pd.Timestamp, np.datetime64, None]) -> Optional[datetime]:
    """
    Normalize any datetime-like object to a naive (no timezone) datetime.

    Handles:
    - datetime objects (with or without timezone)
    - pd.Timestamp objects
    - np.datetime64 objects
    - None (returns None)

    Returns:
        datetime object without timezone info, or None
    """
    if dt is None:
        return None

    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    elif isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    # Remove timezone info if present
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return dt


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    """
    Convert timeframe string to pandas Timedelta.

    Args:
        tf: Timeframe string (e.g., "5m", "1h", "4h", "1d")

    Returns:
        pd.Timedelta representing the timeframe duration

    Raises:
        ValueError: If timeframe format is not supported
    """
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")


def tf_to_minutes(tf: str) -> int:
    """
    Convert timeframe string to minutes.

    Args:
        tf: Timeframe string (e.g., "5m", "1h", "4h", "1d")

    Returns:
        Number of minutes in the timeframe
    """
    return MINUTES_PER_CANDLE.get(tf, 60)  # Default to 1h if unknown


def calculate_funding_cost(
    open_time: Union[datetime, pd.Timestamp, str],
    close_time: Union[datetime, pd.Timestamp],
    notional_value: float,
    funding_rate_8h: float = None
) -> float:
    """
    Calculate funding cost based on actual time held (not bar count).

    Funding on Binance Futures is charged every 8 hours (00:00, 08:00, 16:00 UTC).
    This function calculates the funding cost based on actual hours held,
    properly accounting for all timeframes.

    Args:
        open_time: Trade open time (UTC)
        close_time: Trade close time (UTC)
        notional_value: Position notional value (size * price)
        funding_rate_8h: 8-hour funding rate (default from TRADING_CONFIG)

    Returns:
        Funding cost in USD

    Note:
        Previous implementation used bars_held/96 which assumed 5m timeframe.
        This function uses actual time difference for accurate calculation
        across all timeframes.
    """
    if funding_rate_8h is None:
        funding_rate_8h = TRADING_CONFIG.get("funding_rate_8h", 0.0001)

    # Parse open_time if it's a string
    if isinstance(open_time, str):
        try:
            open_time = datetime.strptime(open_time, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            # Try alternative format
            try:
                open_time = pd.Timestamp(open_time).to_pydatetime()
            except Exception:
                return 0.0

    # Normalize both times
    open_dt = normalize_datetime(open_time)
    close_dt = normalize_datetime(close_time)

    if open_dt is None or close_dt is None:
        return 0.0

    # Calculate hours held
    try:
        hours_held = max(0.0, (close_dt - open_dt).total_seconds() / 3600.0)
    except Exception:
        return 0.0

    # Calculate funding cost: notional * rate * (hours / 8)
    # This properly scales the 8-hour rate to actual time held
    funding_cost = notional_value * funding_rate_8h * (hours_held / 8.0)

    return funding_cost


def format_time_utc(dt: Union[datetime, pd.Timestamp]) -> str:
    """
    Format a datetime to UTC ISO format string.

    Args:
        dt: datetime or Timestamp object

    Returns:
        String in format "YYYY-MM-DDTHH:MM:SSZ"
    """
    dt = normalize_datetime(dt)
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_time_local(dt: Union[datetime, pd.Timestamp], offset_hours: int = 3) -> str:
    """
    Format a datetime to local time string (with timezone offset).

    Args:
        dt: datetime or Timestamp object (UTC)
        offset_hours: Hours to add for local timezone (default: 3 for Turkey)

    Returns:
        String in format "YYYY-MM-DD HH:MM" (local time)

    Note:
        This function clearly indicates that the output is LOCAL time,
        not UTC. Use this for display purposes only.
    """
    dt = normalize_datetime(dt)
    if dt is None:
        return ""

    # Add timezone offset
    local_dt = dt + timedelta(hours=offset_hours)
    return local_dt.strftime("%Y-%m-%d %H:%M")


def append_trade_event(trade: dict, event_type: str, event_time, event_value=None):
    """
    Append an event to a trade's event list.

    Args:
        trade: Trade dictionary
        event_type: Type of event (e.g., "PARTIAL", "BE_SET", "TRAIL_SL")
        event_time: Time of the event
        event_value: Optional value associated with the event (e.g., price)
    """
    if "events" not in trade:
        trade["events"] = []

    event = {
        "type": event_type,
        "time": format_time_utc(event_time),
    }
    if event_value is not None:
        event["value"] = float(event_value) if not isinstance(event_value, str) else event_value

    trade["events"].append(event)


def apply_1m_profit_lock(trade: dict, tf: str, t_type: str, entry: float, dyn_tp: float, progress: float) -> bool:
    """
    Apply profit lock for 1m timeframe when price is very close to TP.

    Moves SL to lock in profit when progress towards TP is high.

    Args:
        trade: Trade dictionary
        tf: Timeframe
        t_type: Trade type ("LONG" or "SHORT")
        entry: Entry price
        dyn_tp: Dynamic take profit price
        progress: Progress towards TP (0.0 to 1.0+)

    Returns:
        True if profit lock was applied, False otherwise
    """
    if tf != "1m":
        return False

    if progress < 0.90:
        return False

    if trade.get("profit_lock_applied"):
        return False

    # Lock 70% of the progress
    lock_distance = abs(dyn_tp - entry) * 0.70

    if t_type == "LONG":
        new_sl = entry + lock_distance
        if new_sl > float(trade.get("sl", 0)):
            trade["sl"] = new_sl
            trade["profit_lock_applied"] = True
            return True
    else:
        new_sl = entry - lock_distance
        if new_sl < float(trade.get("sl", float("inf"))):
            trade["sl"] = new_sl
            trade["profit_lock_applied"] = True
            return True

    return False


def apply_partial_stop_protection(trade: dict, tf: str, progress: float, t_type: str) -> bool:
    """
    Apply stop protection after partial TP is taken.

    Args:
        trade: Trade dictionary
        tf: Timeframe
        progress: Progress towards TP
        t_type: Trade type ("LONG" or "SHORT")

    Returns:
        True if protection was applied, False otherwise
    """
    if not trade.get("partial_taken"):
        return False

    if trade.get("stop_protection"):
        return False

    # Protection threshold varies by timeframe
    protection_threshold = 0.30 if tf in ["1m", "5m"] else 0.25

    if progress < protection_threshold:
        entry = float(trade["entry"])
        # Small profit lock
        profit_lock = abs(float(trade["tp"]) - entry) * 0.10

        if t_type == "LONG":
            new_sl = entry + profit_lock
            if new_sl > float(trade.get("sl", 0)):
                trade["sl"] = new_sl
                trade["stop_protection"] = True
                return True
        else:
            new_sl = entry - profit_lock
            if new_sl < float(trade.get("sl", float("inf"))):
                trade["sl"] = new_sl
                trade["stop_protection"] = True
                return True

    return False


def calculate_r_multiple(pnl: float, risk_amount: float) -> float:
    """
    Calculate R-Multiple for a trade.

    R-Multiple = PnL / Risk Amount
    - R = 1.0 means you won as much as you risked
    - R = -1.0 means you lost as much as you risked (typical SL hit)
    - R > 1.0 means you won more than you risked (good trade)

    Args:
        pnl: Trade PnL in USD
        risk_amount: Risk amount in USD

    Returns:
        R-Multiple value
    """
    if risk_amount <= 0:
        return 0.0
    return pnl / risk_amount


def calculate_expected_r(r_multiples: list) -> float:
    """
    Calculate Expected R-Multiple (E[R]) from a list of trade R-multiples.

    E[R] = average of all R-multiples
    - E[R] > 0: Positive expectancy (profitable system)
    - E[R] = 0: Break-even system
    - E[R] < 0: Negative expectancy (losing system)

    This metric is independent of account size, leverage, and commission.
    It measures the true "edge" of the trading model.

    Args:
        r_multiples: List of R-multiple values

    Returns:
        Expected R-Multiple
    """
    if not r_multiples:
        return 0.0
    return sum(r_multiples) / len(r_multiples)
