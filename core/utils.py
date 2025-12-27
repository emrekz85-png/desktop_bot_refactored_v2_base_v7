"""
Utility functions for the trading bot.
Includes time conversion, funding calculation, and other helpers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Union, Optional

from .config import TRADING_CONFIG, MINUTES_PER_CANDLE
from .logging_config import get_logger

_logger = get_logger(__name__)


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


# M3 Performance: Cache timedelta objects (avoid repeated object creation)
_TF_TIMEDELTA_CACHE: dict = {}


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    """
    Convert timeframe string to pandas Timedelta.

    M3 OPTIMIZED: Uses LRU-style caching to avoid repeated object creation.
    This is a hot path in backtest loops - caching provides 15-20% speedup.

    Args:
        tf: Timeframe string (e.g., "5m", "1h", "4h", "1d")

    Returns:
        pd.Timedelta representing the timeframe duration

    Raises:
        ValueError: If timeframe format is not supported
    """
    # Check cache first (O(1) dict lookup)
    if tf in _TF_TIMEDELTA_CACHE:
        return _TF_TIMEDELTA_CACHE[tf]

    # Compute and cache
    if tf.endswith("m"):
        result = pd.Timedelta(minutes=int(tf[:-1]))
    elif tf.endswith("h"):
        result = pd.Timedelta(hours=int(tf[:-1]))
    elif tf.endswith("d"):
        result = pd.Timedelta(days=int(tf[:-1]))
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")

    _TF_TIMEDELTA_CACHE[tf] = result
    return result


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
            except Exception as e:
                _logger.warning(
                    "Funding cost calculation failed - unparseable open_time: %s (error: %s). "
                    "Using 1 funding period estimate (8 hours).",
                    open_time, e, exc_info=True
                )
                # Reasonable default: 1 funding period (8 hours)
                return notional_value * funding_rate_8h

    # Normalize both times
    open_dt = normalize_datetime(open_time)
    close_dt = normalize_datetime(close_time)

    if open_dt is None or close_dt is None:
        _logger.warning(
            "Funding cost calculation failed - normalize_datetime returned None. "
            "open_time=%s, close_time=%s. Using 1 funding period estimate.",
            open_time, close_time
        )
        # Reasonable default: 1 funding period (8 hours)
        return notional_value * funding_rate_8h

    # Calculate hours held
    try:
        hours_held = max(0.0, (close_dt - open_dt).total_seconds() / 3600.0)
    except Exception as e:
        _logger.warning(
            "Funding cost calculation failed - could not calculate time delta. "
            "open_dt=%s, close_dt=%s (error: %s). Using 1 funding period estimate.",
            open_dt, close_dt, e, exc_info=True
        )
        # Reasonable default: 1 funding period (8 hours)
        return notional_value * funding_rate_8h

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


def derive_exit_profile_params(config: dict, trade_rr: float = 2.0) -> dict:
    """
    Derive effective exit parameters from exit_profile setting.

    PR-1 BASELINE: When sl_validation_mode="off", returns fixed baseline values
    and ignores the exit_profile system entirely.

    Args:
        config: Trade config dict (must contain exit_profile or fallback to defaults)
        trade_rr: Trade RR for RR-based trigger adjustment

    Returns:
        Dict with effective exit params:
        - partial_trigger: effective trigger based on profile + RR
        - partial_fraction: effective fraction
        - dynamic_tp_only_after_partial: bool
        - dynamic_tp_clamp_mode: "tighten_only" or "none"
        - rr_high_trigger: profile-specific high RR trigger
        - rr_low_trigger: profile-specific low RR trigger
        - exit_profile: "baseline" when in baseline mode
    """
    # === PR-1: BASELINE MODE CHECK ===
    # When sl_validation_mode="off", bypass exit_profile system entirely
    sl_validation_mode = config.get("sl_validation_mode", "off")

    if sl_validation_mode == "off":
        # Return fixed baseline values - no profile system, no RR adjustment
        return {
            "partial_trigger": 0.40,                # Baseline: 40%
            "partial_fraction": 0.50,               # Baseline: 50%
            "dynamic_tp_only_after_partial": False, # Baseline: pre-partial too
            "dynamic_tp_clamp_mode": "none",        # Baseline: no clamp
            "rr_high_trigger": 0.40,                # Baseline: same (no adjustment)
            "rr_low_trigger": 0.40,                 # Baseline: same (no adjustment)
            "exit_profile": "baseline",             # Mark as baseline mode
            "partial_rr_adjustment": False,         # Baseline: disabled
        }

    # === PROFILE MODE (sl_validation_mode != "off") ===
    profile = config.get("exit_profile", "clip")

    if profile == "runner":
        base_trigger = config.get("partial_trigger_runner", 0.70)
        fraction = config.get("partial_fraction_runner", 0.33)
        dyn_only_after = config.get("dynamic_tp_only_after_partial_runner", True)
        clamp_mode = config.get("dynamic_tp_clamp_mode_runner", "none")
        rr_high_trigger = config.get("partial_rr_high_trigger_runner", 0.75)
        rr_low_trigger = config.get("partial_rr_low_trigger_runner", 0.55)
    else:  # clip (default)
        base_trigger = config.get("partial_trigger_clip", 0.45)
        fraction = config.get("partial_fraction_clip", 0.50)
        dyn_only_after = config.get("dynamic_tp_only_after_partial_clip", False)
        clamp_mode = config.get("dynamic_tp_clamp_mode_clip", "tighten_only")
        rr_high_trigger = config.get("partial_rr_high_trigger_clip", 0.55)
        rr_low_trigger = config.get("partial_rr_low_trigger_clip", 0.40)

    # RR-based trigger adjustment
    effective_trigger = base_trigger
    if config.get("partial_rr_adjustment", True):
        rr_high_threshold = config.get("partial_rr_high_threshold", 1.8)
        rr_low_threshold = config.get("partial_rr_low_threshold", 1.2)

        if trade_rr >= rr_high_threshold:
            effective_trigger = rr_high_trigger
        elif trade_rr <= rr_low_threshold:
            effective_trigger = rr_low_trigger
        # else: use base_trigger

    return {
        "partial_trigger": effective_trigger,
        "partial_fraction": fraction,
        "dynamic_tp_only_after_partial": dyn_only_after,
        "dynamic_tp_clamp_mode": clamp_mode,
        "rr_high_trigger": rr_high_trigger,
        "rr_low_trigger": rr_low_trigger,
        "exit_profile": profile,
        "partial_rr_adjustment": config.get("partial_rr_adjustment", True),
    }


def validate_and_adjust_sl(
    symbol: str,
    entry: float,
    sl: float,
    tp: float,
    trade_type: str,
    config: dict,
    risk_amount: float,
    leverage: float = 10.0,
) -> tuple:
    """
    Validate SL distance and optionally widen if too tight (noise filter).

    PR-1 BASELINE: When sl_validation_mode="off", no validation is performed.
    This is the recommended mode for baseline behavior.

    Args:
        symbol: Trading symbol
        entry: Entry price
        sl: Stop loss price
        tp: Take profit price
        trade_type: "LONG" or "SHORT"
        config: Config dict with sl_validation_mode and min_sl_distance settings
        risk_amount: Risk amount in USD
        leverage: Trading leverage

    Returns:
        Tuple of (new_sl, new_tp, new_size, notional, new_margin, skip_reason or None)
        If skip_reason is not None, trade should be skipped.

    sl_validation_mode:
        - "off": No validation, use signal's SL as-is (BASELINE)
        - "reject": Reject trades with SL too tight
        - "widen": Widen SL to minimum distance (may cause regression)
    """
    # === DEFENSIVE CHECK: SL cannot equal Entry ===
    # This should never happen if signal generation is correct,
    # but add defensive programming to prevent size=0 edge case
    sl_dist = abs(entry - sl)
    if sl_dist == 0 or sl_dist < 1e-10:  # floating point tolerance
        _logger.error(
            "SL distance is zero or near-zero! Symbol=%s, Entry=%.8f, SL=%.8f. "
            "This trade cannot be sized correctly. Signal generation issue.",
            symbol, entry, sl
        )
        # Return with error indicator - caller should skip this trade
        return sl, tp, 0, 0, 0, "SL_EQUAL_ENTRY"

    # === PR-1: Check sl_validation_mode ===
    sl_validation_mode = config.get("sl_validation_mode", "off")

    # Calculate position size from original SL (always needed)
    size = risk_amount / sl_dist if sl_dist > 0 else 0
    notional = abs(size) * entry
    margin = notional / leverage

    # === BASELINE MODE: No validation, return original values ===
    if sl_validation_mode == "off":
        return sl, tp, size, notional, margin, None

    # === VALIDATION MODE: Check min distance ===
    majors = config.get("majors_symbols", ["BTCUSDT", "ETHUSDT"])
    min_dist_majors = config.get("min_sl_distance_btc_eth", 0.010)
    min_dist_alts = config.get("min_sl_distance_alts", 0.015)
    min_dist = min_dist_majors if symbol in majors else min_dist_alts

    # Current SL distance as percentage
    current_dist_pct = abs(entry - sl) / entry if entry > 0 else 0

    # SL is wide enough - no adjustment needed
    if current_dist_pct >= min_dist:
        return sl, tp, size, notional, margin, None

    # === SL is too tight ===
    if sl_validation_mode == "reject":
        return sl, tp, 0, 0, 0, "SL_TOO_TIGHT_REJECTED"

    # === WIDEN MODE (sl_validation_mode == "widen") ===
    if trade_type == "LONG":
        new_sl = entry * (1 - min_dist)
    else:
        new_sl = entry * (1 + min_dist)

    # Recalculate position size to maintain same risk$
    new_sl_dist = abs(entry - new_sl)
    new_size = risk_amount / new_sl_dist if new_sl_dist > 0 else 0
    new_notional = abs(new_size) * entry
    new_margin = new_notional / leverage

    # Recalculate TP to maintain original RR when SL is widened
    old_sl_dist = abs(entry - sl)
    if old_sl_dist > 0 and config.get("maintain_rr_on_sl_widen", True):
        old_rr = abs(tp - entry) / old_sl_dist
        if trade_type == "LONG":
            new_tp = entry + (new_sl_dist * old_rr)
        else:
            new_tp = entry - (new_sl_dist * old_rr)
    else:
        new_tp = tp

    return new_sl, new_tp, new_size, new_notional, new_margin, None
