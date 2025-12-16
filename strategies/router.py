# strategies/router.py
# Strategy Router - Dispatches to appropriate strategy based on config
#
# This module provides the main check_signal function that routes to
# the correct strategy implementation based on strategy_mode config.

from typing import Tuple, Union
import pandas as pd

from .base import SignalResult, SignalResultWithDebug, DEFAULT_STRATEGY_MODE
from .keltner_bounce import check_keltner_bounce_signal
from .pbema_reaction import check_pbema_reaction_signal

# Import config for default values
try:
    from core.config import DEFAULT_STRATEGY_CONFIG
except ImportError:
    # Fallback defaults if config not available
    DEFAULT_STRATEGY_CONFIG = {
        "rr": 2.0,
        "rsi": 65,
        "slope": 0.4,
        "at_active": False,
        "hold_n": 4,
        "min_hold_frac": 0.50,
        "pb_touch_tolerance": 0.0025,
        "body_tolerance": 0.0025,
        "cloud_keltner_gap_min": 0.0015,
        "tp_min_dist_ratio": 0.0008,
        "tp_max_dist_ratio": 0.040,
        "adx_min": 8.0,
        "strategy_mode": "keltner_bounce",
        "pbema_approach_tolerance": 0.003,
        "pbema_frontrun_margin": 0.002,
    }


def check_signal(
        df: pd.DataFrame,
        config: dict,
        index: int = -2,
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    Wrapper function - routes to appropriate strategy based on strategy_mode.

    strategy_mode values:
    - "keltner_bounce" (default): Keltner band bounce strategy
    - "pbema_reaction": PBEMA reaction strategy

    Args:
        df: OHLCV + indicator dataframe
        config: Strategy configuration (rr, rsi, slope, strategy_mode, etc.)
        index: Candle index for signal check
        return_debug: Return debug info

    Returns:
        (s_type, entry, tp, sl, reason) or with debug info
    """
    strategy_mode = config.get("strategy_mode", DEFAULT_STRATEGY_CONFIG.get("strategy_mode", DEFAULT_STRATEGY_MODE))

    if strategy_mode == "pbema_reaction":
        return check_pbema_reaction_signal(
            df,
            index=index,
            min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
            rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
            slope_thresh=config.get("slope", DEFAULT_STRATEGY_CONFIG["slope"]),
            use_alphatrend=config.get("at_active", DEFAULT_STRATEGY_CONFIG["at_active"]),
            pbema_approach_tolerance=config.get("pbema_approach_tolerance", DEFAULT_STRATEGY_CONFIG["pbema_approach_tolerance"]),
            pbema_frontrun_margin=config.get("pbema_frontrun_margin", DEFAULT_STRATEGY_CONFIG["pbema_frontrun_margin"]),
            tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
            adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
            return_debug=return_debug,
        )
    else:
        # Default: keltner_bounce strategy
        return check_keltner_bounce_signal(
            df,
            index=index,
            min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
            rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
            slope_thresh=config.get("slope", DEFAULT_STRATEGY_CONFIG["slope"]),
            use_alphatrend=config.get("at_active", DEFAULT_STRATEGY_CONFIG["at_active"]),
            hold_n=config.get("hold_n", DEFAULT_STRATEGY_CONFIG["hold_n"]),
            min_hold_frac=config.get("min_hold_frac", DEFAULT_STRATEGY_CONFIG["min_hold_frac"]),
            pb_touch_tolerance=config.get("pb_touch_tolerance", DEFAULT_STRATEGY_CONFIG["pb_touch_tolerance"]),
            body_tolerance=config.get("body_tolerance", DEFAULT_STRATEGY_CONFIG["body_tolerance"]),
            cloud_keltner_gap_min=config.get("cloud_keltner_gap_min", DEFAULT_STRATEGY_CONFIG["cloud_keltner_gap_min"]),
            tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
            adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
            return_debug=return_debug,
        )
