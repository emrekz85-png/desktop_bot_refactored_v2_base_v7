# strategies/router.py
# Strategy Router - Dispatches to appropriate strategy based on config
#
# This module provides the main check_signal function that routes to
# the correct strategy implementation based on strategy_mode config.
#
# Available strategies:
# - ssl_flow: Trend following with SSL HYBRID baseline (TP at PBEMA) [DEFAULT/ACTIVE]
# - keltner_bounce: Mean reversion from Keltner bands (TP at PBEMA) [DISABLED]

from typing import Union
import pandas as pd

from .base import SignalResult, SignalResultWithDebug, DEFAULT_STRATEGY_MODE
from .ssl_flow import check_ssl_flow_signal
from .keltner_bounce import check_keltner_bounce_signal

# Import config for default values
try:
    from core.config import DEFAULT_STRATEGY_CONFIG
except ImportError:
    # Fallback defaults if config not available
    DEFAULT_STRATEGY_CONFIG = {
        "rr": 2.0,
        "rsi": 70,
        "at_active": True,
        "tp_min_dist_ratio": 0.0008,
        "tp_max_dist_ratio": 0.050,
        "adx_min": 15.0,
        "strategy_mode": "ssl_flow",
        # SSL Flow specific parameters
        "ssl_touch_tolerance": 0.002,
        "ssl_body_tolerance": 0.003,
        "min_pbema_distance": 0.004,
        "lookback_candles": 5,
        # Keltner Bounce parameters (DISABLED)
        "hold_n": 4,
        "min_hold_frac": 0.50,
        "pb_touch_tolerance": 0.0025,
        "body_tolerance": 0.0025,
        "cloud_keltner_gap_min": 0.0015,
        "slope": 0.4,
    }


def check_signal(
        df: pd.DataFrame,
        config: dict,
        index: int = -2,
        return_debug: bool = False,
        timeframe: str = "15m",
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    Wrapper function - routes to appropriate strategy based on strategy_mode.

    strategy_mode values:
    - "ssl_flow" (default): SSL HYBRID trend following strategy [ACTIVE]
    - "keltner_bounce": Keltner band bounce / mean reversion strategy [DISABLED]

    Args:
        df: OHLCV + indicator dataframe
        config: Strategy configuration (rr, rsi, at_active, strategy_mode, etc.)
        index: Candle index for signal check
        return_debug: Return debug info
        timeframe: Timeframe string for TF-adaptive thresholds (e.g., "5m", "15m", "1h")

    Returns:
        (s_type, entry, tp, sl, reason) or with debug info
    """
    strategy_mode = config.get("strategy_mode", DEFAULT_STRATEGY_CONFIG.get("strategy_mode", DEFAULT_STRATEGY_MODE))

    if strategy_mode == "keltner_bounce":
        # Keltner Bounce: Mean reversion from Keltner bands [DISABLED - no symbol uses this]
        return check_keltner_bounce_signal(
            df,
            index=index,
            min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
            rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
            slope_thresh=config.get("slope", DEFAULT_STRATEGY_CONFIG.get("slope", 0.4)),
            use_alphatrend=config.get("at_active", DEFAULT_STRATEGY_CONFIG["at_active"]),
            hold_n=config.get("hold_n", DEFAULT_STRATEGY_CONFIG.get("hold_n", 4)),
            min_hold_frac=config.get("min_hold_frac", DEFAULT_STRATEGY_CONFIG.get("min_hold_frac", 0.50)),
            pb_touch_tolerance=config.get("pb_touch_tolerance", DEFAULT_STRATEGY_CONFIG.get("pb_touch_tolerance", 0.0025)),
            body_tolerance=config.get("body_tolerance", DEFAULT_STRATEGY_CONFIG.get("body_tolerance", 0.0025)),
            cloud_keltner_gap_min=config.get("cloud_keltner_gap_min", DEFAULT_STRATEGY_CONFIG.get("cloud_keltner_gap_min", 0.0015)),
            tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
            adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
            return_debug=return_debug,
        )
    else:
        # Default: ssl_flow strategy - Trend following with SSL HYBRID [ACTIVE]
        # NOTE: AlphaTrend is now MANDATORY for SSL_Flow (no use_alphatrend parameter)
        #
        # TF-Adaptive Thresholds:
        # - If config provides explicit values for ssl_touch_tolerance, min_pbema_distance,
        #   lookback_candles, those values are used (backward compatible).
        # - If config values are None or not provided, TF-adaptive defaults are used
        #   based on the timeframe parameter.
        return check_ssl_flow_signal(
            df,
            index=index,
            min_rr=config.get("rr", DEFAULT_STRATEGY_CONFIG["rr"]),
            rsi_limit=config.get("rsi", DEFAULT_STRATEGY_CONFIG["rsi"]),
            # use_alphatrend REMOVED - AlphaTrend is now MANDATORY for SSL_Flow
            # TF-adaptive: pass None to use TF-adaptive, or explicit value to override
            ssl_touch_tolerance=config.get("ssl_touch_tolerance"),  # None = TF-adaptive
            ssl_body_tolerance=config.get("ssl_body_tolerance", DEFAULT_STRATEGY_CONFIG.get("ssl_body_tolerance", 0.003)),
            min_pbema_distance=config.get("min_pbema_distance"),  # None = TF-adaptive
            tp_min_dist_ratio=config.get("tp_min_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_min_dist_ratio"]),
            tp_max_dist_ratio=config.get("tp_max_dist_ratio", DEFAULT_STRATEGY_CONFIG["tp_max_dist_ratio"]),
            adx_min=config.get("adx_min", DEFAULT_STRATEGY_CONFIG["adx_min"]),
            lookback_candles=config.get("lookback_candles"),  # None = TF-adaptive
            # v1.7.2: Regime gating parameters for grid search optimization
            regime_adx_threshold=config.get("regime_adx_avg", 20.0),  # Note: Grid uses "regime_adx_avg" key
            regime_lookback=config.get("regime_lookback", 50),
            # Filter Discovery: allow skipping filters
            skip_overlap_check=config.get("skip_overlap_check", False),
            skip_wick_rejection=config.get("skip_wick_rejection", False),
            # NEW: Scoring system parameters
            use_scoring=config.get("use_scoring", DEFAULT_STRATEGY_CONFIG.get("use_scoring", False)),
            score_threshold=config.get("score_threshold", DEFAULT_STRATEGY_CONFIG.get("score_threshold", 6.0)),
            # v46.x: SSL Never Lost filter (can be disabled for comparison)
            use_ssl_never_lost_filter=config.get("use_ssl_never_lost_filter", True),
            ssl_never_lost_lookback=config.get("ssl_never_lost_lookback", 20),
            # P2: Confirmation candle - wait for momentum confirmation before entry
            use_confirmation_candle=config.get("use_confirmation_candle", False),
            confirmation_candle_mode=config.get("confirmation_candle_mode", "close"),
            # THREE-TIER AT ARCHITECTURE
            at_mode=config.get("at_mode", "binary"),  # "binary", "score", "off" (regime removed)
            at_regime_lookback=config.get("at_regime_lookback", 20),
            at_score_weight=config.get("at_score_weight", 2.0),
            # REGIME FILTER (AT Scenario Analysis 2026-01-03)
            regime_filter=config.get("regime_filter", "skip_neutral"),  # "off", "skip_neutral", "aligned", "veto"
            # SSL Flip Grace Period (addresses AT lag issue)
            use_ssl_flip_grace=config.get("use_ssl_flip_grace", False),
            ssl_flip_grace_bars=config.get("ssl_flip_grace_bars", 3),
            # Skip filters
            skip_body_position=config.get("skip_body_position", False),
            skip_adx_filter=config.get("skip_adx_filter", False),
            skip_at_flat_filter=config.get("skip_at_flat_filter", False),
            # TF-Adaptive: pass timeframe for threshold resolution
            timeframe=timeframe,
            return_debug=return_debug,
        )
