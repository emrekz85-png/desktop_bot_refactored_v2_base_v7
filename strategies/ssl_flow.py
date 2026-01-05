# strategies/ssl_flow.py
# SSL Flow Strategy - Trend following with SSL HYBRID baseline
#
# Core Concept:
# "SSL HYBRID'den PBEMA bulutuna bir yol vardir!"
#
# Entry Logic:
# - LONG: Price above SSL baseline (HMA60) + AlphaTrend buyers > sellers
# - SHORT: Price below SSL baseline (HMA60) + AlphaTrend sellers > buyers
#
# Key Components:
# 1. SSL HYBRID (HMA60): Determines flow direction (support/resistance)
# 2. AlphaTrend: Confirms buyer/seller dominance (filters fake SSL signals)
# 3. PBEMA Cloud (EMA200): Take profit target
#
# Flow Detection:
# - SSL HYBRID alone can give fake signals in sideways markets
# - AlphaTrend dual-line system (buyers vs sellers) confirms real flow
# - If SSL turns bullish BUT AlphaTrend doesn't confirm -> NO TRADE
#
# Avoidance:
# - Don't trade when PBEMA and SSL baseline are too close (no room for profit)
# - Don't trade when AlphaTrend is flat (at_is_flat = True)

from typing import Tuple, Union, Dict, Optional
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug

# TF-Adaptive Thresholds
# Use lazy import to avoid circular dependency:
# strategies.ssl_flow -> core -> core.trading_engine -> strategies
#
# Fallback values (15m baseline) are used if the import fails
_FALLBACK_TF_THRESHOLDS = {
    "flat_threshold": 0.002,
    "ssl_touch_tolerance": 0.003,
    "min_pbema_distance": 0.004,
    "overlap_threshold": 0.005,
    "lookback_candles": 5,
}

# Cache for the imported config module
_config_module_cache = None


def _get_config_module():
    """Lazy import of core.config to avoid circular dependency."""
    global _config_module_cache
    if _config_module_cache is None:
        try:
            import importlib
            _config_module_cache = importlib.import_module('core.config')
        except (ImportError, AttributeError):
            _config_module_cache = False  # Mark as failed
    return _config_module_cache if _config_module_cache else None


def get_tf_threshold(name: str, tf: str) -> float:
    """
    Get TF-adaptive threshold value.

    Uses lazy import to avoid circular dependency issues.
    Falls back to 15m baseline values if import fails.
    """
    config_mod = _get_config_module()
    if config_mod and hasattr(config_mod, 'get_tf_threshold'):
        return config_mod.get_tf_threshold(name, tf)
    return _FALLBACK_TF_THRESHOLDS.get(name, 0.002)


def get_tf_thresholds(tf: str) -> dict:
    """Get all TF-adaptive thresholds for a timeframe."""
    config_mod = _get_config_module()
    if config_mod and hasattr(config_mod, 'get_tf_thresholds'):
        return config_mod.get_tf_thresholds(tf)
    return _FALLBACK_TF_THRESHOLDS.copy()


# For backward compatibility
BASE_TF_THRESHOLDS = _FALLBACK_TF_THRESHOLDS
HAS_TF_THRESHOLDS = True  # Always True since we have fallback

# Import enhanced regime filter (Priority 2 implementation)
try:
    from core.regime_filter import RegimeFilter, RegimeType, check_regime_for_trade
    HAS_REGIME_FILTER = True
except ImportError:
    HAS_REGIME_FILTER = False
    RegimeFilter = None
    RegimeType = None

# Import Volatility Regime (Sinclair's 3-Tier System from Expert Panel)
try:
    from core.indicators import classify_volatility_regime
    HAS_VOL_REGIME = True
except ImportError:
    HAS_VOL_REGIME = False
    classify_volatility_regime = None

# Import Market Structure + FVG (v2.2.0 - FVG Bonus)
try:
    from core.market_structure import MarketStructure, get_structure_score, TrendType
    HAS_MARKET_STRUCTURE = True
except ImportError:
    HAS_MARKET_STRUCTURE = False
    MarketStructure = None
    get_structure_score = None
    TrendType = None

try:
    from core.fvg_detector import FVGDetector, FVGType
    HAS_FVG_DETECTOR = True
except ImportError:
    HAS_FVG_DETECTOR = False
    FVGDetector = None
    FVGType = None

# Import Momentum Pattern Detection (v2.3.0 - Session 2026-01-05)
# Addresses: "Bot is missing 90% of trades" - captures momentum exhaustion patterns
# NOTE: Uses lazy import to avoid circular dependency with core.trading_engine
HAS_MOMENTUM_PATTERNS = None  # Lazy-loaded
_momentum_patterns_module = None


def _get_momentum_patterns():
    """Lazy import of momentum_patterns to avoid circular dependency."""
    global HAS_MOMENTUM_PATTERNS, _momentum_patterns_module
    if HAS_MOMENTUM_PATTERNS is None:
        try:
            from core import momentum_patterns as mp
            _momentum_patterns_module = mp
            HAS_MOMENTUM_PATTERNS = True
        except ImportError:
            HAS_MOMENTUM_PATTERNS = False
            _momentum_patterns_module = None
    return _momentum_patterns_module


def detect_htf_trend(
    htf_df: pd.DataFrame,
    method: str = "baseline",
    lookback: int = 3,
) -> Tuple[str, float]:
    """
    Detect Higher Timeframe (4H) trend direction.

    This is used to filter out counter-trend entries on lower timeframes.
    If 4H trend is UP, only LONG trades are allowed on 15m.
    If 4H trend is DOWN, only SHORT trades are allowed on 15m.

    Args:
        htf_df: DataFrame with 4H candles and indicators (must have 'close', 'ssl_baseline')
        method: "baseline" (price vs HMA60) or "ema" (EMA50 vs EMA200)
        lookback: Number of candles to confirm trend (default 3)

    Returns:
        Tuple of (trend_direction, confidence)
        - trend_direction: "UP", "DOWN", or "NEUTRAL"
        - confidence: 0.0 to 1.0 (how strong the trend is)
    """
    if htf_df is None or len(htf_df) < lookback + 10:
        return "NEUTRAL", 0.0

    try:
        if method == "baseline":
            # Use SSL baseline (HMA60) - same indicator as entry timeframe
            # Column name is "baseline" from calculate_indicators()
            baseline_col = "baseline" if "baseline" in htf_df.columns else "ssl_baseline"
            if baseline_col not in htf_df.columns:
                return "NEUTRAL", 0.0

            recent_closes = htf_df["close"].iloc[-lookback:].values
            recent_baselines = htf_df[baseline_col].iloc[-lookback:].values

            # Count how many candles are above/below baseline
            above_count = sum(1 for c, b in zip(recent_closes, recent_baselines) if c > b)
            below_count = sum(1 for c, b in zip(recent_closes, recent_baselines) if c < b)

            # Calculate trend strength (% of candles in direction)
            if above_count == lookback:
                # All candles above baseline = strong uptrend
                avg_distance = np.mean([(c - b) / b for c, b in zip(recent_closes, recent_baselines)])
                confidence = min(1.0, abs(avg_distance) * 100)  # 1% distance = full confidence
                return "UP", confidence
            elif below_count == lookback:
                # All candles below baseline = strong downtrend
                avg_distance = np.mean([(b - c) / b for c, b in zip(recent_closes, recent_baselines)])
                confidence = min(1.0, abs(avg_distance) * 100)
                return "DOWN", confidence
            else:
                # Mixed = no clear trend
                return "NEUTRAL", 0.0

        elif method == "ema":
            # Use EMA50 vs EMA200 crossover
            if "ema_50" not in htf_df.columns or "ema_200" not in htf_df.columns:
                # Calculate EMAs if not present
                htf_df = htf_df.copy()
                htf_df["ema_50"] = htf_df["close"].ewm(span=50, adjust=False).mean()
                htf_df["ema_200"] = htf_df["close"].ewm(span=200, adjust=False).mean()

            ema_50 = htf_df["ema_50"].iloc[-1]
            ema_200 = htf_df["ema_200"].iloc[-1]
            close = htf_df["close"].iloc[-1]

            # Trend direction based on EMA alignment
            if ema_50 > ema_200 and close > ema_50:
                distance = (ema_50 - ema_200) / ema_200
                confidence = min(1.0, distance * 50)  # 2% EMA gap = full confidence
                return "UP", confidence
            elif ema_50 < ema_200 and close < ema_50:
                distance = (ema_200 - ema_50) / ema_200
                confidence = min(1.0, distance * 50)
                return "DOWN", confidence
            else:
                return "NEUTRAL", 0.0

        else:
            return "NEUTRAL", 0.0

    except Exception as e:
        return "NEUTRAL", 0.0


def calculate_signal_score(
    adx: float,
    baseline_touch: bool,
    at_dominant: bool,
    at_is_flat: bool,
    pbema_distance: float,
    wick_ratio: float,
    no_overlap: bool,
    body_position_ok: bool,
    regime_ok: bool,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate composite signal score for SSL Flow strategy.

    Converts binary AND logic to weighted scoring system:
    - Each filter contributes points based on quality
    - Total score compared against threshold for signal

    Args:
        adx: ADX value (trend strength)
        baseline_touch: Whether baseline was touched/retested
        at_dominant: AlphaTrend dominance in signal direction
        at_is_flat: Whether AlphaTrend is flat (no flow)
        pbema_distance: Distance to PBEMA target (ratio)
        wick_ratio: Rejection wick size (ratio)
        no_overlap: SSL-PBEMA bands don't overlap
        body_position_ok: Candle body on correct side of baseline
        regime_ok: Regime is trending (not ranging)

    Returns:
        (score, max_score, breakdown_dict)
    """
    score = 0.0
    max_score = 10.0
    breakdown = {}

    # 1. ADX Trend Strength (max 2.0)
    # Strong trends = best entries for trend-following
    if adx > 30:
        s = 2.0
    elif adx > 25:
        s = 1.5
    elif adx > 20:
        s = 1.0
    elif adx > 15:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['adx'] = s

    # 2. Regime Gating (max 1.0)
    # Bonus if regime is trending (filters choppy markets)
    if regime_ok:
        s = 1.0
    else:
        s = 0.0
    score += s
    breakdown['regime'] = s

    # 3. Baseline Touch (max 2.0)
    # Critical: ensures we're entering on retest, not chasing
    if baseline_touch:
        s = 2.0
    else:
        s = 0.0
    score += s
    breakdown['baseline_touch'] = s

    # 4. AlphaTrend Confirmation (max 2.0)
    # Confirms real flow (not fake SSL signals)
    if at_dominant and not at_is_flat:
        s = 2.0
    elif at_dominant:
        s = 1.5
    elif not at_is_flat:
        s = 1.0
    else:
        s = 0.0
    score += s
    breakdown['alphatrend'] = s

    # 5. PBEMA Distance (max 1.0)
    # Room for profit - more distance = better
    if pbema_distance >= 0.006:
        s = 1.0
    elif pbema_distance >= 0.004:
        s = 0.75
    elif pbema_distance >= 0.003:
        s = 0.5
    else:
        s = 0.25
    score += s
    breakdown['pbema_distance'] = s

    # 6. Wick Rejection (max 1.0)
    # Strong rejection = higher quality setup
    if wick_ratio >= 0.15:
        s = 1.0
    elif wick_ratio >= 0.10:
        s = 0.75
    elif wick_ratio >= 0.05:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['wick_rejection'] = s

    # 7. Body Position (max 0.5)
    # Body on correct side confirms support/resistance
    if body_position_ok:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['body_position'] = s

    # 8. No Overlap (max 0.5)
    # SSL-PBEMA overlap = no room for flow
    if no_overlap:
        s = 0.5
    else:
        s = 0.0
    score += s
    breakdown['no_overlap'] = s

    return score, max_score, breakdown


def check_ssl_flow_signal(
        df: pd.DataFrame,
        index: int = -2,
        min_rr: float = 2.0,
        rsi_limit: float = 70.0,
        # use_alphatrend REMOVED - AlphaTrend is now MANDATORY for SSL_Flow strategy
        # This prevents LONG trades when SELLERS are dominant (and vice versa)
        ssl_touch_tolerance: float = None,  # TF-adaptive if None
        ssl_body_tolerance: float = 0.003,
        min_pbema_distance: float = None,   # TF-adaptive if None
        tp_min_dist_ratio: float = 0.0015,
        tp_max_dist_ratio: float = 0.05,
        adx_min: float = 15.0,
        adx_max: float = 40.0,
        lookback_candles: int = None,       # TF-adaptive if None
        regime_adx_threshold: float = 20.0,  # v1.7.2: Now configurable for grid search
        regime_lookback: int = 50,  # v1.7.2: Now configurable
        skip_overlap_check: bool = False,  # Filter Discovery: skip SSL-PBEMA overlap check
        skip_wick_rejection: bool = False,  # Filter Discovery: skip wick rejection check
        skip_body_position: bool = False,  # BASELINE: Check body position
        skip_adx_filter: bool = False,     # BASELINE: Check ADX filter
        skip_at_flat_filter: bool = False, # BASELINE: Check AlphaTrend flat
        # === THREE-TIER AT ARCHITECTURE ===
        # Mode: "binary" (old), "regime" (Tier 1), "score" (Tier 3), "off" (disable AT)
        at_mode: str = "binary",           # Default: binary (proven baseline)
        at_regime_lookback: int = 20,      # Tier 1: Bars to calculate regime
        at_score_weight: float = 2.0,      # Tier 3: AT contribution to score
        # === REGIME FILTER (AT Scenario Analysis 2026-01-03) ===
        # Key finding: Neutral regime has 19.7% win rate - skip it!
        # - "off": No regime filtering (backward compatible)
        # - "skip_neutral": Skip trades in neutral regime (RECOMMENDED - adds +$17 value)
        # - "aligned": Only trade when signal matches regime
        # - "veto": Block only opposing signals
        regime_filter: str = "skip_neutral",  # NEW: Skip neutral regime
        # === SSL FLIP GRACE PERIOD (addresses AT lag issue) ===
        use_ssl_flip_grace: bool = False,  # Allow signals when SSL just flipped even if AT hasn't confirmed
        ssl_flip_grace_bars: int = 3,      # Number of bars after SSL flip to allow grace
        # === SSL NEVER LOST FILTER (from user annotations) ===
        use_ssl_never_lost_filter: bool = False, # BASELINE: Disabled (conflicts with baseline_touch)
        ssl_never_lost_lookback: int = 20,  # Lookback period to check if baseline was ever crossed
        use_scoring: bool = False,  # NEW: Enable scoring system (vs AND logic)
        score_threshold: float = 6.0,  # NEW: Minimum score (out of 10.0) for signal
        # === PRIORITY 2: Enhanced Regime Filter ===
        use_btc_regime_filter: bool = False,  # P2: Use BTC as market leader filter
        btc_df: Optional[pd.DataFrame] = None,  # P2: BTC DataFrame for leader check
        symbol: str = None,  # P2: Current symbol (for BTC check bypass)
        regime_min_confidence: float = 0.5,  # P2: Minimum regime confidence for trade
        # === TF-ADAPTIVE THRESHOLDS ===
        timeframe: str = "15m",  # Timeframe for TF-adaptive thresholds
        # === CONFIRMATION CANDLE (P2 - Entry Timing) ===
        use_confirmation_candle: bool = False,  # P2: Wait for confirmation before entry
        confirmation_candle_mode: str = "close",  # "close" or "body"
        # === HTF TREND FILTER (Counter-trend prevention) ===
        use_htf_filter: bool = False,  # Enable 4H trend filter to prevent counter-trend entries
        htf_df: Optional[pd.DataFrame] = None,  # 4H DataFrame with indicators
        htf_trend_method: str = "baseline",  # "baseline" (price vs HMA60) or "ema" (EMA50/200 cross)
        # === MARKET STRUCTURE + FVG BONUS (v2.2.0) ===
        use_market_structure: bool = True,   # Required filter: MS trend alignment
        min_ms_score: float = 1.0,           # Minimum MS score (0-2)
        ms_swing_length: int = 5,            # Swing detection lookback
        use_fvg_bonus: bool = True,          # Bonus: Use FVG for tighter SL
        fvg_min_gap_percent: float = 0.08,   # Minimum FVG size
        fvg_max_age_bars: int = 50,          # Max FVG age
        fvg_mitigation_lookback: int = 5,    # Mitigation detection lookback
        # === VOLATILITY-NORMALIZED PBEMA DISTANCE (v2.3.0) ===
        # Encodes human intuition: "Is there enough room for profit given current volatility?"
        # Normalizes PBEMA distance by ATR to make threshold adaptive
        use_vol_normalized_pbema: bool = True,   # Enable volatility normalization
        vol_norm_min_atr: float = 1.0,           # Minimum: target >= 1.0 ATR away
        vol_norm_max_atr: float = 20.0,          # Maximum: target <= 20 ATR away (relaxed for trend-following)
        # === VOLATILITY REGIME (Sinclair's 3-Tier System - Expert Panel) ===
        # Addresses AT lag issue: In HIGH_VOL, allow SSL flip grace period
        # In LOW_VOL, require strict AT confirmation (SSL whipsaws)
        use_volatility_regime: bool = True,      # Enable volatility-based regime adaptation
        vol_regime_lookback: int = 50,           # Lookback for ATR percentile calculation
        vol_low_threshold: float = 40.0,         # Below this = LOW_VOL (conservative)
        vol_high_threshold: float = 75.0,        # Above this = HIGH_VOL (aggressive)
        # === FILTER HIERARCHY (Clenow's Tier System - Expert Panel) ===
        # Controls which filters are required vs optional
        # Tier 1 (Core): SSL direction, AT aligned, PBEMA path - ALWAYS required
        # Tier 2 (Quality): Baseline touch, PBEMA distance, ADX/RSI - configurable
        # Tier 3 (Risk): Wick rejection, body position, overlap - configurable
        filter_tier_level: int = 3,              # 1=core only, 2=+quality, 3=+risk (full)
        # === MOMENTUM PATTERN DETECTION (v2.3.0 - Session 2026-01-05) ===
        # Addresses: "Bot is missing 90% of trades"
        # When momentum exhaustion pattern detected, relax AT flat filter
        # This captures pattern sequences: Stairstepping → Sharp Break → Fakeout → SSL Sideways
        use_momentum_pattern: bool = False,      # Enable momentum pattern detection
        momentum_min_quality: str = "MODERATE",  # Minimum pattern quality (MODERATE, GOOD, EXCELLENT)
        momentum_skip_at_flat: bool = True,      # Skip AT flat filter when pattern detected
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    SSL Flow Strategy - Trend following with SSL HYBRID baseline direction.

    This strategy follows the flow/trend direction using:
    1. SSL HYBRID (baseline = HMA60) for trend direction
    2. AlphaTrend dual-lines for flow confirmation
    3. PBEMA cloud (EMA200) as TP target

    Entry Logic:
    - LONG: Price above SSL baseline + AlphaTrend buyers dominant + retest/bounce from baseline
    - SHORT: Price below SSL baseline + AlphaTrend sellers dominant + retest/bounce from baseline

    Mode Selection (use_scoring parameter):
    - use_scoring=False (default): Binary AND logic - all filters must pass (strict, fewer trades)
    - use_scoring=True: Weighted scoring - filters contribute points, signal if score >= threshold

    Args:
        df: DataFrame with OHLCV + indicators
        index: Candle index for signal check (default: -2, second to last candle)
        min_rr: Minimum risk/reward ratio
        rsi_limit: RSI threshold (LONG: not overbought, SHORT: not oversold)
        ssl_touch_tolerance: Tolerance for SSL baseline touch detection (0.003 = 0.3%)
        ssl_body_tolerance: Tolerance for candle body position relative to baseline
        min_pbema_distance: Minimum distance between price and PBEMA for valid TP
        tp_min_dist_ratio: Minimum TP distance ratio
        tp_max_dist_ratio: Maximum TP distance ratio
        adx_min: Minimum ADX value (trend strength filter)
        adx_max: Maximum ADX value (filters overly strong trends that may reverse)
        lookback_candles: Number of candles to check for baseline interaction
        regime_adx_threshold: Average ADX threshold for regime detection
        regime_lookback: Number of candles for regime detection
        skip_overlap_check: Skip SSL-PBEMA overlap check (filter discovery)
        skip_wick_rejection: Skip wick rejection check (filter discovery)
        skip_body_position: [P3] Skip body position check (99.9% pass rate = useless filter)
        use_ssl_never_lost_filter: Skip counter-trend trades if SSL baseline was never broken
            (derived from user annotation: "SSL HYBRID not even lost. Should be no short trade here")
        ssl_never_lost_lookback: Number of candles to check if baseline was ever crossed
        use_scoring: Enable weighted scoring system (vs binary AND logic)
        score_threshold: Minimum score required for signal (out of 10.0)
        use_btc_regime_filter: [P2] Use BTC as market leader - if BTC is ranging, skip all alts
        btc_df: [P2] BTC DataFrame for leader check (required if use_btc_regime_filter=True)
        symbol: [P2] Current symbol name (for BTC bypass logic)
        regime_min_confidence: [P2] Minimum regime confidence to allow trading (0.0-1.0)
        return_debug: Whether to return debug info

    Returns:
        SignalResult or SignalResultWithDebug tuple

    Priority 2 Implementation:
        The BTC regime filter addresses a critical finding from the hedge fund analysis:
        - Strategy only works in TRENDING regimes (H1 lost money, H2 made money)
        - BTC leads the crypto market - if BTC is ranging, alts usually range too
        - This filter skips trades when BTC is not in a clear trend

    TF-Adaptive Thresholds:
        When ssl_touch_tolerance, min_pbema_distance, or lookback_candles are None,
        they are resolved using TF-adaptive values from core.config.get_tf_threshold().
        This ensures thresholds scale appropriately across timeframes:
        - 5m: Tighter thresholds (more noise filtering)
        - 15m: Baseline (proven working)
        - 1h+: Looser thresholds (larger price moves)
    """

    # ================= TF-ADAPTIVE THRESHOLD RESOLUTION =================
    # Resolve None values to TF-adaptive thresholds
    # Explicit values passed by caller override TF-adaptive (backward compatible)
    if ssl_touch_tolerance is None:
        ssl_touch_tolerance = get_tf_threshold("ssl_touch_tolerance", timeframe)
    if min_pbema_distance is None:
        min_pbema_distance = get_tf_threshold("min_pbema_distance", timeframe)
    if lookback_candles is None:
        lookback_candles = int(get_tf_threshold("lookback_candles", timeframe))

    # Get overlap threshold (always TF-adaptive, not a function parameter)
    overlap_threshold = get_tf_threshold("overlap_threshold", timeframe)

    debug_info = {
        "adx_ok": None,
        "price_above_baseline": None,
        "price_below_baseline": None,
        "baseline_touch_long": None,
        "baseline_touch_short": None,
        "at_buyers_dominant": None,
        "at_sellers_dominant": None,
        "at_is_flat": None,
        "pbema_distance_ok": None,
        "long_rsi_ok": None,
        "short_rsi_ok": None,
        "rr_value": None,
        "tp_dist_ratio": None,
        # TF-adaptive thresholds (for debugging)
        "timeframe": timeframe,
        "tf_ssl_touch_tolerance": ssl_touch_tolerance,
        "tf_min_pbema_distance": min_pbema_distance,
        "tf_overlap_threshold": overlap_threshold,
        "tf_lookback_candles": lookback_candles,
        # HTF filter info
        "htf_filter_enabled": use_htf_filter,
        "htf_trend": None,
        "htf_confidence": None,
        "htf_blocked_long": False,
        "htf_blocked_short": False,
        # Volatility Regime info (Sinclair's 3-Tier)
        "use_volatility_regime": use_volatility_regime,
        "vol_regime": None,
        "vol_atr_percentile": None,
        "vol_position_multiplier": None,
        "vol_allow_at_grace": None,
        # Filter Tier info (Clenow's Tier System)
        "filter_tier_level": filter_tier_level,
        # Momentum Pattern info (v2.3.0)
        "use_momentum_pattern": use_momentum_pattern,
        "momentum_pattern_detected": False,
        "momentum_pattern_quality": "NONE",
        "momentum_pattern_confidence": 0.0,
        "momentum_phases": {},
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    # Validate input
    if df is None or df.empty:
        return _ret(None, None, None, None, "No Data")

    required_cols = [
        "open", "high", "low", "close",
        "rsi", "adx",
        "baseline",  # SSL HYBRID (HMA60)
        "pb_ema_top", "pb_ema_bot",  # PBEMA cloud (EMA200)
    ]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return _ret(None, None, None, None, "Index Error")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < 60:  # Need enough history for HMA60 and other indicators
        return _ret(None, None, None, None, "Not Enough Data")

    # OPTIMIZATION 4: Cache column arrays for vectorized operations
    _open_arr = df["open"].values
    _high_arr = df["high"].values
    _low_arr = df["low"].values
    _close_arr = df["close"].values
    _baseline_arr = df["baseline"].values
    _pb_top_arr = df["pb_ema_top"].values
    _pb_bot_arr = df["pb_ema_bot"].values
    _adx_arr = df["adx"].values
    _rsi_arr = df["rsi"].values

    # Extract current values
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    baseline = float(curr["baseline"])  # SSL HYBRID (HMA60)
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    adx_val = float(curr["adx"])
    rsi_val = float(curr["rsi"])

    # Check for NaN values
    if any(pd.isna([open_, high, low, close, baseline, pb_top, pb_bot, adx_val, rsi_val])):
        return _ret(None, None, None, None, "NaN Values")

    # ================= VOLATILITY REGIME (Sinclair's 3-Tier System) =================
    # Expert Panel Recommendation: Adapt trading rules based on volatility regime
    # - LOW_VOL: Conservative mode, strict AT confirmation (SSL whipsaws)
    # - NORMAL_VOL: Standard mode, baseline parameters
    # - HIGH_VOL: Aggressive mode, allow AT lag grace period (strong trends)
    vol_regime_result = None
    vol_allow_at_grace_override = False

    if use_volatility_regime and HAS_VOL_REGIME:
        vol_regime_result = classify_volatility_regime(
            df,
            index=abs_index,
            lookback=vol_regime_lookback,
            low_vol_threshold=vol_low_threshold,
            high_vol_threshold=vol_high_threshold,
        )

        debug_info["vol_regime"] = vol_regime_result["regime"]
        debug_info["vol_atr_percentile"] = vol_regime_result["atr_percentile"]
        debug_info["vol_position_multiplier"] = vol_regime_result["position_multiplier"]
        debug_info["vol_allow_at_grace"] = vol_regime_result["allow_at_grace"]
        debug_info["vol_details"] = vol_regime_result["details"]

        # In HIGH_VOL, automatically enable SSL flip grace period
        # This addresses AT lag issue - in strong trends, let AT catch up
        if vol_regime_result["allow_at_grace"]:
            vol_allow_at_grace_override = True

        # In LOW_VOL, be more conservative with AT flat filter
        # (AT lag is actually helpful here as quality filter)
        if vol_regime_result["regime"] == "LOW_VOL":
            # Force AT flat check even if skip_at_flat_filter is True
            skip_at_flat_filter = False

    # ================= PRIORITY 2: BTC LEADER REGIME FILTER =================
    # If BTC is ranging, skip trades for all altcoins (market leader principle)
    # This addresses the hedge fund finding: strategy only works in TRENDING regimes
    if use_btc_regime_filter and HAS_REGIME_FILTER:
        is_btc = symbol and symbol.upper() == "BTCUSDT"

        if not is_btc and btc_df is not None and len(btc_df) > 0:
            # Check BTC's regime
            btc_regime_filter = RegimeFilter(
                adx_trending_threshold=regime_adx_threshold,
                require_btc_trend=False,  # Don't recursively check BTC
                min_confidence=regime_min_confidence,
            )
            btc_result = btc_regime_filter.detect_regime(btc_df, index, symbol="BTCUSDT")

            debug_info["btc_regime"] = btc_result.regime.value if btc_result.regime else "UNKNOWN"
            debug_info["btc_regime_confidence"] = btc_result.confidence
            debug_info["btc_should_trade"] = btc_result.should_trade

            if not btc_result.should_trade:
                return _ret(None, None, None, None,
                           f"BTC Not Trending ({btc_result.regime.value}, conf={btc_result.confidence:.2f})")

        elif is_btc:
            # For BTC itself, use the enhanced regime filter
            symbol_regime_filter = RegimeFilter(
                adx_trending_threshold=regime_adx_threshold,
                require_btc_trend=False,
                min_confidence=regime_min_confidence,
            )
            symbol_result = symbol_regime_filter.detect_regime(df, index, symbol="BTCUSDT")

            debug_info["enhanced_regime"] = symbol_result.regime.value if symbol_result.regime else "UNKNOWN"
            debug_info["enhanced_regime_confidence"] = symbol_result.confidence

            if not symbol_result.should_trade:
                return _ret(None, None, None, None,
                           f"Regime Filter ({symbol_result.regime.value}, conf={symbol_result.confidence:.2f})")

    # ================= ADX FILTER =================
    # Note: ADX max filter REMOVED in v1.6.2-restored
    # Reason: ADX max affects optimizer, causes "0 configs found"
    # SSL Flow is trend-following - strong trends (ADX>40) are BEST opportunities
    # P5: ADX filter is REDUNDANT with Regime Gating (which uses ADX average)
    # skip_adx_filter=True bypasses this check, letting Regime Gating handle trend strength
    debug_info["adx_ok"] = adx_val >= adx_min
    debug_info["skip_adx_filter"] = skip_adx_filter
    if not skip_adx_filter and not debug_info["adx_ok"]:
        return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

    # ================= REGIME GATING (v1.7.0, v1.7.2 configurable) =================
    # Window-level regime detection using ADX average over lookback period
    # RANGING markets (ADX_avg < threshold) = skip trade entirely
    # This prevents trades during sideways/choppy markets where SSL Flow struggles
    # v1.7.2: regime_adx_threshold and regime_lookback are now function parameters

    regime_start = max(0, abs_index - regime_lookback)
    # FIX: Look-ahead bias - exclude current bar from regime calculation
    # At signal time, we don't know the current bar's final ADX yet
    adx_window = df["adx"].iloc[regime_start:abs_index]
    adx_avg = float(adx_window.mean()) if len(adx_window) > 0 else adx_val

    regime = "TRENDING" if adx_avg >= regime_adx_threshold else "RANGING"
    debug_info["adx_avg"] = adx_avg
    debug_info["regime"] = regime
    debug_info["regime_lookback"] = regime_lookback
    debug_info["regime_adx_threshold"] = regime_adx_threshold

    if regime == "RANGING":
        return _ret(None, None, None, None, f"RANGING Regime (ADX_avg={adx_avg:.1f})")

    # ================= MOMENTUM PATTERN DETECTION (v2.3.0) =================
    # Detects momentum exhaustion patterns from user's real trade analysis:
    # Pattern: Stairstepping → Sharp Selloff → Fakeout → SSL Sideways → Entry
    #
    # When pattern detected:
    # - Skip AT flat filter (momentum signals override AT lag)
    # - Use pattern quality for confidence scoring
    #
    # This addresses the core finding: "Bot is missing 90% of trades"
    # Human trades use pattern sequences, not static filters
    momentum_pattern_result = None
    momentum_pattern_detected = False
    momentum_pattern_quality = "NONE"
    momentum_pattern_confidence = 0.0
    momentum_skip_at_flat_effective = False

    # Lazy load momentum patterns module
    mp_module = _get_momentum_patterns() if use_momentum_pattern else None

    if use_momentum_pattern and mp_module is not None:
        # Quality hierarchy: EXCELLENT > GOOD > MODERATE > POOR
        quality_order = {"EXCELLENT": 4, "GOOD": 3, "MODERATE": 2, "POOR": 1, "NONE": 0}
        min_quality_level = quality_order.get(momentum_min_quality, 2)

        # Detect pattern for both directions (we'll use appropriate one later)
        # Check SHORT pattern (momentum exhaustion after uptrend)
        momentum_pattern_short = mp_module.detect_momentum_exhaustion_pattern(
            df, abs_index, signal_type="SHORT", require_all_phases=False
        )
        # Check LONG pattern (momentum exhaustion after downtrend)
        momentum_pattern_long = mp_module.detect_momentum_exhaustion_pattern(
            df, abs_index, signal_type="LONG", require_all_phases=False
        )

        # Use the pattern that matches potential signal direction
        # For SHORT: price below baseline → check SHORT pattern
        # For LONG: price above baseline → check LONG pattern
        if close < baseline:  # Potential SHORT
            momentum_pattern_result = momentum_pattern_short
        else:  # Potential LONG
            momentum_pattern_result = momentum_pattern_long

        if momentum_pattern_result and momentum_pattern_result.get('pattern_detected'):
            pattern_quality = momentum_pattern_result.get('quality', 'NONE')
            pattern_quality_level = quality_order.get(pattern_quality, 0)

            if pattern_quality_level >= min_quality_level:
                momentum_pattern_detected = True
                momentum_pattern_quality = pattern_quality
                momentum_pattern_confidence = momentum_pattern_result.get('confidence', 0.0)

                # Key effect: Skip AT flat filter when momentum pattern detected
                if momentum_skip_at_flat:
                    momentum_skip_at_flat_effective = True

        # Store in debug info
        debug_info["momentum_pattern_detected"] = momentum_pattern_detected
        debug_info["momentum_pattern_quality"] = momentum_pattern_quality
        debug_info["momentum_pattern_confidence"] = momentum_pattern_confidence
        debug_info["momentum_phases"] = momentum_pattern_result.get('phases', {}) if momentum_pattern_result else {}
        debug_info["momentum_skip_at_flat_effective"] = momentum_skip_at_flat_effective

    # ================= SSL NEVER LOST FILTER =================
    # Derived from user annotations: "SSL HYBRID not even lost. Should be no short trade here"
    # If baseline has NEVER been broken in lookback period, trend is too strong to trade against
    #
    # For SHORT: Check if price EVER went BELOW baseline (baseline was "lost" to bears)
    #            If never below -> bullish trend too strong -> don't short
    # For LONG:  Check if price EVER went ABOVE baseline (baseline was "lost" to bulls)
    #            If never above -> bearish trend too strong -> don't long

    if use_ssl_never_lost_filter:
        never_lost_start = max(0, abs_index - ssl_never_lost_lookback)

        # Check if baseline was ever broken (price crossed to other side)
        lookback_lows_nl = _low_arr[never_lost_start:abs_index]
        lookback_highs_nl = _high_arr[never_lost_start:abs_index]
        lookback_baselines_nl = _baseline_arr[never_lost_start:abs_index]

        # For SHORT: Was baseline ever broken downward? (price went below baseline)
        baseline_ever_lost_bearish = np.any(lookback_lows_nl < lookback_baselines_nl)
        # For LONG: Was baseline ever broken upward? (price went above baseline)
        baseline_ever_lost_bullish = np.any(lookback_highs_nl > lookback_baselines_nl)

        debug_info["ssl_never_lost_filter"] = use_ssl_never_lost_filter
        debug_info["ssl_never_lost_lookback"] = ssl_never_lost_lookback
        debug_info["baseline_ever_lost_bearish"] = bool(baseline_ever_lost_bearish)
        debug_info["baseline_ever_lost_bullish"] = bool(baseline_ever_lost_bullish)
    else:
        # Filter disabled - allow all trades
        baseline_ever_lost_bearish = True
        baseline_ever_lost_bullish = True

    # ================= SSL BASELINE DIRECTION =================
    # Price position relative to SSL baseline determines flow direction
    price_above_baseline = close > baseline
    price_below_baseline = close < baseline

    debug_info["price_above_baseline"] = price_above_baseline
    debug_info["price_below_baseline"] = price_below_baseline
    debug_info["baseline"] = baseline
    debug_info["close"] = close

    # ================= BASELINE TOUCH/RETEST DETECTION =================
    # Check if price has touched or come close to baseline in recent candles
    # This ensures we're entering on a retest, not chasing

    lookback_start = max(0, abs_index - lookback_candles)

    # OPTIMIZATION 3: Vectorize baseline touch detection with NumPy
    # For LONG: Check if low touched baseline (retest from above)
    lookback_lows = _low_arr[lookback_start:abs_index + 1]
    lookback_baselines_long = _baseline_arr[lookback_start:abs_index + 1]
    baseline_touch_long = np.any(lookback_lows <= lookback_baselines_long * (1 + ssl_touch_tolerance))

    # For SHORT: Check if high touched baseline (retest from below)
    lookback_highs = _high_arr[lookback_start:abs_index + 1]
    lookback_baselines_short = _baseline_arr[lookback_start:abs_index + 1]
    baseline_touch_short = np.any(lookback_highs >= lookback_baselines_short * (1 - ssl_touch_tolerance))

    debug_info["baseline_touch_long"] = baseline_touch_long
    debug_info["baseline_touch_short"] = baseline_touch_short

    # ================= CANDLE BODY POSITION =================
    # For LONG: Body should be above baseline (confirmation of support)
    # For SHORT: Body should be below baseline (confirmation of resistance)
    candle_body_min = min(open_, close)
    candle_body_max = max(open_, close)

    body_above_baseline = candle_body_min > baseline * (1 - ssl_body_tolerance)
    body_below_baseline = candle_body_max < baseline * (1 + ssl_body_tolerance)

    debug_info["body_above_baseline"] = body_above_baseline
    debug_info["body_below_baseline"] = body_below_baseline

    # ================= ALPHATREND FLOW CONFIRMATION =================
    # CRITICAL: AlphaTrend is MANDATORY for SSL_Flow strategy
    # This prevents LONG trades when SELLERS are dominant (and vice versa)
    # Without this, fake SSL signals lead to wrong-direction trades

    # Check for required AlphaTrend columns
    required_at_cols = ['alphatrend', 'alphatrend_2', 'at_buyers_dominant', 'at_sellers_dominant', 'at_is_flat']
    has_at_cols = all(col in df.columns for col in required_at_cols)

    if not has_at_cols:
        return _ret(None, None, None, None, "AlphaTrend columns missing (REQUIRED)")

    # Get AlphaTrend values for logging
    alphatrend_val = float(curr.get("alphatrend", 0))
    alphatrend_2_val = float(curr.get("alphatrend_2", 0))
    at_is_flat = bool(curr.get("at_is_flat", False))

    # USE PRE-CALCULATED DOMINANCE based on LINE DIRECTION
    # Buyers dominant = AlphaTrend line is RISING (blue in TradingView)
    # Sellers dominant = AlphaTrend line is FALLING (red in TradingView)
    at_buyers_dominant = bool(curr.get("at_buyers_dominant", False))
    at_sellers_dominant = bool(curr.get("at_sellers_dominant", False))

    # Also get legacy at_buyers/at_sellers for backward compat logging
    at_buyers = float(curr.get("at_buyers", 0))
    at_sellers = float(curr.get("at_sellers", 0))

    debug_info["alphatrend"] = alphatrend_val
    debug_info["alphatrend_2"] = alphatrend_2_val
    debug_info["at_buyers"] = at_buyers
    debug_info["at_sellers"] = at_sellers
    debug_info["at_buyers_dominant"] = at_buyers_dominant
    debug_info["at_sellers_dominant"] = at_sellers_dominant
    debug_info["at_is_flat"] = at_is_flat
    debug_info["skip_at_flat_filter"] = skip_at_flat_filter

    # ================= SSL FLIP GRACE PERIOD =================
    # When SSL baseline is crossed (flip), allow signals for N bars even if AT hasn't confirmed
    # This addresses AT lag (2-5 bars behind SSL) that causes missed entries at trend start
    # Condition: SSL just flipped + AT is NOT opposing (not sellers_dominant for long)
    #
    # Volatility Regime Integration (Sinclair's recommendation):
    # - In HIGH_VOL, automatically enable grace period (strong trends, AT will catch up)
    # - In LOW_VOL, disable grace period (SSL whipsaws, need strict AT confirmation)
    ssl_flip_grace_long = False
    ssl_flip_grace_short = False

    # Determine effective grace setting (user setting OR volatility override)
    effective_ssl_flip_grace = use_ssl_flip_grace or vol_allow_at_grace_override
    debug_info["effective_ssl_flip_grace"] = effective_ssl_flip_grace
    debug_info["vol_allow_at_grace_override"] = vol_allow_at_grace_override

    if effective_ssl_flip_grace and abs_index >= ssl_flip_grace_bars:
        # Detect SSL flip in recent bars
        # For LONG: price was below baseline, now above (bullish flip)
        # For SHORT: price was above baseline, now below (bearish flip)
        for lookback in range(1, ssl_flip_grace_bars + 1):
            if abs_index - lookback < 0:
                break
            prev_close = float(_close_arr[abs_index - lookback])
            prev_baseline = float(_baseline_arr[abs_index - lookback])

            # Bullish flip: was below, now above
            if prev_close < prev_baseline and close > baseline:
                # Grace for LONG: allow if AT is NOT opposing (not sellers dominant)
                if not at_sellers_dominant:
                    ssl_flip_grace_long = True
                break

            # Bearish flip: was above, now below
            if prev_close > prev_baseline and close < baseline:
                # Grace for SHORT: allow if AT is NOT opposing (not buyers dominant)
                if not at_buyers_dominant:
                    ssl_flip_grace_short = True
                break

    debug_info["use_ssl_flip_grace"] = use_ssl_flip_grace
    debug_info["ssl_flip_grace_long"] = ssl_flip_grace_long
    debug_info["ssl_flip_grace_short"] = ssl_flip_grace_short

    # ================= THREE-TIER AT ARCHITECTURE =================
    # Mode: "binary" (old), "regime" (Tier 1), "score" (Tier 3), "off" (disable)
    #
    # - "binary": Original behavior - AT blocks signals if not dominant
    # - "regime": AT only blocks if OVERALL regime is opposing (more stable)
    # - "score": AT contributes to signal score, doesn't block
    # - "off": AT completely disabled

    # Calculate AT regime (Tier 1) - also needed for regime_filter
    at_regime = "neutral_regime"
    at_regime_blocks_long = False
    at_regime_blocks_short = False

    # Calculate regime if at_mode uses it OR if regime_filter is active
    if at_mode in ("regime", "score") or regime_filter != "off":
        # Lazy import to avoid circular dependency
        try:
            from core.indicators import calculate_at_regime
            # CRITICAL: Pass abs_index to avoid look-ahead bias!
            at_regime = calculate_at_regime(df, index=abs_index, lookback=at_regime_lookback)
        except ImportError:
            at_regime = "neutral_regime"

        # Regime only blocks if STRONGLY opposing (for at_mode="regime")
        at_regime_blocks_long = (at_regime == "bearish_regime")
        at_regime_blocks_short = (at_regime == "bullish_regime")

    debug_info["at_mode"] = at_mode
    debug_info["at_regime"] = at_regime
    debug_info["regime_filter"] = regime_filter
    debug_info["at_regime_blocks_long"] = at_regime_blocks_long
    debug_info["at_regime_blocks_short"] = at_regime_blocks_short

    # ================= REGIME FILTER CHECK (AT Scenario Analysis 2026-01-03) =================
    # Key finding: Neutral regime has 19.7% win rate - skip it!
    # This check happens BEFORE other filters because regime is the most predictive factor.
    if regime_filter == "skip_neutral":
        if at_regime == "neutral_regime":
            return _ret(None, None, None, None, "Regime Filter: Neutral (19.7% WR)")
    elif regime_filter == "aligned":
        # LONG only in bullish, SHORT only in bearish, block all in neutral
        # (will be checked below when we know signal direction)
        pass  # Direction-specific check done later
    elif regime_filter == "veto":
        # LONG blocked in bearish, SHORT blocked in bullish
        # (will be checked below when we know signal direction)
        pass  # Direction-specific check done later

    # Calculate AT score (Tier 3)
    at_score_long = 0.0
    at_score_short = 0.0

    if at_mode == "score":
        try:
            from core.indicators import calculate_at_score
            at_score_long = calculate_at_score(df, abs_index, "long") * at_score_weight
            at_score_short = calculate_at_score(df, abs_index, "short") * at_score_weight
        except ImportError:
            pass

    debug_info["at_score_long"] = at_score_long
    debug_info["at_score_short"] = at_score_short

    # AT Flat check - only in binary mode
    # In regime/score mode, flat market is handled by regime detection
    # MOMENTUM PATTERN OVERRIDE (v2.3.0): Skip AT flat when momentum pattern detected
    # Rationale: Momentum exhaustion pattern signals override AT lag issues
    if at_mode == "binary":
        # Effective skip: either explicitly disabled OR momentum pattern overrides
        effective_skip_at_flat = skip_at_flat_filter or momentum_skip_at_flat_effective
        debug_info["effective_skip_at_flat"] = effective_skip_at_flat

        if at_is_flat and not effective_skip_at_flat:
            return _ret(None, None, None, None, "AlphaTrend Flat (No Flow)")

    # ================= PBEMA DISTANCE CHECK =================
    # Ensure there's enough room between price and PBEMA for profitable trade
    pbema_mid = (pb_top + pb_bot) / 2

    # For LONG: PBEMA should be above price (TP target)
    long_pbema_distance = (pb_bot - close) / close if close > 0 else 0
    # For SHORT: PBEMA should be below price (TP target)
    short_pbema_distance = (close - pb_top) / close if close > 0 else 0

    debug_info["long_pbema_distance"] = long_pbema_distance
    debug_info["short_pbema_distance"] = short_pbema_distance

    # ================= VOLATILITY-NORMALIZED PBEMA DISTANCE (v2.3.0) =================
    # Human intuition encoded: "Is there enough room for profit given current volatility?"
    # A 0.4% distance means nothing in high-volatility markets (noise)
    # But in low-volatility markets, 0.4% is significant
    # Solution: Normalize PBEMA distance by ATR
    #
    # Rule: Target should be 1.0-4.0 ATR away
    # - < 1.0 ATR: Too close (likely noise, SL will get hit)
    # - > 4.0 ATR: Too far (unrealistic for single swing)

    vol_norm_long_ok = True
    vol_norm_short_ok = True
    long_atr_normalized = 0.0
    short_atr_normalized = 0.0

    if use_vol_normalized_pbema:
        # Get ATR value (should be in dataframe)
        atr_val = float(curr.get("atr", 0))

        if atr_val > 0 and close > 0:
            atr_percent = atr_val / close  # ATR as percentage of price

            # Normalize PBEMA distance by ATR
            # long_pbema_distance is already a ratio (e.g., 0.004 = 0.4%)
            # atr_percent is also a ratio (e.g., 0.002 = 0.2%)
            # normalized = 0.004 / 0.002 = 2.0 ATR units

            if atr_percent > 0:
                long_atr_normalized = long_pbema_distance / atr_percent
                short_atr_normalized = short_pbema_distance / atr_percent

                # Check if within acceptable range
                vol_norm_long_ok = vol_norm_min_atr <= long_atr_normalized <= vol_norm_max_atr
                vol_norm_short_ok = vol_norm_min_atr <= short_atr_normalized <= vol_norm_max_atr

        debug_info["atr_val"] = atr_val
        debug_info["atr_percent"] = atr_percent if atr_val > 0 and close > 0 else 0
        debug_info["long_atr_normalized"] = long_atr_normalized
        debug_info["short_atr_normalized"] = short_atr_normalized
        debug_info["vol_norm_long_ok"] = vol_norm_long_ok
        debug_info["vol_norm_short_ok"] = vol_norm_short_ok
        debug_info["vol_norm_min_atr"] = vol_norm_min_atr
        debug_info["vol_norm_max_atr"] = vol_norm_max_atr

    debug_info["use_vol_normalized_pbema"] = use_vol_normalized_pbema

    # ================= PBEMA-SSL BASELINE OVERLAP CHECK =================
    # "PBEMA ve SSL Hybrid bantlari IC ICE oldugunda islem ALINMAZ"
    # LONG icin: PBEMA baseline'in USTUNDE olmali (yukariya gidecek yol var)
    # SHORT icin: PBEMA baseline'in ALTINDA olmali (asagiya gidecek yol var)

    # TF-adaptive overlap threshold (resolved earlier in function)
    # 15m baseline: 0.5%, HTF: looser (up to 1.5% for 1d)
    baseline_pbema_distance = abs(baseline - pbema_mid) / pbema_mid if pbema_mid > 0 else 0
    is_overlapping = baseline_pbema_distance < overlap_threshold

    # LONG için: PBEMA hedefi baseline'ın üstünde olmalı
    pbema_above_baseline = pbema_mid > baseline
    # SHORT için: PBEMA hedefi baseline'ın altında olmalı
    pbema_below_baseline = pbema_mid < baseline

    debug_info["baseline_pbema_distance"] = baseline_pbema_distance
    debug_info["is_overlapping"] = is_overlapping
    debug_info["pbema_above_baseline"] = pbema_above_baseline
    debug_info["pbema_below_baseline"] = pbema_below_baseline

    # İç içe durumunda işlem alma - flow yok
    # Filter Discovery: skip_overlap_check allows disabling this filter
    if not skip_overlap_check and is_overlapping:
        return _ret(None, None, None, None, "SSL-PBEMA Overlap (No Flow)")

    # ================= WICK REJECTION QUALITY =================
    candle_range = high - low
    if candle_range > 0:
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range
    else:
        upper_wick_ratio = 0.0
        lower_wick_ratio = 0.0

    min_wick_ratio = 0.10  # At least 10% wick for rejection signal
    long_rejection = lower_wick_ratio >= min_wick_ratio
    short_rejection = upper_wick_ratio >= min_wick_ratio

    debug_info["lower_wick_ratio"] = lower_wick_ratio
    debug_info["upper_wick_ratio"] = upper_wick_ratio
    debug_info["long_rejection"] = long_rejection
    debug_info["short_rejection"] = short_rejection

    # ================= LONG SIGNAL =================
    # Conditions:
    # 1. Price above SSL baseline (bullish flow)
    # 2. AlphaTrend buyers dominant (flow confirmation)
    # 3. Recent baseline touch/retest (entry opportunity)
    # 4. Candle body above baseline (support confirmation)
    # 5. PBEMA above price (room for TP)
    # 6. Rejection wick (bounce confirmation)
    # 7. PBEMA above baseline (target reachable - "yol var")
    # 8. SSL baseline was "lost" at some point (not buying into parabolic downtrend)

    # THREE-TIER AT CHECK for LONG
    # - "binary": AT must confirm (old behavior)
    # - "regime": AT regime must not be opposing
    # - "score": AT contributes to score, doesn't block here
    # - "off": AT completely bypassed
    if at_mode == "binary":
        at_allows_long = (at_buyers_dominant or ssl_flip_grace_long)
    elif at_mode == "regime":
        at_allows_long = not at_regime_blocks_long  # Only block if bearish regime
    elif at_mode == "score":
        at_allows_long = True  # Score mode - AT contributes to score, not filter
    else:  # "off"
        at_allows_long = True

    # REGIME FILTER CHECK for LONG (direction-specific)
    regime_allows_long = True
    if regime_filter == "aligned":
        # LONG only allowed in bullish regime
        regime_allows_long = (at_regime == "bullish_regime")
    elif regime_filter == "veto":
        # LONG blocked only in bearish regime
        regime_allows_long = (at_regime != "bearish_regime")
    # "skip_neutral" already handled above, "off" = always True
    debug_info["regime_allows_long"] = regime_allows_long

    # ================= FILTER HIERARCHY (Clenow's Tier System) =================
    # Tier 1 (Core): SSL direction, AT aligned, PBEMA path - ALWAYS required
    # Tier 2 (Quality): Baseline touch, PBEMA distance, ADX/RSI - required if tier >= 2
    # Tier 3 (Risk): Wick rejection, body position, overlap - required if tier >= 3
    #
    # Lower tier = more trades, lower quality
    # Higher tier = fewer trades, higher quality

    # TIER 1 - CORE FILTERS (always required)
    tier1_long = (
        price_above_baseline and           # Core: SSL direction
        at_allows_long and                 # Core: AT confirmation
        regime_allows_long and             # Core: Regime filter (skip_neutral etc.)
        pbema_above_baseline               # Core: PBEMA path exists
    )

    # TIER 2 - QUALITY FILTERS (required if filter_tier_level >= 2)
    tier2_long = (
        baseline_touch_long and            # Quality: Entry timing
        long_pbema_distance >= min_pbema_distance and  # Quality: TP distance
        (not use_vol_normalized_pbema or vol_norm_long_ok) and  # Quality: ATR-normalized distance
        baseline_ever_lost_bullish         # Quality: Not counter-trend
    )

    # TIER 3 - RISK FILTERS (required if filter_tier_level >= 3)
    tier3_long = (
        (skip_body_position or body_above_baseline) and  # Risk: Body confirmation
        (skip_wick_rejection or long_rejection)          # Risk: Wick rejection
    )

    # Combine tiers based on filter_tier_level
    if filter_tier_level == 1:
        is_long = tier1_long  # Core only - highest trade frequency
    elif filter_tier_level == 2:
        is_long = tier1_long and tier2_long  # Core + Quality
    else:  # filter_tier_level >= 3 (default)
        is_long = tier1_long and tier2_long and tier3_long  # Full filters

    debug_info["tier1_long"] = tier1_long
    debug_info["tier2_long"] = tier2_long
    debug_info["tier3_long"] = tier3_long
    debug_info["at_allows_long"] = at_allows_long

    # RSI filter for LONG: not overbought
    long_rsi_ok = rsi_val <= rsi_limit
    debug_info["long_rsi_ok"] = long_rsi_ok
    debug_info["rsi_value"] = rsi_val

    if is_long and not long_rsi_ok:
        is_long = False
        debug_info["long_rejected_rsi"] = True

    # Check if LONG was blocked by SSL Never Lost filter
    # (all other conditions passed but baseline_ever_lost_bullish was False)
    long_blocked_by_ssl_never_lost = (
        price_above_baseline and
        at_buyers_dominant and
        baseline_touch_long and
        long_pbema_distance >= min_pbema_distance and
        pbema_above_baseline and
        long_rsi_ok and
        not baseline_ever_lost_bullish
    )
    debug_info["long_blocked_by_ssl_never_lost"] = long_blocked_by_ssl_never_lost

    debug_info["is_long_candidate"] = is_long

    # ================= SHORT SIGNAL =================
    # Conditions:
    # 1. Price below SSL baseline (bearish flow)
    # 2. AlphaTrend sellers dominant (flow confirmation)
    # 3. Recent baseline touch/retest (entry opportunity)
    # 4. Candle body below baseline (resistance confirmation)
    # 5. PBEMA below price (room for TP)
    # 6. Rejection wick (bounce confirmation)
    # 7. PBEMA below baseline (target reachable - "yol var")
    # 8. SSL baseline was "lost" at some point (not shorting into parabolic uptrend)

    # THREE-TIER AT CHECK for SHORT
    if at_mode == "binary":
        at_allows_short = (at_sellers_dominant or ssl_flip_grace_short)
    elif at_mode == "regime":
        at_allows_short = not at_regime_blocks_short  # Only block if bullish regime
    elif at_mode == "score":
        at_allows_short = True  # Score mode - AT contributes to score, not filter
    else:  # "off"
        at_allows_short = True

    # REGIME FILTER CHECK for SHORT (direction-specific)
    regime_allows_short = True
    if regime_filter == "aligned":
        # SHORT only allowed in bearish regime
        regime_allows_short = (at_regime == "bearish_regime")
    elif regime_filter == "veto":
        # SHORT blocked only in bullish regime
        regime_allows_short = (at_regime != "bullish_regime")
    # "skip_neutral" already handled above, "off" = always True
    debug_info["regime_allows_short"] = regime_allows_short

    # SHORT FILTER HIERARCHY (same tier structure as LONG)
    # TIER 1 - CORE FILTERS (always required)
    tier1_short = (
        price_below_baseline and           # Core: SSL direction
        at_allows_short and                # Core: AT confirmation
        regime_allows_short and            # Core: Regime filter (skip_neutral etc.)
        pbema_below_baseline               # Core: PBEMA path exists
    )

    # TIER 2 - QUALITY FILTERS (required if filter_tier_level >= 2)
    tier2_short = (
        baseline_touch_short and           # Quality: Entry timing
        short_pbema_distance >= min_pbema_distance and  # Quality: TP distance
        (not use_vol_normalized_pbema or vol_norm_short_ok) and  # Quality: ATR-normalized distance
        baseline_ever_lost_bearish         # Quality: Not counter-trend
    )

    # TIER 3 - RISK FILTERS (required if filter_tier_level >= 3)
    tier3_short = (
        (skip_body_position or body_below_baseline) and  # Risk: Body confirmation
        (skip_wick_rejection or short_rejection)         # Risk: Wick rejection
    )

    # Combine tiers based on filter_tier_level
    if filter_tier_level == 1:
        is_short = tier1_short  # Core only - highest trade frequency
    elif filter_tier_level == 2:
        is_short = tier1_short and tier2_short  # Core + Quality
    else:  # filter_tier_level >= 3 (default)
        is_short = tier1_short and tier2_short and tier3_short  # Full filters

    debug_info["tier1_short"] = tier1_short
    debug_info["tier2_short"] = tier2_short
    debug_info["tier3_short"] = tier3_short
    debug_info["at_allows_short"] = at_allows_short

    # RSI filter for SHORT: not oversold
    short_rsi_limit = 100.0 - rsi_limit
    short_rsi_ok = rsi_val >= short_rsi_limit
    debug_info["short_rsi_ok"] = short_rsi_ok
    debug_info["short_rsi_limit"] = short_rsi_limit

    if is_short and not short_rsi_ok:
        is_short = False
        debug_info["short_rejected_rsi"] = True

    # Check if SHORT was blocked by SSL Never Lost filter
    # (all other conditions passed but baseline_ever_lost_bearish was False)
    short_blocked_by_ssl_never_lost = (
        price_below_baseline and
        at_sellers_dominant and
        baseline_touch_short and
        short_pbema_distance >= min_pbema_distance and
        pbema_below_baseline and
        short_rsi_ok and
        not baseline_ever_lost_bearish
    )
    debug_info["short_blocked_by_ssl_never_lost"] = short_blocked_by_ssl_never_lost

    debug_info["is_short_candidate"] = is_short

    # ================= SCORING MODE (ALTERNATIVE TO AND LOGIC) =================
    # If use_scoring=True, override AND logic with weighted scoring system
    if use_scoring:
        # Calculate scores for LONG and SHORT separately
        long_score, long_max, long_breakdown = calculate_signal_score(
            adx=adx_val,
            baseline_touch=baseline_touch_long,
            at_dominant=at_buyers_dominant,
            at_is_flat=at_is_flat,
            pbema_distance=long_pbema_distance,
            wick_ratio=lower_wick_ratio,
            no_overlap=(not is_overlapping or skip_overlap_check),
            body_position_ok=body_above_baseline,
            regime_ok=(regime == "TRENDING"),
        )

        short_score, short_max, short_breakdown = calculate_signal_score(
            adx=adx_val,
            baseline_touch=baseline_touch_short,
            at_dominant=at_sellers_dominant,
            at_is_flat=at_is_flat,
            pbema_distance=short_pbema_distance,
            wick_ratio=upper_wick_ratio,
            no_overlap=(not is_overlapping or skip_overlap_check),
            body_position_ok=body_below_baseline,
            regime_ok=(regime == "TRENDING"),
        )

        # Store in debug info
        debug_info["use_scoring"] = True
        debug_info["score_threshold"] = score_threshold
        debug_info["long_score"] = long_score
        debug_info["long_score_breakdown"] = long_breakdown
        debug_info["short_score"] = short_score
        debug_info["short_score_breakdown"] = short_breakdown

        # CRITICAL FILTERS (always checked even in scoring mode):
        # 1. Price position relative to baseline (determines direction)
        # 2. AlphaTrend direction (confirms buyers vs sellers)
        # 3. RSI bounds (avoid extreme overbought/oversold)

        # Override is_long/is_short based on score + critical filters
        # In score mode, add AT score to signal score
        effective_long_score = long_score + (at_score_long if at_mode == "score" else 0)
        effective_short_score = short_score + (at_score_short if at_mode == "score" else 0)

        is_long_scoring = (
            price_above_baseline and  # CORE: price direction
            at_allows_long and  # THREE-TIER: Mode-dependent AT check
            long_rsi_ok and  # CORE: not overbought
            effective_long_score >= score_threshold  # SCORING: composite quality + AT score
        )

        is_short_scoring = (
            price_below_baseline and  # CORE: price direction
            at_allows_short and  # THREE-TIER: Mode-dependent AT check
            short_rsi_ok and  # CORE: not oversold
            effective_short_score >= score_threshold  # SCORING: composite quality + AT score
        )

        # Override AND logic candidates with scoring results
        is_long = is_long_scoring
        is_short = is_short_scoring

        debug_info["is_long_candidate_scoring"] = is_long
        debug_info["is_short_candidate_scoring"] = is_short
        debug_info["effective_long_score"] = effective_long_score
        debug_info["effective_short_score"] = effective_short_score
        debug_info["at_score_long"] = at_score_long
        debug_info["at_score_short"] = at_score_short

    # ================= CONFIRMATION CANDLE CHECK (P2) =================
    # If enabled, require the current candle to confirm the signal direction
    # This addresses "entry too early" - we wait for momentum confirmation
    #
    # For LONG: Current candle should close higher than previous (bullish continuation)
    # For SHORT: Current candle should close lower than previous (bearish continuation)

    if use_confirmation_candle and abs_index >= 1:
        prev_close = float(_close_arr[abs_index - 1])

        if confirmation_candle_mode == "close":
            # Simple: current close vs previous close
            long_confirmed = close > prev_close
            short_confirmed = close < prev_close
        else:  # "body" mode
            # Stricter: current candle body in signal direction
            curr_body_mid = (open_ + close) / 2
            prev_body_mid = (float(_open_arr[abs_index - 1]) + prev_close) / 2
            long_confirmed = curr_body_mid > prev_body_mid and close > open_  # Green candle, body higher
            short_confirmed = curr_body_mid < prev_body_mid and close < open_  # Red candle, body lower

        debug_info["use_confirmation_candle"] = True
        debug_info["confirmation_mode"] = confirmation_candle_mode
        debug_info["long_confirmed"] = long_confirmed
        debug_info["short_confirmed"] = short_confirmed
        debug_info["prev_close"] = prev_close

        # Apply confirmation filter
        if is_long and not long_confirmed:
            is_long = False
            debug_info["long_rejected_confirmation"] = True

        if is_short and not short_confirmed:
            is_short = False
            debug_info["short_rejected_confirmation"] = True
    else:
        debug_info["use_confirmation_candle"] = use_confirmation_candle

    debug_info["is_long_after_confirmation"] = is_long
    debug_info["is_short_after_confirmation"] = is_short

    # ================= HTF TREND FILTER (Counter-trend prevention) =================
    # If enabled, only allow trades that align with the higher timeframe trend
    # - HTF UP trend → only LONG trades allowed
    # - HTF DOWN trend → only SHORT trades allowed
    # - HTF NEUTRAL → both directions allowed (no filter)

    if use_htf_filter and htf_df is not None:
        htf_trend, htf_confidence = detect_htf_trend(htf_df, method=htf_trend_method)
        debug_info["htf_trend"] = htf_trend
        debug_info["htf_confidence"] = htf_confidence

        if htf_trend == "UP" and is_short:
            # Block SHORT in uptrend
            is_short = False
            debug_info["htf_blocked_short"] = True

        if htf_trend == "DOWN" and is_long:
            # Block LONG in downtrend
            is_long = False
            debug_info["htf_blocked_long"] = True

    debug_info["is_long_after_htf"] = is_long
    debug_info["is_short_after_htf"] = is_short

    # ================= MARKET STRUCTURE FILTER (v2.2.0) =================
    # Required filter: Check if trade aligns with market structure
    # LONG: Requires bullish structure (HH/HL pattern)
    # SHORT: Requires bearish structure (LH/LL pattern)

    ms_score_long = 0.0
    ms_score_short = 0.0
    fvg_result_long = None
    fvg_result_short = None

    if use_market_structure and HAS_MARKET_STRUCTURE and (is_long or is_short):
        try:
            # Get Market Structure score for the candidate direction
            if is_long:
                ms_score_long, ms_result = get_structure_score(df, "LONG", abs_index)
                debug_info["ms_score_long"] = ms_score_long
                debug_info["ms_trend"] = ms_result.trend.value if ms_result else "UNKNOWN"

                if ms_score_long < min_ms_score:
                    is_long = False
                    debug_info["long_rejected_ms"] = True
                    debug_info["ms_rejection_reason"] = f"MS score {ms_score_long:.1f} < {min_ms_score}"

            if is_short:
                ms_score_short, ms_result = get_structure_score(df, "SHORT", abs_index)
                debug_info["ms_score_short"] = ms_score_short
                debug_info["ms_trend"] = ms_result.trend.value if ms_result else "UNKNOWN"

                if ms_score_short < min_ms_score:
                    is_short = False
                    debug_info["short_rejected_ms"] = True
                    debug_info["ms_rejection_reason"] = f"MS score {ms_score_short:.1f} < {min_ms_score}"
        except Exception as e:
            debug_info["ms_error"] = str(e)
            # Don't block trade on MS error

    debug_info["is_long_after_ms"] = is_long
    debug_info["is_short_after_ms"] = is_short
    debug_info["use_market_structure"] = use_market_structure

    # ================= FVG BONUS DETECTION (v2.2.0) =================
    # Bonus: Check for FVG mitigation (price returned to FVG)
    # If FVG is found, use its boundary for tighter SL

    if use_fvg_bonus and HAS_FVG_DETECTOR and (is_long or is_short):
        try:
            fvg_detector = FVGDetector(
                min_gap_percent=fvg_min_gap_percent,
                max_fvgs=50,
                lookback=200,
            )

            if is_long:
                fvg_result_long = fvg_detector.analyze(df, abs_index, "LONG")
                debug_info["fvg_is_mitigation_long"] = fvg_result_long.is_mitigation
                debug_info["fvg_in_fvg_long"] = fvg_result_long.in_fvg
                debug_info["fvg_score_long"] = fvg_result_long.score

            if is_short:
                fvg_result_short = fvg_detector.analyze(df, abs_index, "SHORT")
                debug_info["fvg_is_mitigation_short"] = fvg_result_short.is_mitigation
                debug_info["fvg_in_fvg_short"] = fvg_result_short.in_fvg
                debug_info["fvg_score_short"] = fvg_result_short.score
        except Exception as e:
            debug_info["fvg_error"] = str(e)

    debug_info["use_fvg_bonus"] = use_fvg_bonus

    # ================= EXECUTE LONG =================
    if is_long:
        # Entry: current close
        entry = close

        # TP: PBEMA cloud bottom (pb_ema_bot)
        tp = pb_bot

        # SL: Below recent swing low or below baseline
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_low = float(_low_arr[start:abs_index].min())

        # SL candidates: swing low or baseline
        sl_swing = swing_low * 0.998
        sl_baseline = baseline * 0.998
        sl = min(sl_swing, sl_baseline)

        # FVG BONUS: Use FVG boundary for tighter SL if available
        fvg_sl_used = False
        if (use_fvg_bonus and fvg_result_long is not None and
            fvg_result_long.is_mitigation and fvg_result_long.mitigation_fvg is not None):
            fvg = fvg_result_long.mitigation_fvg
            # Only use FVG SL if it's a bullish FVG and provides tighter SL
            if fvg.fvg_type == FVGType.BULLISH:
                fvg_sl = fvg.low * 0.998
                if fvg_sl < entry and fvg_sl > sl:  # Tighter than current SL
                    sl = fvg_sl
                    fvg_sl_used = True
                    debug_info["fvg_sl_used_long"] = True
                    debug_info["fvg_sl_value"] = fvg_sl

        if tp <= entry:
            return _ret(None, None, None, None, "TP Below Entry (LONG)")
        if sl >= entry:
            sl = min(swing_low * 0.995, baseline * 0.995)

        risk = entry - sl
        reward = tp - entry

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (LONG)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "rr_value": rr,
            "tp_dist_ratio": tp_dist_ratio,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        # Build reason with optional FVG tag
        fvg_tag = "+FVG" if fvg_sl_used else ""
        reason = f"ACCEPTED(SSL_Flow{fvg_tag},R:{rr:.2f})"
        return _ret("LONG", entry, tp, sl, reason)

    # ================= EXECUTE SHORT =================
    if is_short:
        # Entry: current close
        entry = close

        # TP: PBEMA cloud top (pb_ema_top)
        tp = pb_top

        # SL: Above recent swing high or above baseline
        swing_n = 20
        start = max(0, abs_index - swing_n)
        swing_high = float(_high_arr[start:abs_index].max())

        # SL candidates: swing high or baseline
        sl_swing = swing_high * 1.002
        sl_baseline = baseline * 1.002
        sl = max(sl_swing, sl_baseline)

        # FVG BONUS: Use FVG boundary for tighter SL if available
        fvg_sl_used = False
        if (use_fvg_bonus and fvg_result_short is not None and
            fvg_result_short.is_mitigation and fvg_result_short.mitigation_fvg is not None):
            fvg = fvg_result_short.mitigation_fvg
            # Only use FVG SL if it's a bearish FVG and provides tighter SL
            if fvg.fvg_type == FVGType.BEARISH:
                fvg_sl = fvg.high * 1.002
                if fvg_sl > entry and fvg_sl < sl:  # Tighter than current SL
                    sl = fvg_sl
                    fvg_sl_used = True
                    debug_info["fvg_sl_used_short"] = True
                    debug_info["fvg_sl_value"] = fvg_sl

        if tp >= entry:
            return _ret(None, None, None, None, "TP Above Entry (SHORT)")
        if sl <= entry:
            sl = max(swing_high * 1.005, baseline * 1.005)

        risk = sl - entry
        reward = entry - tp

        if risk <= 0 or reward <= 0:
            return _ret(None, None, None, None, "Invalid RR (SHORT)")

        rr = reward / risk
        tp_dist_ratio = reward / entry

        debug_info.update({
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "rr_value": rr,
            "tp_dist_ratio": tp_dist_ratio,
        })

        if tp_dist_ratio < tp_min_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
        if tp_dist_ratio > tp_max_dist_ratio:
            return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
        if rr < min_rr:
            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

        # Build reason with optional FVG tag
        fvg_tag = "+FVG" if fvg_sl_used else ""
        reason = f"ACCEPTED(SSL_Flow{fvg_tag},R:{rr:.2f})"
        return _ret("SHORT", entry, tp, sl, reason)

    # Check if SSL Never Lost filter blocked the trade
    if long_blocked_by_ssl_never_lost:
        return _ret(None, None, None, None, "SSL Never Lost (LONG blocked - bullish trend too strong)")
    if short_blocked_by_ssl_never_lost:
        return _ret(None, None, None, None, "SSL Never Lost (SHORT blocked - bullish trend too strong)")

    return _ret(None, None, None, None, "No Signal")
