# strategies/pbema_retest_v2.py
# PBEMA Retest Strategy V2 - Corrected Logic Based on Real Trade Analysis
#
# ============================================================================
# CRITICAL FIX (2026-01-04): Previous version had WRONG logic!
# ============================================================================
#
# OLD (WRONG) LOGIC:
#   - Fiyat PBEMA'yı yukarı kırdı → Hemen LONG al
#   - Problem: Retest beklemeden entry, fakeout'larda kayıp
#
# NEW (CORRECT) LOGIC:
#   - PBEMA'nın rolü = Fiyatın GELİŞ YÖNÜNE bağlı
#   - Fiyat AŞAĞIDAN gelip PBEMA'ya değdi → PBEMA = RESISTANCE → SHORT
#   - Fiyat YUKARIDAN gelip PBEMA'ya değdi → PBEMA = SUPPORT → LONG
#   - Kırılma olursa → RETEST BEKLE → Onay sonrası entry
#
# ============================================================================
# STRATEJI KURALLARI:
# ============================================================================
#
# SENARYO 1: PBEMA RESISTANCE (Fiyat aşağıdan geliyor)
#   - Son N mum PBEMA altında
#   - Fiyat yukarı hareket edip PBEMA'ya değiyor
#   - PBEMA resistance olarak çalışır → SHORT entry
#   - TP: Aşağıda (baseline veya swing low)
#   - SL: PBEMA üstü
#
# SENARYO 2: PBEMA SUPPORT (Fiyat yukarıdan geliyor)
#   - Son N mum PBEMA üstünde
#   - Fiyat aşağı hareket edip PBEMA'ya değiyor
#   - PBEMA support olarak çalışır → LONG entry
#   - TP: Yukarıda (baseline veya swing high)
#   - SL: PBEMA altı
#
# SENARYO 3: KIRILMA + RETEST CONFIRMATION
#   - Fiyat PBEMA'yı net kırar (breakout)
#   - Kırılma sonrası geri dönüp PBEMA'yı RETEST eder
#   - Retest başarılı (bounce) → Entry (kırılma yönünde)
#   - Bu senaryoda PBEMA'nın rolü DEĞİŞİR
#
# ============================================================================

from typing import Tuple, Union, Dict, Optional
import pandas as pd
import numpy as np

from .base import SignalResult, SignalResultWithDebug


def check_pbema_retest_signal_v2(
        df: pd.DataFrame,
        index: int = -2,
        # === Approach Direction Detection ===
        approach_lookback: int = 10,  # Son N mum'a bak (fiyat nereden geliyor?)
        approach_threshold: float = 0.7,  # %70'i PBEMA'nın bir tarafında olmalı
        # === Touch Detection ===
        touch_tolerance: float = 0.003,  # PBEMA'ya değme toleransı (%0.3)
        min_wick_ratio: float = 0.15,  # Minimum wick rejection (%15)
        # === Breakout + Retest Confirmation ===
        breakout_threshold: float = 0.005,  # Net kırılma için min mesafe (%0.5)
        retest_after_breakout: bool = True,  # Kırılma sonrası retest bekle
        min_candles_after_breakout: int = 3,  # Kırılmadan sonra min mum sayısı
        # === TP/SL Configuration ===
        tp_target: str = "baseline",  # "baseline", "percentage", "swing"
        tp_percentage: float = 0.015,  # %1.5 TP
        sl_buffer: float = 0.003,  # SL buffer (%0.3)
        min_sl_distance: float = 0.01,  # Min SL mesafesi (%1)
        use_atr_sl: bool = True,
        atr_sl_multiplier: float = 1.5,
        # === Filters (OPTIMIZED 2026-01-04) ===
        # Best config: +long_only +regime(ADX>25) = +$1073 PnL, 43.7% WR
        long_only: bool = True,  # Only LONG signals (SHORT is unprofitable)
        regime_filter: bool = True,  # Require ADX > threshold
        regime_adx_min: float = 25.0,  # Minimum ADX for regime filter
        require_trend_alignment: bool = False,  # Baseline trend filter (optional)
        min_rr: float = 1.0,
        # === Debug ===
        return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """
    PBEMA Retest Strategy V2 - Corrected approach-based logic.

    This version fixes the critical bug in V1 where the bot would:
    - Take LONG when price breaks above PBEMA (wrong - should wait for retest)
    - Ignore the approach direction (where price is coming FROM)

    Correct Logic:
    1. Determine approach direction (is price coming from above or below PBEMA?)
    2. If approaching from below → PBEMA acts as RESISTANCE → SHORT on touch
    3. If approaching from above → PBEMA acts as SUPPORT → LONG on touch
    4. If breakout occurs → Wait for RETEST confirmation before entry

    Args:
        df: DataFrame with OHLCV + indicators (pb_ema_top, pb_ema_bot, baseline)
        index: Candle index for signal check
        approach_lookback: Candles to determine approach direction
        approach_threshold: % of candles that must be on one side of PBEMA
        touch_tolerance: Tolerance for PBEMA touch detection
        min_wick_ratio: Minimum wick size for rejection confirmation
        breakout_threshold: Minimum distance for valid breakout
        retest_after_breakout: Require retest after breakout
        min_candles_after_breakout: Min candles between breakout and retest
        tp_target: TP calculation mode
        tp_percentage: TP percentage if using percentage mode
        sl_buffer: SL buffer beyond PBEMA
        min_sl_distance: Minimum SL distance
        use_atr_sl: Use ATR-based SL
        atr_sl_multiplier: ATR multiplier for SL
        require_trend_alignment: Require baseline trend alignment
        min_rr: Minimum risk/reward ratio
        return_debug: Return debug info

    Returns:
        SignalResult or SignalResultWithDebug
    """

    debug_info = {
        "approach_direction": None,  # "FROM_BELOW" or "FROM_ABOVE"
        "pbema_role": None,  # "RESISTANCE" or "SUPPORT"
        "touch_detected": False,
        "wick_rejection": False,
        "breakout_detected": False,
        "breakout_confirmed": False,  # Retest sonrası onay
        "candles_below_pbema": 0,
        "candles_above_pbema": 0,
        "signal_scenario": None,  # "RESISTANCE_SHORT", "SUPPORT_LONG", "BREAKOUT_RETEST"
    }

    def _ret(s_type, entry, tp, sl, reason):
        if return_debug:
            return s_type, entry, tp, sl, reason, debug_info
        return s_type, entry, tp, sl, reason

    # === Validate Input ===
    if df is None or df.empty:
        return _ret(None, None, None, None, "No Data")

    required_cols = ["open", "high", "low", "close", "pb_ema_top", "pb_ema_bot"]
    for col in required_cols:
        if col not in df.columns:
            return _ret(None, None, None, None, f"Missing {col}")

    try:
        curr = df.iloc[index]
    except (IndexError, KeyError):
        return _ret(None, None, None, None, "Index Error")

    abs_index = index if index >= 0 else (len(df) + index)
    if abs_index < approach_lookback + 20:
        return _ret(None, None, None, None, "Not Enough Data")

    # === Extract Current Values ===
    open_ = float(curr["open"])
    high = float(curr["high"])
    low = float(curr["low"])
    close = float(curr["close"])
    pb_top = float(curr["pb_ema_top"])
    pb_bot = float(curr["pb_ema_bot"])
    pb_mid = (pb_top + pb_bot) / 2

    if any(pd.isna([open_, high, low, close, pb_top, pb_bot])):
        return _ret(None, None, None, None, "NaN Values")

    # === REGIME FILTER (ADX) ===
    # Best config: ADX > 25 gives +$1073 PnL vs +$48 without
    if regime_filter:
        if "adx" in df.columns:
            adx = curr.get("adx")
            if pd.isna(adx) or float(adx) < regime_adx_min:
                return _ret(None, None, None, None, f"Regime Filter: ADX {adx:.1f} < {regime_adx_min}")

    # Get baseline for TP
    baseline = None
    if "baseline" in df.columns:
        baseline = float(curr["baseline"]) if not pd.isna(curr.get("baseline")) else None

    # =========================================================================
    # STEP 1: DETERMINE APPROACH DIRECTION
    # =========================================================================
    # Son N mum'un çoğunluğu PBEMA'nın hangi tarafında?
    # Bu bize fiyatın "nereden geldiğini" söyler.

    candles_below = 0
    candles_above = 0

    for i in range(abs_index - approach_lookback, abs_index):
        if i < 0:
            continue
        candle = df.iloc[i]
        candle_close = float(candle["close"])
        candle_pb_mid = (float(candle["pb_ema_top"]) + float(candle["pb_ema_bot"])) / 2

        if candle_close < candle_pb_mid:
            candles_below += 1
        else:
            candles_above += 1

    debug_info["candles_below_pbema"] = candles_below
    debug_info["candles_above_pbema"] = candles_above

    total_candles = candles_below + candles_above
    if total_candles == 0:
        return _ret(None, None, None, None, "No Valid Candles")

    below_ratio = candles_below / total_candles
    above_ratio = candles_above / total_candles

    # Determine approach direction
    if below_ratio >= approach_threshold:
        approach_direction = "FROM_BELOW"
        pbema_role = "RESISTANCE"
    elif above_ratio >= approach_threshold:
        approach_direction = "FROM_ABOVE"
        pbema_role = "SUPPORT"
    else:
        # Mixed - no clear direction, check for breakout scenario
        approach_direction = "MIXED"
        pbema_role = "UNDEFINED"

    debug_info["approach_direction"] = approach_direction
    debug_info["pbema_role"] = pbema_role

    # =========================================================================
    # STEP 2: DETECT TOUCH / INTERACTION WITH PBEMA
    # =========================================================================

    # Current candle touching PBEMA?
    touching_top = high >= pb_top * (1 - touch_tolerance) and low <= pb_top * (1 + touch_tolerance)
    touching_bot = high >= pb_bot * (1 - touch_tolerance) and low <= pb_bot * (1 + touch_tolerance)
    touching_cloud = low <= pb_top and high >= pb_bot  # Inside cloud

    touch_detected = touching_top or touching_bot or touching_cloud
    debug_info["touch_detected"] = touch_detected

    if not touch_detected:
        return _ret(None, None, None, None, "No PBEMA Touch")

    # =========================================================================
    # STEP 3: CHECK FOR WICK REJECTION
    # =========================================================================

    candle_range = high - low
    if candle_range <= 0:
        return _ret(None, None, None, None, "Zero Range Candle")

    candle_body_high = max(open_, close)
    candle_body_low = min(open_, close)
    upper_wick = high - candle_body_high
    lower_wick = candle_body_low - low

    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    debug_info["upper_wick_ratio"] = upper_wick_ratio
    debug_info["lower_wick_ratio"] = lower_wick_ratio

    # =========================================================================
    # SCENARIO A: PBEMA RESISTANCE (Fiyat aşağıdan geliyor)
    # =========================================================================

    if approach_direction == "FROM_BELOW":
        # Fiyat aşağıdan gelip PBEMA'ya değdi
        # PBEMA resistance olarak çalışır → SHORT

        # Check for upper wick rejection (fiyat PBEMA'dan geri döndü)
        wick_rejection = upper_wick_ratio >= min_wick_ratio
        debug_info["wick_rejection"] = wick_rejection

        # Additional confirmation: close should be below PBEMA top
        close_below_pbema = close < pb_top

        if wick_rejection and close_below_pbema:
            debug_info["signal_scenario"] = "RESISTANCE_SHORT"

            entry = close

            # TP: Baseline (aşağıda) veya percentage
            if tp_target == "baseline" and baseline is not None and baseline < close:
                tp = baseline
            else:
                tp = close * (1 - tp_percentage)

            # SL: PBEMA üstü
            if use_atr_sl and 'atr' in df.columns:
                atr = float(curr.get('atr', 0))
                if atr > 0:
                    sl = entry + (atr * atr_sl_multiplier)
                else:
                    sl = pb_top * (1 + sl_buffer)
            else:
                sl = pb_top * (1 + sl_buffer)

            # Ensure minimum SL distance
            sl_distance_pct = (sl - entry) / entry
            if sl_distance_pct < min_sl_distance:
                sl = entry * (1 + min_sl_distance)

            # Validate
            if tp >= entry:
                return _ret(None, None, None, None, "TP Above Entry (SHORT)")

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (SHORT)")

            rr = reward / risk
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            # Optional trend alignment
            if require_trend_alignment and baseline is not None:
                if close > baseline:  # Counter-trend
                    return _ret(None, None, None, None, "Counter-trend SHORT")

            # === LONG_ONLY FILTER ===
            # SHORT is unprofitable (-$107 vs LONG +$156)
            if long_only:
                return _ret(None, None, None, None, "Long Only Filter: SHORT blocked")

            reason = f"PBEMA_RESIST_SHORT(R:{rr:.2f})"
            return _ret("SHORT", entry, tp, sl, reason)

        return _ret(None, None, None, None, "FROM_BELOW: No rejection")

    # =========================================================================
    # SCENARIO B: PBEMA SUPPORT (Fiyat yukarıdan geliyor)
    # =========================================================================

    if approach_direction == "FROM_ABOVE":
        # Fiyat yukarıdan gelip PBEMA'ya değdi
        # PBEMA support olarak çalışır → LONG

        # Check for lower wick rejection (fiyat PBEMA'dan geri döndü)
        wick_rejection = lower_wick_ratio >= min_wick_ratio
        debug_info["wick_rejection"] = wick_rejection

        # Additional confirmation: close should be above PBEMA bottom
        close_above_pbema = close > pb_bot

        if wick_rejection and close_above_pbema:
            debug_info["signal_scenario"] = "SUPPORT_LONG"

            entry = close

            # TP: Baseline (yukarıda) veya percentage
            if tp_target == "baseline" and baseline is not None and baseline > close:
                tp = baseline
            else:
                tp = close * (1 + tp_percentage)

            # SL: PBEMA altı
            if use_atr_sl and 'atr' in df.columns:
                atr = float(curr.get('atr', 0))
                if atr > 0:
                    sl = entry - (atr * atr_sl_multiplier)
                else:
                    sl = pb_bot * (1 - sl_buffer)
            else:
                sl = pb_bot * (1 - sl_buffer)

            # Ensure minimum SL distance
            sl_distance_pct = (entry - sl) / entry
            if sl_distance_pct < min_sl_distance:
                sl = entry * (1 - min_sl_distance)

            # Validate
            if tp <= entry:
                return _ret(None, None, None, None, "TP Below Entry (LONG)")

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (LONG)")

            rr = reward / risk
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            # Optional trend alignment
            if require_trend_alignment and baseline is not None:
                if close < baseline:  # Counter-trend
                    return _ret(None, None, None, None, "Counter-trend LONG")

            reason = f"PBEMA_SUPPORT_LONG(R:{rr:.2f})"
            return _ret("LONG", entry, tp, sl, reason)

        return _ret(None, None, None, None, "FROM_ABOVE: No rejection")

    # =========================================================================
    # SCENARIO C: BREAKOUT + RETEST CONFIRMATION
    # =========================================================================
    # Bu senaryo şu durumda geçerli:
    # - Fiyat PBEMA'yı NET kırmış (güçlü breakout)
    # - Şimdi geri dönüp PBEMA'yı test ediyor (retest)
    # - Retest başarılı (bounce) → Entry

    if approach_direction == "MIXED" or not retest_after_breakout:
        # Mixed direction veya retest gerekmiyorsa, breakout pattern ara

        # Son 20 mum'da breakout var mı?
        breakout_lookback = 20
        breakout_type = None  # "BULLISH" or "BEARISH"
        breakout_candle_idx = None

        for i in range(abs_index - breakout_lookback, abs_index - min_candles_after_breakout):
            if i < 10:
                continue

            candle = df.iloc[i]
            candle_close = float(candle["close"])
            candle_pb_top = float(candle["pb_ema_top"])
            candle_pb_bot = float(candle["pb_ema_bot"])

            prev_candle = df.iloc[i - 1]
            prev_close = float(prev_candle["close"])
            prev_pb_top = float(prev_candle["pb_ema_top"])
            prev_pb_bot = float(prev_candle["pb_ema_bot"])

            # BULLISH BREAKOUT: Fiyat PBEMA'yı yukarı kırdı
            if prev_close < prev_pb_top and candle_close > candle_pb_top:
                breakout_dist = (candle_close - candle_pb_top) / candle_pb_top
                if breakout_dist >= breakout_threshold:
                    breakout_type = "BULLISH"
                    breakout_candle_idx = i
                    break

            # BEARISH BREAKOUT: Fiyat PBEMA'yı aşağı kırdı
            if prev_close > prev_pb_bot and candle_close < candle_pb_bot:
                breakout_dist = (candle_pb_bot - candle_close) / candle_pb_bot
                if breakout_dist >= breakout_threshold:
                    breakout_type = "BEARISH"
                    breakout_candle_idx = i
                    break

        if breakout_type:
            debug_info["breakout_detected"] = True
            debug_info["breakout_type"] = breakout_type

            # Şimdi retest mi oluyor?
            if breakout_type == "BULLISH":
                # Bullish breakout sonrası PBEMA artık SUPPORT olmalı
                # Fiyat geri dönüp PBEMA'yı test ediyorsa → LONG

                # Current candle PBEMA'ya değiyor mu?
                retest_touch = low <= pb_top * (1 + touch_tolerance)
                close_above = close > pb_mid

                if retest_touch and close_above:
                    # Lower wick rejection var mı?
                    if lower_wick_ratio >= min_wick_ratio:
                        debug_info["breakout_confirmed"] = True
                        debug_info["signal_scenario"] = "BREAKOUT_RETEST_LONG"

                        entry = close

                        if tp_target == "baseline" and baseline is not None and baseline > close:
                            tp = baseline
                        else:
                            tp = close * (1 + tp_percentage)

                        if use_atr_sl and 'atr' in df.columns:
                            atr = float(curr.get('atr', 0))
                            if atr > 0:
                                sl = entry - (atr * atr_sl_multiplier)
                            else:
                                sl = pb_bot * (1 - sl_buffer)
                        else:
                            sl = pb_bot * (1 - sl_buffer)

                        sl_distance_pct = (entry - sl) / entry
                        if sl_distance_pct < min_sl_distance:
                            sl = entry * (1 - min_sl_distance)

                        if tp <= entry:
                            return _ret(None, None, None, None, "TP Below Entry (BREAKOUT LONG)")

                        risk = entry - sl
                        reward = tp - entry
                        if risk <= 0 or reward <= 0:
                            return _ret(None, None, None, None, "Invalid RR")

                        rr = reward / risk
                        if rr < min_rr:
                            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

                        reason = f"PBEMA_BREAKOUT_LONG(R:{rr:.2f})"
                        return _ret("LONG", entry, tp, sl, reason)

            elif breakout_type == "BEARISH":
                # Bearish breakout sonrası PBEMA artık RESISTANCE olmalı
                # Fiyat geri dönüp PBEMA'yı test ediyorsa → SHORT

                # Current candle PBEMA'ya değiyor mu?
                retest_touch = high >= pb_bot * (1 - touch_tolerance)
                close_below = close < pb_mid

                if retest_touch and close_below:
                    # Upper wick rejection var mı?
                    if upper_wick_ratio >= min_wick_ratio:
                        debug_info["breakout_confirmed"] = True
                        debug_info["signal_scenario"] = "BREAKOUT_RETEST_SHORT"

                        entry = close

                        if tp_target == "baseline" and baseline is not None and baseline < close:
                            tp = baseline
                        else:
                            tp = close * (1 - tp_percentage)

                        if use_atr_sl and 'atr' in df.columns:
                            atr = float(curr.get('atr', 0))
                            if atr > 0:
                                sl = entry + (atr * atr_sl_multiplier)
                            else:
                                sl = pb_top * (1 + sl_buffer)
                        else:
                            sl = pb_top * (1 + sl_buffer)

                        sl_distance_pct = (sl - entry) / entry
                        if sl_distance_pct < min_sl_distance:
                            sl = entry * (1 + min_sl_distance)

                        if tp >= entry:
                            return _ret(None, None, None, None, "TP Above Entry (BREAKOUT SHORT)")

                        risk = sl - entry
                        reward = entry - tp
                        if risk <= 0 or reward <= 0:
                            return _ret(None, None, None, None, "Invalid RR")

                        rr = reward / risk
                        if rr < min_rr:
                            return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

                        # === LONG_ONLY FILTER ===
                        if long_only:
                            return _ret(None, None, None, None, "Long Only Filter: SHORT blocked")

                        reason = f"PBEMA_BREAKOUT_SHORT(R:{rr:.2f})"
                        return _ret("SHORT", entry, tp, sl, reason)

        return _ret(None, None, None, None, "No valid breakout+retest")

    return _ret(None, None, None, None, "No Signal")


# Convenience alias
check_pbema_v2 = check_pbema_retest_signal_v2
