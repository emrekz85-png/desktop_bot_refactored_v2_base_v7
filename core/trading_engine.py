# core/trading_engine.py
# Trading Engine - Core trading logic, data fetching, and signal generation
#
# This module provides:
# - TradingEngine class with API data fetching (with retry logic)
# - Signal detection for multiple strategies (keltner_bounce, pbema_reaction)
# - Indicator calculation wrappers
#
# Plotting/GUI-dependent functions remain in the main file.

import os
import time
import json
import itertools
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests

from .config import (
    DATA_DIR,
    DEFAULT_STRATEGY_CONFIG,
    TRADING_CONFIG,
)
from .indicators import (
    calculate_indicators as core_calculate_indicators,
    calculate_alphatrend as core_calculate_alphatrend,
)
from .telegram import send_telegram as _core_send_telegram


class TradingEngine:
    """Core trading engine for data fetching, indicator calculation, and signal detection.

    This class provides:
    - Binance API data fetching with retry logic
    - Parallel data fetching for multiple symbols/timeframes
    - Technical indicator calculations
    - Signal detection for keltner_bounce and pbema_reaction strategies

    Note: Plotting/GUI-dependent methods (create_chart_data_json, debug_plot_backtest_trade)
    are implemented as standalone functions in the main application file.
    """

    # Network cooldown to prevent repeated DNS failures
    _network_cooldown_until = 0

    @staticmethod
    def send_telegram(token, chat_id, message):
        """Send Telegram message asynchronously.

        Uses core.telegram module which provides:
        - Thread pool (prevents thread accumulation)
        - Rate limiting (prevents API throttling)
        - Retry logic
        - Environment variable support for credentials

        Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables
        for better security (instead of storing in config.json).
        """
        if not token or not chat_id:
            return

        # Use core package version (has thread pool and rate limiting)
        _core_send_telegram(token, chat_id, message)

    @staticmethod
    def http_get_with_retry(url, params, max_retries=3, timeout=10):
        """Safe HTTP GET with retry logic for handling API failures."""

        # DNS resolution failure cooldown
        now = time.time()
        if now < TradingEngine._network_cooldown_until:
            cooldown_left = int(TradingEngine._network_cooldown_until - now)
            print(f"BAĞLANTI HATASI: Ağ erişimi yok. {cooldown_left}s sonra yeniden denenecek.")
            return None

        delay = 1
        for attempt in range(max_retries):
            try:
                res = requests.get(url, params=params, timeout=timeout)

                # Handle rate limits (429) and server errors (5xx)
                if res.status_code == 429 or res.status_code >= 500:
                    print(f"API HATA {res.status_code} (Deneme {attempt + 1}/{max_retries}). Bekleniyor...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue

                TradingEngine._network_cooldown_until = 0
                return res
            except requests.exceptions.RequestException as e:
                is_dns_error = isinstance(e, requests.exceptions.ConnectionError) and "NameResolutionError" in str(e)
                print(f"BAĞLANTI HATASI (Deneme {attempt + 1}/{max_retries}): {e}")

                # DNS failure: 5 minute cooldown
                if is_dns_error:
                    TradingEngine._network_cooldown_until = time.time() + 300
                    break

                time.sleep(delay)
                delay *= 2

        return None

    @staticmethod
    def get_data(symbol, interval, limit=500):
        """Fetch OHLCV data from Binance Futures API."""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}

            res = TradingEngine.http_get_with_retry(url, params)

            if res is None:
                return pd.DataFrame()

            data = res.json()
            if not data or not isinstance(data, list):
                return pd.DataFrame()

            df = pd.DataFrame(data).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            print(f"VERİ ÇEKME HATASI ({symbol}): {e}")
            error_log_path = os.path.join(DATA_DIR, "error_log.txt")
            try:
                from datetime import datetime
                with open(error_log_path, "a") as f:
                    f.write(f"\n[{datetime.now()}] GET_DATA HATA ({symbol}): {str(e)}\n")
            except Exception:
                pass
            return pd.DataFrame()

    @staticmethod
    def fetch_worker(args):
        """Worker function for parallel data fetching."""
        symbol, tf = args
        try:
            df = TradingEngine.get_data(symbol, tf, limit=500)
            return (symbol, tf, df)
        except Exception:
            return (symbol, tf, pd.DataFrame())

    @staticmethod
    def get_all_candles_parallel(symbol_list, timeframe_list):
        """Fetch candle data for multiple symbols/timeframes in parallel."""
        tasks = list(itertools.product(symbol_list, timeframe_list))
        results = {}

        # Limited to 5 workers for API rate limit protection
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {executor.submit(TradingEngine.fetch_worker, t): t for t in tasks}
            for future in future_to_task:
                try:
                    sym, tf, df = future.result()
                    results[(sym, tf)] = df
                except Exception as e:
                    print(f"Paralel Veri Hatası: {e}")
        return results

    @staticmethod
    def get_latest_prices(symbols):
        """Lightweight ticker fetcher to refresh UI prices without heavy kline calls."""
        prices = {}
        url = "https://fapi.binance.com/fapi/v1/ticker/price"

        for sym in symbols:
            try:
                res = TradingEngine.http_get_with_retry(url, {"symbol": sym}, max_retries=2)
                if res is None:
                    continue
                data = res.json()
                if isinstance(data, dict) and "price" in data:
                    prices[sym] = float(data["price"])
            except Exception as e:
                print(f"[PRICE] {sym} fiyatı alınamadı: {e}")

        return prices

    @staticmethod
    def get_historical_data_pagination(symbol, interval, total_candles=5000, start_date=None, end_date=None):
        """
        Fetch historical kline data from Binance with pagination.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (e.g., "5m", "1h")
            total_candles: Number of candles to fetch (used if no date range)
            start_date: Start date (str "YYYY-MM-DD" or datetime)
            end_date: End date (str "YYYY-MM-DD" or datetime)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        from datetime import datetime, timezone

        all_data = []
        limit_per_req = 1000

        # Date range mode
        if start_date is not None:
            # Convert string to datetime
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            elif start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)

            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59, tzinfo=timezone.utc
                    )
                elif end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            else:
                end_date = datetime.now(timezone.utc)

            start_time_ms = int(start_date.timestamp() * 1000)
            end_time_ms = int(end_date.timestamp() * 1000)

            current_start = start_time_ms

            while current_start < end_time_ms:
                try:
                    url = "https://fapi.binance.com/fapi/v1/klines"
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'limit': limit_per_req,
                        'startTime': current_start,
                        'endTime': end_time_ms
                    }

                    res = TradingEngine.http_get_with_retry(url, params)
                    if res is None:
                        break

                    data = res.json()
                    if not data or not isinstance(data, list):
                        break

                    all_data.extend(data)

                    last_candle_time = data[-1][0]
                    current_start = last_candle_time + 1

                    if len(data) < limit_per_req:
                        break

                    time.sleep(0.1)  # Rate limit courtesy
                except Exception:
                    break
        else:
            # Candle count mode (legacy behavior)
            end_time = int(time.time() * 1000)
            loops = int(np.ceil(total_candles / limit_per_req))

            for _ in range(loops):
                try:
                    url = "https://fapi.binance.com/fapi/v1/klines"
                    params = {'symbol': symbol, 'interval': interval, 'limit': limit_per_req, 'endTime': end_time}

                    res = TradingEngine.http_get_with_retry(url, params)
                    if res is None:
                        break

                    data = res.json()
                    if not data or not isinstance(data, list):
                        break

                    all_data = data + all_data
                    end_time = data[0][0] - 1
                    time.sleep(0.1)
                except Exception:
                    break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if start_date is not None:
            return df.reset_index(drop=True)
        return df.tail(total_candles).reset_index(drop=True)

    @staticmethod
    def calculate_alphatrend(df, coeff=1, ap=14):
        """Calculate AlphaTrend indicator.

        NOTE: Implementation delegated to core.indicators.calculate_alphatrend for single source of truth.
        """
        return core_calculate_alphatrend(df, coeff=coeff, ap=ap)

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for Base Setup.

        - RSI(14)
        - ADX(14)
        - PBEMA cloud: EMA200(high) and EMA200(close)
        - SSL baseline: HMA60(close)
        - Keltner bands: baseline +/- EMA60(TrueRange) * 0.2
        - AlphaTrend: optional filter

        PERFORMANCE NOTE: This function modifies the DataFrame in-place.
        If you need to preserve the original DataFrame, make a copy before calling.

        NOTE: Implementation delegated to core.indicators.calculate_indicators for single source of truth.
        """
        return core_calculate_indicators(df)

    @staticmethod
    def check_signal_diagnostic(
            df: pd.DataFrame,
            index: int = -2,
            min_rr: float = 2.0,
            rsi_limit: float = 60.0,
            slope_thresh: float = 0.5,
            use_alphatrend: bool = True,
            hold_n: int = 5,
            min_hold_frac: float = 0.8,
            pb_touch_tolerance: float = 0.0012,
            body_tolerance: float = 0.0015,
            cloud_keltner_gap_min: float = 0.003,
            tp_min_dist_ratio: float = 0.0015,
            tp_max_dist_ratio: float = 0.03,
            adx_min: float = 12.0,
            return_debug: bool = False,
    ) -> Tuple:
        """
        Base Setup signal detection for LONG / SHORT.

        Filters:
        - ADX minimum threshold
        - Keltner holding + retest
        - PBEMA cloud alignment
        - Minimum distance between Keltner band and PBEMA TP target
        - TP not too close / too far
        - RR >= min_rr (RR = reward / risk)

        Note: This is a mean reversion approach using PBEMA cloud as magnet;
        Keltner touches trigger both SHORT from top and LONG from bottom.
        """

        debug_info = {
            "adx_ok": None,
            "trend_up_strong": None,
            "trend_down_strong": None,
            "holding_long": None,
            "retest_long": None,
            "pb_target_long": None,
            "long_rsi_ok": None,
            "holding_short": None,
            "retest_short": None,
            "pb_target_short": None,
            "short_rsi_ok": None,
            "tp_dist_ratio": None,
            "rr_value": None,
            "long_rr_ok": None,
            "short_rr_ok": None,
        }

        def _ret(s_type, entry, tp, sl, reason):
            if return_debug:
                return s_type, entry, tp, sl, reason, debug_info
            return s_type, entry, tp, sl, reason

        if df is None or df.empty:
            return _ret(None, None, None, None, "No Data")

        required_cols = [
            "open", "high", "low", "close",
            "rsi", "adx",
            "pb_ema_top", "pb_ema_bot",
            "keltner_upper", "keltner_lower",
        ]
        for col in required_cols:
            if col not in df.columns:
                return _ret(None, None, None, None, f"Missing {col}")

        try:
            curr = df.iloc[index]
        except Exception:
            return _ret(None, None, None, None, "Index Error")

        for c in required_cols:
            v = curr.get(c)
            if pd.isna(v):
                return _ret(None, None, None, None, f"NaN in {c}")

        abs_index = index if index >= 0 else (len(df) + index)
        if abs_index < 0 or abs_index >= len(df):
            return _ret(None, None, None, None, "Index Out of Range")

        # --- Parameters ---
        hold_n = int(max(1, hold_n or 1))
        min_hold_frac = float(min_hold_frac if min_hold_frac is not None else 0.8)
        touch_tol = float(pb_touch_tolerance if pb_touch_tolerance is not None else 0.0012)
        body_tol = float(body_tolerance if body_tolerance is not None else 0.0015)
        cloud_keltner_gap_min = float(cloud_keltner_gap_min if cloud_keltner_gap_min is not None else 0.003)
        tp_min_dist_ratio = float(tp_min_dist_ratio if tp_min_dist_ratio is not None else 0.0015)
        tp_max_dist_ratio = float(tp_max_dist_ratio if tp_max_dist_ratio is not None else 0.03)
        adx_min = float(adx_min if adx_min is not None else 12.0)

        # ADX filter
        debug_info["adx_ok"] = float(curr["adx"]) >= adx_min
        if not debug_info["adx_ok"]:
            return _ret(None, None, None, None, "ADX Low")

        if abs_index < hold_n + 1:
            return _ret(None, None, None, None, "Warmup")

        slc = slice(abs_index - hold_n, abs_index)
        closes_slice = df["close"].iloc[slc]
        upper_slice = df["keltner_upper"].iloc[slc]
        lower_slice = df["keltner_lower"].iloc[slc]

        close = float(curr["close"])
        open_ = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])
        upper_band = float(curr["keltner_upper"])
        lower_band = float(curr["keltner_lower"])
        pb_top = float(curr["pb_ema_top"])
        pb_bot = float(curr["pb_ema_bot"])

        # --- Mean reversion: Slope filter DISABLED ---
        # PBEMA (200 EMA) moves too slowly - slope filter is wrong for mean reversion
        slope_top = float(curr.get("slope_top", 0.0) or 0.0)
        slope_bot = float(curr.get("slope_bot", 0.0) or 0.0)

        # Keep for debug, but NO filtering
        debug_info["slope_top"] = slope_top
        debug_info["slope_bot"] = slope_bot

        # Mean reversion = no direction restriction
        long_direction_ok = True
        short_direction_ok = True
        debug_info["long_direction_ok"] = long_direction_ok
        debug_info["short_direction_ok"] = short_direction_ok

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

        min_wick_ratio = 0.15
        long_rejection_quality = lower_wick_ratio >= min_wick_ratio
        short_rejection_quality = upper_wick_ratio >= min_wick_ratio

        debug_info.update({
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "long_rejection_quality": long_rejection_quality,
            "short_rejection_quality": short_rejection_quality,
        })

        # ================= PRICE-PBEMA POSITION CHECK =================
        price_below_pbema = close < pb_bot
        price_above_pbema = close > pb_top

        debug_info.update({
            "price_below_pbema": price_below_pbema,
            "price_above_pbema": price_above_pbema,
        })

        # ================= KELTNER PENETRATION (TRAP) DETECTION =================
        penetration_lookback = min(3, len(df) - abs_index - 1) if abs_index < len(df) - 1 else 0

        # Long: check if any recent candle broke below lower Keltner
        long_penetration = False
        if penetration_lookback > 0:
            for i in range(1, penetration_lookback + 1):
                if abs_index - i >= 0:
                    past_low = float(df["low"].iloc[abs_index - i])
                    past_lower_band = float(df["keltner_lower"].iloc[abs_index - i])
                    if past_low < past_lower_band:
                        long_penetration = True
                        break

        # Short: check if any recent candle broke above upper Keltner
        short_penetration = False
        if penetration_lookback > 0:
            for i in range(1, penetration_lookback + 1):
                if abs_index - i >= 0:
                    past_high = float(df["high"].iloc[abs_index - i])
                    past_upper_band = float(df["keltner_upper"].iloc[abs_index - i])
                    if past_high > past_upper_band:
                        short_penetration = True
                        break

        debug_info.update({
            "long_penetration": long_penetration,
            "short_penetration": short_penetration,
        })

        # ================= LONG =================
        holding_long = (closes_slice > lower_slice).mean() >= min_hold_frac

        retest_long = (
                (low <= lower_band * (1 + touch_tol))
                and (close > lower_band)
                and (min(open_, close) > lower_band * (1 - body_tol))
        )

        keltner_pb_gap_long = (pb_bot - lower_band) / lower_band if lower_band != 0 else 0.0

        pb_target_long = (
                long_direction_ok and
                (keltner_pb_gap_long >= cloud_keltner_gap_min)
        )

        long_quality_ok = long_rejection_quality or long_penetration

        # SOFT VERSION: Only core filters are mandatory
        is_long = holding_long and retest_long and pb_target_long
        debug_info.update({
            "holding_long": holding_long,
            "retest_long": retest_long,
            "pb_target_long": pb_target_long,
            "long_quality_ok": long_quality_ok,
            "price_below_pbema": price_below_pbema,
        })

        # ================= SHORT =================
        holding_short = (closes_slice < upper_slice).mean() >= min_hold_frac

        retest_short = (
                (high >= upper_band * (1 - touch_tol))
                and (close < upper_band)
                and (max(open_, close) < upper_band * (1 + body_tol))
        )

        keltner_pb_gap_short = (upper_band - pb_top) / upper_band if upper_band != 0 else 0.0

        pb_target_short = (
                short_direction_ok and
                (keltner_pb_gap_short >= cloud_keltner_gap_min)
        )

        short_quality_ok = short_rejection_quality or short_penetration

        # SOFT VERSION: Only core filters are mandatory
        is_short = holding_short and retest_short and pb_target_short
        debug_info.update({
            "holding_short": holding_short,
            "retest_short": retest_short,
            "pb_target_short": pb_target_short,
            "short_quality_ok": short_quality_ok,
            "price_above_pbema": price_above_pbema,
        })

        # --- RSI filters (symmetric for LONG and SHORT) ---
        rsi_val = float(curr["rsi"])
        debug_info["rsi_value"] = rsi_val

        # LONG: RSI should not be too high (overbought territory)
        long_rsi_limit = rsi_limit + 10.0
        long_rsi_ok = rsi_val <= long_rsi_limit
        debug_info["long_rsi_ok"] = long_rsi_ok
        debug_info["long_rsi_limit"] = long_rsi_limit
        if is_long and not long_rsi_ok:
            is_long = False

        # SHORT: RSI should not be too low (oversold territory)
        short_rsi_limit = 100.0 - long_rsi_limit
        short_rsi_ok = rsi_val >= short_rsi_limit
        debug_info["short_rsi_ok"] = short_rsi_ok
        debug_info["short_rsi_limit"] = short_rsi_limit
        if is_short and not short_rsi_ok:
            is_short = False

        # --- AlphaTrend (optional) ---
        if use_alphatrend and "alphatrend" in df.columns:
            at_val = float(curr["alphatrend"])
            if is_long and close < at_val:
                is_long = False
            if is_short and close > at_val:
                is_short = False

        # ---------- LONG ----------
        debug_info["long_candidate"] = is_long
        if is_long:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_low = float(df["low"].iloc[start:abs_index].min())
            if swing_low <= 0:
                return _ret(None, None, None, None, "Invalid Swing Low")

            sl_candidate = swing_low * 0.997
            band_sl = lower_band * 0.998
            sl = min(sl_candidate, band_sl)

            entry = close
            tp = pb_bot

            if tp <= entry:
                return _ret(None, None, None, None, "TP Below Entry")
            if sl >= entry:
                sl = min(swing_low * 0.995, entry * 0.997)

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
                "long_rr_ok": rr >= min_rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return _ret("LONG", entry, tp, sl, reason)

        # ---------- SHORT ----------
        debug_info["short_candidate"] = is_short
        if is_short:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_high = float(df["high"].iloc[start:abs_index].max())
            if swing_high <= 0:
                return _ret(None, None, None, None, "Invalid Swing High")

            sl_candidate = swing_high * 1.003
            band_sl = upper_band * 1.002
            sl = max(sl_candidate, band_sl)

            entry = close
            tp = pb_top

            if tp >= entry:
                return _ret(None, None, None, None, "TP Above Entry")
            if sl <= entry:
                sl = max(swing_high * 1.005, entry * 1.003)

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
                "short_rr_ok": rr >= min_rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(Base,R:{rr:.2f})"
            return _ret("SHORT", entry, tp, sl, reason)

        return _ret(None, None, None, None, "No Signal")

    @staticmethod
    def check_pbema_reaction_signal(
            df: pd.DataFrame,
            index: int = -2,
            min_rr: float = 2.0,
            rsi_limit: float = 60.0,
            slope_thresh: float = 0.5,
            use_alphatrend: bool = False,
            pbema_approach_tolerance: float = 0.003,
            pbema_frontrun_margin: float = 0.002,
            tp_min_dist_ratio: float = 0.0015,
            tp_max_dist_ratio: float = 0.04,
            adx_min: float = 12.0,
            return_debug: bool = False,
    ) -> Tuple:
        """
        PBEMA Reaction Strategy - Trade when price approaches/touches PBEMA cloud.

        Concept:
        - PBEMA cloud acts as strong support/resistance
        - Expect reaction when price approaches PBEMA
        - SHORT: Price approaches PBEMA from below -> sell pressure expected
        - LONG: Price approaches PBEMA from above -> buy pressure expected

        Parameters:
        - pbema_approach_tolerance: How close to PBEMA to generate signal (e.g., 0.003 = 0.3%)
        - pbema_frontrun_margin: Frontrun margin (SL = PBEMA + this margin)
        """

        debug_info = {
            "adx_ok": None,
            "price_near_pbema_top": None,
            "price_near_pbema_bot": None,
            "approaching_from_below": None,
            "approaching_from_above": None,
            "short_rsi_ok": None,
            "long_rsi_ok": None,
            "tp_dist_ratio": None,
            "rr_value": None,
            "min_wick_ratio_pbema": None,
            "upper_wick_ratio": None,
            "lower_wick_ratio": None,
            "wick_quality_short": None,
            "wick_quality_long": None,
        }

        def _ret(s_type, entry, tp, sl, reason):
            if return_debug:
                return s_type, entry, tp, sl, reason, debug_info
            return s_type, entry, tp, sl, reason

        if df is None or df.empty:
            return _ret(None, None, None, None, "No Data")

        required_cols = [
            "open", "high", "low", "close",
            "rsi", "adx",
            "pb_ema_top_150", "pb_ema_bot_150",
            "keltner_upper", "keltner_lower",
        ]
        for col in required_cols:
            if col not in df.columns:
                return _ret(None, None, None, None, f"Missing {col}")

        try:
            curr = df.iloc[index]
        except Exception:
            return _ret(None, None, None, None, "Index Error")

        abs_index = index if index >= 0 else (len(df) + index)
        if abs_index < 30:  # Need enough history for swing detection
            return _ret(None, None, None, None, "Not Enough Data")

        # Extract current values
        open_ = float(curr["open"])
        high = float(curr["high"])
        low = float(curr["low"])
        close = float(curr["close"])
        pb_top = float(curr["pb_ema_top_150"])
        pb_bot = float(curr["pb_ema_bot_150"])
        lower_band = float(curr["keltner_lower"])
        upper_band = float(curr["keltner_upper"])
        adx_val = float(curr["adx"])
        rsi_val = float(curr["rsi"])

        # Check for NaN values
        if any(pd.isna([open_, high, low, close, pb_top, pb_bot, lower_band, upper_band, adx_val, rsi_val])):
            return _ret(None, None, None, None, "NaN Values")

        # Wick-quality filter calculations
        min_wick_ratio_pbema = 0.12
        candle_range = high - low
        if candle_range <= 0:
            upper_wick_ratio = 0.0
            lower_wick_ratio = 0.0
        else:
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            upper_wick_ratio = upper_wick / candle_range
            lower_wick_ratio = lower_wick / candle_range

        debug_info["min_wick_ratio_pbema"] = min_wick_ratio_pbema
        debug_info["upper_wick_ratio"] = upper_wick_ratio
        debug_info["lower_wick_ratio"] = lower_wick_ratio

        # ADX filter
        adx_ok = adx_val >= adx_min
        debug_info["adx_ok"] = adx_ok
        if not adx_ok:
            return _ret(None, None, None, None, f"ADX Too Low ({adx_val:.1f})")

        # Calculate distances to PBEMA cloud
        dist_to_pb_top = abs(high - pb_top) / pb_top if pb_top > 0 else 1.0
        dist_to_pb_bot = abs(low - pb_bot) / pb_bot if pb_bot > 0 else 1.0

        # Check if price is approaching PBEMA from below (for SHORT)
        price_below_pbema = close < pb_bot
        price_near_pbema_top = (high >= pb_top * (1 - pbema_approach_tolerance)) and (high <= pb_top * (1 + pbema_frontrun_margin))
        approaching_from_below = (
            not price_below_pbema and
            (dist_to_pb_top <= pbema_approach_tolerance or high >= pb_top)
        )

        # Check if price is approaching PBEMA from above (for LONG)
        price_above_pbema = close > pb_top
        price_near_pbema_bot = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (low >= pb_bot * (1 - pbema_frontrun_margin))
        approaching_from_above = (
            not price_above_pbema and
            (dist_to_pb_bot <= pbema_approach_tolerance or low <= pb_bot)
        )

        debug_info.update({
            "price_near_pbema_top": price_near_pbema_top,
            "price_near_pbema_bot": price_near_pbema_bot,
            "approaching_from_below": approaching_from_below,
            "approaching_from_above": approaching_from_above,
        })

        # ================= SHORT (PBEMA Resistance) =================
        is_short = price_near_pbema_top and close < pb_top

        # Rejection candle check
        if is_short:
            rejection_wick_short = (high >= pb_top * (1 - pbema_approach_tolerance)) and (close < pb_top)
            candle_body_below = max(open_, close) < pb_top
            wick_quality_short = (upper_wick_ratio >= min_wick_ratio_pbema)
            debug_info["wick_quality_short"] = wick_quality_short
            is_short = rejection_wick_short and candle_body_below and wick_quality_short

        # RSI filter for SHORT
        short_rsi_limit = 100.0 - (rsi_limit + 10.0)
        short_rsi_ok = rsi_val >= short_rsi_limit
        debug_info["short_rsi_ok"] = short_rsi_ok
        if is_short and not short_rsi_ok:
            is_short = False

        # Slope filter
        if is_short and "slope_top_150" in curr.index:
            slope_val = float(curr["slope_top_150"])
            if slope_val > slope_thresh:
                is_short = False

        if is_short:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_low = float(df["low"].iloc[start:abs_index].min())

            tp = swing_low * 0.998
            entry = close
            sl = pb_top * (1 + pbema_frontrun_margin + 0.002)

            if tp >= entry:
                return _ret(None, None, None, None, "TP Above Entry (SHORT)")
            if sl <= entry:
                sl = pb_top * (1 + pbema_frontrun_margin + 0.005)

            risk = sl - entry
            reward = entry - tp
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (SHORT)")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
            return _ret("SHORT", entry, tp, sl, reason)

        # ================= LONG (PBEMA Support) =================
        is_long = price_near_pbema_bot and close > pb_bot

        # Rejection candle check
        if is_long:
            rejection_wick_long = (low <= pb_bot * (1 + pbema_approach_tolerance)) and (close > pb_bot)
            candle_body_above = min(open_, close) > pb_bot
            wick_quality_long = (lower_wick_ratio >= min_wick_ratio_pbema)
            debug_info["wick_quality_long"] = wick_quality_long
            is_long = rejection_wick_long and candle_body_above and wick_quality_long

        # RSI filter for LONG
        long_rsi_limit = rsi_limit + 10.0
        long_rsi_ok = rsi_val <= long_rsi_limit
        debug_info["long_rsi_ok"] = long_rsi_ok
        if is_long and not long_rsi_ok:
            is_long = False

        # Slope filter
        if is_long and "slope_bot_150" in curr.index:
            slope_val = float(curr["slope_bot_150"])
            if slope_val < -slope_thresh:
                is_long = False

        if is_long:
            swing_n = 20
            start = max(0, abs_index - swing_n)
            swing_high = float(df["high"].iloc[start:abs_index].max())

            tp = swing_high * 1.002
            entry = close
            sl = pb_bot * (1 - pbema_frontrun_margin - 0.002)

            if tp <= entry:
                return _ret(None, None, None, None, "TP Below Entry (LONG)")
            if sl >= entry:
                sl = pb_bot * (1 - pbema_frontrun_margin - 0.005)

            risk = entry - sl
            reward = tp - entry
            if risk <= 0 or reward <= 0:
                return _ret(None, None, None, None, "Invalid RR (LONG)")

            rr = reward / risk
            tp_dist_ratio = reward / entry

            debug_info.update({
                "tp_dist_ratio": tp_dist_ratio,
                "rr_value": rr,
            })

            if tp_dist_ratio < tp_min_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Close ({tp_dist_ratio:.4f})")
            if tp_dist_ratio > tp_max_dist_ratio:
                return _ret(None, None, None, None, f"TP Too Far ({tp_dist_ratio:.4f})")
            if rr < min_rr:
                return _ret(None, None, None, None, f"RR Too Low ({rr:.2f})")

            reason = f"ACCEPTED(PBEMA_Reaction,R:{rr:.2f})"
            return _ret("LONG", entry, tp, sl, reason)

        return _ret(None, None, None, None, "No Signal")

    @staticmethod
    def check_signal(
            df: pd.DataFrame,
            config: dict,
            index: int = -2,
            return_debug: bool = False,
    ) -> Tuple:
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
        strategy_mode = config.get("strategy_mode", DEFAULT_STRATEGY_CONFIG.get("strategy_mode", "keltner_bounce"))

        if strategy_mode == "pbema_reaction":
            return TradingEngine.check_pbema_reaction_signal(
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
            return TradingEngine.check_signal_diagnostic(
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

    @staticmethod
    def debug_base_short(df, index):
        """
        Debug function to inspect Base SHORT conditions for a specific candle.
        index: df.iloc[index] style (e.g., -1 for last candle)
        """
        curr = df.iloc[index]
        abs_index = index if index >= 0 else (len(df) + index)

        hold_n = 4
        min_hold_frac = 0.50
        touch_tol = 0.0012
        slope_thresh = 0.5

        # ADX and slope
        adx_ok = curr["adx"] >= 15
        slope_ok_short = curr["slope_top"] <= slope_thresh

        if abs_index >= hold_n + 1:
            hold_slice = slice(abs_index - hold_n, abs_index)
            holding_short = (
                    (df["close"].iloc[hold_slice] < df["keltner_upper"].iloc[hold_slice])
                    .mean() >= min_hold_frac
            )
        else:
            holding_short = False

        retest_short = (
                               curr["high"] >= curr["keltner_upper"] * (1 - touch_tol)
                       ) and (
                               curr["close"] < curr["keltner_upper"]
                       )

        pb_target_short = curr["pb_ema_top"] < curr["close"]

        # RSI limit (Base short)
        rsi_limit = 60
        rsi_thresh = (100 - rsi_limit) - 10
        rsi_ok = curr["rsi"] >= rsi_thresh

        return {
            "time": str(getattr(curr, "name", "")),
            "adx": float(curr["adx"]),
            "rsi": float(curr["rsi"]),
            "adx_ok": adx_ok,
            "slope_ok_short": slope_ok_short,
            "holding_short": bool(holding_short),
            "retest_short": bool(retest_short),
            "pb_target_short": bool(pb_target_short),
            "rsi_ok": rsi_ok,
        }
