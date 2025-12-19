# core/trading_engine.py
# Trading Engine - Core trading logic, data fetching, and signal generation
#
# This module provides:
# - TradingEngine class with API data fetching (with retry logic)
# - Signal detection for multiple strategies (ssl_flow, keltner_bounce)
# - Indicator calculation wrappers
#
# Signal detection is delegated to the strategies/ module for modularity.
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

# Import strategy signal detection from strategies module
from strategies import (
    check_keltner_bounce_signal,
    check_ssl_flow_signal,
    check_signal as strategies_check_signal,
)


class TradingEngine:
    """Core trading engine for data fetching, indicator calculation, and signal detection.

    This class provides:
    - Binance API data fetching with retry logic
    - Parallel data fetching for multiple symbols/timeframes
    - Technical indicator calculations
    - Signal detection for ssl_flow and keltner_bounce strategies

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
        Keltner Bounce signal detection for LONG / SHORT.

        Delegates to strategies.keltner_bounce.check_keltner_bounce_signal.

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
        return check_keltner_bounce_signal(
            df=df,
            index=index,
            min_rr=min_rr,
            rsi_limit=rsi_limit,
            slope_thresh=slope_thresh,
            use_alphatrend=use_alphatrend,
            hold_n=hold_n,
            min_hold_frac=min_hold_frac,
            pb_touch_tolerance=pb_touch_tolerance,
            body_tolerance=body_tolerance,
            cloud_keltner_gap_min=cloud_keltner_gap_min,
            tp_min_dist_ratio=tp_min_dist_ratio,
            tp_max_dist_ratio=tp_max_dist_ratio,
            adx_min=adx_min,
            return_debug=return_debug,
        )

    @staticmethod
    def check_ssl_flow_signal(
            df: pd.DataFrame,
            index: int = -2,
            min_rr: float = 2.0,
            rsi_limit: float = 70.0,
            use_alphatrend: bool = True,
            ssl_touch_tolerance: float = 0.002,
            ssl_body_tolerance: float = 0.003,
            min_pbema_distance: float = 0.004,
            tp_min_dist_ratio: float = 0.0015,
            tp_max_dist_ratio: float = 0.05,
            adx_min: float = 15.0,
            lookback_candles: int = 5,
            return_debug: bool = False,
    ) -> Tuple:
        """
        SSL Flow Strategy - Trend following with SSL HYBRID baseline.

        Delegates to strategies.ssl_flow.check_ssl_flow_signal.

        Concept:
        - SSL HYBRID (HMA60) determines flow direction (price above = bullish, below = bearish)
        - AlphaTrend confirms buyer/seller dominance (filters fake SSL signals)
        - Entry when price retests SSL baseline as support/resistance
        - TP target at PBEMA cloud (EMA200)

        Entry Logic:
        - LONG: Price above SSL baseline + AlphaTrend buyers dominant + retest
        - SHORT: Price below SSL baseline + AlphaTrend sellers dominant + retest
        """
        return check_ssl_flow_signal(
            df=df,
            index=index,
            min_rr=min_rr,
            rsi_limit=rsi_limit,
            use_alphatrend=use_alphatrend,
            ssl_touch_tolerance=ssl_touch_tolerance,
            ssl_body_tolerance=ssl_body_tolerance,
            min_pbema_distance=min_pbema_distance,
            tp_min_dist_ratio=tp_min_dist_ratio,
            tp_max_dist_ratio=tp_max_dist_ratio,
            adx_min=adx_min,
            lookback_candles=lookback_candles,
            return_debug=return_debug,
        )

    @staticmethod
    def check_signal(
            df: pd.DataFrame,
            config: dict,
            index: int = -2,
            return_debug: bool = False,
    ) -> Tuple:
        """
        Wrapper function - routes to appropriate strategy based on strategy_mode.

        Delegates to strategies.router.check_signal.

        strategy_mode values:
        - "ssl_flow" (default): SSL HYBRID trend following strategy
        - "keltner_bounce": Keltner band bounce / mean reversion strategy

        Args:
            df: OHLCV + indicator dataframe
            config: Strategy configuration (rr, rsi, slope, strategy_mode, etc.)
            index: Candle index for signal check
            return_debug: Return debug info

        Returns:
            (s_type, entry, tp, sl, reason) or with debug info
        """
        return strategies_check_signal(
            df=df,
            config=config,
            index=index,
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
