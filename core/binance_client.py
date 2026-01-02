"""
Binance API client module.

Provides reliable API access with:
- Retry logic with exponential backoff
- Rate limit handling
- Network cooldown for DNS failures
- Parallel data fetching
"""

import time
import logging
import itertools
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Binance Futures API client with robust error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limit (429) and server error (5xx) handling
    - DNS failure detection with cooldown
    - Parallel data fetching for multiple symbols/timeframes
    """

    # Network cooldown timestamp (class-level to persist across instances)
    _network_cooldown_until = 0

    BASE_URL = "https://fapi.binance.com/fapi/v1"

    def __init__(self, max_retries: int = 3, timeout: int = 10):
        """
        Initialize the Binance client.

        Args:
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout

    def _get_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = None,
        timeout: int = None
    ) -> Optional[requests.Response]:
        """
        Make HTTP GET request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            max_retries: Override default max_retries
            timeout: Override default timeout

        Returns:
            Response object or None if all retries failed
        """
        max_retries = max_retries or self.max_retries
        timeout = timeout or self.timeout

        # Check network cooldown
        now = time.time()
        if now < BinanceClient._network_cooldown_until:
            cooldown_left = int(BinanceClient._network_cooldown_until - now)
            print(f"Network error: No network access. Retrying in {cooldown_left}s.")
            return None

        delay = 1
        for attempt in range(max_retries):
            try:
                res = requests.get(url, params=params, timeout=timeout)

                # Handle rate limiting and server errors
                if res.status_code == 429 or res.status_code >= 500:
                    print(f"API Error {res.status_code} (Attempt {attempt + 1}/{max_retries}). Waiting...")
                    time.sleep(delay)
                    delay *= 2
                    continue

                # Success or client error (4xx except 429)
                BinanceClient._network_cooldown_until = 0
                return res

            except requests.exceptions.RequestException as e:
                is_dns_error = (
                    isinstance(e, requests.exceptions.ConnectionError) and
                    "NameResolutionError" in str(e)
                )
                print(f"Connection error (Attempt {attempt + 1}/{max_retries}): {e}")

                # DNS failure: cooldown for 5 minutes
                if is_dns_error:
                    BinanceClient._network_cooldown_until = time.time() + 300
                    break

                time.sleep(delay)
                delay *= 2

        return None

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get kline (candlestick) data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (e.g., "5m", "1h")
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            url = f"{self.BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}

            res = self._get_with_retry(url, params)
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
            print(f"Data fetch error ({symbol}): {e}")
            return pd.DataFrame()

    def get_klines_paginated(
        self,
        symbol: str,
        interval: str,
        total_candles: int = 5000
    ) -> pd.DataFrame:
        """
        Get historical kline data with pagination.

        Fetches data in batches of 1000 candles going back in time.

        Args:
            symbol: Trading pair
            interval: Timeframe
            total_candles: Total number of candles to fetch

        Returns:
            DataFrame with historical kline data
        """
        all_data = []
        end_time = int(time.time() * 1000)
        limit_per_req = 1000
        loops = int(np.ceil(total_candles / limit_per_req))

        for _ in range(loops):
            try:
                url = f"{self.BASE_URL}/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit_per_req,
                    'endTime': end_time
                }

                res = self._get_with_retry(url, params)
                if res is None:
                    break

                data = res.json()
                if not data or not isinstance(data, list):
                    break

                all_data = data + all_data  # Prepend older data
                end_time = data[0][0] - 1
                time.sleep(0.1)  # Rate limit courtesy

            except (requests.RequestException, ValueError, KeyError) as e:
                logger.debug(f"Klines fetch stopped: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.tail(total_candles).reset_index(drop=True)

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current price or None if failed
        """
        try:
            url = f"{self.BASE_URL}/ticker/price"
            res = self._get_with_retry(url, {"symbol": symbol}, max_retries=2)

            if res is None:
                return None

            data = res.json()
            if isinstance(data, dict) and "price" in data:
                return float(data["price"])

        except Exception as e:
            print(f"[PRICE] Failed to get {symbol} price: {e}")

        return None

    def get_ticker_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict of {symbol: price}
        """
        prices = {}
        for sym in symbols:
            price = self.get_ticker_price(sym)
            if price is not None:
                prices[sym] = price
        return prices

    def get_klines_parallel(
        self,
        symbols: List[str],
        timeframes: List[str],
        limit: int = 500,
        max_workers: int = 5
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Fetch kline data for multiple symbols and timeframes in parallel.

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            limit: Candles per request
            max_workers: Max parallel threads (keep low for rate limits)

        Returns:
            Dict of {(symbol, timeframe): DataFrame}
        """
        tasks = list(itertools.product(symbols, timeframes))
        results = {}

        def fetch_one(args):
            symbol, tf = args
            try:
                df = self.get_klines(symbol, tf, limit)
                return (symbol, tf, df)
            except (requests.RequestException, ValueError, KeyError) as e:
                logger.debug(f"Fetch {symbol}-{tf} failed: {e}")
                return (symbol, tf, pd.DataFrame())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    sym, tf, df = future.result()
                    results[(sym, tf)] = df
                except (TimeoutError, RuntimeError) as e:
                    logger.warning(f"Parallel fetch error: {e}")

        return results


# Global client instance (lazy initialization)
_client: Optional[BinanceClient] = None


def get_client() -> BinanceClient:
    """Get the global Binance client instance."""
    global _client
    if _client is None:
        _client = BinanceClient()
    return _client
