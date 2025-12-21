"""
Performance optimization module for rolling walk-forward tests.

Provides:
- Master data cache with disk persistence (parquet/feather)
- Efficient NumPy-based date slicing (no DataFrame copies)
- Timestamp and timedelta caching
- Parallel data fetching with IO/CPU separation
"""

import os
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "perf_cache")
USE_PARQUET = True  # Use parquet for compressed cache (smaller, slower)
USE_FEATHER = False  # Use feather for fast cache (larger, faster)

# Thread pool settings
MAX_IO_WORKERS = 10  # For network/disk I/O
MAX_CPU_WORKERS = None  # None = os.cpu_count()

# Timedelta cache (computed once)
_TIMEDELTA_CACHE: Dict[str, pd.Timedelta] = {}
_TIMEDELTA_NS_CACHE: Dict[str, int] = {}  # Nanoseconds for int64 comparison

# Master data cache (in-memory)
_MASTER_CACHE: Dict[Tuple[str, str, str, str], Any] = {}  # (sym, tf, start, end) -> data
_CACHE_LOCK = threading.Lock()


# ============================================================================
# TIMEDELTA CACHING
# ============================================================================

def get_timedelta(tf: str) -> pd.Timedelta:
    """Get cached timedelta for a timeframe string."""
    if tf not in _TIMEDELTA_CACHE:
        if tf.endswith("m"):
            _TIMEDELTA_CACHE[tf] = pd.Timedelta(minutes=int(tf[:-1]))
        elif tf.endswith("h"):
            _TIMEDELTA_CACHE[tf] = pd.Timedelta(hours=int(tf[:-1]))
        elif tf.endswith("d"):
            _TIMEDELTA_CACHE[tf] = pd.Timedelta(days=int(tf[:-1]))
        elif tf.endswith("w"):
            _TIMEDELTA_CACHE[tf] = pd.Timedelta(weeks=int(tf[:-1]))
        else:
            # Default fallback
            _TIMEDELTA_CACHE[tf] = pd.Timedelta(minutes=15)
    return _TIMEDELTA_CACHE[tf]


def get_timedelta_ns(tf: str) -> int:
    """Get cached timedelta in nanoseconds for int64 timestamp comparison."""
    if tf not in _TIMEDELTA_NS_CACHE:
        td = get_timedelta(tf)
        _TIMEDELTA_NS_CACHE[tf] = int(td.value)  # pd.Timedelta.value is nanoseconds
    return _TIMEDELTA_NS_CACHE[tf]


# ============================================================================
# NUMPY-BASED DATE SLICING
# ============================================================================

class StreamArrays:
    """
    Pre-extracted NumPy arrays for a stream with efficient slicing.

    Avoids DataFrame copies by using NumPy searchsorted for index lookup.
    """

    __slots__ = (
        'sym', 'tf', 'timestamps_ns', 'timestamps_raw',
        'highs', 'lows', 'closes', 'opens',
        'pb_tops', 'pb_bots', 'df_ref', 'is_tz_aware'
    )

    def __init__(self, sym: str, tf: str, df: pd.DataFrame,
                 pb_top_col: str = "pb_ema_top", pb_bot_col: str = "pb_ema_bot"):
        """
        Initialize stream arrays from DataFrame.

        Args:
            sym: Symbol
            tf: Timeframe
            df: DataFrame with OHLCV and indicators
            pb_top_col: Column name for PBEMA top
            pb_bot_col: Column name for PBEMA bottom
        """
        self.sym = sym
        self.tf = tf
        self.df_ref = df  # Keep reference for signal detection

        # Convert timestamps to int64 nanoseconds for fast comparison
        ts_series = pd.to_datetime(df["timestamp"])
        self.is_tz_aware = ts_series.dt.tz is not None
        self.timestamps_raw = ts_series.values  # Keep raw for display

        # Convert to int64 nanoseconds (works for both tz-aware and naive)
        if self.is_tz_aware:
            self.timestamps_ns = ts_series.view('int64')
        else:
            self.timestamps_ns = ts_series.astype('int64').values

        # Pre-extract price arrays
        self.highs = df["high"].values.astype(np.float64)
        self.lows = df["low"].values.astype(np.float64)
        self.closes = df["close"].values.astype(np.float64)
        self.opens = df["open"].values.astype(np.float64)

        # PBEMA columns
        if pb_top_col in df.columns:
            self.pb_tops = df[pb_top_col].values.astype(np.float64)
        else:
            self.pb_tops = self.closes.copy()

        if pb_bot_col in df.columns:
            self.pb_bots = df[pb_bot_col].values.astype(np.float64)
        else:
            self.pb_bots = self.closes.copy()

    def find_start_index(self, start_ts_ns: int, min_warmup: int = 250) -> int:
        """
        Find the first index >= start_ts using binary search.

        Args:
            start_ts_ns: Start timestamp in nanoseconds
            min_warmup: Minimum index to ensure indicator warmup

        Returns:
            Index to start from (with warmup guarantee)
        """
        idx = np.searchsorted(self.timestamps_ns, start_ts_ns, side='left')
        return max(idx, min_warmup)

    def find_end_index(self, end_ts_ns: int) -> int:
        """
        Find the last index < end_ts using binary search.

        Args:
            end_ts_ns: End timestamp in nanoseconds

        Returns:
            Index of last candle before end
        """
        return np.searchsorted(self.timestamps_ns, end_ts_ns, side='left')

    def get_slice_indices(self, start_ts_ns: int, end_ts_ns: int,
                          min_warmup: int = 250) -> Tuple[int, int]:
        """
        Get slice indices for a date range.

        Args:
            start_ts_ns: Start timestamp in nanoseconds
            end_ts_ns: End timestamp in nanoseconds
            min_warmup: Minimum start index for warmup

        Returns:
            (start_idx, end_idx) tuple
        """
        start_idx = self.find_start_index(start_ts_ns, min_warmup)
        end_idx = self.find_end_index(end_ts_ns)
        return start_idx, end_idx

    def __len__(self) -> int:
        return len(self.timestamps_ns)


def datetime_to_ns(dt: datetime, is_tz_aware: bool = False) -> int:
    """Convert datetime to nanoseconds since epoch."""
    if is_tz_aware:
        ts = pd.Timestamp(dt).tz_localize('UTC')
    else:
        ts = pd.Timestamp(dt)
    return int(ts.value)


# ============================================================================
# MASTER DATA CACHE
# ============================================================================

def get_cache_key(sym: str, tf: str, start_date: str, end_date: str) -> str:
    """Generate a cache key for disk storage."""
    key_str = f"{sym}_{tf}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def get_cache_path(cache_key: str) -> str:
    """Get the file path for a cache entry."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if USE_PARQUET:
        return os.path.join(CACHE_DIR, f"{cache_key}.parquet")
    elif USE_FEATHER:
        return os.path.join(CACHE_DIR, f"{cache_key}.feather")
    else:
        return os.path.join(CACHE_DIR, f"{cache_key}.pkl")


def load_from_disk_cache(sym: str, tf: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Load data from disk cache if available."""
    cache_key = get_cache_key(sym, tf, start_date, end_date)
    cache_path = get_cache_path(cache_key)

    if not os.path.exists(cache_path):
        return None

    try:
        # Check if cache is stale (older than 1 hour for recent data)
        cache_mtime = os.path.getmtime(cache_path)
        cache_age_hours = (datetime.now().timestamp() - cache_mtime) / 3600

        # If end_date is recent (within 7 days), use shorter cache expiry
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_ago = (datetime.now() - end_dt).days

        if days_ago < 7 and cache_age_hours > 1:
            # Stale cache for recent data
            return None
        elif cache_age_hours > 24:
            # Stale cache for older data
            return None

        if USE_PARQUET:
            return pd.read_parquet(cache_path)
        elif USE_FEATHER:
            return pd.read_feather(cache_path)
        else:
            return pd.read_pickle(cache_path)
    except Exception:
        return None


def save_to_disk_cache(df: pd.DataFrame, sym: str, tf: str, start_date: str, end_date: str):
    """Save data to disk cache."""
    cache_key = get_cache_key(sym, tf, start_date, end_date)
    cache_path = get_cache_path(cache_key)

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if USE_PARQUET:
            df.to_parquet(cache_path, index=False)
        elif USE_FEATHER:
            df.to_feather(cache_path)
        else:
            df.to_pickle(cache_path)
    except Exception:
        pass  # Silently fail cache writes


def clear_disk_cache(max_age_hours: float = 24):
    """Clear disk cache entries older than max_age_hours."""
    if not os.path.exists(CACHE_DIR):
        return

    now = datetime.now().timestamp()
    for filename in os.listdir(CACHE_DIR):
        filepath = os.path.join(CACHE_DIR, filename)
        try:
            if os.path.isfile(filepath):
                mtime = os.path.getmtime(filepath)
                age_hours = (now - mtime) / 3600
                if age_hours > max_age_hours:
                    os.remove(filepath)
        except Exception:
            pass


# ============================================================================
# MASTER DATA FETCHER
# ============================================================================

class MasterDataCache:
    """
    Master data cache for efficient rolling walk-forward testing.

    Features:
    - Fetches all data once for entire test period
    - Uses NumPy-based slicing for window extraction
    - Disk persistence with parquet/feather
    - Thread-safe for parallel access
    """

    def __init__(self, symbols: List[str], timeframes: List[str],
                 start_date: str, end_date: str,
                 fetch_func=None, indicator_func=None,
                 buffer_days: int = 50):
        """
        Initialize master cache.

        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fetch_func: Function to fetch data (sym, tf, start, end) -> DataFrame
            indicator_func: Function to calculate indicators (df) -> df
            buffer_days: Days of buffer before start_date for warmup
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.fetch_func = fetch_func
        self.indicator_func = indicator_func
        self.buffer_days = buffer_days

        # Calculate fetch range with buffer
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.fetch_start = (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        # Storage
        self._dataframes: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._stream_arrays: Dict[Tuple[str, str], StreamArrays] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def load_all(self, max_workers: int = None, progress_callback=None) -> int:
        """
        Load all data into cache.

        Args:
            max_workers: Max parallel workers (default: MAX_IO_WORKERS)
            progress_callback: Optional callback(loaded, total) for progress

        Returns:
            Number of streams loaded
        """
        if self._loaded:
            return len(self._dataframes)

        if max_workers is None:
            max_workers = MAX_IO_WORKERS

        jobs = [(s, t) for s in self.symbols for t in self.timeframes]
        total = len(jobs)
        loaded = 0

        def fetch_one(sym: str, tf: str) -> Optional[Tuple[str, str, pd.DataFrame]]:
            # Try disk cache first
            df = load_from_disk_cache(sym, tf, self.fetch_start, self.end_date)

            if df is None and self.fetch_func is not None:
                # Fetch from API
                try:
                    df = self.fetch_func(sym, tf, self.fetch_start, self.end_date)
                    if df is not None and not df.empty and len(df) >= 250:
                        # Calculate indicators
                        if self.indicator_func is not None:
                            df = self.indicator_func(df)
                        df = df.reset_index(drop=True)
                        # Save to disk cache
                        save_to_disk_cache(df, sym, tf, self.fetch_start, self.end_date)
                except Exception:
                    return None

            if df is not None and not df.empty and len(df) >= 250:
                return (sym, tf, df)
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, s, t): (s, t) for s, t in jobs}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    sym, tf, df = result
                    with self._lock:
                        self._dataframes[(sym, tf)] = df
                        self._stream_arrays[(sym, tf)] = StreamArrays(sym, tf, df)
                        loaded += 1

                if progress_callback:
                    progress_callback(loaded, total)

        self._loaded = True
        return loaded

    def get_stream_arrays(self, sym: str, tf: str) -> Optional[StreamArrays]:
        """Get pre-extracted arrays for a stream."""
        return self._stream_arrays.get((sym, tf))

    def get_dataframe(self, sym: str, tf: str) -> Optional[pd.DataFrame]:
        """Get raw DataFrame for a stream (needed for signal detection)."""
        return self._dataframes.get((sym, tf))

    def get_all_stream_arrays(self) -> Dict[Tuple[str, str], StreamArrays]:
        """Get all stream arrays."""
        return self._stream_arrays.copy()

    def get_window_indices(self, start_dt: datetime, end_dt: datetime,
                           min_warmup: int = 250) -> Dict[Tuple[str, str], Tuple[int, int]]:
        """
        Get slice indices for all streams in a date window.

        Args:
            start_dt: Window start datetime
            end_dt: Window end datetime
            min_warmup: Minimum index for warmup

        Returns:
            Dict of (sym, tf) -> (start_idx, end_idx)
        """
        # Determine if tz-aware from first stream
        is_tz_aware = False
        if self._stream_arrays:
            first_arr = next(iter(self._stream_arrays.values()))
            is_tz_aware = first_arr.is_tz_aware

        start_ns = datetime_to_ns(start_dt, is_tz_aware)
        end_ns = datetime_to_ns(end_dt, is_tz_aware)

        indices = {}
        for (sym, tf), arr in self._stream_arrays.items():
            start_idx, end_idx = arr.get_slice_indices(start_ns, end_ns, min_warmup)
            if end_idx > start_idx:  # Valid slice
                indices[(sym, tf)] = (start_idx, end_idx)

        return indices

    @property
    def stream_count(self) -> int:
        """Number of loaded streams."""
        return len(self._dataframes)

    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded."""
        return self._loaded


# ============================================================================
# HEAP OPTIMIZATION
# ============================================================================

class FastEventHeap:
    """
    Optimized event heap using int64 timestamps.

    Avoids pd.Timestamp creation in hot loop by using raw nanosecond values.
    """

    __slots__ = ('_heap', '_td_ns_cache')

    def __init__(self):
        import heapq
        self._heap = []
        self._td_ns_cache = {}

    def push(self, time_ns: int, sym: str, tf: str):
        """Push event with nanosecond timestamp."""
        import heapq
        heapq.heappush(self._heap, (time_ns, sym, tf))

    def pop(self) -> Tuple[int, str, str]:
        """Pop next event."""
        import heapq
        return heapq.heappop(self._heap)

    def push_next(self, current_ns: int, sym: str, tf: str, end_ns: int):
        """Push next candle event if before end time."""
        if tf not in self._td_ns_cache:
            self._td_ns_cache[tf] = get_timedelta_ns(tf)

        next_ns = current_ns + self._td_ns_cache[tf]
        if next_ns < end_ns:
            self.push(next_ns, sym, tf)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)


# ============================================================================
# OPEN TRADE LOOKUP OPTIMIZATION
# ============================================================================

class OpenTradeIndex:
    """
    Fast lookup for open trades by (symbol, timeframe).

    Replaces `any(t.get("symbol") == sym and t.get("timeframe") == tf for t in open_trades)`
    with O(1) dict lookup.
    """

    __slots__ = ('_index', '_trades')

    def __init__(self):
        self._index: Dict[Tuple[str, str], int] = {}  # (sym, tf) -> trade_id
        self._trades: Dict[int, dict] = {}  # trade_id -> trade

    def add(self, trade: dict):
        """Add a trade to the index."""
        key = (trade.get("symbol"), trade.get("timeframe"))
        trade_id = id(trade)
        self._index[key] = trade_id
        self._trades[trade_id] = trade

    def remove(self, trade: dict):
        """Remove a trade from the index."""
        key = (trade.get("symbol"), trade.get("timeframe"))
        trade_id = id(trade)
        self._index.pop(key, None)
        self._trades.pop(trade_id, None)

    def has_open(self, sym: str, tf: str) -> bool:
        """Check if there's an open trade for (sym, tf)."""
        return (sym, tf) in self._index

    def get_trade(self, sym: str, tf: str) -> Optional[dict]:
        """Get the open trade for (sym, tf)."""
        trade_id = self._index.get((sym, tf))
        return self._trades.get(trade_id) if trade_id else None

    def clear(self):
        """Clear all trades."""
        self._index.clear()
        self._trades.clear()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_candle_count(timeframe: str, days: int) -> int:
    """Estimate number of candles for a given timeframe and day count."""
    tf_map = {
        "1m": 1440,
        "3m": 480,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "2h": 12,
        "4h": 6,
        "6h": 4,
        "8h": 3,
        "12h": 2,
        "1d": 1,
    }
    candles_per_day = tf_map.get(timeframe, 24)
    return days * candles_per_day
