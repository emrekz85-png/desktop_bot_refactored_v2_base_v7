# Core package for trading bot
# Provides modular components: config, utils, trade_manager, trading_engine, etc.
#
# Module Structure:
# - config.py: Constants, trading configuration, blacklist management
# - utils.py: Helper functions (time conversion, funding calculation, etc.)
# - config_loader.py: Load/save optimized strategy configurations
# - trade_manager.py: BaseTradeManager and SimTradeManager classes
# - telegram.py: Telegram notification handling (secure, efficient)
# - binance_client.py: Binance API client with retry logic
# - indicators.py: Technical indicator calculations
# - logging_config.py: Centralized logging with file rotation
#
# This modular structure eliminates code duplication and improves maintainability.

from .config import (
    # Environment detection
    IS_COLAB, IS_HEADLESS, IS_NOTEBOOK, HAS_TQDM,
    # Symbols and timeframes
    SYMBOLS, TIMEFRAMES, LOWER_TIMEFRAMES, HTF_TIMEFRAMES,
    # Paths
    DATA_DIR, CSV_FILE, CONFIG_FILE, BEST_CONFIGS_FILE,
    DYNAMIC_BLACKLIST_FILE, BACKTEST_META_FILE, POT_LOG_FILE,
    # Candle limits
    BACKTEST_CANDLE_LIMITS, DAILY_REPORT_CANDLE_LIMITS,
    # Trading config
    TRADING_CONFIG, CANDLES_PER_DAY, MINUTES_PER_CANDLE,
    # Functions
    days_to_candles, days_to_candles_map,
    # Blacklist
    is_stream_blacklisted, load_dynamic_blacklist, update_dynamic_blacklist,
    save_dynamic_blacklist, DYNAMIC_BLACKLIST_CONFIG, DYNAMIC_BLACKLIST_CACHE,
    POST_PORTFOLIO_BLACKLIST,
    # Thresholds
    MIN_EXPECTANCY_R_MULTIPLE, MIN_SCORE_THRESHOLD, CONFIDENCE_RISK_MULTIPLIER,
)

from .utils import (
    normalize_datetime, tf_to_timedelta, tf_to_minutes,
    calculate_funding_cost, format_time_utc, format_time_local,
    append_trade_event, calculate_r_multiple, calculate_expected_r,
    apply_1m_profit_lock, apply_partial_stop_protection,
)

from .config import (
    # Strategy configs (single source of truth)
    DEFAULT_STRATEGY_CONFIG, SYMBOL_PARAMS,
)

from .config_loader import (
    load_optimized_config, save_best_configs, invalidate_config_cache,
)

from .trade_manager import (
    BaseTradeManager, SimTradeManager,
)

from .telegram import (
    TelegramNotifier, get_notifier, send_telegram,
    save_telegram_config, load_telegram_config,
)

from .binance_client import (
    BinanceClient, get_client,
)

from .indicators import (
    calculate_indicators, calculate_alphatrend,
    get_indicator_value, get_candle_data,
    calculate_rr_ratio, check_wick_rejection,
)

from .logging_config import (
    get_logger, set_log_level,
    log_trade_event, log_signal, log_api_call,
    print_to_log, info, debug, warning, error,
)

from .trading_engine import TradingEngine

__all__ = [
    # Environment
    'IS_COLAB', 'IS_HEADLESS', 'IS_NOTEBOOK', 'HAS_TQDM',
    # Symbols and timeframes
    'SYMBOLS', 'TIMEFRAMES', 'LOWER_TIMEFRAMES', 'HTF_TIMEFRAMES',
    # Paths
    'DATA_DIR', 'CSV_FILE', 'CONFIG_FILE', 'BEST_CONFIGS_FILE',
    'DYNAMIC_BLACKLIST_FILE', 'BACKTEST_META_FILE', 'POT_LOG_FILE',
    # Candle limits
    'BACKTEST_CANDLE_LIMITS', 'DAILY_REPORT_CANDLE_LIMITS',
    # Trading config
    'TRADING_CONFIG', 'CANDLES_PER_DAY', 'MINUTES_PER_CANDLE',
    'days_to_candles', 'days_to_candles_map',
    # Blacklist
    'is_stream_blacklisted', 'load_dynamic_blacklist', 'update_dynamic_blacklist',
    'save_dynamic_blacklist', 'DYNAMIC_BLACKLIST_CONFIG', 'DYNAMIC_BLACKLIST_CACHE',
    'POST_PORTFOLIO_BLACKLIST',
    # Thresholds
    'MIN_EXPECTANCY_R_MULTIPLE', 'MIN_SCORE_THRESHOLD', 'CONFIDENCE_RISK_MULTIPLIER',
    # Utils
    'normalize_datetime', 'tf_to_timedelta', 'tf_to_minutes',
    'calculate_funding_cost', 'format_time_utc', 'format_time_local',
    'append_trade_event', 'calculate_r_multiple', 'calculate_expected_r',
    'apply_1m_profit_lock', 'apply_partial_stop_protection',
    # Config loader
    'load_optimized_config', 'save_best_configs', 'invalidate_config_cache',
    'SYMBOL_PARAMS', 'DEFAULT_STRATEGY_CONFIG',
    # Trade manager
    'BaseTradeManager', 'SimTradeManager',
    # Telegram
    'TelegramNotifier', 'get_notifier', 'send_telegram',
    'save_telegram_config', 'load_telegram_config',
    # Binance client
    'BinanceClient', 'get_client',
    # Indicators
    'calculate_indicators', 'calculate_alphatrend',
    'get_indicator_value', 'get_candle_data',
    'calculate_rr_ratio', 'check_wick_rejection',
    # Logging
    'get_logger', 'set_log_level',
    'log_trade_event', 'log_signal', 'log_api_call',
    'print_to_log', 'info', 'debug', 'warning', 'error',
    # Trading Engine
    'TradingEngine',
]
