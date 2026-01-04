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
# - momentum_exit.py: Pattern 1 - Momentum exhaustion exit detection
#
# This modular structure eliminates code duplication and improves maintainability.

from .config import (
    # Version
    VERSION,
    # Environment detection
    IS_COLAB, IS_HEADLESS, IS_NOTEBOOK, HAS_TQDM, HTF_ONLY_MODE,
    # M3 Performance settings
    M3_PERFORMANCE_CONFIG,
    # Symbols and timeframes
    SYMBOLS, TIMEFRAMES, LOWER_TIMEFRAMES, HTF_TIMEFRAMES,
    # Paths
    DATA_DIR, CSV_FILE, CONFIG_FILE, BEST_CONFIGS_FILE,
    DYNAMIC_BLACKLIST_FILE, BACKTEST_META_FILE, POT_LOG_FILE,
    # Candle limits
    BACKTEST_CANDLE_LIMITS, DAILY_REPORT_CANDLE_LIMITS,
    # Trading config
    TRADING_CONFIG, CANDLES_PER_DAY, MINUTES_PER_CANDLE, ALPHATREND_CONFIG,
    # Functions
    days_to_candles, days_to_candles_map,
    # Blacklist
    is_stream_blacklisted, load_dynamic_blacklist, update_dynamic_blacklist,
    save_dynamic_blacklist, DYNAMIC_BLACKLIST_CONFIG, DYNAMIC_BLACKLIST_CACHE,
    POST_PORTFOLIO_BLACKLIST,
    # Thresholds
    MIN_EXPECTANCY_R_MULTIPLE, MIN_SCORE_THRESHOLD, CONFIDENCE_RISK_MULTIPLIER,
    # Walk-forward and circuit breaker configs
    WALK_FORWARD_CONFIG, MIN_OOS_TRADES_BY_TF,
    CIRCUIT_BREAKER_CONFIG, ROLLING_ER_CONFIG,
    # TF-Adaptive Thresholds
    TF_THRESHOLDS, BASE_TF_THRESHOLDS,
    get_tf_threshold, get_tf_thresholds,
    # AlphaTrend 3-Layer Validation Thresholds
    AT_VALIDATION_THRESHOLDS, BASE_AT_VALIDATION_THRESHOLDS,
    get_at_validation_thresholds,
)

from .utils import (
    utcnow,  # Replacement for deprecated datetime.utcnow()
    normalize_datetime, tf_to_timedelta, tf_to_minutes,
    calculate_funding_cost, format_time_utc, format_time_local,
    append_trade_event, calculate_r_multiple, calculate_expected_r,
    apply_1m_profit_lock, apply_partial_stop_protection,
    derive_exit_profile_params, validate_and_adjust_sl,
)

from .config import (
    # Strategy configs (single source of truth)
    DEFAULT_STRATEGY_CONFIG, SYMBOL_PARAMS,
    # PR-1: Baseline config for regression fix
    BASELINE_CONFIG,
    # PR-2: Risk management & config continuity
    PR2_CONFIG,
)

from .config_loader import (
    load_optimized_config, save_best_configs, invalidate_config_cache,
    _strategy_signature,  # Export for use in main file
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
    add_at_validation_columns,  # AlphaTrend 3-layer validation
    get_indicator_value, get_candle_data,
    calculate_rr_ratio, check_wick_rejection,
    detect_regime, add_regime_column, get_regime_multiplier,
    # Volatility Regime (Expert Panel - Sinclair's 3-Tier System)
    classify_volatility_regime,
)

# Pattern 1: Momentum Exhaustion Exit (Real Trade Analysis 2026-01-04)
from .momentum_exit import (
    detect_momentum_exhaustion,
    calculate_dynamic_tp_from_momentum,
    should_exit_on_momentum,
)

# Patterns 3-7: Additional Trading Patterns (Real Trade Analysis 2026-01-04)
from .pattern_filters import (
    # Pattern 3: Liquidity Grab
    detect_liquidity_grab,
    # Pattern 4: SSL Slope Filter (Original)
    is_ssl_baseline_ranging,
    # Pattern 4B-D: NEW SSL Filters (Professional Analysis 2026-01-04)
    check_ssl_slope_direction,  # Priority 1A: Directional slope filter
    check_ssl_stability,        # Priority 1C: Stability check
    predict_quick_failure,      # Combined quick failure predictor
    # Pattern 5: HTF Bounce
    detect_htf_bounce,
    # Pattern 6: Momentum Loss
    detect_momentum_loss_after_trend,
    # Pattern 7: SSL Dynamic Support
    is_ssl_acting_as_dynamic_support,
)

# Enhanced Regime Filter (Priority 2 - Hedge Fund Recommendation)
from .regime_filter import (
    RegimeFilter, RegimeType, RegimeResult,
    check_regime_for_trade, get_btc_regime,
    analyze_regime_distribution,
)

# Correlation Management (Priority 4 - Hedge Fund Recommendation)
from .correlation_manager import (
    CorrelationManager, CorrelationCheckResult,
    check_correlation_risk, calculate_portfolio_effective_positions,
    DEFAULT_CORRELATION_MATRIX,
    adjust_kelly_for_correlation, calculate_portfolio_risk,
)

# Kelly Criterion Risk Management
from .kelly_calculator import (
    calculate_kelly, calculate_kelly_from_history,
    calculate_growth_rate, trades_to_double,
    kelly_comparison, edge_exists, minimum_win_rate_for_edge,
    MIN_KELLY_FRACTION, MAX_KELLY_FRACTION, MIN_TRADES_FOR_KELLY,
)

# Drawdown Management
from .drawdown_tracker import (
    DrawdownTracker, DrawdownStatus, DrawdownState,
    calculate_drawdown, get_drawdown_kelly_multiplier,
    get_recovery_status, estimate_recovery_time,
)

# Risk Manager (Main Interface)
from .risk_manager import (
    RiskManager, PositionSizeResult, TradeRecord,
    calculate_r_multiple as rm_calculate_r_multiple,
    calculate_trade_rr, RISK_CONFIG,
)

# Market Structure Analysis
from .market_structure import (
    MarketStructure, StructureResult, SwingPoint,
    SwingType, TrendType, StructureBreak,
    detect_structure, is_trade_aligned, get_structure_score,
    add_structure_columns,
)

# Fair Value Gap (FVG) Detection
from .fvg_detector import (
    FVGDetector, FairValueGap, FVGResult,
    FVGType, FVGStatus,
    detect_fvgs, get_fvg_score, is_in_fvg,
    add_fvg_columns,
)

from .logging_config import (
    get_logger, set_log_level, print_to_log,
)

from .trading_engine import (
    TradingEngine,
    set_backtest_mode,
    is_backtest_mode,
)

# Performance optimization (v40.x)
from .perf_cache import (
    MasterDataCache, StreamArrays, FastEventHeap, OpenTradeIndex,
    get_timedelta, get_timedelta_ns, datetime_to_ns,
    clear_disk_cache, estimate_candle_count,
)

# Optimizer functions (v40.5 - modular)
from .optimizer import (
    _optimize_backtest_configs,
    _generate_candidate_configs,
    _generate_quick_candidate_configs,
    _get_min_trades_for_timeframe,
    _split_data_walk_forward,
    _validate_config_oos,
    _check_overfit,
    _compute_optimizer_score,
    _score_config_for_stream,
    MIN_EXPECTANCY_PER_TRADE,
    STRATEGY_BLACKLIST,
)

# Optuna-based Optimizer (v2.3.0 - Bayesian optimization)
try:
    from .optuna_optimizer import (
        OPTUNA_AVAILABLE,
        SSLFlowOptimizer,
        optimize_multiple_streams,
        ParameterSpace,
        DEFAULT_PARAM_SPACE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    SSLFlowOptimizer = None
    optimize_multiple_streams = None
    ParameterSpace = None
    DEFAULT_PARAM_SPACE = None

__all__ = [
    # Environment
    'IS_COLAB', 'IS_HEADLESS', 'IS_NOTEBOOK', 'HAS_TQDM', 'HTF_ONLY_MODE',
    # M3 Performance
    'M3_PERFORMANCE_CONFIG',
    # Symbols and timeframes
    'SYMBOLS', 'TIMEFRAMES', 'LOWER_TIMEFRAMES', 'HTF_TIMEFRAMES',
    # Paths
    'DATA_DIR', 'CSV_FILE', 'CONFIG_FILE', 'BEST_CONFIGS_FILE',
    'DYNAMIC_BLACKLIST_FILE', 'BACKTEST_META_FILE', 'POT_LOG_FILE',
    # Candle limits
    'BACKTEST_CANDLE_LIMITS', 'DAILY_REPORT_CANDLE_LIMITS',
    # Trading config
    'TRADING_CONFIG', 'CANDLES_PER_DAY', 'MINUTES_PER_CANDLE', 'ALPHATREND_CONFIG',
    'days_to_candles', 'days_to_candles_map',
    # Blacklist
    'is_stream_blacklisted', 'load_dynamic_blacklist', 'update_dynamic_blacklist',
    'save_dynamic_blacklist', 'DYNAMIC_BLACKLIST_CONFIG', 'DYNAMIC_BLACKLIST_CACHE',
    'POST_PORTFOLIO_BLACKLIST',
    # Thresholds
    'MIN_EXPECTANCY_R_MULTIPLE', 'MIN_SCORE_THRESHOLD', 'CONFIDENCE_RISK_MULTIPLIER',
    # Walk-forward and circuit breaker configs
    'WALK_FORWARD_CONFIG', 'MIN_OOS_TRADES_BY_TF',
    'CIRCUIT_BREAKER_CONFIG', 'ROLLING_ER_CONFIG',
    # TF-Adaptive Thresholds
    'TF_THRESHOLDS', 'BASE_TF_THRESHOLDS',
    'get_tf_threshold', 'get_tf_thresholds',
    # AlphaTrend 3-Layer Validation Thresholds
    'AT_VALIDATION_THRESHOLDS', 'BASE_AT_VALIDATION_THRESHOLDS',
    'get_at_validation_thresholds',
    # Utils
    'normalize_datetime', 'tf_to_timedelta', 'tf_to_minutes',
    'calculate_funding_cost', 'format_time_utc', 'format_time_local',
    'append_trade_event', 'calculate_r_multiple', 'calculate_expected_r',
    'apply_1m_profit_lock', 'apply_partial_stop_protection',
    # Config loader
    'load_optimized_config', 'save_best_configs', 'invalidate_config_cache', '_strategy_signature',
    'SYMBOL_PARAMS', 'DEFAULT_STRATEGY_CONFIG', 'BASELINE_CONFIG', 'PR2_CONFIG',
    # Trade manager
    'BaseTradeManager', 'SimTradeManager',
    # Telegram
    'TelegramNotifier', 'get_notifier', 'send_telegram',
    'save_telegram_config', 'load_telegram_config',
    # Binance client
    'BinanceClient', 'get_client',
    # Indicators
    'calculate_indicators', 'calculate_alphatrend',
    'add_at_validation_columns',  # AlphaTrend 3-layer validation
    'get_indicator_value', 'get_candle_data',
    'calculate_rr_ratio', 'check_wick_rejection',
    'classify_volatility_regime',  # Volatility Regime (Sinclair's 3-Tier)
    # Regime Filter (Priority 2)
    'RegimeFilter', 'RegimeType', 'RegimeResult',
    'check_regime_for_trade', 'get_btc_regime',
    'analyze_regime_distribution',
    # Correlation Management (Priority 4)
    'CorrelationManager', 'CorrelationCheckResult',
    'check_correlation_risk', 'calculate_portfolio_effective_positions',
    'DEFAULT_CORRELATION_MATRIX',
    'adjust_kelly_for_correlation', 'calculate_portfolio_risk',
    # Kelly Criterion Risk Management
    'calculate_kelly', 'calculate_kelly_from_history',
    'calculate_growth_rate', 'trades_to_double',
    'kelly_comparison', 'edge_exists', 'minimum_win_rate_for_edge',
    'MIN_KELLY_FRACTION', 'MAX_KELLY_FRACTION', 'MIN_TRADES_FOR_KELLY',
    # Drawdown Management
    'DrawdownTracker', 'DrawdownStatus', 'DrawdownState',
    'calculate_drawdown', 'get_drawdown_kelly_multiplier',
    'get_recovery_status', 'estimate_recovery_time',
    # Risk Manager
    'RiskManager', 'PositionSizeResult', 'TradeRecord',
    'calculate_trade_rr', 'RISK_CONFIG',
    # Market Structure Analysis
    'MarketStructure', 'StructureResult', 'SwingPoint',
    'SwingType', 'TrendType', 'StructureBreak',
    'detect_structure', 'is_trade_aligned', 'get_structure_score',
    'add_structure_columns',
    # Fair Value Gap (FVG) Detection
    'FVGDetector', 'FairValueGap', 'FVGResult',
    'FVGType', 'FVGStatus',
    'detect_fvgs', 'get_fvg_score', 'is_in_fvg',
    'add_fvg_columns',
    # Logging
    'get_logger', 'set_log_level', 'print_to_log',
    # Trading Engine
    'TradingEngine',
    'set_backtest_mode',
    'is_backtest_mode',
    # Performance optimization
    'MasterDataCache', 'StreamArrays', 'FastEventHeap', 'OpenTradeIndex',
    'get_timedelta', 'get_timedelta_ns', 'datetime_to_ns',
    'clear_disk_cache', 'estimate_candle_count',
    # Optimizer (v40.5)
    '_optimize_backtest_configs',
    '_generate_candidate_configs',
    '_generate_quick_candidate_configs',
    '_get_min_trades_for_timeframe',
    '_split_data_walk_forward',
    '_validate_config_oos',
    '_check_overfit',
    '_compute_optimizer_score',
    '_score_config_for_stream',
    'MIN_EXPECTANCY_PER_TRADE',
    'STRATEGY_BLACKLIST',
    # Optuna Optimizer (v2.3.0)
    'OPTUNA_AVAILABLE',
    'SSLFlowOptimizer',
    'optimize_multiple_streams',
    'ParameterSpace',
    'DEFAULT_PARAM_SPACE',
]
