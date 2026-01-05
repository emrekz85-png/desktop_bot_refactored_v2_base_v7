# config_core_only.py
# CORE ONLY Configuration - Expert Panel Phase 1 Recommendation
#
# Purpose: Establish statistical foundation with 100+ trades
# Method: Strip ALL optional filters, keep only essential signal criteria
#
# Expected Results:
# - Trades: 80-150 (vs current 13)
# - PnL: Unknown (possibly negative, but statistically valid!)
# - Win Rate: 35-45% expected
# - Goal: STATISTICAL SIGNIFICANCE, not optimization
#
# Expert Panel Quote (Andreas Clenow):
# "13 trades is not a backtest, it's a coin flip session. Get to 100+ trades first,
#  then we can talk about optimization. You're building a house on sand right now."

from core.config import (
    TRADING_CONFIG,
    SYMBOLS,
    TIMEFRAMES,
    TF_THRESHOLDS,
    BASE_TF_THRESHOLDS,
)

# ==========================================
# CORE ONLY STRATEGY CONFIG
# ==========================================

CORE_ONLY_STRATEGY_CONFIG = {
    # === MINIMAL REQUIREMENTS ===
    # Only 4 essential checks (vs current 15+ filters)

    "strategy_mode": "ssl_flow",

    # 1. RISK/REWARD - Lowered from 2.0 to 1.0 for more signals
    "rr": 1.0,  # LOOSENED (was 2.0)

    # 2. RSI - Loosened significantly
    "rsi": 75,  # LOOSENED (was 70)

    # 3. AlphaTrend - KEEP (essential for momentum confirmation)
    "at_active": True,
    "at_mode": "binary",  # Simple binary mode (not regime/score)

    # === ALL OPTIONAL FILTERS DISABLED ===

    # Skip restrictive checks (set to True = SKIP the check)
    "skip_body_position": True,      # DISABLED (was False) - 99.9% pass rate = useless
    "skip_adx_filter": True,         # DISABLED (was False) - regime will handle this later
    "skip_overlap_check": True,      # DISABLED (was False) - removes <0.5% distance requirement
    "skip_at_flat_filter": True,     # DISABLED (was False) - AlphaTrend flat allowed
    "skip_wick_rejection": True,     # Already True - keep disabled

    # Disable advanced features
    "use_ssl_never_lost_filter": False,   # DISABLED (was True)
    "use_ssl_flip_grace": False,          # Already False
    "use_confirmation_candle": False,     # Already False
    "use_market_structure": False,        # DISABLED (was True)
    "use_fvg_bonus": False,              # DISABLED (was True)
    "use_scoring": False,                # Keep AND logic
    "use_htf_filter": False,             # Disable higher timeframe filter
    "use_btc_regime_filter": False,      # Disable BTC regime filter

    # === MINIMAL THRESHOLDS ===

    # PBEMA distance - SIGNIFICANTLY LOOSENED
    "min_pbema_distance": 0.002,  # 0.2% (was 0.4%) - allow closer targets

    # SSL touch - LOOSENED
    "ssl_touch_tolerance": 0.005,  # 0.5% (was 0.3%) - more lenient touch detection
    "lookback_candles": 7,         # 7 candles (was 5) - longer lookback window

    # ADX - Not used (skip_adx_filter=True)
    "adx_min": 10.0,  # Very low (not enforced)
    "adx_max": 100.0,  # No upper limit

    # Regime - Not used in core-only
    "regime_adx_threshold": 15.0,  # Low threshold (not used)

    # === EXIT MANAGEMENT (Keep simple) ===
    "use_trailing": False,
    "use_partial": True,  # Basic partial TP
    "use_dynamic_pbema_tp": False,  # Disable dynamic TP
    "exit_profile": "clip",

    # Partial TP settings (simple)
    "partial_trigger": 0.50,   # Take profit at 50% of TP
    "partial_fraction": 0.50,  # Close 50% of position

    # === TF-ADAPTIVE THRESHOLDS (Inherit) ===
    "ssl_body_tolerance": 0.005,  # Loosened
    "overlap_threshold": 0.010,   # Loosened (not used due to skip)
    "flat_threshold": 0.003,      # Loosened

    # === POSITION SIZING (Keep standard) ===
    "sl_validation_mode": "off",  # No SL validation
    "use_progressive_partial": False,
    "partial_rr_adjustment": False,

    # === REGIME DETECTION (Phase 2 - not yet) ===
    "at_regime_lookback": 20,
    "at_score_weight": 2.0,

    # === VOLATILITY (Phase 2 - not yet) ===
    "use_vol_normalized_pbema": False,  # Disable volatility normalization
    "vol_norm_min_atr": 1.0,
    "vol_norm_max_atr": 4.0,

    # === SMART FEATURES (All disabled) ===
    "use_smart_reentry": False,
    "use_time_invalidation": False,
    "use_roc_filter": False,
    "avoid_hours": None,
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_core_only_config():
    """
    Get the core-only configuration for baseline testing.

    Returns:
        dict: Stripped-down config with only essential filters
    """
    return CORE_ONLY_STRATEGY_CONFIG.copy()


def print_config_comparison():
    """
    Print comparison between DEFAULT and CORE_ONLY configs.
    Shows which filters were disabled.
    """
    from core.config import DEFAULT_STRATEGY_CONFIG

    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON: DEFAULT vs CORE ONLY")
    print("="*70)
    print("\nFilter Status Changes:")
    print("-"*70)

    changes = [
        ("skip_body_position", False, True, "Body position check DISABLED"),
        ("skip_adx_filter", False, True, "ADX filter DISABLED"),
        ("skip_overlap_check", False, True, "SSL-PBEMA overlap check DISABLED"),
        ("skip_at_flat_filter", False, True, "AlphaTrend flat filter DISABLED"),
        ("use_ssl_never_lost_filter", True, False, "SSL never lost filter DISABLED"),
        ("use_market_structure", True, False, "Market structure filter DISABLED"),
        ("use_fvg_bonus", True, False, "FVG bonus DISABLED"),
    ]

    for param, old_val, new_val, desc in changes:
        print(f"  {param:30s}: {str(old_val):5s} → {str(new_val):5s}  | {desc}")

    print("\nThreshold Changes:")
    print("-"*70)

    threshold_changes = [
        ("rr", 2.0, 1.0, "Min risk/reward LOOSENED"),
        ("rsi", 70, 75, "RSI limit LOOSENED"),
        ("min_pbema_distance", 0.004, 0.002, "PBEMA distance LOOSENED (0.4%→0.2%)"),
        ("ssl_touch_tolerance", 0.003, 0.005, "SSL touch tolerance LOOSENED"),
        ("lookback_candles", 5, 7, "Lookback window EXTENDED"),
    ]

    for param, old_val, new_val, desc in threshold_changes:
        print(f"  {param:30s}: {old_val:5.3f} → {new_val:5.3f}  | {desc}")

    print("\nExpected Impact:")
    print("-"*70)
    print("  Current Trades:  13")
    print("  Expected Trades: 80-150 (7-10x increase)")
    print("  Current PnL:     -$39.90")
    print("  Expected PnL:    Unknown (possibly negative)")
    print("  Win Rate:        35-45% expected")
    print("")
    print("  GOAL: Establish statistical foundation for meaningful optimization")
    print("="*70)


def get_active_filter_count(config):
    """Count how many filters are active in a config."""
    count = 0

    # Filters that are active when skip_* is False
    if not config.get("skip_body_position", True):
        count += 1
    if not config.get("skip_adx_filter", True):
        count += 1
    if not config.get("skip_overlap_check", True):
        count += 1
    if not config.get("skip_at_flat_filter", True):
        count += 1
    if not config.get("skip_wick_rejection", True):
        count += 1

    # Filters that are active when use_* is True
    if config.get("use_ssl_never_lost_filter", False):
        count += 1
    if config.get("use_confirmation_candle", False):
        count += 1
    if config.get("use_market_structure", False):
        count += 1
    if config.get("use_fvg_bonus", False):
        count += 1
    if config.get("use_htf_filter", False):
        count += 1
    if config.get("use_btc_regime_filter", False):
        count += 1
    if config.get("use_vol_normalized_pbema", False):
        count += 1

    return count


if __name__ == "__main__":
    print_config_comparison()

    print("\nActive Filter Counts:")
    print(f"  DEFAULT config:   {get_active_filter_count(__import__('core.config').config.DEFAULT_STRATEGY_CONFIG)} filters")
    print(f"  CORE ONLY config: {get_active_filter_count(CORE_ONLY_STRATEGY_CONFIG)} filters")
    print("\n")
