# config_ultra_minimal.py
# ULTRA-MINIMAL Configuration - Scientific AT Analysis
#
# Purpose: Test SSL Baseline with minimal constraints
# Method: AlphaTrend in SCORE mode (contributes data but doesn't block)
# Goal: 50-100 trades with AT scores recorded for deep analysis
#
# Scientific Approach:
# 1. Let trades happen based on SSL baseline + minimal filters
# 2. Record AlphaTrend score for each trade (but don't use it to block)
# 3. Analyze correlation between AT scores and trade outcomes
# 4. Determine optimal AT integration strategy based on DATA

from core.config import (
    TRADING_CONFIG,
    SYMBOLS,
    TIMEFRAMES,
    TF_THRESHOLDS,
    BASE_TF_THRESHOLDS,
)

# ==========================================
# ULTRA-MINIMAL STRATEGY CONFIG
# ==========================================

ULTRA_MINIMAL_STRATEGY_CONFIG = {
    # === STRATEGY MODE ===
    "strategy_mode": "ssl_flow",

    # === ALPHATREND: SCORE MODE (Data Collection) ===
    # CRITICAL: AT is ENABLED but in SCORE mode
    # - Will calculate AT scores for each signal
    # - Will NOT block any signals
    # - Scores saved with each trade for analysis
    # - Post-analysis will determine optimal usage

    "at_active": True,           # ENABLED for scoring
    "at_mode": "score",          # SCORE mode (doesn't block!)
    "at_score_weight": 2.0,      # Weight for scoring (not used for blocking)
    "at_regime_lookback": 20,    # For regime calculation

    # === ULTRA-LOOSE THRESHOLDS ===
    # Goal: Accept almost ANY setup that meets basic direction

    # 1. RISK/REWARD - Accept ANY positive RR
    "rr": 0.3,  # ULTRA-LOOSE (was 1.0) - even 0.3:1 RR is acceptable

    # 2. RSI - Almost never reject
    "rsi": 90,  # ULTRA-LOOSE (was 75) - only reject extreme overbought

    # 3. PBEMA DISTANCE - No minimum
    "min_pbema_distance": 0.0,  # ZERO minimum (was 0.002)

    # 4. SSL TOUCH - Very lenient
    "ssl_touch_tolerance": 0.010,  # 1.0% (was 0.5%)
    "lookback_candles": 10,        # Wide window (was 7)

    # === ALL OPTIONAL FILTERS DISABLED ===
    # Only SSL baseline direction + minimal validation

    "skip_body_position": True,      # DISABLED
    "skip_adx_filter": True,         # DISABLED
    "skip_overlap_check": True,      # DISABLED
    "skip_at_flat_filter": True,     # DISABLED (AT flat won't block)
    "skip_wick_rejection": True,     # DISABLED

    "use_ssl_never_lost_filter": False,   # DISABLED
    "use_ssl_flip_grace": False,          # DISABLED
    "use_confirmation_candle": False,     # DISABLED
    "use_market_structure": False,        # DISABLED
    "use_fvg_bonus": False,               # DISABLED
    "use_scoring": False,                 # Not needed (AT uses score mode internally)
    "use_htf_filter": False,              # DISABLED
    "use_btc_regime_filter": False,       # DISABLED
    "use_vol_normalized_pbema": False,    # DISABLED

    # === EXIT MANAGEMENT (Keep simple) ===
    "use_trailing": False,
    "use_partial": True,
    "use_dynamic_pbema_tp": False,
    "exit_profile": "clip",

    # Partial TP settings (simple)
    "partial_trigger": 0.50,
    "partial_fraction": 0.50,

    # === OTHER THRESHOLDS (Very loose) ===
    "ssl_body_tolerance": 0.010,  # 1.0% (very loose)
    "overlap_threshold": 0.020,   # 2.0% (very loose, not used due to skip)
    "flat_threshold": 0.005,      # 0.5% (very loose)

    "adx_min": 5.0,   # Very low (not enforced due to skip)
    "adx_max": 100.0,

    "regime_adx_threshold": 10.0,  # Very low

    # === POSITION SIZING ===
    "sl_validation_mode": "off",
    "use_progressive_partial": False,
    "partial_rr_adjustment": False,

    # === SMART FEATURES (All disabled) ===
    "use_smart_reentry": False,
    "use_time_invalidation": False,
    "use_roc_filter": False,
    "avoid_hours": None,

    # === METADATA FOR ANALYSIS ===
    # These will be saved with each trade for deep analysis
    "save_at_metadata": True,  # Custom flag for our analysis
    "save_rejection_reasons": True,
}

# ==========================================
# AT SCORE INTERPRETATION
# ==========================================

AT_SCORE_INTERPRETATION = {
    "description": "AlphaTrend contribution to signal quality",
    "range": "0.0 to 2.0+ (with weight=2.0)",
    "calculation": {
        "score_mode": {
            "buyers_dominant": 2.0,      # Full score
            "neutral": 0.5,              # Partial score
            "sellers_dominant": -1.0,    # Negative score (opposing)
        },
        "regime_mode": {
            "bullish_regime": 2.0,
            "neutral_regime": 0.5,
            "bearish_regime": -1.0,
        }
    },
    "questions_to_answer": [
        "Do trades with AT score > 1.5 have higher win rate?",
        "What's the optimal AT score threshold?",
        "Should we use binary (block) or score (weight) mode?",
        "Are there regimes where AT helps vs hurts?",
        "Does AT flat (score ~0.5) indicate ranging = good for reversals?",
    ]
}

# ==========================================
# EXPECTED OUTCOMES
# ==========================================

EXPECTED_OUTCOMES = {
    "trade_frequency": {
        "target": "50-100 trades",
        "current": "4 trades (core-only failed)",
        "multiplier": "12-25x increase expected",
    },
    "performance": {
        "win_rate": "30-45% (lower but valid)",
        "pnl": "Unknown (could be negative)",
        "goal": "Statistical foundation, NOT profit yet",
    },
    "data_collection": {
        "at_scores": "Saved for each trade",
        "at_regime": "Recorded for analysis",
        "at_buyers_dominant": "Boolean flag saved",
        "at_sellers_dominant": "Boolean flag saved",
        "at_is_flat": "Boolean flag saved",
    },
    "analysis_goals": [
        "Correlation between AT score and win rate",
        "Optimal AT threshold (if any)",
        "Binary vs Score vs Off mode comparison",
        "Regime-specific AT effectiveness",
        "False signal identification",
    ]
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_ultra_minimal_config():
    """
    Get the ultra-minimal configuration for baseline testing.

    Returns:
        dict: Ultra-minimal config with AT in score mode
    """
    return ULTRA_MINIMAL_STRATEGY_CONFIG.copy()


def print_config_comparison():
    """
    Print comparison between Core-Only and Ultra-Minimal configs.
    """
    from core.config_core_only import CORE_ONLY_STRATEGY_CONFIG

    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON: CORE-ONLY vs ULTRA-MINIMAL")
    print("="*80)
    print("\nKey Differences:")
    print("-"*80)

    changes = [
        ("AlphaTrend Mode", "binary (blocks)", "score (data only)", "CRITICAL CHANGE"),
        ("at_active", "True", "True", "Enabled for scoring"),
        ("at_mode", "binary", "score", "Won't block signals"),
        ("", "", "", ""),
        ("rr (min RR)", "1.0", "0.3", "Accept ANY positive RR"),
        ("rsi (limit)", "75", "90", "Almost never reject"),
        ("min_pbema_distance", "0.002 (0.2%)", "0.0 (none)", "No minimum distance"),
        ("ssl_touch_tolerance", "0.005 (0.5%)", "0.010 (1.0%)", "More lenient"),
        ("lookback_candles", "7", "10", "Wider window"),
    ]

    for param, core_val, ultra_val, desc in changes:
        if param == "":
            print("")
            continue
        print(f"  {param:25s}: {core_val:20s} → {ultra_val:20s} | {desc}")

    print("\nExpected Impact:")
    print("-"*80)
    print("  Core-Only Result: 4 trades (FAILED)")
    print("  Ultra-Minimal Target: 50-100 trades")
    print("  Multiplier: 12-25x increase")
    print("")
    print("  AT Scoring: Enabled for data collection")
    print("  AT Blocking: DISABLED (won't reject signals)")
    print("  Analysis: Post-test correlation analysis")
    print("="*80)


def get_analysis_metadata_fields():
    """
    Get list of metadata fields to save with each trade for analysis.

    Returns:
        list: Field names to extract and save
    """
    return [
        # AlphaTrend metrics
        "at_score_long",
        "at_score_short",
        "at_buyers_dominant",
        "at_sellers_dominant",
        "at_is_flat",
        "at_regime",

        # Price/indicator values
        "close",
        "ssl_baseline",
        "pbema_top",
        "pbema_bot",
        "pbema_mid",
        "rsi",
        "adx",

        # Distance metrics
        "baseline_pbema_distance",
        "long_pbema_distance",
        "short_pbema_distance",

        # Rejection flags (for analysis of filtered signals)
        "would_reject_at_binary",  # Would this be rejected in binary mode?
        "would_reject_rr_1_0",     # Would this be rejected with RR=1.0?
        "would_reject_rsi_75",     # Would this be rejected with RSI=75?
    ]


if __name__ == "__main__":
    print_config_comparison()

    print("\nAlphaTrend Score Interpretation:")
    print("-"*80)
    for key, value in AT_SCORE_INTERPRETATION.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")

    print("\n" + "="*80)
    print("SCIENTIFIC APPROACH")
    print("="*80)
    print("""
1. Run test with AT in SCORE mode (doesn't block)
2. Collect 50-100 trades with AT scores
3. Analyze correlation: AT score vs trade outcome
4. Answer questions:
   - Do high AT scores predict winners?
   - What's the optimal threshold?
   - Binary vs Score vs Off?
5. Make DATA-DRIVEN decision on AT integration
    """)
    print("="*80 + "\n")
