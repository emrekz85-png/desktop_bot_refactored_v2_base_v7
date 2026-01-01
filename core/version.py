"""
Version tracking module for SSL Flow Trading Bot.

Bu modül versiyon bilgilerini merkezi olarak tutar ve test scriptleri
tarafından kullanılır.
"""

# Current version info
VERSION = "2.0.0"
VERSION_NAME = "indicator-parity-fix"
VERSION_DATE = "2026-01-01"

# Active changes in this version
CHANGES = {
    "ATR_METHOD": "RMA",           # Was: SMA
    "MOMENTUM_SOURCE": "MFI",      # MFI if volume, else RSI
    "SKIP_WICK_REJECTION": True,   # Was: False
    "FLAT_THRESHOLD": 0.002,       # Was: 0.001
}

# Version history for reference
VERSION_HISTORY = [
    {
        "version": "2.0.0",
        "name": "indicator-parity-fix",
        "date": "2026-01-01",
        "changes": [
            "ATR: SMA → RMA (TradingView uyumu)",
            "skip_wick_rejection: True (gereksiz filtre kaldırıldı)",
            "flat_threshold: 0.001 → 0.002 (daha az kısıtlayıcı)",
        ],
        "test_results": {
            "pnl": -39.90,
            "trades": 13,
            "win_rate": 30.8,
            "max_dd": 98.06,
            "baseline_pnl": -161.99,
            "improvement": 122.09,
        }
    },
    {
        "version": "1.0.0",
        "name": "original-baseline",
        "date": "2025-12-31",
        "changes": ["Original baseline configuration"],
        "test_results": {
            "pnl": -161.99,
            "trades": 51,
            "win_rate": 41.0,
            "max_dd": 208.0,
        }
    },
]


def get_version_string() -> str:
    """Get formatted version string."""
    return f"v{VERSION} ({VERSION_NAME})"


def get_version_banner(symbols: list = None, timeframes: list = None,
                       lookback_days: int = None) -> str:
    """
    Get version banner for test output.

    Args:
        symbols: List of symbols being tested
        timeframes: List of timeframes being tested
        lookback_days: Lookback period in days

    Returns:
        Formatted banner string
    """
    lines = [
        "",
        "=" * 70,
        f"📊 TEST - Version: v{VERSION} ({VERSION_NAME})",
        f"   Date: {VERSION_DATE}",
        f"   Değişiklikler: ATR={CHANGES['ATR_METHOD']}, "
        f"skip_wick={CHANGES['SKIP_WICK_REJECTION']}, "
        f"flat={CHANGES['FLAT_THRESHOLD']}",
    ]

    if symbols:
        lines.append(f"   Semboller: {'+'.join(symbols)}")
    if timeframes:
        lines.append(f"   Timeframes: {', '.join(timeframes)}")
    if lookback_days:
        lines.append(f"   Lookback: {lookback_days} gün")

    lines.extend([
        "=" * 70,
        "",
    ])

    return "\n".join(lines)


def print_version_banner(symbols: list = None, timeframes: list = None,
                         lookback_days: int = None) -> None:
    """Print version banner to console."""
    print(get_version_banner(symbols, timeframes, lookback_days))


def get_changes_summary() -> str:
    """Get summary of changes in current version."""
    changes = VERSION_HISTORY[0]["changes"]
    return " | ".join(changes)


def compare_with_baseline() -> dict:
    """Compare current version with baseline."""
    current = VERSION_HISTORY[0]["test_results"]
    baseline = VERSION_HISTORY[1]["test_results"]

    return {
        "pnl_change": current["pnl"] - baseline["pnl"],
        "trades_change": current["trades"] - baseline["trades"],
        "win_rate_change": current["win_rate"] - baseline["win_rate"],
        "dd_change": current["max_dd"] - baseline["max_dd"],
    }
