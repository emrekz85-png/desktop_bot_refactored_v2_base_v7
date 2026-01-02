"""
Version tracking module for SSL Flow Trading Bot.

Bu modül versiyon bilgilerini merkezi olarak tutar ve test scriptleri
tarafından kullanılır.

Kullanım:
    from core.version import VERSION, print_version_banner, get_current_config

    # Test başında versiyon banner'ı yazdır
    print_version_banner(symbols=['BTCUSDT'], timeframes=['15m'])

    # Mevcut konfigürasyonu al
    config = get_current_config()
"""

from typing import Dict, List

# =============================================================================
# CURRENT VERSION
# =============================================================================

VERSION = "2.0.0"
VERSION_NAME = "indicator-parity-fix"
VERSION_DATE = "2026-01-01"

# =============================================================================
# ACTIVE CONFIGURATION
# Bu değerler mevcut versiyondaki aktif ayarları gösterir
# =============================================================================

CURRENT_CONFIG = {
    # Indicator Settings
    "ATR_METHOD": "RMA",              # Was: SMA (v1.0.0)
    "MOMENTUM_SOURCE": "MFI",         # MFI if volume, else RSI

    # Filter Settings
    "skip_wick_rejection": True,      # Was: False (v1.0.0)
    "skip_body_position": False,
    "skip_adx_filter": False,
    "skip_overlap_check": False,
    "skip_at_flat_filter": False,

    # Threshold Settings
    "flat_threshold": 0.002,          # Was: 0.001 (v1.0.0)
    "min_pbema_distance": 0.004,
    "regime_adx_threshold": 20.0,
    "ssl_touch_tolerance": 0.003,

    # Optimizer Settings
    "lookback_days": 60,
    "forward_days": 7,
    "hard_min_trades": 5,

    # Trade Management
    "partial_trigger_1": 0.40,
    "partial_fraction_1": 0.33,
    "partial_trigger_2": 0.70,
    "partial_fraction_2": 0.50,
    "be_after_tranche": 1,
    "be_atr_multiplier": 0.5,
}

# =============================================================================
# BASELINE REFERENCE (v1.0.0)
# Tüm karşılaştırmalar için referans
# =============================================================================

BASELINE_CONFIG = {
    "ATR_METHOD": "SMA",
    "MOMENTUM_SOURCE": "MFI",
    "skip_wick_rejection": False,
    "flat_threshold": 0.001,
    "lookback_days": 60,
}

BASELINE_RESULTS = {
    "pnl": -161.99,
    "trades": 51,
    "win_rate": 41.0,
    "max_dd": 208.0,
}

# =============================================================================
# VERSION HISTORY
# =============================================================================

VERSION_HISTORY = [
    {
        "version": "2.0.0",
        "name": "indicator-parity-fix",
        "date": "2026-01-01",
        "summary": "TradingView ile indicator uyumu + filter optimizasyonu",
        "changes": [
            {"setting": "ATR_METHOD", "from": "SMA", "to": "RMA", "reason": "TradingView uyumu"},
            {"setting": "skip_wick_rejection", "from": False, "to": True, "reason": "+$30 test sonucu"},
            {"setting": "flat_threshold", "from": 0.001, "to": 0.002, "reason": "Daha az kısıtlayıcı"},
        ],
        "tried_and_reverted": [
            {"setting": "MOMENTUM_SOURCE", "tried": "RSI", "reason": "Trade sayısı %70 düştü"},
            {"setting": "partial_trigger_1", "tried": 0.65, "reason": "PnL $68 düştü"},
        ],
        "results": {
            "pnl": -39.90,
            "trades": 13,
            "win_rate": 30.8,
            "max_dd": 98.06,
        },
        "comparison": {
            "pnl_change": 122.09,
            "trades_change": -38,
            "win_rate_change": -10.2,
            "max_dd_change": -109.94,
        },
        "verdict": "partial_success",
        "pros": [
            "PnL $122 iyileşti",
            "Drawdown yarıya indi",
            "TradingView ATR uyumu sağlandı",
        ],
        "cons": [
            "Trade sayısı çok düştü (51→13)",
            "Win rate düştü (41%→31%)",
            "Hala negatif PnL",
            "TRENDING'de kayıp",
        ],
    },
    {
        "version": "1.0.0",
        "name": "original-baseline",
        "date": "2025-12-31",
        "summary": "Orijinal baseline konfigürasyonu",
        "changes": [],
        "results": BASELINE_RESULTS.copy(),
        "verdict": "baseline",
    },
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_version_string() -> str:
    """Get formatted version string."""
    return f"v{VERSION} ({VERSION_NAME})"


def get_current_config() -> Dict:
    """Get current configuration dict."""
    return CURRENT_CONFIG.copy()


def get_baseline_config() -> Dict:
    """Get baseline configuration dict."""
    return BASELINE_CONFIG.copy()


def get_changes_from_baseline() -> List[Dict]:
    """Get list of changes from baseline."""
    current_version = VERSION_HISTORY[0]
    return current_version.get("changes", [])


def get_version_banner(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    lookback_days: int = None
) -> str:
    """
    Get version banner for test output.

    Args:
        symbols: List of symbols being tested
        timeframes: List of timeframes being tested
        lookback_days: Lookback period in days

    Returns:
        Formatted banner string
    """
    # Build changes string
    changes = get_changes_from_baseline()
    change_strs = []
    for c in changes[:3]:  # Show max 3 changes
        change_strs.append(f"{c['setting']}={c['to']}")
    changes_line = ", ".join(change_strs)

    lines = [
        "",
        "=" * 70,
        f"📊 TEST - Version: v{VERSION} ({VERSION_NAME})",
        f"   Date: {VERSION_DATE}",
        f"   Değişiklikler: {changes_line}",
    ]

    if symbols:
        lines.append(f"   Semboller: {'+'.join(symbols)}")
    if timeframes:
        lines.append(f"   Timeframes: {', '.join(timeframes)}")
    if lookback_days:
        lines.append(f"   Lookback: {lookback_days} gün")

    # Add comparison with baseline
    current = VERSION_HISTORY[0]
    if "comparison" in current:
        comp = current["comparison"]
        lines.append(f"   vs Baseline: PnL {comp['pnl_change']:+.2f}, DD {comp['max_dd_change']:+.2f}")

    lines.extend([
        "=" * 70,
        "",
    ])

    return "\n".join(lines)


def print_version_banner(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    lookback_days: int = None
) -> None:
    """Print version banner to console."""
    print(get_version_banner(symbols, timeframes, lookback_days))


def compare_with_baseline() -> Dict:
    """Compare current version with baseline."""
    current = VERSION_HISTORY[0]
    return current.get("comparison", {
        "pnl_change": 0,
        "trades_change": 0,
        "win_rate_change": 0,
        "max_dd_change": 0,
    })


def get_version_summary() -> str:
    """Get human-readable version summary."""
    current = VERSION_HISTORY[0]

    lines = [
        f"Version: v{VERSION} ({VERSION_NAME})",
        f"Date: {VERSION_DATE}",
        "",
        "Değişiklikler:",
    ]

    for change in current.get("changes", []):
        lines.append(f"  - {change['setting']}: {change['from']} → {change['to']}")
        lines.append(f"    Sebep: {change['reason']}")

    if current.get("tried_and_reverted"):
        lines.append("")
        lines.append("Denenen ve Geri Alınan:")
        for rev in current["tried_and_reverted"]:
            lines.append(f"  - {rev['setting']}: {rev['tried']} denendi")
            lines.append(f"    Geri alındı: {rev['reason']}")

    results = current.get("results", {})
    comp = current.get("comparison", {})

    lines.extend([
        "",
        "Test Sonuçları:",
        f"  PnL: ${results.get('pnl', 0):.2f} ({comp.get('pnl_change', 0):+.2f} vs baseline)",
        f"  Trades: {results.get('trades', 0)} ({comp.get('trades_change', 0):+d})",
        f"  Win Rate: {results.get('win_rate', 0):.1f}% ({comp.get('win_rate_change', 0):+.1f}%)",
        f"  Max DD: ${results.get('max_dd', 0):.2f} ({comp.get('max_dd_change', 0):+.2f})",
        "",
        f"Verdict: {current.get('verdict', 'N/A').upper()}",
    ])

    if current.get("pros"):
        lines.append("")
        lines.append("Avantajlar:")
        for pro in current["pros"]:
            lines.append(f"  ✅ {pro}")

    if current.get("cons"):
        lines.append("")
        lines.append("Dezavantajlar:")
        for con in current["cons"]:
            lines.append(f"  ⚠️ {con}")

    return "\n".join(lines)


def print_version_summary() -> None:
    """Print detailed version summary."""
    print(get_version_summary())


# =============================================================================
# CHANGES dict for backward compatibility
# =============================================================================

CHANGES = {
    "ATR_METHOD": CURRENT_CONFIG["ATR_METHOD"],
    "MOMENTUM_SOURCE": CURRENT_CONFIG["MOMENTUM_SOURCE"],
    "SKIP_WICK_REJECTION": CURRENT_CONFIG["skip_wick_rejection"],
    "FLAT_THRESHOLD": CURRENT_CONFIG["flat_threshold"],
}
