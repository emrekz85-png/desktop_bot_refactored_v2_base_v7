# strategies/__init__.py
# Strategy Signal Detection Module
#
# This module contains trading strategy implementations for signal detection.
# Each strategy is in its own module for modularity and maintainability.
#
# Available strategies:
# - ssl_flow: Trend following with SSL HYBRID baseline (TP at PBEMA) [ACTIVE]
# - pbema_retest: Trade PBEMA as support/resistance after breakout [DEPRECATED - use v2]
# - pbema_retest_v2: PBEMA strategy with corrected approach-direction logic [ACTIVE]
# - keltner_bounce: Mean reversion using Keltner bands (TP at PBEMA) [DISABLED]

from .base import SignalResult, STRATEGY_MODES
from .ssl_flow import check_ssl_flow_signal
from .pbema_retest import check_pbema_retest_signal  # DEPRECATED
from .pbema_retest_v2 import check_pbema_retest_signal_v2, check_pbema_v2  # NEW
from .keltner_bounce import check_keltner_bounce_signal  # DISABLED - kept for future use
from .router import check_signal

# Strategy registry for dynamic lookup
# ssl_flow and pbema_retest_v2 are active
STRATEGY_REGISTRY = {
    "ssl_flow": check_ssl_flow_signal,
    "pbema_retest": check_pbema_retest_signal_v2,  # V2 is now default
    "pbema_retest_v1": check_pbema_retest_signal,  # Old version for comparison
    "pbema_retest_v2": check_pbema_retest_signal_v2,
    # "keltner_bounce": check_keltner_bounce_signal,  # DISABLED
}

__all__ = [
    # Types
    "SignalResult",
    "STRATEGY_MODES",
    # Strategy functions
    "check_ssl_flow_signal",
    "check_pbema_retest_signal",  # DEPRECATED - use check_pbema_retest_signal_v2
    "check_pbema_retest_signal_v2",  # NEW - corrected logic
    "check_pbema_v2",  # Alias for V2
    "check_keltner_bounce_signal",  # DISABLED but exported for potential future use
    # Router
    "check_signal",
    # Registry
    "STRATEGY_REGISTRY",
]
