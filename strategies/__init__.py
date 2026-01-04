# strategies/__init__.py
# Strategy Signal Detection Module
#
# This module contains trading strategy implementations for signal detection.
# Each strategy is in its own module for modularity and maintainability.
#
# Available strategies:
# - ssl_flow: Trend following with SSL HYBRID baseline (TP at PBEMA) [ACTIVE]
# - pbema_retest: Trade PBEMA as support/resistance after breakout [ACTIVE]
# - keltner_bounce: Mean reversion using Keltner bands (TP at PBEMA) [DISABLED]

from .base import SignalResult, STRATEGY_MODES
from .ssl_flow import check_ssl_flow_signal
from .pbema_retest import check_pbema_retest_signal
from .keltner_bounce import check_keltner_bounce_signal  # DISABLED - kept for future use
from .router import check_signal

# Strategy registry for dynamic lookup
# ssl_flow and pbema_retest are active
STRATEGY_REGISTRY = {
    "ssl_flow": check_ssl_flow_signal,
    "pbema_retest": check_pbema_retest_signal,
    # "keltner_bounce": check_keltner_bounce_signal,  # DISABLED
}

__all__ = [
    # Types
    "SignalResult",
    "STRATEGY_MODES",
    # Strategy functions
    "check_ssl_flow_signal",
    "check_pbema_retest_signal",
    "check_keltner_bounce_signal",  # DISABLED but exported for potential future use
    # Router
    "check_signal",
    # Registry
    "STRATEGY_REGISTRY",
]
