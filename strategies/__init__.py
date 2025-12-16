# strategies/__init__.py
# Strategy Signal Detection Module
#
# This module contains trading strategy implementations for signal detection.
# Each strategy is in its own module for modularity and maintainability.
#
# Available strategies:
# - keltner_bounce: Mean reversion using Keltner bands with PBEMA cloud target
# - pbema_reaction: Trade when price approaches/touches PBEMA cloud

from .base import SignalResult, STRATEGY_MODES
from .keltner_bounce import check_keltner_bounce_signal
from .pbema_reaction import check_pbema_reaction_signal
from .router import check_signal

# Strategy registry for dynamic lookup
STRATEGY_REGISTRY = {
    "keltner_bounce": check_keltner_bounce_signal,
    "pbema_reaction": check_pbema_reaction_signal,
}

__all__ = [
    # Types
    "SignalResult",
    "STRATEGY_MODES",
    # Strategy functions
    "check_keltner_bounce_signal",
    "check_pbema_reaction_signal",
    # Router
    "check_signal",
    # Registry
    "STRATEGY_REGISTRY",
]
