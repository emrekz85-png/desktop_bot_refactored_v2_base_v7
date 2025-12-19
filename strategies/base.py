# strategies/base.py
# Base types and constants for strategy signal detection
#
# This module provides:
# - SignalResult type alias for strategy return values
# - Strategy mode constants
# - Shared utility functions

from typing import Tuple, Optional, Dict, Any, Union

# Type alias for signal detection return values
# Format: (signal_type, entry_price, take_profit, stop_loss, reason)
# signal_type: "LONG", "SHORT", or None
# With debug: adds debug_info dict as 6th element
SignalResult = Tuple[
    Optional[str],      # signal_type: "LONG", "SHORT", or None
    Optional[float],    # entry_price
    Optional[float],    # take_profit
    Optional[float],    # stop_loss
    str,                # reason: "ACCEPTED(...)" or failure reason
]

SignalResultWithDebug = Tuple[
    Optional[str],      # signal_type
    Optional[float],    # entry_price
    Optional[float],    # take_profit
    Optional[float],    # stop_loss
    str,                # reason
    Dict[str, Any],     # debug_info
]

# Available strategy modes
STRATEGY_MODES = {
    "ssl_flow": "SSL Flow - Trend following with SSL HYBRID baseline (TP at PBEMA)",
    "keltner_bounce": "Keltner Bounce - Mean reversion with Keltner bands (TP at PBEMA)",
}

# Default strategy mode
DEFAULT_STRATEGY_MODE = "ssl_flow"


def make_signal_result(
    signal_type: Optional[str],
    entry: Optional[float],
    tp: Optional[float],
    sl: Optional[float],
    reason: str,
    debug_info: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
) -> Union[SignalResult, SignalResultWithDebug]:
    """Helper to create consistent signal result tuples."""
    if return_debug:
        return signal_type, entry, tp, sl, reason, debug_info or {}
    return signal_type, entry, tp, sl, reason
