"""
Centralized logging configuration for the trading bot.

Provides:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file handlers
- Log rotation to prevent disk space issues
- Easy migration from print() statements

Usage:
    from core.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Trade opened")
    logger.debug("Detailed info: %s", data)
    logger.warning("Low balance")
    logger.error("API error: %s", error)
"""

import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# ==========================================
# LOGGING CONFIGURATION
# ==========================================

# Default log levels
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

# Log file settings
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
LOG_FILE_NAME = "trading_bot.log"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files

# Log format
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global logger cache
_loggers = {}
_initialized = False
_file_handler = None
_console_handler = None


def _ensure_log_dir():
    """Create log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def _get_console_handler() -> logging.StreamHandler:
    """Get or create the console handler."""
    global _console_handler
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setLevel(DEFAULT_CONSOLE_LEVEL)
        console_formatter = logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT)
        _console_handler.setFormatter(console_formatter)
    return _console_handler


def _get_file_handler() -> Optional[RotatingFileHandler]:
    """Get or create the rotating file handler."""
    global _file_handler
    if _file_handler is None:
        try:
            _ensure_log_dir()
            log_file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)
            _file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=BACKUP_COUNT,
                encoding='utf-8'
            )
            _file_handler.setLevel(DEFAULT_FILE_LEVEL)
            file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)
            _file_handler.setFormatter(file_formatter)
        except Exception as e:
            print(f"[LOGGING] Could not create file handler: {e}")
            return None
    return _file_handler


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    if name is None:
        name = "trading_bot"

    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already has handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set to DEBUG, handlers filter

        # Add console handler
        console_handler = _get_console_handler()
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)

        # Add file handler
        file_handler = _get_file_handler()
        if file_handler and file_handler not in logger.handlers:
            logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


def set_log_level(level: int, console: bool = True, file: bool = True):
    """
    Change the log level dynamically.

    Args:
        level: logging level (e.g., logging.DEBUG, logging.INFO)
        console: Apply to console handler
        file: Apply to file handler
    """
    if console:
        handler = _get_console_handler()
        if handler:
            handler.setLevel(level)

    if file:
        handler = _get_file_handler()
        if handler:
            handler.setLevel(level)


def log_trade_event(
    event_type: str,
    symbol: str,
    timeframe: str,
    details: dict = None,
    level: int = logging.INFO
):
    """
    Log a structured trade event.

    Args:
        event_type: Type of event (OPEN, CLOSE, SL_HIT, TP_HIT, etc.)
        symbol: Trading symbol
        timeframe: Timeframe
        details: Additional details dict
        level: Log level
    """
    logger = get_logger("trade_events")

    msg_parts = [f"[{event_type}] {symbol}-{timeframe}"]

    if details:
        for key, value in details.items():
            if isinstance(value, float):
                msg_parts.append(f"{key}={value:.6f}")
            else:
                msg_parts.append(f"{key}={value}")

    logger.log(level, " | ".join(msg_parts))


def log_signal(
    signal_type: str,
    symbol: str,
    timeframe: str,
    entry: float,
    tp: float,
    sl: float,
    reason: str = None
):
    """
    Log a trading signal.

    Args:
        signal_type: LONG or SHORT
        symbol: Trading symbol
        timeframe: Timeframe
        entry: Entry price
        tp: Take profit price
        sl: Stop loss price
        reason: Signal generation reason
    """
    logger = get_logger("signals")

    rr = abs(tp - entry) / abs(entry - sl) if entry != sl else 0

    msg = f"[SIGNAL] {signal_type} {symbol}-{timeframe} | entry={entry:.6f} tp={tp:.6f} sl={sl:.6f} RR={rr:.2f}"
    if reason:
        msg += f" | reason={reason}"

    logger.info(msg)


def log_api_call(
    endpoint: str,
    success: bool,
    response_time_ms: float = None,
    error: str = None
):
    """
    Log API calls for debugging and monitoring.

    Args:
        endpoint: API endpoint
        success: Whether the call succeeded
        response_time_ms: Response time in milliseconds
        error: Error message if failed
    """
    logger = get_logger("api")

    if success:
        msg = f"[API] {endpoint} | OK"
        if response_time_ms:
            msg += f" | {response_time_ms:.0f}ms"
        logger.debug(msg)
    else:
        msg = f"[API] {endpoint} | FAILED"
        if error:
            msg += f" | {error}"
        logger.warning(msg)


# ==========================================
# PRINT REPLACEMENT HELPERS
# ==========================================

def print_to_log(msg: str, level: str = "INFO"):
    """
    Helper function to migrate from print() to logging.

    Args:
        msg: Message to log
        level: Log level as string (DEBUG, INFO, WARNING, ERROR)
    """
    logger = get_logger("main")
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    logger.log(log_level, msg)


# Convenience shorthand
info = lambda msg: get_logger("main").info(msg)
debug = lambda msg: get_logger("main").debug(msg)
warning = lambda msg: get_logger("main").warning(msg)
error = lambda msg: get_logger("main").error(msg)
