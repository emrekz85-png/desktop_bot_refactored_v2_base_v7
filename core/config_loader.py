"""
Configuration loader for trading strategies.

Handles loading and saving optimized configurations for each symbol/timeframe pair.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional

from .config import (
    BEST_CONFIGS_FILE, BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS,
    DATA_DIR, TRADING_CONFIG,
    # Strategy configs from central location (single source of truth)
    DEFAULT_STRATEGY_CONFIG, SYMBOL_PARAMS,
    # Thread safety lock for config cache
    _BEST_CONFIG_LOCK,
)
from .logging_config import get_logger

# Module logger
_logger = get_logger(__name__)


def _strategy_signature() -> str:
    """
    Generate a deterministic fingerprint of the current strategy inputs.

    The hash combines the live trading configuration, the default strategy
    parameters and the symbol-specific overrides so that backtest results are
    only consumed when they were produced with the exact same settings.

    IMPORTANT: This must match the _strategy_signature() in the main file
    to ensure saved configs are properly validated on load.
    """
    payload = {
        "trading": TRADING_CONFIG,
        "strategy": DEFAULT_STRATEGY_CONFIG,
        "symbol_params": SYMBOL_PARAMS,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _is_best_config_signature_valid(best_cfgs: dict) -> bool:
    """
    Check if cached configs belong to the current strategy signature.

    Returns False if:
    - No signature stored (old cache format)
    - Signature doesn't match current strategy
    """
    global BEST_CONFIG_WARNING_FLAGS

    if not isinstance(best_cfgs, dict):
        return False

    # Empty dict means no config file exists - return False silently
    # to avoid warning spam during backtest
    if not best_cfgs:
        return False

    meta = best_cfgs.get("_meta", {}) if isinstance(best_cfgs.get("_meta"), dict) else {}
    stored_sig = meta.get("strategy_signature")

    if not stored_sig:
        # Only warn if there are actual configs but no signature (old format)
        has_any_config = any(k != "_meta" for k in best_cfgs.keys())
        if has_any_config and not BEST_CONFIG_WARNING_FLAGS.get("missing_signature", False):
            _logger.warning("No saved backtest signature found. Best configs will be ignored.")
            BEST_CONFIG_WARNING_FLAGS["missing_signature"] = True
        return False

    current_sig = _strategy_signature()
    if stored_sig != current_sig:
        if not BEST_CONFIG_WARNING_FLAGS.get("signature_mismatch", False):
            _logger.warning(
                "Backtest config signature doesn't match current strategy. "
                "Using default configs; please re-run backtest."
            )
            BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = True
        return False

    return True


def _load_best_configs() -> dict:
    """Load best configs from file, using cache if available.

    Thread-safe: Uses lock to prevent race conditions during cache access.
    """
    global BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS

    with _BEST_CONFIG_LOCK:
        # Check cache first (inside lock to prevent TOCTOU)
        if BEST_CONFIG_CACHE:
            return BEST_CONFIG_CACHE.copy()

        if os.path.exists(BEST_CONFIGS_FILE):
            try:
                with open(BEST_CONFIGS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    BEST_CONFIG_CACHE.clear()
                    BEST_CONFIG_CACHE.update(raw)
                    # Başarılı yükleme - hata flag'lerini sıfırla
                    BEST_CONFIG_WARNING_FLAGS["json_error"] = False
                    BEST_CONFIG_WARNING_FLAGS["load_error"] = False
            except json.JSONDecodeError as e:
                if not BEST_CONFIG_WARNING_FLAGS.get("json_error", False):
                    _logger.error("Config file corrupted (JSON error): %s", e)
                    _logger.info("Delete corrupted file and re-run backtest: %s", BEST_CONFIGS_FILE)
                    BEST_CONFIG_WARNING_FLAGS["json_error"] = True
            except Exception as e:
                if not BEST_CONFIG_WARNING_FLAGS.get("load_error", False):
                    _logger.error("Config load error: %s", e)
                    BEST_CONFIG_WARNING_FLAGS["load_error"] = True

        return BEST_CONFIG_CACHE.copy()


def load_optimized_config(symbol: str, timeframe: str) -> dict:
    """
    Return optimized config for given symbol/timeframe with safe defaults.

    Priority order:
    1. Best configs from backtest (if signature valid)
    2. Manual settings in SYMBOL_PARAMS
    3. Safe defaults

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "15m")

    Returns:
        Configuration dictionary
    """
    defaults = DEFAULT_STRATEGY_CONFIG.copy()

    best_cfgs = _load_best_configs()
    signature_ok = _is_best_config_signature_valid(best_cfgs)

    # Get symbol-specific params
    symbol_cfg = SYMBOL_PARAMS.get(symbol, {})
    tf_cfg = symbol_cfg.get(timeframe, {}) if isinstance(symbol_cfg, dict) else {}

    # Merge with best configs if signature valid
    if isinstance(best_cfgs, dict) and signature_ok:
        sym_dict = best_cfgs.get(symbol, {}) if isinstance(best_cfgs.get(symbol), dict) else {}
        if isinstance(sym_dict, dict) and timeframe in sym_dict:
            tf_cfg = {**tf_cfg, **sym_dict.get(timeframe, {})}

    merged = {**defaults, **tf_cfg}

    # Ensure all default keys exist
    for k, v in defaults.items():
        merged.setdefault(k, v)

    return merged


def save_best_configs(best_configs: dict):
    """
    Persist best backtest configs to disk and cache.

    Thread-safe: Uses lock to prevent race conditions during cache updates.

    Args:
        best_configs: Dictionary of {(symbol, timeframe): config} or nested format
    """
    global BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS

    # Kritik metadata alanları - underscore ile başlasa bile korunmalı
    # _oos_start_time: Backtest trade'lerini OOS dönemden filtrelemek için
    # _expected_r, _oos_expected_r: E[R] metrikleri (istatistik için)
    METADATA_TO_PRESERVE = {"_oos_start_time", "_expected_r", "_oos_expected_r", "_walk_forward_validated"}

    cleaned = {}
    for (key, cfg) in best_configs.items():
        if isinstance(key, tuple) and len(key) == 2:
            sym, tf = key
            cleaned.setdefault(sym, {})
            # Underscore ile başlamayan + kritik metadata alanlarını koru
            cleaned[sym][tf] = {
                k: v for k, v in cfg.items()
                if not str(k).startswith("_") or k in METADATA_TO_PRESERVE
            }
        elif isinstance(cfg, dict):
            cleaned[key] = cfg

    cleaned["_meta"] = {
        "strategy_signature": _strategy_signature(),
        "saved_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }

    # Convert numpy/pandas types to Python native types for JSON serialization
    def _convert_to_native(obj):
        """Convert numpy/pandas types to Python native types for JSON serialization."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: _convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_to_native(v) for v in obj]
        # pandas Timestamp, datetime, date objects -> ISO string
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # numpy scalar types (np.float64, np.int64, etc.) - check AFTER Timestamp
        elif hasattr(obj, 'item') and not hasattr(obj, 'isoformat'):
            return obj.item()
        # numpy arrays
        elif hasattr(obj, 'tolist') and not hasattr(obj, 'isoformat'):
            return obj.tolist()
        # pandas NaT (Not a Time) -> None
        elif str(type(obj).__name__) == 'NaTType':
            return None
        return obj

    cleaned = _convert_to_native(cleaned)

    with _BEST_CONFIG_LOCK:
        BEST_CONFIG_CACHE.clear()
        BEST_CONFIG_CACHE.update(cleaned)
        # Reset all flags in-place instead of reassigning
        BEST_CONFIG_WARNING_FLAGS["missing_signature"] = False
        BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = False
        BEST_CONFIG_WARNING_FLAGS["json_error"] = False
        BEST_CONFIG_WARNING_FLAGS["load_error"] = False

        try:
            with open(BEST_CONFIGS_FILE, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)
            _logger.info("Best configs saved: %s", BEST_CONFIGS_FILE)
        except Exception as e:
            _logger.error("Config save error: %s", e)


def invalidate_config_cache():
    """Clear the config cache to force reload from file.

    Thread-safe: Uses lock to prevent race conditions during cache clear.
    """
    global BEST_CONFIG_CACHE
    with _BEST_CONFIG_LOCK:
        BEST_CONFIG_CACHE.clear()
