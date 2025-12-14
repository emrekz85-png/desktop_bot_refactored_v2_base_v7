"""
Configuration loader for trading strategies.

Handles loading and saving optimized configurations for each symbol/timeframe pair.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional

from .config import (
    BEST_CONFIGS_FILE, BEST_CONFIG_CACHE, BEST_CONFIG_WARNING_FLAGS,
    DATA_DIR,
)


# Default strategy configuration
DEFAULT_STRATEGY_CONFIG = {
    "rr": 3.0,
    "rsi": 60,
    "slope": 0.5,
    "at_active": False,
    "use_trailing": False,
    "use_dynamic_pbema_tp": True,
    "strategy_mode": "keltner_bounce",
}

# Symbol-specific parameters
# These can be overridden by optimizer results
SYMBOL_PARAMS = {
    "BTCUSDT": {
        "5m": {"rr": 2.4, "rsi": 35, "slope": 0.2, "at_active": False},
        "15m": {"rr": 2.8, "rsi": 40, "slope": 0.3, "at_active": False},
        "30m": {"rr": 3.0, "rsi": 45, "slope": 0.3, "at_active": False},
        "1h": {"rr": 3.2, "rsi": 50, "slope": 0.4, "at_active": False},
        "4h": {"rr": 3.5, "rsi": 55, "slope": 0.5, "at_active": False},
        "12h": {"rr": 3.8, "rsi": 60, "slope": 0.5, "at_active": False},
        "1d": {"rr": 4.0, "rsi": 60, "slope": 0.5, "at_active": False},
    },
    # Add more symbols as needed
}


def _strategy_signature() -> str:
    """
    Generate a signature hash for the current strategy configuration.

    This is used to detect when the strategy has changed and cached
    configs should be invalidated.
    """
    sig_data = {
        "default_config": DEFAULT_STRATEGY_CONFIG,
        "version": "v39.0",
    }
    sig_str = json.dumps(sig_data, sort_keys=True)
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]


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

    meta = best_cfgs.get("_meta", {}) if isinstance(best_cfgs.get("_meta"), dict) else {}
    stored_sig = meta.get("strategy_signature")

    if not stored_sig:
        if not BEST_CONFIG_WARNING_FLAGS.get("missing_signature", False):
            print("[CFG] Warning: No saved backtest signature found. Best configs will be ignored.")
            BEST_CONFIG_WARNING_FLAGS["missing_signature"] = True
        return False

    current_sig = _strategy_signature()
    if stored_sig != current_sig:
        if not BEST_CONFIG_WARNING_FLAGS.get("signature_mismatch", False):
            print(
                "[CFG] Warning: Backtest config signature doesn't match current strategy. "
                "Using default configs; please re-run backtest."
            )
            BEST_CONFIG_WARNING_FLAGS["signature_mismatch"] = True
        return False

    return True


def _load_best_configs() -> dict:
    """Load best configs from file, using cache if available."""
    global BEST_CONFIG_CACHE

    if BEST_CONFIG_CACHE:
        return BEST_CONFIG_CACHE

    if os.path.exists(BEST_CONFIGS_FILE):
        try:
            with open(BEST_CONFIGS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                BEST_CONFIG_CACHE.update(raw)
        except Exception:
            pass

    return BEST_CONFIG_CACHE


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
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    BEST_CONFIG_CACHE.clear()
    BEST_CONFIG_CACHE.update(cleaned)
    BEST_CONFIG_WARNING_FLAGS = {
        "missing_signature": False,
        "signature_mismatch": False,
    }

    try:
        with open(BEST_CONFIGS_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def invalidate_config_cache():
    """Clear the config cache to force reload from file."""
    global BEST_CONFIG_CACHE
    BEST_CONFIG_CACHE.clear()
