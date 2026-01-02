"""
Trade Manager module.

Provides BaseTradeManager with common logic, and two implementations:
- TradeManager: For live trading with persistence and Telegram notifications
- SimTradeManager: For backtesting without file I/O

This eliminates code duplication between live and backtest trade managers.
"""

import json
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union
from abc import ABC, abstractmethod

from .config import (
    TRADING_CONFIG, TIMEFRAMES, CONFIDENCE_RISK_MULTIPLIER,
    CSV_FILE, DATA_DIR,
    CIRCUIT_BREAKER_CONFIG, ROLLING_ER_CONFIG,
)
from .utils import (
    utcnow,  # Replacement for deprecated utcnow()
    normalize_datetime, format_time_utc, format_time_local,
    calculate_funding_cost, append_trade_event,
    apply_1m_profit_lock, apply_partial_stop_protection,
    calculate_r_multiple, derive_exit_profile_params, validate_and_adjust_sl,
)
from .logging_config import get_logger

# Correlation Management (Priority 4 - Hedge Fund Recommendation)
try:
    from .correlation_manager import CorrelationManager, CorrelationCheckResult
    HAS_CORRELATION_MANAGER = True
except ImportError:
    HAS_CORRELATION_MANAGER = False
    CorrelationManager = None

_logger = get_logger(__name__)


class BaseTradeManager(ABC):
    """
    Base class for trade management with common logic.

    This class contains all the shared functionality between
    TradeManager (live) and SimTradeManager (backtest).
    """

    def __init__(self, initial_balance: float = None):
        """
        Initialize the base trade manager.

        Args:
            initial_balance: Starting balance (default from TRADING_CONFIG)
        """
        self.open_trades: List[Dict] = []
        self.history: List[Dict] = []
        self.cooldowns: Dict[Tuple[str, str], datetime] = {}

        # Wallet management
        init_bal = float(initial_balance if initial_balance is not None
                        else TRADING_CONFIG["initial_balance"])
        self.wallet_balance = init_bal
        self.locked_margin = 0.0
        self.total_pnl = 0.0

        # Config from central settings
        self.slippage_pct = TRADING_CONFIG["slippage_rate"]
        self.sl_slippage_multiplier = TRADING_CONFIG.get("sl_slippage_multiplier", 1.0)  # P5: SL exits have higher slippage
        self.risk_per_trade_pct = TRADING_CONFIG.get("risk_per_trade_pct", 0.01)
        self.max_portfolio_risk_pct = TRADING_CONFIG.get("max_portfolio_risk_pct", 0.03)
        self.leverage = TRADING_CONFIG.get("leverage", 10)

        # R-Multiple tracking
        self.trade_r_multiples: List[float] = []

        # Trade ID counter
        self._trade_id = 1

        # ==========================================
        # CIRCUIT BREAKER TRACKING (v40.2)
        # ==========================================
        # Shared between backtest and live for consistent behavior
        # Stream-level: {(sym, tf): {"cumulative_pnl": 0, "peak_pnl": 0, "trades": 0, "r_multiples": []}}
        self._stream_pnl_tracker: Dict[Tuple[str, str], Dict] = {}
        # Killed streams: {(sym, tf): {"reason": str, "at_pnl": float, "at_trade": int}}
        self._circuit_breaker_killed: Dict[Tuple[str, str], Dict] = {}
        # Global tracking
        self._global_cumulative_pnl = 0.0
        self._global_peak_pnl = 0.0
        # Weekly tracking (v40.4)
        self._global_weekly_pnl = 0.0
        self._current_week_start: Optional[datetime] = None

        # ==========================================
        # THREAD SAFETY (v44.x - Race Condition Fix)
        # ==========================================
        # Lock to protect circuit breaker check-and-update operations
        # Prevents race condition where two threads could both pass is_stream_killed()
        # and both open trades before _update_circuit_breaker_tracking() is called
        self._circuit_breaker_lock = threading.Lock()

        # ==========================================
        # CORRELATION MANAGEMENT (v45.x - Priority 4)
        # ==========================================
        # Prevents over-concentration in highly correlated assets
        # BTC/ETH/LINK correlation = 0.85-0.95, 3 positions = 1.07 effective positions
        self._use_correlation_management = False  # BASELINE: Disabled
        self._correlation_manager = None
        if HAS_CORRELATION_MANAGER:
            self._correlation_manager = CorrelationManager(
                max_positions_same_direction=2,
                max_total_positions=4,
                high_correlation_threshold=0.80,
                position_reduction_factor=0.50,
                enable_size_reduction=True,
                enable_direction_limit=True,
            )

        # ==========================================
        # SMART RE-ENTRY SYSTEM (v46.x)
        # ==========================================
        # Post-SL recovery system for liquidity grab scenarios
        # Stores recent SL trades and allows quick re-entry if price recovers
        # Key: (symbol, timeframe) -> SL trade info
        self.last_sl_trades: Dict[Tuple[str, str], Dict] = {}
        # Statistics
        self._reentry_stats = {
            "attempts": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
        }

    def _next_id(self) -> int:
        """Generate next trade ID."""
        tid = self._trade_id
        self._trade_id += 1
        return tid

    def check_cooldown(self, symbol: str, timeframe: str, now_utc=None) -> bool:
        """
        Check if a symbol/timeframe pair is in cooldown.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "15m")
            now_utc: Current time (datetime, pd.Timestamp, or None for now)

        Returns:
            True if in cooldown (should not trade), False otherwise
        """
        key = (symbol, timeframe)

        if now_utc is None:
            now_utc = utcnow()

        # Normalize times
        now_naive = normalize_datetime(now_utc)
        if now_naive is None:
            now_naive = utcnow()

        if key not in self.cooldowns:
            return False

        expiry = normalize_datetime(self.cooldowns[key])
        if expiry is None:
            del self.cooldowns[key]
            return False

        # Update stored expiry to normalized value
        self.cooldowns[key] = expiry

        if now_naive < expiry:
            return True  # Still in cooldown

        # Cooldown expired, remove it
        del self.cooldowns[key]
        return False

    # ==========================================
    # SMART RE-ENTRY METHODS (v46.x)
    # ==========================================

    def check_quick_reentry(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        current_time,
        at_dominant: str,
        window_hours: float = 4.0,
        price_threshold_pct: float = 0.01,
        require_at_confirm: bool = True,
    ) -> bool:
        """
        Check if re-entry conditions are met after a recent SL hit.

        Conditions for re-entry:
        1. SL trade exists for this stream
        2. Time since SL <= window_hours (liquidity grabs resolve quickly)
        3. Price within price_threshold_pct of original entry
        4. AlphaTrend confirms same direction (optional)
        5. Re-entry not already used

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            current_price: Current market price
            current_time: Current time (datetime)
            at_dominant: AlphaTrend dominance ("BUYERS" or "SELLERS")
            window_hours: Time window after SL (default: 4h)
            price_threshold_pct: Max price distance from entry (default: 1.0%)
            require_at_confirm: Require AlphaTrend to confirm (default: True)

        Returns:
            True if re-entry allowed, False otherwise
        """
        key = (symbol, timeframe)

        # Check if SL trade exists
        if key not in self.last_sl_trades:
            return False

        sl_trade = self.last_sl_trades[key]

        # Check if re-entry already used
        if sl_trade.get("reentry_used", False):
            return False

        # Check time window
        exit_time = sl_trade.get("exit_time")
        if exit_time is None:
            return False

        exit_time_norm = normalize_datetime(exit_time)
        current_time_norm = normalize_datetime(current_time)

        if exit_time_norm is None or current_time_norm is None:
            return False

        time_since_exit = (current_time_norm - exit_time_norm).total_seconds() / 3600.0
        if time_since_exit > window_hours:
            # Expired - clean up
            del self.last_sl_trades[key]
            return False

        # Check price proximity
        original_entry = float(sl_trade.get("entry", 0))
        if original_entry <= 0:
            return False

        price_diff_pct = abs(current_price - original_entry) / original_entry
        if price_diff_pct > price_threshold_pct:
            return False

        # Check AlphaTrend confirmation (optional)
        if require_at_confirm:
            original_side = sl_trade.get("side", "")
            if original_side == "LONG" and at_dominant != "BUYERS":
                # Direction changed - clean up and reject
                del self.last_sl_trades[key]
                return False
            if original_side == "SHORT" and at_dominant != "SELLERS":
                # Direction changed - clean up and reject
                del self.last_sl_trades[key]
                return False

        return True

    def mark_reentry_used(self, symbol: str, timeframe: str):
        """Mark re-entry as used to prevent duplicate re-entries."""
        key = (symbol, timeframe)
        if key in self.last_sl_trades:
            self.last_sl_trades[key]["reentry_used"] = True
            self._reentry_stats["attempts"] += 1

    def update_reentry_stats(self, trade: Dict):
        """Update statistics when a re-entry trade closes."""
        if not trade.get("is_reentry", False):
            return

        pnl = float(trade.get("pnl", 0))
        self._reentry_stats["total_pnl"] += pnl

        if pnl > 0:
            self._reentry_stats["wins"] += 1
        else:
            self._reentry_stats["losses"] += 1

    def get_reentry_stats(self) -> Dict:
        """Get re-entry system statistics."""
        attempts = self._reentry_stats["attempts"]
        wins = self._reentry_stats["wins"]
        losses = self._reentry_stats["losses"]
        total_pnl = self._reentry_stats["total_pnl"]

        return {
            "reentry_attempts": attempts,
            "reentry_wins": wins,
            "reentry_losses": losses,
            "reentry_total_pnl": total_pnl,
            "reentry_win_rate": wins / attempts if attempts > 0 else 0.0,
            "reentry_avg_pnl": total_pnl / attempts if attempts > 0 else 0.0,
            "pending_reentry_count": len(self.last_sl_trades),
        }

    def _store_sl_trade(
        self,
        symbol: str,
        timeframe: str,
        trade_type: str,
        entry: float,
        sl: float,
        tp: float,
        exit_time,
    ):
        """Store SL trade details for potential re-entry."""
        key = (symbol, timeframe)
        self.last_sl_trades[key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "side": trade_type,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit_time": exit_time,
            "reentry_used": False,
        }

    def _calculate_equity(self, current_prices: dict = None) -> float:
        """
        Calculate total equity = wallet_balance + locked_margin + unrealized_pnl.

        This is the proper base for risk calculations.
        Using wallet_balance alone causes risk % to inflate as margin is locked.

        Args:
            current_prices: Dict of {symbol: price} for unrealized PnL calculation

        Returns:
            Total equity value
        """
        equity = self.wallet_balance + self.locked_margin

        # Add unrealized PnL if current prices available
        if current_prices and self.open_trades:
            for trade in self.open_trades:
                sym = trade.get("symbol")
                if sym not in current_prices:
                    continue
                current_price = current_prices[sym]
                entry = float(trade.get("entry", 0))
                size = float(trade.get("size", 0))
                trade_type = trade.get("type")
                if trade_type == "LONG":
                    unrealized = (current_price - entry) * size
                else:
                    unrealized = (entry - current_price) * size
                equity += unrealized

        return equity

    def _calculate_portfolio_risk_pct(self, wallet_balance: float = None) -> float:
        """
        Calculate portfolio risk as percentage of EQUITY.

        Uses equity instead of wallet_balance to prevent risk %
        from inflating as margin is locked.

        Returns:
            Portfolio risk as decimal (e.g., 0.05 for 5%)
        """
        equity = self._calculate_equity()
        if equity <= 0:
            return 0.0

        total_open_risk = 0.0
        for trade in self.open_trades:
            entry_price = float(trade.get("entry", 0.0))
            sl_price = float(trade.get("sl", entry_price))
            size = abs(float(trade.get("size", 0.0)))
            if entry_price <= 0 or size <= 0:
                continue
            sl_fraction = abs(entry_price - sl_price) / entry_price
            open_risk_amount = sl_fraction * size * entry_price
            total_open_risk += open_risk_amount

        return total_open_risk / equity

    def _calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        risk_amount: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate position size based on risk amount and SL distance.

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            risk_amount: Amount to risk in USD

        Returns:
            Tuple of (position_size, position_notional, required_margin, risk_amount)
            Risk amount may be adjusted if margin is insufficient.
        """
        # FIX: Guard against invalid entry price
        if entry_price <= 0:
            return 0, 0, 0, 0

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 0, 0, 0, 0

        sl_fraction = sl_distance / entry_price
        if sl_fraction <= 0:
            return 0, 0, 0, 0

        position_notional = risk_amount / sl_fraction
        position_size = position_notional / entry_price

        leverage = TRADING_CONFIG["leverage"]
        required_margin = position_notional / leverage

        # Scale down if margin exceeds balance
        if required_margin > self.wallet_balance:
            max_notional = self.wallet_balance * leverage
            if max_notional <= 0:
                return 0, 0, 0, 0
            position_notional = max_notional
            position_size = position_notional / entry_price
            required_margin = position_notional / leverage
            # Recalculate risk amount after scale-down
            risk_amount = sl_fraction * position_notional

        return position_size, position_notional, required_margin, risk_amount

    def _apply_slippage(self, price: float, trade_type: str, is_entry: bool = True, is_stop_loss: bool = False) -> float:
        """
        Apply slippage to price.

        Args:
            price: Raw price
            trade_type: "LONG" or "SHORT"
            is_entry: True for entry, False for exit
            is_stop_loss: True for stop loss exits (P5: higher slippage due to panic selling)

        Returns:
            Price adjusted for slippage
        """
        # P5: Stop loss exits have higher slippage (panic selling, liquidity gaps)
        slip = self.slippage_pct
        if is_stop_loss and not is_entry:
            slip = self.slippage_pct * self.sl_slippage_multiplier

        if is_entry:
            # Entry: LONG pays more, SHORT gets less
            if trade_type == "LONG":
                return price * (1 + slip)
            else:
                return price * (1 - slip)
        else:
            # Exit: LONG gets less, SHORT pays more
            if trade_type == "LONG":
                return price * (1 - slip)
            else:
                return price * (1 + slip)

    def _process_trade_update(
        self,
        trade_idx: int,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        candle_time_utc,
        pb_top: float = None,
        pb_bot: float = None,
        candle_data: dict = None,
    ) -> Optional[Dict]:
        """
        Process update for a single trade.

        This contains the core trade management logic:
        - Dynamic TP adjustment (conditional: only after partial or always)
        - Partial TP with RR-based trigger adjustment
        - Breakeven (tied to effective_partial_trigger)
        - Trailing SL
        - TP/SL hit detection

        Args:
            trade_idx: Index of trade in open_trades
            candle_high: Candle high price
            candle_low: Candle low price
            candle_close: Candle close price
            candle_time_utc: Candle time (UTC)
            pb_top: PBEMA cloud top level (optional)
            pb_bot: PBEMA cloud bottom level (optional)
            candle_data: Dict with additional candle data (AlphaTrend indicators, etc.)

        Returns:
            Closed trade dict if trade was closed, None otherwise
        """
        trade = self.open_trades[trade_idx]
        entry = float(trade["entry"])
        tp = float(trade["tp"])
        sl = float(trade["sl"])
        size = float(trade["size"])
        t_type = trade["type"]
        tf = trade["timeframe"]

        # FIX: Guard against invalid trade data (division by zero prevention)
        if entry <= 0 or size <= 0:
            _logger.error("Invalid trade data: entry=%.8f, size=%.8f", entry, size)
            return None

        # Fallback: margin yoksa doğru hesapla (size * entry / leverage, size / leverage DEĞİL)
        initial_margin = float(trade.get("margin", abs(size) * entry / TRADING_CONFIG["leverage"]))

        # Get config from trade (snapshotted at open)
        use_trailing = trade.get("use_trailing", False)
        use_partial = trade.get("use_partial", True)  # Config'den oku, default True
        use_dynamic_tp = trade.get("use_dynamic_pbema_tp", True)

        # === AŞAMA 3: RR bazlı effective_partial_trigger hesaplama ===
        # Config'den partial TP parametrelerini al
        partial_trigger = float(trade.get("partial_trigger", 0.65))
        partial_fraction = float(trade.get("partial_fraction", 0.50))
        partial_rr_adjustment = trade.get("partial_rr_adjustment", True)

        # RR değerini al (opt_rr veya entry/tp/sl'den hesapla)
        trade_rr = float(trade.get("opt_rr", 0))
        if trade_rr <= 0:
            # Fallback: entry, tp, sl'den hesapla
            tp_dist = abs(tp - entry)
            sl_dist = abs(entry - sl)
            trade_rr = tp_dist / sl_dist if sl_dist > 0 else 2.0

        # Effective partial trigger hesapla (RR'a göre dinamik)
        if partial_rr_adjustment:
            rr_high_threshold = float(trade.get("partial_rr_high_threshold", 1.8))
            rr_high_trigger = float(trade.get("partial_rr_high_trigger", 0.75))
            rr_low_threshold = float(trade.get("partial_rr_low_threshold", 1.2))
            rr_low_trigger = float(trade.get("partial_rr_low_trigger", 0.55))

            if trade_rr >= rr_high_threshold:
                effective_partial_trigger = rr_high_trigger
            elif trade_rr <= rr_low_threshold:
                effective_partial_trigger = rr_low_trigger
            else:
                effective_partial_trigger = partial_trigger
        else:
            effective_partial_trigger = partial_trigger

        # === AŞAMA 5: BE trigger = effective_partial_trigger ===
        effective_be_trigger = effective_partial_trigger

        # === ATR-BASED BE BUFFER (v46.x - BE Fix) ===
        # Use ATR to calculate dynamic BE buffer instead of fixed 0.2%
        # This prevents getting stopped out by normal market noise
        be_atr_multiplier = float(trade.get("be_atr_multiplier", 0.5))  # Default: 0.5x ATR
        be_min_buffer_pct = float(trade.get("be_min_buffer_pct", 0.002))  # Min: 0.2%
        be_max_buffer_pct = float(trade.get("be_max_buffer_pct", 0.01))  # Max: 1.0%

        # Get ATR from candle_data (passed from trading engine)
        atr_value = None
        if candle_data is not None:
            atr_value = candle_data.get("atr")

        # Calculate BE buffer
        if atr_value is not None and atr_value > 0 and entry > 0:
            # ATR-based buffer: atr_value * multiplier / entry (as percentage)
            atr_buffer_pct = (atr_value * be_atr_multiplier) / entry
            # Clamp between min and max
            be_buffer = max(be_min_buffer_pct, min(be_max_buffer_pct, atr_buffer_pct))
        else:
            # Fallback to minimum buffer if ATR not available
            be_buffer = be_min_buffer_pct

        # Calculate prices and progress
        if t_type == "LONG":
            close_price = candle_close
            extreme_price = candle_high
            partial_fill_price = close_price * 0.70 + extreme_price * 0.30
            in_profit = extreme_price > entry
        else:
            close_price = candle_close
            extreme_price = candle_low
            partial_fill_price = close_price * 0.70 + extreme_price * 0.30
            in_profit = extreme_price < entry

        # === AŞAMA 6: Dynamic TP koşullu ve güvenlik kontrolleri ===
        initial_tp = float(trade.get("initial_tp", tp))  # Progress hesabı için sabit referans
        dyn_tp = initial_tp  # Default: initial_tp kullan

        # Dynamic TP sadece partial sonrası mı aktif?
        dynamic_tp_only_after_partial = trade.get("dynamic_tp_only_after_partial", True)
        dynamic_tp_min_distance = float(trade.get("dynamic_tp_min_distance", 0.004))

        # === AŞAMA 5 (v42.x): Profile-dependent clamp mode ===
        # "tighten_only": Only allow TP to get tighter (closer to entry)
        # "none": Allow TP to move freely with market
        dynamic_tp_clamp_mode = trade.get("dynamic_tp_clamp_mode", "tighten_only")
        current_tp = float(trade.get("tp", initial_tp))

        # Dynamic TP hesaplama koşulu
        should_apply_dynamic_tp = use_dynamic_tp and (
            not dynamic_tp_only_after_partial or trade.get("partial_taken", False)
        )

        if should_apply_dynamic_tp and pb_top is not None and pb_bot is not None:
            try:
                candidate_dyn_tp = float(pb_bot) if t_type == "LONG" else float(pb_top)

                # Güvenlik kontrolü 1: Minimum mesafe
                dist = abs(candidate_dyn_tp - entry) / entry if entry > 0 else 0
                if dist < dynamic_tp_min_distance:
                    # Mesafe çok yakın, initial_tp kullan
                    dyn_tp = initial_tp
                # Güvenlik kontrolü 2: Yön kontrolü (kritik bug önleyici)
                elif t_type == "LONG" and candidate_dyn_tp <= entry:
                    # LONG için dyn_tp entry'nin üstünde olmalı
                    dyn_tp = initial_tp
                elif t_type == "SHORT" and candidate_dyn_tp >= entry:
                    # SHORT için dyn_tp entry'nin altında olmalı
                    dyn_tp = initial_tp
                else:
                    # === Clamp mode enforcement ===
                    if dynamic_tp_clamp_mode == "tighten_only":
                        # Tighten only: only update if TP gets closer to entry
                        if t_type == "LONG":
                            # LONG: tighter = lower TP (closer to entry)
                            if candidate_dyn_tp < current_tp:
                                dyn_tp = candidate_dyn_tp
                                self.open_trades[trade_idx]["tp"] = dyn_tp
                            else:
                                dyn_tp = current_tp  # Keep current tighter TP
                        else:
                            # SHORT: tighter = higher TP (closer to entry)
                            if candidate_dyn_tp > current_tp:
                                dyn_tp = candidate_dyn_tp
                                self.open_trades[trade_idx]["tp"] = dyn_tp
                            else:
                                dyn_tp = current_tp  # Keep current tighter TP
                    else:
                        # Clamp mode "none": allow TP to move freely
                        dyn_tp = candidate_dyn_tp
                        self.open_trades[trade_idx]["tp"] = dyn_tp
            except Exception as e:
                _logger.warning(
                    "Dynamic TP calculation failed for trade %s: %s. Using initial_tp=%.2f",
                    trade.get("id", "unknown"), e, initial_tp, exc_info=True
                )
                dyn_tp = initial_tp
        else:
            # Dynamic TP kapalı veya partial öncesi, initial_tp kullan
            dyn_tp = initial_tp

        # Update live PnL
        if t_type == "LONG":
            live_pnl = (close_price - entry) * size
        else:
            live_pnl = (entry - close_price) * size
        self.open_trades[trade_idx]["pnl"] = live_pnl

        # Calculate progress towards target
        # KRITIK: Progress için initial_tp kullan, dinamik TP değil!
        total_dist = abs(initial_tp - entry)
        if total_dist <= 0:
            return None
        current_dist = abs(extreme_price - entry)
        progress = current_dist / total_dist

        # === MOMENTUM TP EXTENSION (v1.5 - EMA15 based) ===
        # When price reaches high progress with strong momentum, extend TP to let winners run
        # v1.5: Using EMA15 instead of AlphaTrend - faster response, lower lag (~7.5 candles vs ~14+)
        use_momentum_tp = trade.get("momentum_tp_extension", False)
        momentum_threshold = float(trade.get("momentum_extension_threshold", 0.80))
        momentum_multiplier = float(trade.get("momentum_extension_multiplier", 1.5))

        if (use_momentum_tp and
            in_profit and
            progress >= momentum_threshold and
            not trade.get("tp_extended", False)):

            # v1.5: Check momentum using EMA15 position
            # For LONG: price > EMA15 = bullish momentum still active
            # For SHORT: price < EMA15 = bearish momentum still active
            ema15 = candle_data.get("ema15") if candle_data else None

            if ema15 is not None:
                if t_type == "LONG":
                    momentum_strong = close_price > ema15
                else:
                    momentum_strong = close_price < ema15
            else:
                # Fallback: at 80% progress if we got this far, momentum was strong
                momentum_strong = True

            if momentum_strong:
                # Extend TP by multiplier
                original_tp_dist = abs(initial_tp - entry)
                extended_dist = original_tp_dist * momentum_multiplier

                if t_type == "LONG":
                    new_tp = entry + extended_dist
                else:
                    new_tp = entry - extended_dist

                self.open_trades[trade_idx]["tp"] = new_tp
                self.open_trades[trade_idx]["tp_extended"] = True
                self.open_trades[trade_idx]["original_tp"] = initial_tp
                dyn_tp = new_tp  # Update dyn_tp for this iteration

                append_trade_event(self.open_trades[trade_idx], "TP_EXTENDED", candle_time_utc, new_tp)

        # === AŞAMA 2, 4: Partial TP + Breakeven (düzeltilmiş sıralama ve fraction) ===
        # === Stage 3: Profile-based partial enforcement ===
        # CLIP profile: FORCED partial (guarantees profit capture)
        # RUNNER profile: Optional partial (allows runners to full TP)
        exit_profile = trade.get("exit_profile", "clip")

        # === PROGRESSIVE PARTIAL TP (v1.3) ===
        use_progressive_partial = trade.get("use_progressive_partial", False)
        partial_tranches = trade.get("partial_tranches", [])
        partials_taken_count = int(trade.get("partials_taken_count", 0))
        progressive_be_after = int(trade.get("progressive_be_after_tranche", 0))

        if use_progressive_partial and partial_tranches and partials_taken_count < len(partial_tranches):
            # Progressive partial mode - check current tranche
            current_tranche = partial_tranches[partials_taken_count]
            tranche_trigger = float(current_tranche.get("trigger", 0.5))
            tranche_fraction = float(current_tranche.get("fraction", 0.33))

            should_take_partial = (
                in_profit and
                progress >= tranche_trigger and
                (exit_profile == "clip" or use_partial)
            )

            if should_take_partial:
                # Calculate size to close for this tranche
                current_size = float(self.open_trades[trade_idx]["size"])
                closed_size = current_size * tranche_fraction
                remaining_size = current_size - closed_size

                # Partial fill with slippage
                partial_fill = self._apply_slippage(partial_fill_price, t_type, is_entry=False)

                if t_type == "LONG":
                    partial_pnl_percent = (partial_fill - entry) / entry
                else:
                    partial_pnl_percent = (entry - partial_fill) / entry

                # PnL calculation
                partial_pnl = partial_pnl_percent * (entry * closed_size)
                partial_notional = abs(closed_size) * abs(partial_fill)
                commission = partial_notional * TRADING_CONFIG["total_fee"]
                net_partial_pnl = partial_pnl - commission

                # Margin calculations (proportional to closed size)
                current_margin = float(self.open_trades[trade_idx].get("margin", initial_margin))
                margin_release = current_margin * tranche_fraction
                remaining_margin = current_margin - margin_release

                # R-Multiple for partial
                trade_risk_amount = float(trade.get("risk_amount", 0))
                current_risk = float(self.open_trades[trade_idx].get("risk_amount", trade_risk_amount))
                partial_risk = current_risk * tranche_fraction
                remaining_risk = current_risk - partial_risk

                # Store original risk for final R-multiple calculation (BUG FIX v1.8.3)
                if "original_risk_amount" not in self.open_trades[trade_idx]:
                    self.open_trades[trade_idx]["original_risk_amount"] = trade_risk_amount
                if partial_risk > 0:
                    partial_r_multiple = net_partial_pnl / partial_risk
                    self.trade_r_multiples.append(partial_r_multiple)
                else:
                    partial_r_multiple = 0.0

                # Update wallet
                self.wallet_balance += margin_release + net_partial_pnl
                self.locked_margin -= margin_release
                self.total_pnl += net_partial_pnl

                # Update trade state
                self.open_trades[trade_idx]["partials_taken_count"] = partials_taken_count + 1
                self.open_trades[trade_idx]["partial_taken"] = True  # Backward compat
                self.open_trades[trade_idx]["partial_price"] = float(partial_fill)

                # Accumulate partial PnL for final trade record (BUG FIX v1.8.3)
                current_partial_total = float(self.open_trades[trade_idx].get("partial_pnl_total", 0.0))
                self.open_trades[trade_idx]["partial_pnl_total"] = current_partial_total + net_partial_pnl

                # Move to BE after specified tranche (using ATR-based buffer)
                if partials_taken_count >= progressive_be_after and not trade.get("breakeven"):
                    # be_buffer already calculated above using ATR
                    if t_type == "LONG":
                        be_sl = entry * (1 + be_buffer)
                    else:
                        be_sl = entry * (1 - be_buffer)
                    self.open_trades[trade_idx]["sl"] = be_sl
                    self.open_trades[trade_idx]["breakeven"] = True
                    append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

                tranche_num = partials_taken_count + 1
                append_trade_event(self.open_trades[trade_idx], f"PARTIAL_T{tranche_num}", candle_time_utc, partial_fill)

                # Create partial record
                partial_record = trade.copy()
                partial_record["size"] = closed_size
                partial_record["notional"] = partial_notional
                partial_record["pnl"] = net_partial_pnl
                partial_record["r_multiple"] = partial_r_multiple
                partial_record["status"] = f"PARTIAL T{tranche_num} ({int(tranche_fraction*100)}%)"
                partial_record["close_time_utc"] = format_time_utc(candle_time_utc)
                partial_record["close_time_local"] = format_time_local(candle_time_utc, offset_hours=3)
                partial_record["close_price"] = float(partial_fill)
                partial_record["pb_ema_top"] = pb_top
                partial_record["pb_ema_bot"] = pb_bot
                partial_record["partial_taken"] = True
                partial_record["events"] = json.dumps(self.open_trades[trade_idx].get("events", []))

                self._on_partial_tp(partial_record)
                self.history.append(partial_record)

                # Update remaining position
                self.open_trades[trade_idx]["size"] = remaining_size
                self.open_trades[trade_idx]["notional"] = remaining_size * entry
                self.open_trades[trade_idx]["margin"] = remaining_margin
                self.open_trades[trade_idx]["risk_amount"] = remaining_risk

        else:
            # Original single partial mode (backward compatibility)
            should_take_partial = (
                in_profit and
                (not trade.get("partial_taken")) and
                progress >= effective_partial_trigger and
                (exit_profile == "clip" or use_partial)  # CLIP always, RUNNER respects use_partial
            )

            if should_take_partial:
                # === AŞAMA 4: partial_fraction config'den oku ===
                closed_size = size * partial_fraction
                remaining_size = size - closed_size

                # Partial fill with slippage
                partial_fill = self._apply_slippage(partial_fill_price, t_type, is_entry=False)

                if t_type == "LONG":
                    partial_pnl_percent = (partial_fill - entry) / entry
                else:
                    partial_pnl_percent = (entry - partial_fill) / entry

                # PnL hesaplaması closed_size üzerinden
                partial_pnl = partial_pnl_percent * (entry * closed_size)
                partial_notional = abs(closed_size) * abs(partial_fill)
                commission = partial_notional * TRADING_CONFIG["total_fee"]
                net_partial_pnl = partial_pnl - commission

                # Margin hesaplamaları
                margin_release = initial_margin * partial_fraction
                remaining_margin = initial_margin - margin_release

                # R-Multiple for partial
                trade_risk_amount = float(trade.get("risk_amount", 0))
                partial_risk = trade_risk_amount * partial_fraction
                remaining_risk = trade_risk_amount - partial_risk

                # Store original risk for final R-multiple calculation (BUG FIX v1.8.3)
                if "original_risk_amount" not in self.open_trades[trade_idx]:
                    self.open_trades[trade_idx]["original_risk_amount"] = trade_risk_amount
                if partial_risk > 0:
                    partial_r_multiple = net_partial_pnl / partial_risk
                    self.trade_r_multiples.append(partial_r_multiple)
                else:
                    partial_r_multiple = 0.0

                # Update wallet
                self.wallet_balance += margin_release + net_partial_pnl
                self.locked_margin -= margin_release
                self.total_pnl += net_partial_pnl

                # === AŞAMA 2: ÖNCE open_trade'i güncelle, SONRA record oluştur ===
                # 1. ÖNCE open trade flaglerini güncelle
                self.open_trades[trade_idx]["partial_taken"] = True
                self.open_trades[trade_idx]["partial_price"] = float(partial_fill)

                # Accumulate partial PnL for final trade record (BUG FIX v1.8.3)
                current_partial_total = float(self.open_trades[trade_idx].get("partial_pnl_total", 0.0))
                self.open_trades[trade_idx]["partial_pnl_total"] = current_partial_total + net_partial_pnl

                # 2. ÖNCE eventleri ekle
                # Move to breakeven (using ATR-based buffer)
                # be_buffer already calculated above using ATR
                if t_type == "LONG":
                    be_sl = entry * (1 + be_buffer)
                else:
                    be_sl = entry * (1 - be_buffer)
                self.open_trades[trade_idx]["sl"] = be_sl
                self.open_trades[trade_idx]["breakeven"] = True

                append_trade_event(self.open_trades[trade_idx], "PARTIAL", candle_time_utc, partial_fill)
                append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

                # 3. SONRA partial_record oluştur (güncel state ile)
                partial_record = trade.copy()
                partial_record["size"] = closed_size
                partial_record["notional"] = partial_notional
                partial_record["pnl"] = net_partial_pnl
                partial_record["r_multiple"] = partial_r_multiple
                partial_record["status"] = f"PARTIAL TP ({int(partial_fraction*100)}%)"
                partial_record["close_time_utc"] = format_time_utc(candle_time_utc)
                partial_record["close_time_local"] = format_time_local(candle_time_utc, offset_hours=3)
                partial_record["close_price"] = float(partial_fill)
                partial_record["pb_ema_top"] = pb_top
                partial_record["pb_ema_bot"] = pb_bot
                # Events güncel olarak ekle (PARTIAL ve BE_SET eventleri dahil)
                partial_record["partial_taken"] = True
                partial_record["events"] = json.dumps(self.open_trades[trade_idx].get("events", []))

                self._on_partial_tp(partial_record)
                self.history.append(partial_record)

                # 4. Kalan pozisyon bilgilerini güncelle
                self.open_trades[trade_idx]["size"] = remaining_size
                self.open_trades[trade_idx]["notional"] = remaining_size * entry
                self.open_trades[trade_idx]["margin"] = remaining_margin
                self.open_trades[trade_idx]["risk_amount"] = remaining_risk

        # Breakeven without partial (only if partial not just taken and BE not set)
        if in_profit and (not trade.get("breakeven")) and progress >= effective_be_trigger:
            # Breakeven without partial (using ATR-based buffer)
            # be_buffer already calculated above using ATR
            if t_type == "LONG":
                be_sl = entry * (1 + be_buffer)
            else:
                be_sl = entry * (1 - be_buffer)
            self.open_trades[trade_idx]["sl"] = be_sl
            self.open_trades[trade_idx]["breakeven"] = True
            append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

        # === AŞAMA 7: Trailing BE to Partial koruması ===
        # Move BE from entry to partial price when 90% progress reached
        if (trade.get("partial_taken") and
            trade.get("breakeven") and
            progress >= 0.90 and
            not trade.get("trailing_be_to_partial")):

            partial_price = trade.get("partial_price")
            if partial_price is not None:
                current_sl = float(self.open_trades[trade_idx]["sl"])
                # be_buffer already calculated above using ATR

                if t_type == "LONG":
                    # New BE at partial price (with buffer above it)
                    new_be = partial_price * (1 + be_buffer)
                    # Only update if new BE is higher than current SL (more protective)
                    if new_be > current_sl:
                        self.open_trades[trade_idx]["sl"] = new_be
                        self.open_trades[trade_idx]["trailing_be_to_partial"] = True
                        append_trade_event(self.open_trades[trade_idx], "BE_TO_PARTIAL", candle_time_utc, new_be)
                else:
                    # SHORT: New BE at partial price (with buffer below it)
                    new_be = partial_price * (1 - be_buffer)
                    # Only update if new BE is lower than current SL (more protective)
                    if new_be < current_sl:
                        self.open_trades[trade_idx]["sl"] = new_be
                        self.open_trades[trade_idx]["trailing_be_to_partial"] = True
                        append_trade_event(self.open_trades[trade_idx], "BE_TO_PARTIAL", candle_time_utc, new_be)

        # 1m profit lock
        if apply_1m_profit_lock(self.open_trades[trade_idx], tf, t_type, entry, dyn_tp, progress):
            append_trade_event(self.open_trades[trade_idx], "PROFIT_LOCK", candle_time_utc,
                             self.open_trades[trade_idx].get("sl"))

        # Trailing SL (uses effective_be_trigger for BE activation)
        if in_profit and use_trailing:
            if (not trade.get("breakeven")) and progress >= effective_be_trigger:
                # be_buffer already calculated above using ATR
                if t_type == "LONG":
                    be_sl = entry * (1 + be_buffer)
                else:
                    be_sl = entry * (1 - be_buffer)
                self.open_trades[trade_idx]["sl"] = be_sl
                self.open_trades[trade_idx]["breakeven"] = True
                append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

            if progress >= 0.50:
                trail_buffer = total_dist * 0.40
                current_sl = float(self.open_trades[trade_idx]["sl"])
                if t_type == "LONG":
                    new_sl = close_price - trail_buffer
                    if new_sl > current_sl:
                        self.open_trades[trade_idx]["sl"] = new_sl
                        self.open_trades[trade_idx]["trailing_active"] = True
                        append_trade_event(self.open_trades[trade_idx], "TRAIL_SL", candle_time_utc, new_sl)
                else:
                    new_sl = close_price + trail_buffer
                    if new_sl < current_sl:
                        self.open_trades[trade_idx]["sl"] = new_sl
                        self.open_trades[trade_idx]["trailing_active"] = True
                        append_trade_event(self.open_trades[trade_idx], "TRAIL_SL", candle_time_utc, new_sl)

        # Stop protection
        if apply_partial_stop_protection(self.open_trades[trade_idx], tf, progress, t_type):
            append_trade_event(self.open_trades[trade_idx], "STOP_PROTECTION", candle_time_utc,
                             self.open_trades[trade_idx].get("sl"))

        # === TIME-BASED TRADE INVALIDATION (v46.x) ===
        # Exit stale trades at breakeven if no meaningful movement after X candles
        # User complaint: "no movement for a while, should have been exited at entry"
        use_time_invalidation = trade.get("use_time_invalidation", True)
        time_invalidation_candles = int(trade.get("time_invalidation_candles", 8))
        time_invalidation_min_move_pct = float(trade.get("time_invalidation_min_move_pct", 0.003))
        time_invalidation_exit_mode = trade.get("time_invalidation_exit_mode", "breakeven")

        # Increment candles counter
        candles_since_entry = int(trade.get("candles_since_entry", 0)) + 1
        self.open_trades[trade_idx]["candles_since_entry"] = candles_since_entry

        # Track max favorable move (for LONG: max high above entry, for SHORT: max low below entry)
        max_favorable_move_pct = float(trade.get("max_favorable_move_pct", 0.0))
        if t_type == "LONG":
            current_favorable_pct = max(0, (candle_high - entry) / entry)
        else:
            current_favorable_pct = max(0, (entry - candle_low) / entry)
        max_favorable_move_pct = max(max_favorable_move_pct, current_favorable_pct)
        self.open_trades[trade_idx]["max_favorable_move_pct"] = max_favorable_move_pct

        # Check invalidation conditions
        time_invalidation_triggered = False
        if (use_time_invalidation and
            candles_since_entry >= time_invalidation_candles and
            not trade.get("partial_taken", False) and  # Don't invalidate if partial already taken
            not trade.get("breakeven", False) and  # Don't invalidate if already at BE
            max_favorable_move_pct < time_invalidation_min_move_pct):  # Hasn't moved enough

            time_invalidation_triggered = True
            append_trade_event(self.open_trades[trade_idx], "TIME_INVALIDATION", candle_time_utc,
                             f"candles={candles_since_entry}, max_move={max_favorable_move_pct:.4f}")

            # Force exit at breakeven or market
            if time_invalidation_exit_mode == "breakeven":
                # Exit at entry price (breakeven)
                exit_level = entry
                reason = "INVALIDATED (Time-BE)"
            else:
                # Exit at current market price
                exit_level = candle_close
                reason = "INVALIDATED (Time-Market)"

            # Close position immediately
            current_size = float(self.open_trades[trade_idx]["size"])
            margin_release = float(self.open_trades[trade_idx].get("margin", initial_margin))
            exit_fill = self._apply_slippage(exit_level, t_type, is_entry=False)

            if t_type == "LONG":
                gross_pnl = (exit_fill - entry) * current_size
            else:
                gross_pnl = (entry - exit_fill) * current_size

            commission_notional = abs(current_size) * abs(exit_fill)
            commission = commission_notional * TRADING_CONFIG["total_fee"]
            funding_cost = calculate_funding_cost(
                open_time=trade.get("open_time_utc", ""),
                close_time=candle_time_utc,
                notional_value=abs(current_size) * entry
            )
            net_pnl = gross_pnl - commission - funding_cost

            # Add partial PnLs if any
            partial_total = float(trade.get("partial_pnl_total", 0.0))
            combined_pnl = net_pnl + partial_total

            self.wallet_balance += margin_release + combined_pnl
            self.locked_margin -= margin_release

            # Build closed trade record
            closed_trade = trade.copy()
            closed_trade.update({
                "exit": float(exit_fill),
                "pnl": combined_pnl,
                "gross_pnl": gross_pnl + partial_total,
                "net_pnl": net_pnl,
                "commission": commission,
                "funding_cost": funding_cost,
                "status": reason,
                "close_time_utc": candle_time_utc,
                "candles_held": candles_since_entry,
            })
            return closed_trade

        # Check TP/SL hit
        sl = float(self.open_trades[trade_idx]["sl"])

        if t_type == "LONG":
            hit_tp = candle_high >= dyn_tp
            hit_sl = candle_low <= sl
        else:
            hit_tp = candle_low <= dyn_tp
            hit_sl = candle_high >= sl

        if not (hit_tp or hit_sl):
            return None

        # Determine exit reason and level
        if hit_tp and hit_sl:
            reason = "STOP (BothHit)"
            exit_level = sl
        elif hit_tp:
            reason = "WIN (TP)"
            exit_level = dyn_tp
        else:
            reason = "STOP"
            exit_level = sl

        # Close position
        current_size = float(self.open_trades[trade_idx]["size"])
        margin_release = float(self.open_trades[trade_idx].get("margin", initial_margin))

        # P5: Stop loss exits have higher slippage (panic selling, liquidity gaps)
        is_stop = "STOP" in reason
        exit_fill = self._apply_slippage(exit_level, t_type, is_entry=False, is_stop_loss=is_stop)

        if t_type == "LONG":
            gross_pnl = (exit_fill - entry) * current_size
        else:
            gross_pnl = (entry - exit_fill) * current_size

        commission_notional = abs(current_size) * abs(exit_fill)
        commission = commission_notional * TRADING_CONFIG["total_fee"]

        # Funding cost based on actual time (fixed calculation)
        funding_cost = calculate_funding_cost(
            open_time=trade.get("open_time_utc", ""),
            close_time=candle_time_utc,
            notional_value=abs(current_size) * entry
        )

        final_net_pnl = gross_pnl - commission - funding_cost

        # R-Multiple calculation (BUG FIX v1.8.3 - use combined PnL and original risk)
        partial_pnl_total = float(trade.get("partial_pnl_total", 0.0))
        combined_pnl_for_r = final_net_pnl + partial_pnl_total

        # Use original risk if partial was taken, otherwise use current risk
        original_risk = float(trade.get("original_risk_amount", 0))
        current_risk = float(trade.get("risk_amount", 0))
        total_risk_for_r = original_risk if original_risk > 0 else current_risk

        if total_risk_for_r > 0:
            r_multiple = combined_pnl_for_r / total_risk_for_r
            self.trade_r_multiples.append(r_multiple)
            trade["r_multiple"] = r_multiple
        else:
            trade["r_multiple"] = 0.0

        # Update wallet
        self.wallet_balance += margin_release + final_net_pnl
        self.locked_margin -= margin_release
        self.total_pnl += final_net_pnl

        # Set cooldown on stops
        if "STOP" in reason:
            wait_minutes = 10 if tf == "1m" else (30 if tf == "5m" else 60)
            cooldown_base = normalize_datetime(candle_time_utc) or utcnow()
            self.cooldowns[(trade["symbol"], tf)] = cooldown_base + pd.Timedelta(minutes=wait_minutes)

            # Store SL trade for potential re-entry (v46.x Smart Re-Entry)
            self._store_sl_trade(
                symbol=trade["symbol"],
                timeframe=tf,
                trade_type=t_type,
                entry=entry,
                sl=sl,
                tp=float(trade.get("tp", 0)),
                exit_time=candle_time_utc,
            )

        # Check for breakeven stop
        if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
            reason = "BE"

        # Add partial PnL to final trade record (BUG FIX v1.8.3)
        partial_pnl_total = float(trade.get("partial_pnl_total", 0.0))
        combined_pnl = final_net_pnl + partial_pnl_total

        # Finalize trade
        trade["status"] = reason
        trade["pnl"] = combined_pnl
        trade["remaining_pnl"] = final_net_pnl  # For debugging
        trade["partial_pnl_total"] = partial_pnl_total  # For debugging
        trade["close_time_utc"] = format_time_utc(candle_time_utc)
        trade["close_time_local"] = format_time_local(candle_time_utc, offset_hours=3)
        trade["close_price"] = float(exit_fill)
        trade["pb_ema_top"] = pb_top
        trade["pb_ema_bot"] = pb_bot
        trade["events"] = json.dumps(trade.get("events", []))

        return trade

    def _on_partial_tp(self, partial_record: Dict):
        """Hook called when partial TP is taken. Override in subclasses."""
        pass

    def _on_trade_closed(self, trade: Dict):
        """Hook called when a trade is closed. Override in subclasses."""
        pass

    # ==========================================
    # CIRCUIT BREAKER METHODS (v40.2, weekly added v40.4)
    # ==========================================

    def _get_week_start(self, dt: datetime) -> datetime:
        """Get the Monday 00:00 UTC of the week containing dt."""
        # Monday = 0, Sunday = 6
        days_since_monday = dt.weekday()
        week_start = dt - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    def _check_and_reset_week(self, trade_time: datetime = None):
        """Check if we've entered a new week and reset weekly PnL if so.

        Args:
            trade_time: The time of the trade (for backtest). None = use current time.
        """
        if trade_time is None:
            trade_time = utcnow()

        # Normalize to naive datetime
        if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is not None:
            trade_time = trade_time.replace(tzinfo=None)

        current_week_start = self._get_week_start(trade_time)

        if self._current_week_start is None:
            # First trade - initialize week
            self._current_week_start = current_week_start
        elif current_week_start > self._current_week_start:
            # New week started - reset weekly PnL
            self._global_weekly_pnl = 0.0
            self._current_week_start = current_week_start

    def _update_circuit_breaker_tracking(self, sym: str, tf: str, pnl: float,
                                          r_multiple: float = None, trade_time: datetime = None):
        """Update circuit breaker tracking after a trade closes.

        Args:
            sym: Symbol
            tf: Timeframe
            pnl: Trade PnL (net)
            r_multiple: Trade R-multiple (optional)
            trade_time: Time of trade close (for backtest weekly tracking)
        """
        # THREAD SAFETY: Protect circuit breaker state updates
        with self._circuit_breaker_lock:
            key = (sym, tf)

            # Initialize tracker if needed
            if key not in self._stream_pnl_tracker:
                self._stream_pnl_tracker[key] = {
                    "cumulative_pnl": 0.0,
                    "peak_pnl": 0.0,
                    "trades": 0,
                    "r_multiples": [],
                    "consecutive_full_stops": 0,  # v42.x: Track consecutive full stops
                }

            tracker = self._stream_pnl_tracker[key]
            tracker["cumulative_pnl"] += pnl
            tracker["trades"] += 1
            tracker["peak_pnl"] = max(tracker["peak_pnl"], tracker["cumulative_pnl"])

            if r_multiple is not None:
                tracker["r_multiples"].append(r_multiple)

                # === v42.x: Full stop tracking ===
                # A "full stop" is R <= -0.95 (trade hit SL for nearly full loss)
                if r_multiple <= -0.95:
                    tracker["consecutive_full_stops"] = tracker.get("consecutive_full_stops", 0) + 1
                else:
                    # Any non-full-stop resets the counter
                    tracker["consecutive_full_stops"] = 0

            # Update global tracking
            self._global_cumulative_pnl += pnl
            self._global_peak_pnl = max(self._global_peak_pnl, self._global_cumulative_pnl)

            # Update weekly tracking (v40.4)
            self._check_and_reset_week(trade_time)
            self._global_weekly_pnl += pnl

    def check_stream_circuit_breaker(self, sym: str, tf: str) -> Tuple[bool, Optional[str]]:
        """Check if stream circuit breaker should trigger.

        Args:
            sym: Symbol
            tf: Timeframe

        Returns:
            (should_kill, reason) - reason is None if not triggered
        """
        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        key = (sym, tf)

        # Already killed?
        if key in self._circuit_breaker_killed:
            return True, self._circuit_breaker_killed[key].get("reason", "already_killed")

        # No tracking yet?
        if key not in self._stream_pnl_tracker:
            return False, None

        tracker = self._stream_pnl_tracker[key]

        # Minimum trades before circuit breaker activates
        min_trades = CIRCUIT_BREAKER_CONFIG.get("stream_min_trades_before_kill", 5)
        if tracker["trades"] < min_trades:
            return False, None

        # Check 1: Absolute loss limit
        max_loss = CIRCUIT_BREAKER_CONFIG.get("stream_max_loss", -200.0)
        if tracker["cumulative_pnl"] < max_loss:
            reason = f"max_loss_exceeded (PnL=${tracker['cumulative_pnl']:.2f} < ${max_loss})"
            self._kill_stream(key, reason, tracker)
            return True, reason

        # Check 2: Drawdown from peak (DOLLAR-BASED)
        max_dd_dollars = CIRCUIT_BREAKER_CONFIG.get("stream_max_drawdown_dollars", 100.0)
        if tracker["peak_pnl"] > 0:
            drawdown_dollars = tracker["peak_pnl"] - tracker["cumulative_pnl"]
            if drawdown_dollars > max_dd_dollars:
                reason = f"drawdown_exceeded (${drawdown_dollars:.2f} drop from peak ${tracker['peak_pnl']:.2f})"
                self._kill_stream(key, reason, tracker)
                return True, reason

        # Check 3: Consecutive full stops (v42.x)
        # Kill stream after N consecutive full SL hits (R <= -0.95)
        max_full_stops = CIRCUIT_BREAKER_CONFIG.get("max_full_stops", 2)
        consecutive_full_stops = tracker.get("consecutive_full_stops", 0)
        if consecutive_full_stops >= max_full_stops:
            reason = f"consecutive_full_stops ({consecutive_full_stops} >= {max_full_stops})"
            self._kill_stream(key, reason, tracker)
            return True, reason

        # Check 4: Rolling E[R] check
        if ROLLING_ER_CONFIG.get("enabled", True):
            r_multiples = tracker.get("r_multiples", [])
            min_trades_er = ROLLING_ER_CONFIG.get("min_trades_before_check", 10)

            if len(r_multiples) >= min_trades_er:
                window = ROLLING_ER_CONFIG.get("window_by_tf", {}).get(tf, 15)
                recent_r = r_multiples[-window:] if len(r_multiples) >= window else r_multiples

                if ROLLING_ER_CONFIG.get("use_confidence_band", True) and len(recent_r) >= 5:
                    import statistics
                    mean_r = statistics.mean(recent_r)
                    stdev_r = statistics.stdev(recent_r) if len(recent_r) > 1 else 0
                    factor = ROLLING_ER_CONFIG.get("confidence_band_factor", 0.5)
                    lower_bound = mean_r - (stdev_r * factor)

                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if lower_bound < kill_thresh:
                        reason = f"rolling_er_negative (E[R]={mean_r:.3f}, lower_bound={lower_bound:.3f})"
                        self._kill_stream(key, reason, tracker)
                        return True, reason
                else:
                    # Simple threshold check
                    rolling_er = sum(recent_r) / len(recent_r)
                    kill_thresh = ROLLING_ER_CONFIG.get("kill_threshold", -0.05)
                    if rolling_er < kill_thresh:
                        reason = f"rolling_er_below_threshold (E[R]={rolling_er:.3f} < {kill_thresh})"
                        self._kill_stream(key, reason, tracker)
                        return True, reason

        return False, None

    def check_global_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """Check if global circuit breaker should trigger.

        Returns:
            (should_kill, reason) - reason is None if not triggered
        """
        if not CIRCUIT_BREAKER_CONFIG.get("enabled", True):
            return False, None

        # Check daily loss limit (session-based for live)
        daily_max = CIRCUIT_BREAKER_CONFIG.get("global_daily_max_loss", -400.0)
        if self._global_cumulative_pnl < daily_max:
            return True, f"daily_loss_exceeded (${self._global_cumulative_pnl:.2f} < ${daily_max})"

        # Check weekly loss limit (v40.4)
        weekly_max = CIRCUIT_BREAKER_CONFIG.get("global_weekly_max_loss", -800.0)
        if self._global_weekly_pnl < weekly_max:
            return True, f"weekly_loss_exceeded (${self._global_weekly_pnl:.2f} < ${weekly_max})"

        # Check global drawdown (equity-based)
        initial_balance = TRADING_CONFIG.get("initial_balance", 2000.0)
        peak_equity = initial_balance + self._global_peak_pnl
        current_equity = initial_balance + self._global_cumulative_pnl
        max_dd_pct = CIRCUIT_BREAKER_CONFIG.get("global_max_drawdown_pct", 0.20)

        if peak_equity > initial_balance:  # Only check if we've had profits
            dd_pct = (peak_equity - current_equity) / peak_equity
            if dd_pct > max_dd_pct:
                return True, f"global_drawdown_exceeded ({dd_pct:.1%} > {max_dd_pct:.1%})"

        return False, None

    def _kill_stream(self, key: Tuple[str, str], reason: str, tracker: Dict):
        """Mark a stream as killed by circuit breaker."""
        self._circuit_breaker_killed[key] = {
            "reason": reason,
            "at_pnl": tracker.get("cumulative_pnl", 0),
            "at_trade": tracker.get("trades", 0),
        }

    def is_stream_killed(self, sym: str, tf: str) -> bool:
        """Check if a stream has been killed by circuit breaker.

        Use this before opening new trades to skip killed streams.
        """
        return (sym, tf) in self._circuit_breaker_killed

    def get_circuit_breaker_report(self) -> Dict:
        """Get circuit breaker status report."""
        return {
            "killed_streams": dict(self._circuit_breaker_killed),
            "stream_trackers": dict(self._stream_pnl_tracker),
            "global_pnl": self._global_cumulative_pnl,
            "global_peak": self._global_peak_pnl,
            "global_weekly_pnl": self._global_weekly_pnl,
            "current_week_start": self._current_week_start.isoformat() if self._current_week_start else None,
        }

    def reset_circuit_breaker(self, force: bool = False) -> None:
        """Reset circuit breaker state.

        For rolling window backtests, each window should start with clean
        circuit breaker state. However, in live trading, circuit breaker
        should NOT be reset automatically.

        Args:
            force: If True, reset even in live mode (USE WITH CAUTION).
                   Only use this for testing or explicit manual reset.

        Safety:
            - SimTradeManager (backtest): Always safe to reset
            - TradeManager (live): Requires force=True to prevent accidental resets
        """
        # Safety check: Prevent accidental reset in live mode
        # SimTradeManager is for backtesting, so reset is safe
        # For base class or live subclass, require force=True
        if not force and not isinstance(self, SimTradeManager):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Attempted circuit breaker reset in non-simulation mode - BLOCKED. "
                "Use force=True if this is intentional."
            )
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Resetting circuit breaker state (force=%s)", force)

        self._stream_pnl_tracker.clear()
        self._circuit_breaker_killed.clear()
        self._global_cumulative_pnl = 0.0
        self._global_peak_pnl = 0.0
        self._global_weekly_pnl = 0.0
        self._current_week_start = None

    def get_exit_profile_stats(self) -> Dict:
        """Get exit profile and SL widening statistics (v42.x).

        Returns:
            Dict with profile breakdown and SL widening counts
        """
        all_trades = self.history + self.open_trades

        # Profile breakdown
        profile_counts = {"clip": 0, "runner": 0, "unknown": 0}
        profile_pnl = {"clip": 0.0, "runner": 0.0, "unknown": 0.0}

        # SL widening stats
        sl_widened_count = 0
        sl_widened_wins = 0
        sl_widened_losses = 0

        for trade in all_trades:
            # Profile stats
            profile = trade.get("exit_profile", "unknown")
            if profile not in profile_counts:
                profile = "unknown"
            profile_counts[profile] += 1
            profile_pnl[profile] += float(trade.get("pnl", 0))

            # SL widening stats
            if trade.get("sl_widened"):
                sl_widened_count += 1
                pnl = float(trade.get("pnl", 0))
                if pnl > 0:
                    sl_widened_wins += 1
                elif pnl < 0:
                    sl_widened_losses += 1

        return {
            "profile_counts": profile_counts,
            "profile_pnl": profile_pnl,
            "sl_widened_total": sl_widened_count,
            "sl_widened_wins": sl_widened_wins,
            "sl_widened_losses": sl_widened_losses,
            "total_trades": len(all_trades),
        }

    def get_correlation_stats(self) -> Dict:
        """Get correlation management statistics (v45.x - Priority 4).

        Returns:
            Dict with correlation-reduced trade stats and portfolio summary
        """
        all_trades = self.history + self.open_trades

        # Correlation reduction stats
        corr_reduced_count = 0
        corr_reduced_wins = 0
        corr_reduced_losses = 0
        corr_reduced_pnl = 0.0
        normal_pnl = 0.0

        for trade in all_trades:
            pnl = float(trade.get("pnl", 0))
            if trade.get("correlation_reduced"):
                corr_reduced_count += 1
                corr_reduced_pnl += pnl
                if pnl > 0:
                    corr_reduced_wins += 1
                elif pnl < 0:
                    corr_reduced_losses += 1
            else:
                normal_pnl += pnl

        # Get portfolio summary from correlation manager
        portfolio_summary = {}
        if self._correlation_manager is not None:
            portfolio_summary = self._correlation_manager.get_portfolio_summary()

        return {
            "correlation_reduced_trades": corr_reduced_count,
            "correlation_reduced_wins": corr_reduced_wins,
            "correlation_reduced_losses": corr_reduced_losses,
            "correlation_reduced_pnl": corr_reduced_pnl,
            "normal_pnl": normal_pnl,
            "total_trades": len(all_trades),
            "portfolio_summary": portfolio_summary,
        }

    @abstractmethod
    def open_trade(self, trade_data: dict) -> bool:
        """
        Open a new trade.

        Args:
            trade_data: Trade signal data containing symbol, timeframe, type,
                       entry, tp, sl, etc.

        Returns:
            True if trade was opened, False otherwise
        """
        pass

    @abstractmethod
    def update_trades(
        self,
        symbol: str,
        tf: str,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        candle_time_utc=None,
        pb_top: float = None,
        pb_bot: float = None,
        candle_data: dict = None,
    ) -> List[Dict]:
        """
        Update all open trades for a symbol/timeframe.

        Args:
            symbol: Trading symbol
            tf: Timeframe
            candle_high: Current candle high
            candle_low: Current candle low
            candle_close: Current candle close
            candle_time_utc: Candle time (UTC)
            pb_top: PBEMA cloud top
            pb_bot: PBEMA cloud bottom
            candle_data: Dict with additional candle data (AlphaTrend indicators, etc.)

        Returns:
            List of closed trades
        """
        pass


class SimTradeManager(BaseTradeManager):
    """
    Trade manager for backtesting (no file I/O).

    Uses the same economic model as TradeManager but without
    persistence or Telegram notifications.
    """

    def __init__(self, initial_balance: float = None):
        super().__init__(initial_balance)

    def open_trade(self, trade_data: dict) -> bool:
        """Open a new trade for backtesting."""
        # ==========================================
        # INPUT VALIDATION (v45.x - Security Fix)
        # ==========================================
        # Validate required fields exist and have valid values
        required_fields = ["symbol", "timeframe", "type", "entry", "tp", "sl"]
        for field in required_fields:
            if field not in trade_data:
                _logger.error("Trade REJECTED: Missing required field '%s'", field)
                return False

        tf = trade_data["timeframe"]
        sym = trade_data["symbol"]
        trade_type = trade_data.get("type", "")

        # Validate trade type
        if trade_type not in ("LONG", "SHORT"):
            _logger.error("Trade REJECTED: Invalid type '%s' (must be LONG or SHORT)", trade_type)
            return False

        # Validate numeric fields
        try:
            entry_val = float(trade_data["entry"])
            tp_val = float(trade_data["tp"])
            sl_val = float(trade_data["sl"])
        except (ValueError, TypeError) as e:
            _logger.error("Trade REJECTED: Invalid numeric value: %s", e)
            return False

        # Validate positive entry price
        if entry_val <= 0:
            _logger.error("Trade REJECTED: Entry price must be positive (got %.8f)", entry_val)
            return False

        # Validate TP/SL direction
        if trade_type == "LONG":
            if tp_val <= entry_val:
                _logger.error("Trade REJECTED: LONG TP (%.8f) must be above entry (%.8f)", tp_val, entry_val)
                return False
            if sl_val >= entry_val:
                _logger.error("Trade REJECTED: LONG SL (%.8f) must be below entry (%.8f)", sl_val, entry_val)
                return False
        else:  # SHORT
            if tp_val >= entry_val:
                _logger.error("Trade REJECTED: SHORT TP (%.8f) must be below entry (%.8f)", tp_val, entry_val)
                return False
            if sl_val <= entry_val:
                _logger.error("Trade REJECTED: SHORT SL (%.8f) must be above entry (%.8f)", sl_val, entry_val)
                return False

        # ==========================================
        # ATOMIC CIRCUIT BREAKER CHECK (v44.x)
        # ==========================================
        # CRITICAL: Lock protects against race condition where two threads
        # could both pass is_stream_killed() before either marks the stream as active.
        # This ensures check-and-update is atomic in multi-threaded environments.

        # === SMART RE-ENTRY CHECK (v46.x) ===
        # Check if this is a valid re-entry after a recent SL hit
        # Re-entry bypasses normal cooldown if conditions are met
        is_reentry = False
        cooldown_ref_time = trade_data.get("open_time_utc") or utcnow()

        # Get re-entry config from trade_data or DEFAULT_STRATEGY_CONFIG
        from core.config import DEFAULT_STRATEGY_CONFIG
        use_smart_reentry = trade_data.get("use_smart_reentry", DEFAULT_STRATEGY_CONFIG.get("use_smart_reentry", False))
        reentry_window_hours = float(trade_data.get("reentry_window_hours", DEFAULT_STRATEGY_CONFIG.get("reentry_window_hours", 4.0)))
        reentry_price_threshold_pct = float(trade_data.get("reentry_price_threshold_pct", DEFAULT_STRATEGY_CONFIG.get("reentry_price_threshold_pct", 0.01)))
        reentry_require_at_confirm = trade_data.get("reentry_require_at_confirm", DEFAULT_STRATEGY_CONFIG.get("reentry_require_at_confirm", True))

        # Get current price and AlphaTrend from trade_data
        current_price = float(trade_data.get("entry", 0))
        indicators = trade_data.get("indicators_at_entry", {})
        at_dominant = indicators.get("at_dominant", trade_data.get("at_dominant", ""))

        with self._circuit_breaker_lock:
            # Circuit breaker check - defense in depth
            # This guarantees no trade opens even if signal-side check was bypassed
            if self.is_stream_killed(sym, tf):
                return False

            # === SMART RE-ENTRY: Check before cooldown ===
            # If re-entry conditions are met, bypass cooldown
            if use_smart_reentry and current_price > 0:
                is_reentry = self.check_quick_reentry(
                    symbol=sym,
                    timeframe=tf,
                    current_price=current_price,
                    current_time=cooldown_ref_time,
                    at_dominant=at_dominant,
                    window_hours=reentry_window_hours,
                    price_threshold_pct=reentry_price_threshold_pct,
                    require_at_confirm=reentry_require_at_confirm,
                )
                if is_reentry:
                    _logger.info(
                        "SMART RE-ENTRY triggered: %s %s %s - bypassing cooldown",
                        sym, tf, trade_type
                    )

            # Cooldown check (skipped if re-entry is valid)
            if not is_reentry and self.check_cooldown(sym, tf, cooldown_ref_time):
                return False

            # Check for existing open position on this stream
            if any(t.get("symbol") == sym and t.get("timeframe") == tf
                   for t in self.open_trades):
                return False

            # Mark stream as active by proceeding with trade opening
            # (The actual trade opening continues outside the lock for performance)

        # ==========================================
        # CORRELATION CHECK (v45.x - Priority 4)
        # ==========================================
        # Check if new position violates correlation rules:
        # 1. Max 2 positions in same direction (LONG or SHORT)
        # 2. Position size reduction for correlated assets
        correlation_size_multiplier = 1.0
        if (self._use_correlation_management and
            self._correlation_manager is not None and
            HAS_CORRELATION_MANAGER):

            # Build open positions dict for correlation check
            open_positions_dict = {}
            for t in self.open_trades:
                t_sym = t.get("symbol", "")
                t_dir = t.get("type", "")
                t_size = abs(float(t.get("risk_amount", 0)))  # Use risk amount for sizing
                if t_sym and t_dir:
                    open_positions_dict[t_sym] = {
                        "direction": t_dir,
                        "size": t_size,
                    }

            # Check correlation rules
            corr_result = self._correlation_manager.check_new_position(
                symbol=sym,
                direction=trade_type,
                base_position_size=1.0,  # Normalized, actual size calculated later
                open_positions=open_positions_dict,
            )

            if not corr_result.can_open:
                _logger.info(
                    "Trade BLOCKED by correlation manager: %s %s - %s",
                    sym, trade_type, corr_result.reason
                )
                return False

            # Apply size reduction for correlated assets
            correlation_size_multiplier = corr_result.size_multiplier
            if correlation_size_multiplier < 1.0:
                _logger.info(
                    "Position size reduced %.0f%% due to correlation: %s",
                    (1 - correlation_size_multiplier) * 100, corr_result.reason
                )

        setup_type = trade_data.get("setup", "Unknown")

        # Import here to avoid circular imports
        from .config_loader import load_optimized_config

        # KRITIK: Config snapshot'ı önce trade_data'dan al (sinyal üretiminde kullanılan config)
        # Bu sayede sinyal üretimi ve trade yönetimi aynı config ile yapılır
        # Fallback: trade_data'da yoksa diskten yükle (eski trade'ler için)
        config_snapshot = trade_data.get("config_snapshot") or load_optimized_config(sym, tf)
        use_trailing = config_snapshot.get("use_trailing", False)
        use_partial = config_snapshot.get("use_partial", True)  # Partial TP aktif (default True)
        use_dynamic_pbema_tp = config_snapshot.get("use_dynamic_pbema_tp", True)
        opt_rr = config_snapshot.get("rr", 3.0)
        opt_rsi = config_snapshot.get("rsi", 60)

        # === EXIT PROFILE SYSTEM ===
        # Derive effective exit params from exit_profile setting
        # This replaces individual partial_trigger/partial_fraction extraction
        trade_rr = opt_rr  # RR from config for profile adjustment
        exit_params = derive_exit_profile_params(config_snapshot, trade_rr)

        exit_profile = exit_params["exit_profile"]
        partial_trigger = exit_params["partial_trigger"]
        partial_fraction = exit_params["partial_fraction"]
        partial_rr_adjustment = config_snapshot.get("partial_rr_adjustment", True)
        partial_rr_high_threshold = config_snapshot.get("partial_rr_high_threshold", 1.8)
        partial_rr_high_trigger = exit_params["rr_high_trigger"]
        partial_rr_low_threshold = config_snapshot.get("partial_rr_low_threshold", 1.2)
        partial_rr_low_trigger = exit_params["rr_low_trigger"]

        # Dynamic TP parametreleri (profile-dependent)
        dynamic_tp_only_after_partial = exit_params["dynamic_tp_only_after_partial"]
        dynamic_tp_clamp_mode = exit_params["dynamic_tp_clamp_mode"]
        dynamic_tp_min_distance = config_snapshot.get("dynamic_tp_min_distance", 0.004)

        # === MOMENTUM TP EXTENSION (v1.2) ===
        momentum_tp_extension = config_snapshot.get("momentum_tp_extension", False)
        momentum_extension_threshold = config_snapshot.get("momentum_extension_threshold", 0.80)
        momentum_extension_multiplier = config_snapshot.get("momentum_extension_multiplier", 1.5)

        # === PROGRESSIVE PARTIAL TP (v1.3) ===
        use_progressive_partial = config_snapshot.get("use_progressive_partial", False)
        partial_tranches = config_snapshot.get("partial_tranches", [])
        progressive_be_after_tranche = config_snapshot.get("progressive_be_after_tranche", 0)

        # === ATR-BASED BE BUFFER (v46.x - BE Fix) ===
        # Dynamic BE buffer based on ATR instead of fixed 0.2%
        be_atr_multiplier = config_snapshot.get("be_atr_multiplier", 0.5)  # Default: 0.5x ATR
        be_min_buffer_pct = config_snapshot.get("be_min_buffer_pct", 0.002)  # Min: 0.2%
        be_max_buffer_pct = config_snapshot.get("be_max_buffer_pct", 0.01)  # Max: 1.0%

        # === TIME-BASED TRADE INVALIDATION (v46.x) ===
        # Exit stale trades at breakeven if no meaningful movement
        use_time_invalidation = config_snapshot.get("use_time_invalidation", True)
        time_invalidation_candles = config_snapshot.get("time_invalidation_candles", 8)
        time_invalidation_min_move_pct = config_snapshot.get("time_invalidation_min_move_pct", 0.003)
        time_invalidation_exit_mode = config_snapshot.get("time_invalidation_exit_mode", "breakeven")

        # Confidence-based risk multiplier
        # Backward compat: eski JSON'larda _confidence olabilir
        confidence_level = config_snapshot.get("confidence") or config_snapshot.get("_confidence", "high")
        risk_multiplier = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)

        # NOTE: Per-trade regime multiplier disabled (v43)
        # Testing showed ADX at signal time doesn't correlate with window-level regime losses.
        # The SSL_Flow strategy naturally fires signals in higher ADX conditions.
        # Future work: Consider window-level BTC regime filtering instead.
        # regime_multiplier = float(trade_data.get("regime_multiplier", 1.0))
        # risk_multiplier *= regime_multiplier

        if risk_multiplier <= 0:
            return False

        if self.wallet_balance < 10:
            return False

        raw_entry = float(trade_data["entry"])
        trade_type = trade_data["type"]
        real_entry = self._apply_slippage(raw_entry, trade_type, is_entry=True)
        sl_price = float(trade_data["sl"])

        # Check portfolio risk
        effective_risk_pct = self.risk_per_trade_pct * risk_multiplier
        current_portfolio_risk = self._calculate_portfolio_risk_pct()
        if current_portfolio_risk + effective_risk_pct > self.max_portfolio_risk_pct:
            return False

        risk_amount = self.wallet_balance * effective_risk_pct

        # Apply correlation-based position size reduction (v45.x - Priority 4)
        # This reduces risk when opening positions in highly correlated assets
        if correlation_size_multiplier < 1.0:
            risk_amount *= correlation_size_multiplier

        # === VOLATILITY REGIME POSITION SIZING (Expert Panel - Sinclair) ===
        # Adjust position size based on volatility regime:
        # - LOW_VOL: 0.5x (conservative - SSL whipsaws)
        # - NORMAL_VOL: 1.0x (standard)
        # - HIGH_VOL: 1.5x (aggressive - strong trends)
        vol_regime_multiplier = float(trade_data.get("vol_position_multiplier", 1.0))
        if vol_regime_multiplier != 1.0:
            risk_amount *= vol_regime_multiplier

        tp_price = float(trade_data["tp"])

        # === MIN SL DISTANCE VALIDATION ===
        # Validate SL distance and widen if too tight (noise filter)
        # This prevents getting stopped out by market noise
        validated_sl, validated_tp, position_size, position_notional, required_margin, skip_reason = \
            validate_and_adjust_sl(
                symbol=sym,
                entry=real_entry,
                sl=sl_price,
                tp=tp_price,
                trade_type=trade_type,
                config=config_snapshot,
                risk_amount=risk_amount,
                leverage=self.leverage,
            )

        if skip_reason:
            # Handle validation failures
            if skip_reason == "SL_EQUAL_ENTRY":
                # Critical: SL equals entry - cannot size position
                # This should never happen if signal generation is correct
                self.logger.error(
                    "Trade REJECTED: SL equals Entry! Symbol=%s, TF=%s, Entry=%.8f, SL=%.8f. "
                    "Signal generation bug detected.",
                    sym, tf, real_entry, sl_price
                )
            elif skip_reason == "SL_TOO_TIGHT_REJECTED":
                # SL distance below minimum threshold and sl_validation_mode="reject"
                self.logger.warning(
                    "Trade REJECTED: SL too tight. Symbol=%s, TF=%s, Entry=%.8f, SL=%.8f",
                    sym, tf, real_entry, sl_price
                )
            return False

        if position_size <= 0:
            # Defensive: should not happen after skip_reason check
            self.logger.warning(
                "Trade REJECTED: position_size <= 0. Symbol=%s, TF=%s, size=%.8f",
                sym, tf, position_size
            )
            return False

        # === SL WIDENING TRACKING (v42.x) ===
        original_sl = float(trade_data["sl"])
        sl_was_widened = abs(validated_sl - original_sl) > 1e-8  # Float comparison tolerance

        # Use validated SL (may have been widened)
        sl_price = validated_sl

        # === TP/SL DIRECTION VALIDATION ===
        # Critical: Ensure TP and SL are on correct sides of entry
        # Without this check, trades can get stuck if TP is unreachable
        # This mirrors the dynamic TP direction check at lines 385-390
        if trade_type == "LONG":
            # LONG: TP must be above entry, SL must be below entry
            if tp_price <= real_entry:
                return False
            if sl_price >= real_entry:
                return False
        elif trade_type == "SHORT":
            # SHORT: TP must be below entry, SL must be above entry
            if tp_price >= real_entry:
                return False
            if sl_price <= real_entry:
                return False

        # Format open time
        open_time_val = trade_data.get("open_time_utc") or utcnow()
        open_time_str = format_time_utc(normalize_datetime(open_time_val))

        new_trade = {
            "id": self._next_id(),
            "symbol": sym,
            "timestamp": trade_data.get("timestamp", utcnow().strftime("%Y-%m-%d %H:%M")),
            "open_time_utc": open_time_str,
            "timeframe": tf,
            "type": trade_type,
            "setup": setup_type,
            "entry": real_entry,
            "tp": float(trade_data["tp"]),
            "initial_tp": float(trade_data["tp"]),  # Progress hesabı için sabit referans (dinamik TP değişse bile)
            "sl": sl_price,
            "size": position_size,
            "margin": required_margin,
            "notional": position_notional,
            "status": "OPEN",
            "pnl": 0.0,
            "breakeven": False,
            "trailing_active": False,
            "partial_taken": False,
            "partial_price": None,
            "trailing_be_to_partial": False,
            "has_cash": True,
            "close_time_utc": "",
            "close_time_local": "",
            "close_price": "",
            "events": [],
            "use_trailing": use_trailing,
            "use_partial": use_partial,  # Partial TP config'den
            "use_dynamic_pbema_tp": use_dynamic_pbema_tp,
            "opt_rr": opt_rr,
            "opt_rsi": opt_rsi,
            "risk_amount": risk_amount,
            "indicators_at_entry": trade_data.get("indicators_at_entry", {}),  # Indicator snapshot at entry
            # Partial TP parametreleri (AŞAMA 3-4)
            "partial_trigger": partial_trigger,
            "partial_fraction": partial_fraction,
            "partial_rr_adjustment": partial_rr_adjustment,
            "partial_rr_high_threshold": partial_rr_high_threshold,
            "partial_rr_high_trigger": partial_rr_high_trigger,
            "partial_rr_low_threshold": partial_rr_low_threshold,
            "partial_rr_low_trigger": partial_rr_low_trigger,
            # Dynamic TP parametreleri (AŞAMA 6)
            "dynamic_tp_only_after_partial": dynamic_tp_only_after_partial,
            "dynamic_tp_min_distance": dynamic_tp_min_distance,
            # Exit profile system (v42.x)
            "exit_profile": exit_profile,
            "dynamic_tp_clamp_mode": dynamic_tp_clamp_mode,
            # SL widening tracking (v42.x)
            "sl_widened": sl_was_widened,
            "original_sl": original_sl if sl_was_widened else None,
            # Correlation management tracking (v45.x - Priority 4)
            "correlation_size_multiplier": correlation_size_multiplier,
            "correlation_reduced": correlation_size_multiplier < 1.0,
            # === MOMENTUM TP EXTENSION (v1.2) ===
            "momentum_tp_extension": momentum_tp_extension,
            "momentum_extension_threshold": momentum_extension_threshold,
            "momentum_extension_multiplier": momentum_extension_multiplier,
            # === PROGRESSIVE PARTIAL TP (v1.3) ===
            "use_progressive_partial": use_progressive_partial,
            "partial_tranches": partial_tranches,
            "progressive_be_after_tranche": progressive_be_after_tranche,
            "partials_taken_count": 0,  # Track number of partials taken
            # === ATR-BASED BE BUFFER (v46.x - BE Fix) ===
            "be_atr_multiplier": be_atr_multiplier,
            "be_min_buffer_pct": be_min_buffer_pct,
            "be_max_buffer_pct": be_max_buffer_pct,
            # === TIME-BASED TRADE INVALIDATION (v46.x) ===
            "use_time_invalidation": use_time_invalidation,
            "time_invalidation_candles": time_invalidation_candles,
            "time_invalidation_min_move_pct": time_invalidation_min_move_pct,
            "time_invalidation_exit_mode": time_invalidation_exit_mode,
            "candles_since_entry": 0,  # Counter for candles since trade opened
            "max_favorable_move_pct": 0.0,  # Track max favorable move for invalidation check
            # Config snapshot for source tracking (bootstrap, carry_forward, etc.)
            "config_snapshot": config_snapshot,
            # === SMART RE-ENTRY TRACKING (v46.x) ===
            "is_reentry": is_reentry,  # True if this trade bypassed cooldown via re-entry
        }

        self.wallet_balance -= required_margin
        self.locked_margin += required_margin
        self.open_trades.append(new_trade)

        # === SMART RE-ENTRY: Mark as used (v46.x) ===
        # Prevents duplicate re-entries for the same SL trade
        if is_reentry:
            self.mark_reentry_used(sym, tf)

        # Register position with correlation manager for tracking (v45.x - Priority 4)
        if (self._use_correlation_management and
            self._correlation_manager is not None and
            HAS_CORRELATION_MANAGER):
            self._correlation_manager.register_position(
                symbol=sym,
                direction=trade_type,
                size=risk_amount,  # Use risk amount for correlation tracking
            )

        return True

    def update_trades(
        self,
        symbol: str,
        tf: str,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        candle_time_utc=None,
        pb_top: float = None,
        pb_bot: float = None,
        candle_data: dict = None,
    ) -> List[Dict]:
        """Update trades for backtesting."""
        if candle_time_utc is None:
            candle_time_utc = utcnow()

        closed_indices = []
        just_closed_trades = []

        for i, trade in enumerate(self.open_trades):
            if trade.get("symbol") != symbol or trade.get("timeframe") != tf:
                continue

            closed_trade = self._process_trade_update(
                i, candle_high, candle_low, candle_close,
                candle_time_utc, pb_top, pb_bot, candle_data
            )

            if closed_trade:
                self.history.append(closed_trade)
                just_closed_trades.append(closed_trade)
                closed_indices.append(i)

                # Update circuit breaker tracking
                self._update_circuit_breaker_tracking(
                    symbol, tf,
                    closed_trade.get("pnl", 0),
                    closed_trade.get("r_multiple")
                )

                # Update smart re-entry stats (v46.x)
                self.update_reentry_stats(closed_trade
                )

                # Close position in correlation manager (v45.x - Priority 4)
                if (self._use_correlation_management and
                    self._correlation_manager is not None and
                    HAS_CORRELATION_MANAGER):
                    closed_symbol = closed_trade.get("symbol", symbol)
                    self._correlation_manager.close_position(closed_symbol)

        # Remove closed trades
        for idx in sorted(closed_indices, reverse=True):
            del self.open_trades[idx]

        return just_closed_trades


# Note: TradeManager (live version with persistence) should be implemented
# in the main module or a separate live_trade_manager.py file,
# inheriting from BaseTradeManager and adding:
# - File persistence (save_trades, load_trades)
# - Telegram notifications
# - Thread-safe locking
# - Strategy-based wallets
