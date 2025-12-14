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
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union
from abc import ABC, abstractmethod

from .config import (
    TRADING_CONFIG, TIMEFRAMES, CONFIDENCE_RISK_MULTIPLIER,
    CSV_FILE, DATA_DIR,
)
from .utils import (
    normalize_datetime, format_time_utc, format_time_local,
    calculate_funding_cost, append_trade_event,
    apply_1m_profit_lock, apply_partial_stop_protection,
    calculate_r_multiple,
)


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
        self.risk_per_trade_pct = TRADING_CONFIG.get("risk_per_trade_pct", 0.01)
        self.max_portfolio_risk_pct = TRADING_CONFIG.get("max_portfolio_risk_pct", 0.03)

        # R-Multiple tracking
        self.trade_r_multiples: List[float] = []

        # Trade ID counter
        self._trade_id = 1

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
            now_utc = datetime.utcnow()

        # Normalize times
        now_naive = normalize_datetime(now_utc)
        if now_naive is None:
            now_naive = datetime.utcnow()

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

    def _apply_slippage(self, price: float, trade_type: str, is_entry: bool = True) -> float:
        """
        Apply slippage to price.

        Args:
            price: Raw price
            trade_type: "LONG" or "SHORT"
            is_entry: True for entry, False for exit

        Returns:
            Price adjusted for slippage
        """
        if is_entry:
            # Entry: LONG pays more, SHORT gets less
            if trade_type == "LONG":
                return price * (1 + self.slippage_pct)
            else:
                return price * (1 - self.slippage_pct)
        else:
            # Exit: LONG gets less, SHORT pays more
            if trade_type == "LONG":
                return price * (1 - self.slippage_pct)
            else:
                return price * (1 + self.slippage_pct)

    def _process_trade_update(
        self,
        trade_idx: int,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        candle_time_utc,
        pb_top: float = None,
        pb_bot: float = None,
    ) -> Optional[Dict]:
        """
        Process update for a single trade.

        This contains the core trade management logic:
        - Dynamic TP adjustment
        - Partial TP
        - Breakeven
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
        initial_margin = float(trade.get("margin", size / TRADING_CONFIG["leverage"]))

        # Get config from trade (snapshotted at open)
        use_trailing = trade.get("use_trailing", False)
        use_dynamic_tp = trade.get("use_dynamic_pbema_tp", True)
        use_partial = not use_trailing

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

        # Dynamic PBEMA TP
        dyn_tp = tp
        if use_dynamic_tp and pb_top is not None and pb_bot is not None:
            try:
                dyn_tp = float(pb_bot) if t_type == "LONG" else float(pb_top)
                self.open_trades[trade_idx]["tp"] = dyn_tp
            except Exception:
                dyn_tp = tp

        # Update live PnL
        if t_type == "LONG":
            live_pnl = (close_price - entry) * size
        else:
            live_pnl = (entry - close_price) * size
        self.open_trades[trade_idx]["pnl"] = live_pnl

        # Calculate progress towards target
        total_dist = abs(dyn_tp - entry)
        if total_dist <= 0:
            return None
        current_dist = abs(extreme_price - entry)
        progress = current_dist / total_dist

        # Partial TP + Breakeven
        if in_profit and use_partial:
            if (not trade.get("partial_taken")) and progress >= 0.50:
                partial_size = size / 2.0

                # Partial fill with slippage
                partial_fill = self._apply_slippage(partial_fill_price, t_type, is_entry=False)

                if t_type == "LONG":
                    partial_pnl_percent = (partial_fill - entry) / entry
                else:
                    partial_pnl_percent = (entry - partial_fill) / entry

                partial_pnl = partial_pnl_percent * (entry * partial_size)
                partial_notional = abs(partial_size) * abs(partial_fill)
                commission = partial_notional * TRADING_CONFIG["total_fee"]
                net_partial_pnl = partial_pnl - commission
                margin_release = initial_margin / 2.0

                # R-Multiple for partial
                trade_risk_amount = float(trade.get("risk_amount", 0))
                partial_risk = trade_risk_amount / 2.0
                if partial_risk > 0:
                    partial_r_multiple = net_partial_pnl / partial_risk
                    self.trade_r_multiples.append(partial_r_multiple)
                else:
                    partial_r_multiple = 0.0

                # Update wallet
                self.wallet_balance += margin_release + net_partial_pnl
                self.locked_margin -= margin_release
                self.total_pnl += net_partial_pnl

                # Record partial in history
                partial_record = trade.copy()
                partial_record["size"] = partial_size
                partial_record["notional"] = partial_notional
                partial_record["pnl"] = net_partial_pnl
                partial_record["r_multiple"] = partial_r_multiple
                partial_record["status"] = "PARTIAL TP (50%)"
                partial_record["close_time_utc"] = format_time_utc(candle_time_utc)
                partial_record["close_time_local"] = format_time_local(candle_time_utc, offset_hours=3)
                partial_record["close_price"] = float(partial_fill)
                partial_record["pb_ema_top"] = pb_top
                partial_record["pb_ema_bot"] = pb_bot
                partial_record["events"] = json.dumps(trade.get("events", []))
                self._on_partial_tp(partial_record)
                self.history.append(partial_record)

                # Update open trade
                self.open_trades[trade_idx]["size"] = partial_size
                self.open_trades[trade_idx]["notional"] = partial_notional
                self.open_trades[trade_idx]["margin"] = margin_release
                self.open_trades[trade_idx]["partial_taken"] = True
                self.open_trades[trade_idx]["partial_price"] = float(partial_fill)
                self.open_trades[trade_idx]["risk_amount"] = partial_risk

                # Move to breakeven
                be_buffer = 0.0003
                if t_type == "LONG":
                    be_sl = entry * (1 + be_buffer)
                else:
                    be_sl = entry * (1 - be_buffer)
                self.open_trades[trade_idx]["sl"] = be_sl
                self.open_trades[trade_idx]["breakeven"] = True
                append_trade_event(self.open_trades[trade_idx], "PARTIAL", candle_time_utc, partial_fill)
                append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

            elif (not trade.get("breakeven")) and progress >= 0.40:
                # Breakeven without partial
                be_buffer = 0.0003
                if t_type == "LONG":
                    be_sl = entry * (1 + be_buffer)
                else:
                    be_sl = entry * (1 - be_buffer)
                self.open_trades[trade_idx]["sl"] = be_sl
                self.open_trades[trade_idx]["breakeven"] = True
                append_trade_event(self.open_trades[trade_idx], "BE_SET", candle_time_utc, be_sl)

        # 1m profit lock
        if apply_1m_profit_lock(self.open_trades[trade_idx], tf, t_type, entry, dyn_tp, progress):
            append_trade_event(self.open_trades[trade_idx], "PROFIT_LOCK", candle_time_utc,
                             self.open_trades[trade_idx].get("sl"))

        # Trailing SL
        if in_profit and use_trailing:
            if (not trade.get("breakeven")) and progress >= 0.40:
                be_buffer = 0.0003
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

        exit_fill = self._apply_slippage(exit_level, t_type, is_entry=False)

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

        # R-Multiple
        trade_risk_amount = float(trade.get("risk_amount", 0))
        if trade_risk_amount > 0:
            r_multiple = final_net_pnl / trade_risk_amount
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
            cooldown_base = normalize_datetime(candle_time_utc) or datetime.utcnow()
            self.cooldowns[(trade["symbol"], tf)] = cooldown_base + pd.Timedelta(minutes=wait_minutes)

        # Check for breakeven stop
        if trade.get("breakeven") and abs(final_net_pnl) < 1e-6 and "STOP" in reason:
            reason = "BE"

        # Finalize trade
        trade["status"] = reason
        trade["pnl"] = final_net_pnl
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
        tf = trade_data["timeframe"]
        sym = trade_data["symbol"]

        cooldown_ref_time = trade_data.get("open_time_utc") or datetime.utcnow()
        if self.check_cooldown(sym, tf, cooldown_ref_time):
            return False

        # Check for existing open position
        if any(t.get("symbol") == sym and t.get("timeframe") == tf
               for t in self.open_trades):
            return False

        setup_type = trade_data.get("setup", "Unknown")

        # Import here to avoid circular imports
        from .config_loader import load_optimized_config

        # Snapshot config at trade open
        config_snapshot = load_optimized_config(sym, tf)
        use_trailing = config_snapshot.get("use_trailing", False)
        use_dynamic_pbema_tp = config_snapshot.get("use_dynamic_pbema_tp", True)
        opt_rr = config_snapshot.get("rr", 3.0)
        opt_rsi = config_snapshot.get("rsi", 60)

        # Confidence-based risk multiplier
        confidence_level = config_snapshot.get("_confidence", "high")
        risk_multiplier = CONFIDENCE_RISK_MULTIPLIER.get(confidence_level, 1.0)
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
        position_size, position_notional, required_margin, risk_amount = \
            self._calculate_position_size(real_entry, sl_price, risk_amount)

        if position_size <= 0:
            return False

        # Format open time
        open_time_val = trade_data.get("open_time_utc") or datetime.utcnow()
        open_time_str = format_time_utc(normalize_datetime(open_time_val))

        new_trade = {
            "id": self._next_id(),
            "symbol": sym,
            "timestamp": trade_data.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
            "open_time_utc": open_time_str,
            "timeframe": tf,
            "type": trade_type,
            "setup": setup_type,
            "entry": real_entry,
            "tp": float(trade_data["tp"]),
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
            "has_cash": True,
            "close_time_utc": "",
            "close_time_local": "",
            "close_price": "",
            "events": [],
            "use_trailing": use_trailing,
            "use_dynamic_pbema_tp": use_dynamic_pbema_tp,
            "opt_rr": opt_rr,
            "opt_rsi": opt_rsi,
            "risk_amount": risk_amount,
        }

        self.wallet_balance -= required_margin
        self.locked_margin += required_margin
        self.open_trades.append(new_trade)

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
    ) -> List[Dict]:
        """Update trades for backtesting."""
        if candle_time_utc is None:
            candle_time_utc = datetime.utcnow()

        closed_indices = []
        just_closed_trades = []

        for i, trade in enumerate(self.open_trades):
            if trade.get("symbol") != symbol or trade.get("timeframe") != tf:
                continue

            closed_trade = self._process_trade_update(
                i, candle_high, candle_low, candle_close,
                candle_time_utc, pb_top, pb_bot
            )

            if closed_trade:
                self.history.append(closed_trade)
                just_closed_trades.append(closed_trade)
                closed_indices.append(i)

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
