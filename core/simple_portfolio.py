"""
Simple Portfolio Backtest System

Ultra-conservative settings for $1000 starting balance.
Clean, readable, no complex dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class PortfolioConfig:
    """Portfolio configuration with conservative defaults."""
    initial_balance: float = 1000.0
    risk_per_trade_pct: float = 0.01  # 1% = $10 risk
    max_portfolio_risk_pct: float = 0.03  # Max 3 concurrent trades worth of risk
    leverage: int = 10
    max_position_pct: float = 0.10  # Max 10% of balance per position

    # Costs
    slippage_pct: float = 0.0005  # 0.05%
    fee_pct: float = 0.0007  # 0.07% (maker+taker avg)

    # Drawdown limits
    daily_dd_limit: float = 0.05  # 5%
    weekly_dd_limit: float = 0.10  # 10%
    total_dd_limit: float = 0.25  # 25%

    # Compounding
    compound_mode: str = "none"  # none, full, threshold
    compound_threshold: float = 100.0  # Update base after $100 profit


@dataclass
class Trade:
    """Single trade record."""
    entry_time: str
    exit_time: str
    signal_type: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    position_size: float  # In base currency (BTC)
    position_notional: float  # In USD
    risk_amount: float
    pnl: float
    pnl_pct: float
    win: bool
    exit_reason: str
    bars_held: int
    balance_before: float
    balance_after: float


class SimplePortfolio:
    """
    Simple portfolio backtester.

    Features:
    - Risk-based position sizing
    - Slippage and fee calculation
    - Drawdown tracking
    - No cooldown (can add later)
    """

    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.reset()

    def reset(self):
        """Reset portfolio state."""
        self.balance = self.config.initial_balance
        self.base_balance = self.config.initial_balance  # For compounding
        self.peak_balance = self.config.initial_balance
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.current_day = None
        self.current_week = None
        self.stopped = False
        self.stop_reason = None

    def calculate_position_size(self, entry_price: float, sl_price: float) -> tuple:
        """
        Calculate position size based on risk.

        Returns: (position_size_base, position_notional, risk_amount)
        """
        # Risk amount in USD
        risk_amount = self.balance * self.config.risk_per_trade_pct

        # SL distance percentage
        sl_distance_pct = abs(entry_price - sl_price) / entry_price

        # Minimum SL distance (prevent huge positions on tiny SL)
        min_sl_distance = 0.002  # 0.2%
        sl_distance_pct = max(sl_distance_pct, min_sl_distance)

        # Position notional = Risk / SL%
        position_notional = risk_amount / sl_distance_pct

        # Cap at max position percentage
        max_notional = self.balance * self.config.max_position_pct * self.config.leverage
        position_notional = min(position_notional, max_notional)

        # Position size in base currency (e.g., BTC)
        position_size = position_notional / entry_price

        # Recalculate actual risk
        actual_risk = position_notional * sl_distance_pct

        return position_size, position_notional, actual_risk

    def apply_slippage(self, price: float, is_entry: bool, is_long: bool) -> float:
        """Apply slippage to price."""
        slippage = self.config.slippage_pct

        if is_entry:
            # Entry: worse price
            if is_long:
                return price * (1 + slippage)
            else:
                return price * (1 - slippage)
        else:
            # Exit: worse price
            if is_long:
                return price * (1 - slippage)
            else:
                return price * (1 + slippage)

    def calculate_pnl(self, entry_price: float, exit_price: float,
                      position_size: float, is_long: bool) -> float:
        """Calculate PnL including fees."""
        if is_long:
            gross_pnl = (exit_price - entry_price) * position_size
        else:
            gross_pnl = (entry_price - exit_price) * position_size

        # Entry and exit fees
        entry_notional = entry_price * position_size
        exit_notional = exit_price * position_size
        total_fees = (entry_notional + exit_notional) * self.config.fee_pct

        net_pnl = gross_pnl - total_fees
        return net_pnl

    def check_drawdown_limits(self) -> bool:
        """Check if drawdown limits are hit. Returns True if can continue."""
        if self.stopped:
            return False

        # Total drawdown
        total_dd = (self.peak_balance - self.balance) / self.peak_balance
        if total_dd >= self.config.total_dd_limit:
            self.stopped = True
            self.stop_reason = f"Total DD limit hit: {total_dd*100:.1f}%"
            return False

        return True

    def execute_trade(self, df, idx: int, signal_type: str,
                      entry_price: float, tp_price: float, sl_price: float,
                      min_bars: int = 1) -> Optional[Trade]:
        """
        Execute a single trade simulation.

        Args:
            df: DataFrame with OHLC data
            idx: Entry candle index
            signal_type: "LONG" or "SHORT"
            entry_price: Signal entry price
            tp_price: Take profit price
            sl_price: Stop loss price
            min_bars: Minimum bars before next trade

        Returns:
            Trade object or None if trade rejected
        """
        if self.stopped:
            return None

        if not self.check_drawdown_limits():
            return None

        is_long = signal_type == "LONG"

        # Apply entry slippage
        actual_entry = self.apply_slippage(entry_price, is_entry=True, is_long=is_long)

        # Calculate position size
        position_size, position_notional, risk_amount = self.calculate_position_size(
            actual_entry, sl_price
        )

        # Check if position is viable
        if position_notional < 10:  # Min $10 position
            return None

        entry_time = str(df.index[idx])
        balance_before = self.balance

        # Simulate trade bar by bar
        exit_price = None
        exit_reason = None
        exit_idx = None

        for j in range(idx + 1, len(df)):
            candle = df.iloc[j]
            high = float(candle["high"])
            low = float(candle["low"])

            if is_long:
                # Check SL first (conservative)
                if low <= sl_price:
                    exit_price = self.apply_slippage(sl_price, is_entry=False, is_long=True)
                    exit_reason = "SL"
                    exit_idx = j
                    break
                # Then check TP
                if high >= tp_price:
                    exit_price = self.apply_slippage(tp_price, is_entry=False, is_long=True)
                    exit_reason = "TP"
                    exit_idx = j
                    break
            else:
                # SHORT
                if high >= sl_price:
                    exit_price = self.apply_slippage(sl_price, is_entry=False, is_long=False)
                    exit_reason = "SL"
                    exit_idx = j
                    break
                if low <= tp_price:
                    exit_price = self.apply_slippage(tp_price, is_entry=False, is_long=False)
                    exit_reason = "TP"
                    exit_idx = j
                    break

        # If no exit found, close at last candle
        if exit_price is None:
            exit_idx = len(df) - 1
            last_close = float(df.iloc[-1]["close"])
            exit_price = self.apply_slippage(last_close, is_entry=False, is_long=is_long)
            exit_reason = "EOD"

        exit_time = str(df.index[exit_idx])
        bars_held = exit_idx - idx

        # Calculate PnL
        pnl = self.calculate_pnl(actual_entry, exit_price, position_size, is_long)
        pnl_pct = pnl / balance_before * 100
        win = pnl > 0

        # Update balance
        self.balance += pnl

        # Update peak for drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Create trade record
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            signal_type=signal_type,
            entry_price=actual_entry,
            exit_price=exit_price,
            sl_price=sl_price,
            tp_price=tp_price,
            position_size=position_size,
            position_notional=position_notional,
            risk_amount=risk_amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            win=win,
            exit_reason=exit_reason,
            bars_held=bars_held,
            balance_before=balance_before,
            balance_after=self.balance,
        )

        self.trades.append(trade)
        return trade

    def get_summary(self) -> Dict:
        """Get portfolio performance summary."""
        if not self.trades:
            return {
                "initial_balance": self.config.initial_balance,
                "final_balance": self.balance,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "profit_factor": 0,
                "stopped": self.stopped,
                "stop_reason": self.stop_reason,
            }

        wins = [t for t in self.trades if t.win]
        losses = [t for t in self.trades if not t.win]

        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        # Calculate max drawdown from equity curve
        equity = [self.config.initial_balance]
        for t in self.trades:
            equity.append(t.balance_after)

        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        max_dd_pct = max_dd / self.config.initial_balance * 100

        return {
            "initial_balance": self.config.initial_balance,
            "final_balance": self.balance,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.config.initial_balance * 100,
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100,
            "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl for t in losses) / len(losses) if losses else 0,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }

    def get_trades_as_dicts(self) -> List[Dict]:
        """Get trades as list of dictionaries for JSON export."""
        return [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "signal_type": t.signal_type,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl_price": t.sl_price,
                "tp_price": t.tp_price,
                "position_size": t.position_size,
                "position_notional": t.position_notional,
                "risk_amount": t.risk_amount,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "win": t.win,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "balance_before": t.balance_before,
                "balance_after": t.balance_after,
            }
            for t in self.trades
        ]


def run_portfolio_backtest(df, signals: List[Dict], config: PortfolioConfig = None) -> Dict:
    """
    Run portfolio backtest with given signals.

    Args:
        df: DataFrame with OHLC and indicator data
        signals: List of signal dicts with keys:
            - idx: candle index
            - signal_type: "LONG" or "SHORT"
            - entry: entry price
            - tp: take profit price
            - sl: stop loss price
        config: Portfolio configuration

    Returns:
        Summary dict with performance metrics and trade list
    """
    portfolio = SimplePortfolio(config)

    last_exit_idx = -5  # Min bars between trades

    for signal in signals:
        idx = signal["idx"]

        # Skip if too close to last trade
        if idx <= last_exit_idx + 5:
            continue

        trade = portfolio.execute_trade(
            df=df,
            idx=idx,
            signal_type=signal["signal_type"],
            entry_price=signal["entry"],
            tp_price=signal["tp"],
            sl_price=signal["sl"],
        )

        if trade:
            # Find exit index for spacing
            for i, t in enumerate(df.index):
                if str(t) == trade.exit_time:
                    last_exit_idx = i
                    break

    summary = portfolio.get_summary()
    summary["trades_list"] = portfolio.get_trades_as_dicts()
    summary["config"] = {
        "initial_balance": config.initial_balance if config else 1000,
        "risk_per_trade_pct": config.risk_per_trade_pct if config else 0.01,
        "leverage": config.leverage if config else 10,
    }

    return summary
