"""
Trade Visualization System for Rolling Walk-Forward Results

Creates TradingView-style candlestick charts with technical indicators
to diagnose why individual trades succeed or fail.

Features:
- Professional dark theme (TradingView style)
- SSL Baseline, PBEMA Cloud, AlphaTrend visualization
- RSI, ADX, Volume subplots
- Entry/Exit markers with TP/SL levels
- Trade zone highlighting (win/loss)
- Detailed trade info panel

Author: Claude (Anthropic)
Date: December 28, 2024
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from .binance_client import get_client
from .logging_config import get_logger

# Module logger
_logger = get_logger(__name__)

# TradingView Dark Theme
DARK_THEME = {
    "background": "#131722",
    "chart_bg": "#1e222d",
    "grid": "#363c4e",
    "text": "#d1d4dc",
    "text_secondary": "#787b86",
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "volume_up": "#26a69a40",
    "volume_down": "#ef535040",
    "tp_line": "#00c853",
    "sl_line": "#ff1744",
    "entry_long": "#00e676",
    "entry_short": "#ff5252",
    "baseline": "#2196f3",
    "pbema_fill": "#9c27b080",
    "rsi_line": "#ffeb3b",
    "adx_line": "#ff9800",
    "at_buyers": "#0022fc",  # Blue
    "at_sellers": "#fc0400",  # Red
}


class TradeVisualizer:
    """
    Professional trade visualization system.

    Generates TradingView-style charts for individual trades showing:
    - Candlestick price action
    - Technical indicators (SSL, PBEMA, AlphaTrend, RSI, ADX)
    - Entry/Exit points with TP/SL levels
    - Trade outcome (win/loss) visualization
    """

    def __init__(self, output_dir: str = "trade_charts", dpi: int = 300):
        """
        Initialize the TradeVisualizer.

        Args:
            output_dir: Directory to save chart images
            dpi: Image resolution (default: 300 for high quality)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.client = get_client()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        _logger.info(f"TradeVisualizer initialized. Output: {output_dir}")

    def _fetch_klines_by_time(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch klines for a specific time window.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (e.g., "15m")
            start_time: Start datetime
            end_time: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        url = f"{self.client.BASE_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000
        }

        try:
            import requests

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                _logger.error(f"API error {response.status_code}: {response.text}")
                return pd.DataFrame()

            data = response.json()

            if not data:
                _logger.warning(f"No data returned for {symbol} {interval}")
                return pd.DataFrame()

            # Parse to DataFrame
            df = pd.DataFrame(data).iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            _logger.error(f"Error in _fetch_klines_by_time: {e}")
            return pd.DataFrame()

    def load_trades_from_file(self, filepath: str) -> List[Dict]:
        """
        Load trades from rolling WF results JSON.

        Args:
            filepath: Path to report.json file

        Returns:
            List of trade dictionaries
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            trades = []

            # Extract trades from window_results
            if 'window_results' in data:
                for window in data['window_results']:
                    # Parse trades_detailed.txt to get trade info
                    # For now, we'll need to implement a parser
                    pass

            # For now, return empty - we'll parse from trades_detailed.txt separately
            _logger.warning("Trade parsing from report.json not yet implemented")
            _logger.info("Please use trades_detailed.txt parsing instead")

            return trades

        except Exception as e:
            _logger.error(f"Failed to load trades from {filepath}: {e}")
            return []

    def fetch_ohlcv_for_trade(
        self,
        trade: Dict,
        candles_before: int = 250,
        candles_after: int = 35
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data around trade entry/exit times.

        Args:
            trade: Trade dictionary with symbol, timeframe, open_time, close_time
            candles_before: Number of candles before entry
            candles_after: Number of candles after exit

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        symbol = trade['symbol'].split('-')[0]  # "BTCUSDT-15m" -> "BTCUSDT"
        timeframe = trade['timeframe']

        # Calculate total candles needed
        total_candles = candles_before + candles_after + 50  # Buffer

        try:
            # Parse times
            entry_time = pd.to_datetime(trade['open_time_utc'])
            exit_time = pd.to_datetime(trade['close_time_utc'])

            # Calculate window
            tf_minutes = self.timeframe_to_minutes(timeframe)
            start_time = entry_time - timedelta(minutes=tf_minutes * candles_before)
            end_time = exit_time + timedelta(minutes=tf_minutes * candles_after)

            _logger.info(f"Fetching {total_candles} candles for {symbol}-{timeframe}")
            _logger.debug(f"Time window: {start_time} to {end_time}")

            # Fetch data using time-based approach
            # We need to get historical data around the trade time
            df = self._fetch_klines_by_time(
                symbol=symbol,
                interval=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            if df.empty:
                _logger.error(f"No data fetched for {symbol}-{timeframe}")
                return pd.DataFrame()

            # Set datetime index
            if 'timestamp' not in df.index.names:
                df.set_index('timestamp', inplace=True)

            # Filter to relevant window
            mask = (df.index >= start_time) & (df.index <= end_time)
            df = df[mask].copy()

            if len(df) < 10:
                _logger.warning(f"Insufficient data after filtering: only {len(df)} candles")
                _logger.debug(f"Requested: {start_time} to {end_time}")
                if not df.empty:
                    _logger.debug(f"Got: {df.index.min()} to {df.index.max()}")

                # Try alternate approach: fetch more recent data and hope it includes our window
                _logger.info("Trying alternate fetch strategy...")
                df = self.client.get_klines(symbol, timeframe, limit=1000)

                if not df.empty:
                    if 'timestamp' not in df.index.names:
                        df.set_index('timestamp', inplace=True)

                    mask = (df.index >= start_time) & (df.index <= end_time)
                    df = df[mask].copy()

                if len(df) < 10:
                    return pd.DataFrame()

            _logger.info(f"Fetched {len(df)} candles for {symbol}-{timeframe}")
            return df

        except Exception as e:
            _logger.error(f"Error fetching OHLCV: {e}")
            import traceback
            _logger.error(traceback.format_exc())
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for chart visualization.

        Indicators:
        - HMA(60) for SSL Baseline
        - EMA(200) high/close for PBEMA Cloud
        - RSI(14)
        - ADX(14)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator columns added
        """
        try:
            import pandas_ta as ta

            # SSL Baseline (HMA 60)
            df['baseline'] = ta.hma(df['close'], length=60)

            # PBEMA Cloud (EMA 200)
            df['pb_ema_top'] = ta.ema(df['high'], length=200)
            df['pb_ema_bot'] = ta.ema(df['close'], length=200)

            # RSI(14)
            df['rsi'] = ta.rsi(df['close'], length=14)

            # ADX(14)
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_result is not None and 'ADX_14' in adx_result.columns:
                df['adx'] = adx_result['ADX_14']
            else:
                df['adx'] = 0.0

            # Fill NaN values
            df.ffill(inplace=True)
            df.bfill(inplace=True)

            return df

        except Exception as e:
            _logger.error(f"Error calculating indicators: {e}")
            return df

    def plot_trade(
        self,
        trade: Dict,
        df: pd.DataFrame,
        save_path: str = None
    ) -> str:
        """
        Generate professional chart for a single trade.

        Args:
            trade: Trade dictionary
            df: DataFrame with OHLCV + indicators
            save_path: Custom save path (optional)

        Returns:
            Path to saved chart image
        """
        if df.empty:
            _logger.error("Cannot plot empty DataFrame")
            return None

        # Create figure with subplots - MAIN CHART FOCUSED, minimal indicators
        fig = plt.figure(figsize=(20, 14), facecolor=DARK_THEME['background'])
        gs = fig.add_gridspec(2, 1, height_ratios=[10, 1], hspace=0.02)

        ax_main = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_main)

        # Apply dark theme
        for ax in [ax_main, ax_vol]:
            ax.set_facecolor(DARK_THEME['chart_bg'])
            ax.tick_params(colors=DARK_THEME['text'])
            ax.spines['bottom'].set_color(DARK_THEME['grid'])
            ax.spines['top'].set_color(DARK_THEME['grid'])
            ax.spines['left'].set_color(DARK_THEME['grid'])
            ax.spines['right'].set_color(DARK_THEME['grid'])
            ax.grid(True, alpha=0.15, color=DARK_THEME['grid'])

        # === MAIN CHART: Candlesticks + Indicators ===
        self._plot_candlesticks(ax_main, df)
        self._plot_indicators(ax_main, df, trade)  # Pass trade for PBEMA fallback
        self._plot_rr_tool(ax_main, df, trade)  # Enhanced RR Tool
        self._plot_trade_markers(ax_main, df, trade)

        # === VOLUME SUBPLOT (minimal) ===
        self._plot_volume(ax_vol, df)

        # === INFO PANEL ===
        self._add_info_panel(fig, trade)

        # Format x-axis
        ax_vol.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Save figure
        if save_path is None:
            filename = self._generate_filename(trade)
            save_path = os.path.join(self.output_dir, filename)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor=DARK_THEME['background'])
        plt.close(fig)

        _logger.info(f"Chart saved: {save_path}")
        return save_path

    def _plot_candlesticks(self, ax, df):
        """Plot candlestick chart."""
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        width = 0.6 / 24  # Adjust width for datetime
        width2 = 0.05 / 24

        # Convert datetime to matplotlib date numbers
        up_dates = mdates.date2num(up.index)
        down_dates = mdates.date2num(down.index)

        # Up candles
        ax.bar(up_dates, up['close'] - up['open'], width, bottom=up['open'],
               color=DARK_THEME['candle_up'], edgecolor=DARK_THEME['candle_up'])
        ax.bar(up_dates, up['high'] - up['close'], width2, bottom=up['close'],
               color=DARK_THEME['candle_up'])
        ax.bar(up_dates, up['low'] - up['open'], width2, bottom=up['open'],
               color=DARK_THEME['candle_up'])

        # Down candles
        ax.bar(down_dates, down['open'] - down['close'], width, bottom=down['close'],
               color=DARK_THEME['candle_down'], edgecolor=DARK_THEME['candle_down'])
        ax.bar(down_dates, down['high'] - down['open'], width2, bottom=down['open'],
               color=DARK_THEME['candle_down'])
        ax.bar(down_dates, down['low'] - down['close'], width2, bottom=down['close'],
               color=DARK_THEME['candle_down'])

    def _plot_indicators(self, ax, df, trade: Dict = None):
        """Plot technical indicators on main chart - CLEAN and SIMPLE style."""
        dates = mdates.date2num(df.index)

        # === 1. SSL HYBRID BASELINE - Soft smooth cyan line ===
        if 'baseline' in df.columns:
            baseline = pd.to_numeric(df['baseline'], errors='coerce')
            if not baseline.isna().all():
                ax.plot(dates, baseline, color='#4dd0e1', linewidth=3,
                       label='SSL Hybrid', alpha=0.8, zorder=5, solid_capstyle='round')

        # === 2. PBEMA CLOUD - Soft purple cloud with fill ===
        if 'pb_ema_top' in df.columns and 'pb_ema_bot' in df.columns:
            pb_top = pd.to_numeric(df['pb_ema_top'], errors='coerce')
            pb_bot = pd.to_numeric(df['pb_ema_bot'], errors='coerce')
            if not pb_top.isna().all() and not pb_bot.isna().all():
                # Soft purple cloud - subtle fill
                ax.fill_between(dates, pb_top, pb_bot,
                               color='#ce93d8', alpha=0.2, label='PBEMA Cloud', zorder=2)
                ax.plot(dates, pb_top, color='#ce93d8', linewidth=1, alpha=0.5, zorder=3)
                ax.plot(dates, pb_bot, color='#ce93d8', linewidth=1, alpha=0.5, zorder=3)

        # === 3. ALPHATREND - Blue (buyers) and Red (sellers) lines ===
        try:
            from .indicators import calculate_alphatrend
            df_at = calculate_alphatrend(df.copy(), timeframe='15m')

            if 'at_buyers' in df_at.columns and 'at_sellers' in df_at.columns:
                at_buyers = pd.to_numeric(df_at['at_buyers'], errors='coerce')
                at_sellers = pd.to_numeric(df_at['at_sellers'], errors='coerce')

                # Blue line for buyers (current alphatrend)
                ax.plot(dates, at_buyers, color='#2196f3', linewidth=1.2,
                       label='AT Buyers', alpha=0.9, zorder=4)
                # Red line for sellers (alphatrend shifted by 2)
                ax.plot(dates, at_sellers, color='#f44336', linewidth=1.2,
                       label='AT Sellers', alpha=0.9, zorder=4)
        except Exception as e:
            _logger.debug(f"AlphaTrend calculation skipped: {e}")

        ax.legend(loc='upper left', facecolor=DARK_THEME['chart_bg'],
                 edgecolor=DARK_THEME['grid'], labelcolor=DARK_THEME['text'],
                 fontsize=9, framealpha=0.8)

    def _plot_rr_tool(self, ax, df, trade):
        """
        Plot full RR Tool showing Entry, TP, SL zones with BE and Partial levels.

        This creates a TradingView-style position visualization showing:
        - Green zone: Entry to TP (profit zone)
        - Red zone: Entry to SL (risk zone)
        - BE line: Breakeven level if moved
        - Partial markers: Where partial profit was taken
        """
        entry_time = pd.to_datetime(trade['open_time_utc'])
        exit_time = pd.to_datetime(trade['close_time_utc'])
        entry_price = float(trade['entry'])
        tp = float(trade['tp'])
        sl = float(trade['sl'])
        trade_type = trade['type']

        # Get chart boundaries for full-width display
        chart_start = df.index.min()
        chart_end = df.index.max()

        # Find entry candle index
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
        if entry_idx == -1:
            return
        entry_dt = df.index[entry_idx]

        # Calculate x positions (from entry to end of chart to show full RR)
        x_start = mdates.date2num(entry_dt)
        x_end = mdates.date2num(chart_end)
        width = x_end - x_start

        # === PROFIT ZONE (Entry to TP) - Green ===
        if trade_type == 'LONG':
            # Long: TP above entry
            profit_bottom = entry_price
            profit_height = tp - entry_price
        else:
            # Short: TP below entry
            profit_bottom = tp
            profit_height = entry_price - tp

        profit_rect = Rectangle(
            (x_start, profit_bottom), width, profit_height,
            alpha=0.08, facecolor='#00c853', edgecolor='none', zorder=1
        )
        ax.add_patch(profit_rect)

        # === RISK ZONE (Entry to SL) - Red ===
        if trade_type == 'LONG':
            # Long: SL below entry
            risk_bottom = sl
            risk_height = entry_price - sl
        else:
            # Short: SL above entry
            risk_bottom = entry_price
            risk_height = sl - entry_price

        risk_rect = Rectangle(
            (x_start, risk_bottom), width, risk_height,
            alpha=0.08, facecolor='#ff1744', edgecolor='none', zorder=1
        )
        ax.add_patch(risk_rect)

        # === ENTRY LINE - White dashed ===
        ax.axhline(y=entry_price, color='#ffffff', linestyle='--',
                  linewidth=2, alpha=0.8, zorder=6)
        ax.annotate(f'ENTRY {entry_price:,.2f}',
                   xy=(x_start, entry_price),
                   xytext=(10, 0), textcoords='offset points',
                   color='#ffffff', fontsize=9, fontweight='bold',
                   va='center', zorder=10)

        # === TP LINE - Bright Green ===
        ax.axhline(y=tp, color='#00e676', linestyle='-',
                  linewidth=2.5, alpha=0.9, zorder=6)
        ax.annotate(f'TP {tp:,.2f}',
                   xy=(x_end, tp),
                   xytext=(-60, 0), textcoords='offset points',
                   color='#00e676', fontsize=9, fontweight='bold',
                   va='center', zorder=10)

        # === SL LINE - Bright Red ===
        ax.axhline(y=sl, color='#ff1744', linestyle='-',
                  linewidth=2.5, alpha=0.9, zorder=6)
        ax.annotate(f'SL {sl:,.2f}',
                   xy=(x_end, sl),
                   xytext=(-60, 0), textcoords='offset points',
                   color='#ff1744', fontsize=9, fontweight='bold',
                   va='center', zorder=10)

        # === BREAKEVEN LINE (if BE was set) ===
        breakeven = trade.get('breakeven', False)
        if breakeven and breakeven != entry_price:
            # BE is usually at entry price after partial
            be_price = entry_price  # Standard BE is at entry
            ax.axhline(y=be_price, color='#ffeb3b', linestyle=':',
                      linewidth=2, alpha=0.8, zorder=6)
            ax.annotate('BE',
                       xy=(x_start + width*0.5, be_price),
                       xytext=(0, 5), textcoords='offset points',
                       color='#ffeb3b', fontsize=8, fontweight='bold',
                       va='bottom', ha='center', zorder=10)

        # === PARTIAL PROFIT MARKERS ===
        partial_taken = trade.get('partial_taken', False)
        partial_price = trade.get('partial_price')
        if partial_taken and partial_price:
            partial_price = float(partial_price)
            # Draw partial level
            ax.axhline(y=partial_price, color='#29b6f6', linestyle='-.',
                      linewidth=1.5, alpha=0.7, zorder=6)
            ax.annotate(f'PARTIAL {partial_price:,.2f}',
                       xy=(x_start + width*0.3, partial_price),
                       xytext=(0, -15), textcoords='offset points',
                       color='#29b6f6', fontsize=8,
                       va='top', ha='center', zorder=10)

        # === RR RATIO LABEL ===
        r_multiple = trade.get('r_multiple', 0)
        if r_multiple:
            rr_color = '#00e676' if r_multiple > 0 else '#ff1744'
            ax.annotate(f'R: {r_multiple:+.2f}',
                       xy=(x_start, tp if trade_type == 'LONG' else sl),
                       xytext=(10, 10 if trade_type == 'LONG' else -10),
                       textcoords='offset points',
                       color=rr_color, fontsize=11, fontweight='bold',
                       va='bottom' if trade_type == 'LONG' else 'top', zorder=10)

    def _plot_trade_markers(self, ax, df, trade):
        """Plot entry/exit markers only (TP/SL drawn by _plot_rr_tool)."""
        entry_time = pd.to_datetime(trade['open_time_utc'])
        exit_time = pd.to_datetime(trade['close_time_utc'])
        entry_price = float(trade['entry'])
        exit_price = float(trade['close_price'])
        trade_type = trade['type']

        # Find closest candles
        entry_idx = df.index.get_indexer([entry_time], method='nearest')[0]
        exit_idx = df.index.get_indexer([exit_time], method='nearest')[0]

        if entry_idx == -1 or exit_idx == -1:
            _logger.warning("Could not find entry/exit candles in data")
            return

        entry_dt = df.index[entry_idx]
        exit_dt = df.index[exit_idx]

        # Entry marker - Large triangle
        if trade_type == 'LONG':
            ax.scatter(entry_dt, entry_price, s=500, marker='^',
                      color='#00e676', edgecolors='white',
                      linewidths=2.5, zorder=15)
        else:
            ax.scatter(entry_dt, entry_price, s=500, marker='v',
                      color='#ff5252', edgecolors='white',
                      linewidths=2.5, zorder=15)

        # Exit marker - Based on result
        status = trade.get('status', '')
        if 'WON' in status or 'PARTIAL' in status:
            marker_style = '*'
            marker_color = '#00e676'
            marker_size = 600
        else:
            marker_style = 'X'
            marker_color = '#ff1744'
            marker_size = 500

        ax.scatter(exit_dt, exit_price, s=marker_size, marker=marker_style,
                  color=marker_color, edgecolors='white',
                  linewidths=2.5, zorder=15)

        # Draw vertical line from entry to exit for trade duration
        ax.plot([mdates.date2num(entry_dt), mdates.date2num(exit_dt)],
               [entry_price, exit_price],
               color='#ffffff', linestyle=':', linewidth=1, alpha=0.4, zorder=4)

    def _plot_rsi(self, ax, df):
        """Plot RSI indicator."""
        if 'rsi' not in df.columns:
            return

        dates = mdates.date2num(df.index)

        ax.plot(dates, df['rsi'], color=DARK_THEME['rsi_line'],
               linewidth=1.5, label='RSI(14)')
        ax.axhline(y=70, color=DARK_THEME['sl_line'], linestyle=':',
                  linewidth=1, alpha=0.5)
        ax.axhline(y=30, color=DARK_THEME['tp_line'], linestyle=':',
                  linewidth=1, alpha=0.5)
        ax.fill_between(dates, 30, 70, alpha=0.05,
                       color=DARK_THEME['text_secondary'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', color=DARK_THEME['text'])
        ax.legend(loc='upper left', facecolor=DARK_THEME['chart_bg'],
                 edgecolor=DARK_THEME['grid'], labelcolor=DARK_THEME['text'])

    def _plot_adx(self, ax, df):
        """Plot ADX indicator."""
        if 'adx' not in df.columns:
            return

        dates = mdates.date2num(df.index)

        ax.plot(dates, df['adx'], color=DARK_THEME['adx_line'],
               linewidth=1.5, label='ADX(14)')
        ax.axhline(y=25, color=DARK_THEME['text_secondary'],
                  linestyle=':', linewidth=1, alpha=0.5)
        ax.set_ylim(0, 60)
        ax.set_ylabel('ADX', color=DARK_THEME['text'])
        ax.legend(loc='upper left', facecolor=DARK_THEME['chart_bg'],
                 edgecolor=DARK_THEME['grid'], labelcolor=DARK_THEME['text'])

    def _plot_volume(self, ax, df):
        """Plot volume bars."""
        dates = mdates.date2num(df.index)

        colors = [DARK_THEME['volume_up'] if close >= open_ else DARK_THEME['volume_down']
                 for open_, close in zip(df['open'], df['close'])]

        ax.bar(dates, df['volume'], color=colors, width=0.6/24)
        ax.set_ylabel('Volume', color=DARK_THEME['text'])
        ax.ticklabel_format(style='plain', axis='y')

    def _add_info_panel(self, fig, trade):
        """Add detailed trade information panel to chart."""
        # Calculate RR ratio
        entry = float(trade['entry'])
        tp = float(trade['tp'])
        sl = float(trade['sl'])
        trade_type = trade['type']

        if trade_type == 'LONG':
            reward = tp - entry
            risk = entry - sl
        else:
            reward = entry - tp
            risk = sl - entry

        rr_ratio = reward / risk if risk > 0 else 0

        # Get indicator values at entry
        indicators = trade.get('indicators_at_entry', {})
        rsi_val = indicators.get('rsi', 'N/A')
        adx_val = indicators.get('adx', 'N/A')
        at_dominant = indicators.get('at_dominant', 'N/A')

        if isinstance(rsi_val, float):
            rsi_val = f"{rsi_val:.1f}"
        if isinstance(adx_val, float):
            adx_val = f"{adx_val:.1f}"

        # Partial/BE info
        partial_info = ""
        if trade.get('partial_taken'):
            partial_info = f" | PARTIAL @ {self.format_price(trade.get('partial_price', 0))}"
        if trade.get('breakeven'):
            partial_info += " | BE SET"

        # Status color indicator
        status = trade.get('status', 'UNKNOWN')
        pnl = float(trade['pnl'])
        result_emoji = "✓" if pnl > 0 else "✗"

        info_text = f"""
{trade['symbol']} | {trade['timeframe']} | {trade['type']} {result_emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Entry: {str(trade['open_time_utc'])[:19]} @ {self.format_price(entry)}
Exit:  {str(trade['close_time_utc'])[:19]} @ {self.format_price(trade['close_price'])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TP: {self.format_price(tp)} | SL: {self.format_price(sl)} | RR: {rr_ratio:.1f}:1
Result: ${pnl:+.2f} | R-Multiple: {trade['r_multiple']:+.2f}x
Status: {status}{partial_info}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RSI: {rsi_val} | ADX: {adx_val} | AT: {at_dominant}
        """.strip()

        # Color based on result
        text_color = '#00e676' if pnl > 0 else '#ff5252'

        fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
                fontsize=11, verticalalignment='top',
                color=DARK_THEME['text'], family='monospace',
                bbox=dict(boxstyle='round', facecolor=DARK_THEME['chart_bg'],
                         alpha=0.9, edgecolor=text_color, linewidth=2))

    def _generate_filename(self, trade: Dict) -> str:
        """Generate filename for trade chart."""
        symbol = trade['symbol'].replace('-', '_')
        trade_type = trade['type']
        result = 'WIN' if trade['status'] == 'WON' else 'LOSS'
        timestamp = trade['open_time_utc'].replace(':', '').replace('-', '')[:13]

        return f"{symbol}_{timestamp}_{trade_type}_{result}.png"

    def visualize_trade(
        self,
        trade: Dict,
        candles_before: int = 250,
        candles_after: int = 35
    ) -> str:
        """
        High-level method to visualize a single trade.

        Args:
            trade: Trade dictionary
            candles_before: Candles before entry
            candles_after: Candles after exit

        Returns:
            Path to saved chart
        """
        # Fetch data
        df = self.fetch_ohlcv_for_trade(trade, candles_before, candles_after)

        if df.empty:
            _logger.error(f"Cannot visualize trade: no data")
            return None

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Generate chart
        chart_path = self.plot_trade(trade, df)

        return chart_path

    def visualize_all_trades(self, trades: List[Dict]) -> List[str]:
        """
        Batch visualize all trades.

        Args:
            trades: List of trade dictionaries

        Returns:
            List of saved chart paths
        """
        chart_paths = []
        total = len(trades)

        _logger.info(f"Visualizing {total} trades...")

        for i, trade in enumerate(trades, 1):
            print(f"Processing trade {i}/{total}...", end='\r')

            try:
                chart_path = self.visualize_trade(trade)
                if chart_path:
                    chart_paths.append(chart_path)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                _logger.error(f"Failed to visualize trade {i}: {e}")
                continue

        print()  # New line after progress
        _logger.info(f"Successfully visualized {len(chart_paths)}/{total} trades")

        return chart_paths

    def interactive_browser(self, trades: List[Dict]):
        """
        Interactive trade browser (CLI).

        Args:
            trades: List of trade dictionaries
        """
        if not trades:
            print("No trades to display")
            return

        # Print trade summary table
        print("\n" + "="*80)
        print(f"{'#':<4} {'Symbol':<15} {'Type':<6} {'Entry Time':<20} {'PnL':<10} {'R':<6} {'Result':<8}")
        print("="*80)

        for i, trade in enumerate(trades, 1):
            pnl_str = f"${trade['pnl']:+.2f}"
            r_str = f"{trade['r_multiple']:+.2f}x"

            print(f"{i:<4} {trade['symbol']:<15} {trade['type']:<6} "
                  f"{trade['open_time_utc']:<20} {pnl_str:<10} {r_str:<6} {trade['status']:<8}")

        print("="*80)

        # Interactive loop
        while True:
            try:
                choice = input("\nEnter trade number to visualize (or 'q' to quit): ").strip()

                if choice.lower() == 'q':
                    break

                idx = int(choice) - 1

                if idx < 0 or idx >= len(trades):
                    print(f"Invalid choice. Please enter 1-{len(trades)}")
                    continue

                trade = trades[idx]
                print(f"\nVisualizing trade #{choice}...")

                chart_path = self.visualize_trade(trade)

                if chart_path:
                    print(f"Chart saved: {chart_path}")
                else:
                    print("Failed to generate chart")

            except ValueError:
                print("Invalid input. Please enter a number or 'q'")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    @staticmethod
    def format_price(price: float) -> str:
        """Format price with commas."""
        return f"{price:,.2f}"

    @staticmethod
    def timeframe_to_minutes(tf: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        return mapping.get(tf, 60)  # Default to 1h
