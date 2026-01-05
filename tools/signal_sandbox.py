#!/usr/bin/env python3
"""
Signal Sandbox v2 - Trade outcome'lu hızlı filtre testi

Kullanım:
    # 1. Önce sinyalleri topla (bir kere, ~3-4 dk)
    python tools/signal_sandbox.py --collect

    # 2. Farklı filtreleri test et (saniyeler içinde, WIN RATE dahil!)
    python tools/signal_sandbox.py --test "last_3_momentum < 0"
    python tools/signal_sandbox.py --test "adx > 25"

    # 3. Birden fazla filtre kombinasyonu karşılaştır
    python tools/signal_sandbox.py --compare "adx > 20" "adx > 25" "adx > 30"
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

from core.config import SYMBOLS, TRADING_CONFIG
from core.indicators import calculate_indicators
from core.binance_client import BinanceClient
from core.safe_eval import safe_eval


SANDBOX_DIR = Path(__file__).parent.parent / "data" / "signal_sandbox"
SIGNALS_FILE = SANDBOX_DIR / "all_signals_v2.json"


# ============================================================================
# TRADE OUTCOME SIMULATION
# ============================================================================

def calculate_sl_tp(df: pd.DataFrame, index: int, signal_type: str) -> Tuple[float, float, float]:
    """
    Sinyal için SL ve TP hesapla (strateji mantığına uygun).

    Returns:
        (entry, sl, tp)
    """
    row = df.iloc[index]
    close = row['close']
    baseline = row.get('baseline', close)
    pb_ema_top = row.get('pb_ema_top', close)
    pb_ema_bot = row.get('pb_ema_bot', close)

    # Entry = close price
    entry = close

    # Swing high/low for SL (son 20 bar)
    lookback = min(20, index)
    swing_high = df['high'].iloc[index - lookback:index].max()
    swing_low = df['low'].iloc[index - lookback:index].min()

    if signal_type == "SHORT":
        # SL: swing high veya baseline (hangisi daha yakınsa)
        sl_swing = swing_high * 1.002  # %0.2 buffer
        sl_baseline = baseline * 1.002
        sl = min(sl_swing, sl_baseline)  # Daha yakın olanı kullan

        # Minimum SL mesafesi (%0.5)
        min_sl = entry * 1.005
        sl = max(sl, min_sl)

        # TP: PBEMA bot
        tp = pb_ema_bot * 0.998  # %0.2 buffer

    else:  # LONG
        # SL: swing low veya baseline
        sl_swing = swing_low * 0.998
        sl_baseline = baseline * 0.998
        sl = max(sl_swing, sl_baseline)

        # Minimum SL mesafesi
        max_sl = entry * 0.995
        sl = min(sl, max_sl)

        # TP: PBEMA top
        tp = pb_ema_top * 1.002

    return entry, sl, tp


def simulate_trade_outcome(
    df: pd.DataFrame,
    index: int,
    signal_type: str,
    entry: float,
    sl: float,
    tp: float,
    max_bars: int = 100
) -> Dict:
    """
    Trade'in sonucunu simüle et - TP mi SL mi önce vuruldu?

    Args:
        df: OHLCV DataFrame
        index: Sinyal barının indexi
        signal_type: "LONG" veya "SHORT"
        entry, sl, tp: Giriş, stop loss, take profit seviyeleri
        max_bars: Maksimum bekleme süresi (bar cinsinden)

    Returns:
        {outcome, exit_price, exit_bar, bars_held, r_multiple, pnl_pct}
    """
    risk = abs(entry - sl)
    reward = abs(tp - entry)

    if risk == 0:
        return {
            "outcome": "INVALID",
            "exit_price": entry,
            "exit_bar": index,
            "bars_held": 0,
            "r_multiple": 0,
            "pnl_pct": 0,
            "hit_tp": False,
            "hit_sl": False,
            "max_favorable": 0,
            "max_adverse": 0
        }

    # İleriye doğru simüle et
    max_favorable = 0  # En iyi nokta (MAE için)
    max_adverse = 0    # En kötü nokta (MFE için)

    for i in range(index + 1, min(index + max_bars, len(df))):
        bar = df.iloc[i]
        high = bar['high']
        low = bar['low']

        if signal_type == "SHORT":
            # SHORT için: düşük fiyat iyi, yüksek fiyat kötü
            favorable_move = entry - low
            adverse_move = high - entry
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)

            # SL check (high >= sl)
            if high >= sl:
                pnl = entry - sl
                return {
                    "outcome": "LOSS",
                    "exit_price": sl,
                    "exit_bar": i,
                    "bars_held": i - index,
                    "r_multiple": pnl / risk,
                    "pnl_pct": (pnl / entry) * 100,
                    "hit_tp": False,
                    "hit_sl": True,
                    "max_favorable": max_favorable / entry * 100,
                    "max_adverse": max_adverse / entry * 100
                }

            # TP check (low <= tp)
            if low <= tp:
                pnl = entry - tp
                return {
                    "outcome": "WIN",
                    "exit_price": tp,
                    "exit_bar": i,
                    "bars_held": i - index,
                    "r_multiple": pnl / risk,
                    "pnl_pct": (pnl / entry) * 100,
                    "hit_tp": True,
                    "hit_sl": False,
                    "max_favorable": max_favorable / entry * 100,
                    "max_adverse": max_adverse / entry * 100
                }

        else:  # LONG
            # LONG için: yüksek fiyat iyi, düşük fiyat kötü
            favorable_move = high - entry
            adverse_move = entry - low
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)

            # SL check (low <= sl)
            if low <= sl:
                pnl = sl - entry
                return {
                    "outcome": "LOSS",
                    "exit_price": sl,
                    "exit_bar": i,
                    "bars_held": i - index,
                    "r_multiple": pnl / risk,
                    "pnl_pct": (pnl / entry) * 100,
                    "hit_tp": False,
                    "hit_sl": True,
                    "max_favorable": max_favorable / entry * 100,
                    "max_adverse": max_adverse / entry * 100
                }

            # TP check (high >= tp)
            if high >= tp:
                pnl = tp - entry
                return {
                    "outcome": "WIN",
                    "exit_price": tp,
                    "exit_bar": i,
                    "bars_held": i - index,
                    "r_multiple": pnl / risk,
                    "pnl_pct": (pnl / entry) * 100,
                    "hit_tp": True,
                    "hit_sl": False,
                    "max_favorable": max_favorable / entry * 100,
                    "max_adverse": max_adverse / entry * 100
                }

    # Timeout - ne TP ne SL vurulmadı
    last_bar = df.iloc[min(index + max_bars - 1, len(df) - 1)]
    last_close = last_bar['close']

    if signal_type == "SHORT":
        pnl = entry - last_close
    else:
        pnl = last_close - entry

    return {
        "outcome": "TIMEOUT",
        "exit_price": last_close,
        "exit_bar": min(index + max_bars - 1, len(df) - 1),
        "bars_held": max_bars,
        "r_multiple": pnl / risk if risk > 0 else 0,
        "pnl_pct": (pnl / entry) * 100,
        "hit_tp": False,
        "hit_sl": False,
        "max_favorable": max_favorable / entry * 100,
        "max_adverse": max_adverse / entry * 100
    }


# ============================================================================
# SIGNAL COLLECTION
# ============================================================================

def collect_all_signals(
    symbols: List[str] = None,
    timeframe: str = "15m",
    days: int = 180,
    progress_callback=None
) -> List[Dict]:
    """
    Tüm potansiyel sinyalleri ve trade outcome'larını topla.
    """
    symbols = symbols or ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    all_signals = []
    client = BinanceClient()

    for sym_idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(f"Collecting {symbol}...", sym_idx / len(symbols))

        print(f"  Collecting {symbol}...")

        # Veri çek (paginated for large date ranges)
        target_bars = min(days * 96, 15000)  # 15m = 96/day, max 15000
        df = client.get_klines_paginated(symbol, timeframe, total_candles=target_bars)
        if df is None or len(df) < 500:
            print(f"    Skipping {symbol} - insufficient data ({len(df) if df is not None else 0} bars)")
            continue

        # İndikatörleri hesapla
        df = calculate_indicators(df)

        signal_count = 0
        win_count = 0

        # Her bar için sinyal context'i kaydet
        # Son 100 bar'ı simülasyon için bırak
        for i in range(250, len(df) - 100):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Temel sinyal koşulları (AlphaTrend dominant)
            at_buyers = row.get('at_buyers', 0)
            at_sellers = row.get('at_sellers', 0)
            baseline = row.get('baseline', 0)
            close = row['close']

            # Potansiyel sinyal yönü
            signal_type = None
            if at_sellers > at_buyers and close < baseline:
                signal_type = "SHORT"
            elif at_buyers > at_sellers and close > baseline:
                signal_type = "LONG"

            if signal_type is None:
                continue

            # Entry/SL/TP hesapla
            entry, sl, tp = calculate_sl_tp(df, i, signal_type)

            # Trade outcome simüle et
            outcome = simulate_trade_outcome(df, i, signal_type, entry, sl, tp)

            signal_count += 1
            if outcome['outcome'] == 'WIN':
                win_count += 1

            # Tüm context bilgilerini kaydet
            signal_context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": str(row.name),
                "bar_index": i,
                "signal_type": signal_type,

                # Fiyat bilgileri
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),

                # Trade levels
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "risk_pct": float(abs(entry - sl) / entry * 100),
                "reward_pct": float(abs(tp - entry) / entry * 100),
                "rr_ratio": float(abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 0 else 0,

                # Ana indikatörler
                "baseline": float(baseline),
                "rsi": float(row.get('rsi', 50)),
                "adx": float(row.get('adx', 0)),
                "plus_di": float(row.get('plus_di', 0)),
                "minus_di": float(row.get('minus_di', 0)),
                "at_buyers": float(at_buyers),
                "at_sellers": float(at_sellers),
                "at_dominant": row.get('at_dominant', 'NONE'),
                "at_is_flat": bool(row.get('at_is_flat', False)),

                # PBEMA
                "pb_ema_top": float(row.get('pb_ema_top', 0)),
                "pb_ema_bot": float(row.get('pb_ema_bot', 0)),

                # Momentum hesaplamaları
                "last_3_momentum": float(df['close'].iloc[i-3:i].diff().sum()),
                "last_5_momentum": float(df['close'].iloc[i-5:i].diff().sum()),

                # Breakout detection
                "recent_high_10": float(df['high'].iloc[i-10:i].max()),
                "recent_low_10": float(df['low'].iloc[i-10:i].min()),
                "recent_high_20": float(df['high'].iloc[i-20:i].max()),
                "recent_low_20": float(df['low'].iloc[i-20:i].min()),

                # Candle patterns
                "body_bullish": bool(row['close'] > row['open']),
                "body_bearish": bool(row['close'] < row['open']),
                "prev_body_bullish": bool(prev_row['close'] > prev_row['open']),
                "prev_body_bearish": bool(prev_row['close'] < prev_row['open']),

                # Wick analysis
                "upper_wick_pct": float((row['high'] - max(row['open'], row['close'])) / row['close'] * 100),
                "lower_wick_pct": float((min(row['open'], row['close']) - row['low']) / row['close'] * 100),

                # Volatility
                "atr": float(row.get('atr', 0)),
                "atr_pct": float(row.get('atr', 0) / row['close'] * 100),

                # Distance calculations
                "dist_to_baseline_pct": float((close - baseline) / baseline * 100),
                "dist_to_pbema_top_pct": float((row.get('pb_ema_top', close) - close) / close * 100),
                "dist_to_pbema_bot_pct": float((close - row.get('pb_ema_bot', close)) / close * 100),

                # === SSL NEVER LOST FILTER (YENİ!) ===
                # Check if baseline was ever crossed in last N bars
                "baseline_ever_lost_bearish_20": bool(np.any(df['low'].iloc[max(0,i-20):i].values < df['baseline'].iloc[max(0,i-20):i].values)),
                "baseline_ever_lost_bullish_20": bool(np.any(df['high'].iloc[max(0,i-20):i].values > df['baseline'].iloc[max(0,i-20):i].values)),
                "baseline_ever_lost_bearish_10": bool(np.any(df['low'].iloc[max(0,i-10):i].values < df['baseline'].iloc[max(0,i-10):i].values)),
                "baseline_ever_lost_bullish_10": bool(np.any(df['high'].iloc[max(0,i-10):i].values > df['baseline'].iloc[max(0,i-10):i].values)),
                "baseline_ever_lost_bearish_5": bool(np.any(df['low'].iloc[max(0,i-5):i].values < df['baseline'].iloc[max(0,i-5):i].values)),
                "baseline_ever_lost_bullish_5": bool(np.any(df['high'].iloc[max(0,i-5):i].values > df['baseline'].iloc[max(0,i-5):i].values)),
                # Legacy names for backward compat
                "baseline_ever_lost_bearish": bool(np.any(df['low'].iloc[max(0,i-20):i].values < df['baseline'].iloc[max(0,i-20):i].values)),
                "baseline_ever_lost_bullish": bool(np.any(df['high'].iloc[max(0,i-20):i].values > df['baseline'].iloc[max(0,i-20):i].values)),

                # === TRADE OUTCOME (YENİ!) ===
                "outcome": outcome['outcome'],
                "outcome_win": outcome['outcome'] == 'WIN',
                "outcome_loss": outcome['outcome'] == 'LOSS',
                "exit_price": outcome['exit_price'],
                "bars_held": outcome['bars_held'],
                "r_multiple": outcome['r_multiple'],
                "pnl_pct": outcome['pnl_pct'],
                "hit_tp": outcome['hit_tp'],
                "hit_sl": outcome['hit_sl'],
                "max_favorable_pct": outcome['max_favorable'],
                "max_adverse_pct": outcome['max_adverse'],
            }

            # Breakout flags
            signal_context["breakout_up"] = close > signal_context["recent_high_10"] * 0.998
            signal_context["breakout_down"] = close < signal_context["recent_low_10"] * 1.002

            # DI relationship
            signal_context["di_bullish"] = signal_context["plus_di"] > signal_context["minus_di"]
            signal_context["di_bearish"] = signal_context["minus_di"] > signal_context["plus_di"]
            signal_context["di_ratio"] = (
                signal_context["plus_di"] / max(signal_context["minus_di"], 0.01)
            )

            all_signals.append(signal_context)

        win_rate = (win_count / signal_count * 100) if signal_count > 0 else 0
        print(f"    {symbol}: {signal_count} signals, {win_rate:.1f}% win rate")

    return all_signals


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_signals(signals: List[Dict], filepath: Path = None):
    """Sinyalleri dosyaya kaydet"""
    filepath = filepath or SIGNALS_FILE
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    signals = convert_to_json_serializable(signals)

    # İstatistikleri hesapla
    wins = sum(1 for s in signals if s['outcome'] == 'WIN')
    losses = sum(1 for s in signals if s['outcome'] == 'LOSS')
    timeouts = sum(1 for s in signals if s['outcome'] == 'TIMEOUT')

    with open(filepath, 'w') as f:
        json.dump({
            "version": "2.0",
            "collected_at": datetime.now().isoformat(),
            "total_signals": len(signals),
            "stats": {
                "wins": wins,
                "losses": losses,
                "timeouts": timeouts,
                "win_rate": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            },
            "signals": signals
        }, f, indent=2)

    print(f"Saved {len(signals)} signals to {filepath}")


def load_signals(filepath: Path = None) -> List[Dict]:
    """Kaydedilmiş sinyalleri yükle"""
    filepath = filepath or SIGNALS_FILE

    if not filepath.exists():
        raise FileNotFoundError(f"Signals file not found: {filepath}\nRun with --collect first")

    with open(filepath) as f:
        data = json.load(f)

    version = data.get('version', '1.0')
    stats = data.get('stats', {})

    print(f"Loaded {data['total_signals']} signals (v{version}, collected: {data['collected_at']})")
    if stats:
        print(f"  Baseline: {stats.get('wins', 0)} wins, {stats.get('losses', 0)} losses, {stats.get('win_rate', 0):.1f}% WR")

    return data['signals']


# ============================================================================
# FILTER TESTING WITH OUTCOMES
# ============================================================================

def test_filter(signals: List[Dict], filter_expr: str, signal_type: str = None) -> Dict:
    """
    Filtre ifadesini test et ve WIN RATE dahil sonuçları döndür.
    """
    # Signal type filter
    if signal_type:
        signals = [s for s in signals if s['signal_type'] == signal_type]

    total = len(signals)
    passed_signals = []
    failed_signals = []

    for sig in signals:
        try:
            ctx = sig.copy()
            # SECURITY: Use safe_eval() instead of eval() to prevent code injection
            result = safe_eval(filter_expr, ctx)

            if result:
                passed_signals.append(sig)
            else:
                failed_signals.append(sig)
        except Exception as e:
            print(f"Error evaluating filter on signal: {e}")

    # Passed sinyallerin outcome analizi
    passed_wins = sum(1 for s in passed_signals if s.get('outcome') == 'WIN')
    passed_losses = sum(1 for s in passed_signals if s.get('outcome') == 'LOSS')
    passed_total_outcomes = passed_wins + passed_losses

    # Failed sinyallerin outcome analizi
    failed_wins = sum(1 for s in failed_signals if s.get('outcome') == 'WIN')
    failed_losses = sum(1 for s in failed_signals if s.get('outcome') == 'LOSS')
    failed_total_outcomes = failed_wins + failed_losses

    # R-multiple ortalamaları
    passed_r_avg = np.mean([s.get('r_multiple', 0) for s in passed_signals]) if passed_signals else 0
    failed_r_avg = np.mean([s.get('r_multiple', 0) for s in failed_signals]) if failed_signals else 0

    # PnL tahminleri - Win/Loss bazlı basit model
    # Varsayımlar: WIN = +1R ($35), LOSS = -1R ($35)
    RISK_PER_TRADE = 35.0
    AVG_WIN_R = 1.0   # Ortalama kazanç R
    AVG_LOSS_R = -1.0  # Ortalama kayıp R

    passed_total_r = (passed_wins * AVG_WIN_R) + (passed_losses * AVG_LOSS_R)
    failed_total_r = (failed_wins * AVG_WIN_R) + (failed_losses * AVG_LOSS_R)
    all_total_r = passed_total_r + failed_total_r

    passed_pnl_estimate = passed_total_r * RISK_PER_TRADE
    failed_pnl_estimate = failed_total_r * RISK_PER_TRADE
    all_pnl_estimate = all_total_r * RISK_PER_TRADE

    return {
        "filter": filter_expr,
        "signal_type": signal_type or "ALL",
        "total": total,
        "passed": len(passed_signals),
        "failed": len(failed_signals),
        "pass_rate": len(passed_signals) / total * 100 if total > 0 else 0,

        # Passed signals outcome
        "passed_wins": passed_wins,
        "passed_losses": passed_losses,
        "passed_win_rate": passed_wins / passed_total_outcomes * 100 if passed_total_outcomes > 0 else 0,
        "passed_avg_r": passed_r_avg,
        "passed_total_r": passed_total_r,
        "passed_pnl_estimate": passed_pnl_estimate,

        # Failed signals outcome (ne kaçırdık?)
        "failed_wins": failed_wins,
        "failed_losses": failed_losses,
        "failed_win_rate": failed_wins / failed_total_outcomes * 100 if failed_total_outcomes > 0 else 0,
        "failed_avg_r": failed_r_avg,
        "failed_total_r": failed_total_r,
        "failed_pnl_estimate": failed_pnl_estimate,

        # Toplam
        "all_total_r": all_total_r,
        "all_pnl_estimate": all_pnl_estimate,

        "passed_signals": passed_signals,
        "failed_signals": failed_signals
    }


def analyze_filter_impact(signals: List[Dict], filter_expr: str) -> None:
    """Filtrenin etkisini detaylı analiz et - WIN RATE, PnL ve Trade sayısı dahil"""

    print(f"\n{'='*70}")
    print(f"Filter: {filter_expr}")
    print('='*70)

    # Baseline (filtresiz) istatistikler
    all_wins = sum(1 for s in signals if s.get('outcome') == 'WIN')
    all_losses = sum(1 for s in signals if s.get('outcome') == 'LOSS')
    baseline_wr = all_wins / (all_wins + all_losses) * 100 if (all_wins + all_losses) > 0 else 0
    baseline_trades = all_wins + all_losses

    # Baseline PnL - Basit model: WIN = +1R, LOSS = -1R
    RISK_PER_TRADE = 35.0
    AVG_WIN_R = 1.0
    AVG_LOSS_R = -1.0
    baseline_total_r = (all_wins * AVG_WIN_R) + (all_losses * AVG_LOSS_R)
    baseline_pnl = baseline_total_r * RISK_PER_TRADE

    print(f"\nBASELINE (filtresiz):")
    print(f"  Trades: {baseline_trades} | Win Rate: {baseline_wr:.1f}%")
    print(f"  Total R: {baseline_total_r:.1f} | Est. PnL: ${baseline_pnl:.0f}")

    # SHORT sinyalleri
    short_result = test_filter(signals, filter_expr, "SHORT")
    print(f"\n{'─'*70}")
    print(f"SHORT Signals:")
    print(f"  Total: {short_result['total']} → Passed: {short_result['passed']} ({short_result['pass_rate']:.1f}%)")
    print(f"  ┌─ PASSED: {short_result['passed_wins']}W/{short_result['passed_losses']}L = {short_result['passed_win_rate']:.1f}% WR")
    print(f"  │         Total R: {short_result['passed_total_r']:.1f}, Est. PnL: ${short_result['passed_pnl_estimate']:.0f}")
    print(f"  └─ BLOCKED: {short_result['failed_wins']}W/{short_result['failed_losses']}L, Est. PnL: ${short_result['failed_pnl_estimate']:.0f} (kaçırılan)")

    # LONG sinyalleri
    long_result = test_filter(signals, filter_expr, "LONG")
    print(f"\n{'─'*70}")
    print(f"LONG Signals:")
    print(f"  Total: {long_result['total']} → Passed: {long_result['passed']} ({long_result['pass_rate']:.1f}%)")
    print(f"  ┌─ PASSED: {long_result['passed_wins']}W/{long_result['passed_losses']}L = {long_result['passed_win_rate']:.1f}% WR")
    print(f"  │         Total R: {long_result['passed_total_r']:.1f}, Est. PnL: ${long_result['passed_pnl_estimate']:.0f}")
    print(f"  └─ BLOCKED: {long_result['failed_wins']}W/{long_result['failed_losses']}L, Est. PnL: ${long_result['failed_pnl_estimate']:.0f} (kaçırılan)")

    # Net etki
    print(f"\n{'─'*70}")
    total_passed = short_result['passed'] + long_result['passed']
    total_passed_wins = short_result['passed_wins'] + long_result['passed_wins']
    total_passed_losses = short_result['passed_losses'] + long_result['passed_losses']
    total_blocked_wins = short_result['failed_wins'] + long_result['failed_wins']
    total_blocked_losses = short_result['failed_losses'] + long_result['failed_losses']

    new_wr = total_passed_wins / (total_passed_wins + total_passed_losses) * 100 if (total_passed_wins + total_passed_losses) > 0 else 0

    # PnL hesaplamaları
    passed_pnl = short_result['passed_pnl_estimate'] + long_result['passed_pnl_estimate']
    blocked_pnl = short_result['failed_pnl_estimate'] + long_result['failed_pnl_estimate']
    passed_trades = total_passed_wins + total_passed_losses

    wr_change = new_wr - baseline_wr
    pnl_change = passed_pnl - baseline_pnl
    trade_change = passed_trades - baseline_trades
    trade_change_pct = (trade_change / baseline_trades * 100) if baseline_trades > 0 else 0

    print(f"TAHMİNİ ETKİ:")
    print(f"  ┌─ Trade Sayısı: {baseline_trades} → {passed_trades} ({trade_change:+d}, {trade_change_pct:+.0f}%)")
    print(f"  ├─ Win Rate:     {baseline_wr:.1f}% → {new_wr:.1f}% ({'+' if wr_change >= 0 else ''}{wr_change:.1f}%)")
    print(f"  └─ Est. PnL:     ${baseline_pnl:.0f} → ${passed_pnl:.0f} ({'+' if pnl_change >= 0 else ''}{pnl_change:.0f})")

    print(f"\n  Kaçırılan: {total_blocked_wins}W/{total_blocked_losses}L = ${blocked_pnl:.0f}")

    # Değerlendirme
    print(f"\n{'─'*70}")
    print("DEĞERLENDİRME:")

    score = 0
    if wr_change > 2:
        print(f"  ✅ Win Rate +{wr_change:.1f}% arttı")
        score += 1
    elif wr_change < -2:
        print(f"  ❌ Win Rate {wr_change:.1f}% düştü")
        score -= 1

    if pnl_change > 0:
        print(f"  ✅ PnL +${pnl_change:.0f} artış bekleniyor")
        score += 2
    else:
        print(f"  ❌ PnL ${pnl_change:.0f} düşüş bekleniyor")
        score -= 2

    if trade_change_pct < -50:
        print(f"  ⚠️  Trade sayısı %{abs(trade_change_pct):.0f} azaldı (çok kısıtlayıcı)")
        score -= 1
    elif trade_change_pct < -30:
        print(f"  ➖ Trade sayısı %{abs(trade_change_pct):.0f} azaldı (orta)")

    if score >= 2:
        print(f"\n  🏆 SONUÇ: İYİ FİLTRE - Rolling WF'de test et!")
    elif score <= -2:
        print(f"\n  ⛔ SONUÇ: KÖTÜ FİLTRE - Kullanma")
    else:
        print(f"\n  ➖ SONUÇ: NÖTR - Daha fazla analiz gerekli")


def compare_filters(signals: List[Dict], filters: List[str]) -> pd.DataFrame:
    """Birden fazla filtreyi WIN RATE ve PnL ile karşılaştır"""

    # Baseline hesapla - Basit model: WIN = +1R, LOSS = -1R
    RISK_PER_TRADE = 35.0
    AVG_WIN_R = 1.0
    AVG_LOSS_R = -1.0
    all_wins = sum(1 for s in signals if s.get('outcome') == 'WIN')
    all_losses = sum(1 for s in signals if s.get('outcome') == 'LOSS')
    baseline_wr = all_wins / (all_wins + all_losses) * 100 if (all_wins + all_losses) > 0 else 0
    baseline_total_r = (all_wins * AVG_WIN_R) + (all_losses * AVG_LOSS_R)
    baseline_pnl = baseline_total_r * RISK_PER_TRADE
    baseline_trades = all_wins + all_losses

    results = []
    for filter_expr in filters:
        short_result = test_filter(signals, filter_expr, "SHORT")
        long_result = test_filter(signals, filter_expr, "LONG")

        total_passed = short_result['passed'] + long_result['passed']
        total_wins = short_result['passed_wins'] + long_result['passed_wins']
        total_losses = short_result['passed_losses'] + long_result['passed_losses']
        total_trades = total_wins + total_losses

        new_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        wr_change = new_wr - baseline_wr

        # PnL hesaplama
        passed_pnl = short_result['passed_pnl_estimate'] + long_result['passed_pnl_estimate']
        pnl_change = passed_pnl - baseline_pnl
        trade_change_pct = ((total_trades - baseline_trades) / baseline_trades * 100) if baseline_trades > 0 else 0

        results.append({
            "filter": filter_expr[:30] + "..." if len(filter_expr) > 30 else filter_expr,
            "trades": total_trades,
            "Δtrd%": f"{trade_change_pct:+.0f}%",
            "WR%": f"{new_wr:.1f}%",
            "ΔWR": f"{'+' if wr_change >= 0 else ''}{wr_change:.1f}%",
            "PnL": f"${passed_pnl:.0f}",
            "ΔPnL": f"{'+' if pnl_change >= 0 else ''}{pnl_change:.0f}",
        })

    df = pd.DataFrame(results)
    return df


def interactive_mode(signals: List[Dict]):
    """İnteraktif filtre test modu - WIN RATE dahil"""

    # Baseline stats
    all_wins = sum(1 for s in signals if s.get('outcome') == 'WIN')
    all_losses = sum(1 for s in signals if s.get('outcome') == 'LOSS')
    baseline_wr = all_wins / (all_wins + all_losses) * 100 if (all_wins + all_losses) > 0 else 0

    print("\n" + "="*70)
    print("Signal Sandbox v2 - Interactive Mode (with Trade Outcomes)")
    print("="*70)
    print(f"Loaded {len(signals)} signals")
    print(f"Baseline: {all_wins}W/{all_losses}L = {baseline_wr:.1f}% Win Rate")
    print("\nAvailable variables:")
    print("  Price: open, high, low, close, volume, entry, sl, tp")
    print("  Indicators: rsi, adx, plus_di, minus_di, baseline, atr")
    print("  Momentum: last_3_momentum, last_5_momentum")
    print("  Breakout: breakout_up, breakout_down")
    print("  DI: di_bullish, di_bearish, di_ratio")
    print("  Outcome: outcome_win, outcome_loss, r_multiple, pnl_pct")
    print("\nExamples:")
    print('  > last_3_momentum < 0')
    print('  > adx > 25')
    print('  > not breakout_up and adx > 20')
    print("\nCommands: 'quit', 'compare', 'best' (find best filters)")
    print("="*70)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'compare':
                print("Enter filters (one per line, empty line to finish):")
                filters = []
                while True:
                    f = input("  filter> ").strip()
                    if not f:
                        break
                    filters.append(f)

                if filters:
                    df = compare_filters(signals, filters)
                    print("\n" + df.to_string(index=False))
                continue

            if user_input.lower() == 'best':
                print("\nSearching for best filters...")
                candidate_filters = [
                    "adx > 20",
                    "adx > 25",
                    "adx > 30",
                    "last_3_momentum < 0",
                    "last_5_momentum < 0",
                    "not breakout_up",
                    "not breakout_down",
                    "rsi < 70",
                    "rsi > 30",
                    "di_bearish",
                    "di_bullish",
                    "atr_pct > 0.5",
                    "atr_pct < 2.0",
                    "rr_ratio > 1.5",
                    "rr_ratio > 2.0",
                ]
                df = compare_filters(signals, candidate_filters)
                print("\n" + df.to_string(index=False))
                continue

            # Test the filter
            analyze_filter_impact(signals, user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Signal Sandbox v2 - Fast filter testing with trade outcomes")
    parser.add_argument("--collect", action="store_true", help="Collect all signals with outcomes (run once)")
    parser.add_argument("--test", type=str, help="Test a filter expression")
    parser.add_argument("--compare", nargs="+", help="Compare multiple filters")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "LINKUSDT"])
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=180)

    args = parser.parse_args()

    if args.collect:
        print(f"Collecting signals for {args.symbols} ({args.timeframe}, {args.days} days)...")
        print("This includes trade outcome simulation - may take 3-5 minutes...")
        signals = collect_all_signals(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days
        )
        save_signals(signals)
        print(f"\nCollection complete! {len(signals)} signals with outcomes saved.")
        print("Now run with --test or --interactive to test filters.")
        return

    # Load existing signals
    try:
        signals = load_signals()
    except FileNotFoundError as e:
        print(e)
        return

    if args.test:
        analyze_filter_impact(signals, args.test)

    elif args.compare:
        df = compare_filters(signals, args.compare)
        print("\n" + df.to_string(index=False))

    elif args.interactive or (not args.test and not args.compare):
        interactive_mode(signals)


if __name__ == "__main__":
    main()
