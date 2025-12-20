#!/usr/bin/env python3
"""
Rolling Walk-Forward Test Script

Bu script, Rolling Walk-Forward framework'ünü test eder:
1. Fixed vs Monthly vs Weekly vs 5day vs Triday karşılaştırması yapar
2. 2025 yılı için stitched OOS sonuçlarını hesaplar
3. En iyi modu önerir

Modlar:
- Fixed: Sabit config, re-optimization yok
- Monthly: Aylık re-optimization (60 gün lookback, 30 gün forward)
- Weekly: Haftalık re-optimization (30 gün lookback, 7 gün forward)
- 5day: 5 günlük re-optimization (75 gün lookback, 5 gün forward)
- Triday: 3 günlük re-optimization (90 gün lookback, 3 gün forward)

Kullanım:
    python run_rolling_wf_test.py                    # Varsayılan test (son 6 ay)
    python run_rolling_wf_test.py --full-year       # 2025 tam yıl testi
    python run_rolling_wf_test.py --quick           # Hızlı test (3 ay, az sembol)
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# TRADE LOG WRITER FUNCTIONS
# ============================================================================

def calculate_duration(entry_time, exit_time) -> str:
    """Calculate duration between entry and exit times."""
    try:
        if isinstance(entry_time, str):
            # Parse common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    entry_dt = datetime.strptime(entry_time.split("+")[0].split(".")[0], fmt)
                    break
                except ValueError:
                    continue
            else:
                return "N/A"
        else:
            entry_dt = entry_time

        if isinstance(exit_time, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    exit_dt = datetime.strptime(exit_time.split("+")[0].split(".")[0], fmt)
                    break
                except ValueError:
                    continue
            else:
                return "N/A"
        else:
            exit_dt = exit_time

        duration = exit_dt - entry_dt
        total_minutes = int(duration.total_seconds() / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except Exception:
        return "N/A"


def format_price(value, decimals: int = 4) -> str:
    """Format price value with appropriate decimals."""
    if value is None:
        return "N/A"
    try:
        val = float(value)
        if val >= 1000:
            return f"{val:,.2f}"
        elif val >= 1:
            return f"{val:.4f}"
        else:
            return f"{val:.6f}"
    except (ValueError, TypeError):
        return "N/A"


def format_header(run_id: str, mode: str, start_date: str, end_date: str,
                  total_trades: int, win_rate: float, total_pnl: float) -> str:
    """Format the file header section."""
    wins = int(total_trades * win_rate)
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ROLLING WALK-FORWARD TRADE LOG                            ║
║                                                                              ║
║  Run ID: {run_id:<60}║
║  Period: {start_date} → {end_date:<50}║
║  Mode: {mode.upper():<63}║
║  Total Trades: {total_trades:<55}║
║  Win Rate: {win_rate*100:.1f}% ({wins}/{total_trades}){' ' * (53 - len(f"{win_rate*100:.1f}% ({wins}/{total_trades})"))           }║
║  Total PnL: ${total_pnl:+,.2f}{' ' * (55 - len(f"${total_pnl:+,.2f}"))                                      }║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def format_window_header(window: dict) -> str:
    """Format a window header section."""
    window_id = window.get("window_id", 0)
    trade_start = window.get("trade_start", "N/A")
    trade_end = window.get("trade_end", "N/A")
    pnl = window.get("pnl", 0.0)
    trades = window.get("trades", 0)
    wins = window.get("wins", 0)
    win_pct = (wins / trades * 100) if trades > 0 else 0

    return f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  WINDOW #{window_id}: {trade_start} → {trade_end}{' ' * (50 - len(f"{trade_start} → {trade_end}"))}│
│  Window PnL: ${pnl:+,.2f} | Trades: {trades} | Wins: {wins} ({win_pct:.0f}%){' ' * (33 - len(f"${pnl:+,.2f} | Trades: {trades} | Wins: {wins} ({win_pct:.0f}%)"))}│
└──────────────────────────────────────────────────────────────────────────────┘
"""


def format_trade_box(trade: dict, trade_num: int) -> str:
    """Format a single trade in a box format."""
    pnl = float(trade.get("pnl", 0))
    is_win = pnl > 0
    status_icon = "✅ WIN" if is_win else "❌ LOSS"

    # Exit status
    status = trade.get("status", "")
    if "WIN" in status or "WON" in status:
        exit_status = "TP HIT"
    elif "STOP" in status:
        exit_status = "SL HIT"
    elif "PARTIAL" in status:
        exit_status = "PARTIAL TP"
    elif "BE" in status:
        exit_status = "BREAKEVEN"
    else:
        exit_status = status or "CLOSED"

    # Times
    entry_time = trade.get("open_time_utc", "N/A")
    exit_time = trade.get("close_time_utc", "N/A")
    duration = calculate_duration(entry_time, exit_time)

    # Prices
    entry = float(trade.get("entry", 0))
    tp = float(trade.get("tp", 0))
    sl = float(trade.get("sl", 0))
    close_price = float(trade.get("close_price", 0)) if trade.get("close_price") else 0

    trade_type = trade.get("type", "UNKNOWN")

    # TP/SL percentages
    if entry > 0:
        if trade_type == "LONG":
            tp_pct = ((tp - entry) / entry) * 100
            sl_pct = ((sl - entry) / entry) * 100
        else:
            tp_pct = ((entry - tp) / entry) * 100
            sl_pct = ((entry - sl) / entry) * 100
    else:
        tp_pct = 0
        sl_pct = 0

    # R-Multiple
    r_multiple = trade.get("r_multiple", 0)

    # Symbol/timeframe
    symbol = trade.get("symbol", "UNKNOWN")
    timeframe = trade.get("timeframe", "?")
    setup = trade.get("setup", "N/A")

    # Indicators
    ind = trade.get("indicators_at_entry", {})
    at_buyers = ind.get("at_buyers")
    at_sellers = ind.get("at_sellers")
    at_dominant = ind.get("at_dominant", "N/A")
    at_is_flat = ind.get("at_is_flat", False)
    baseline = ind.get("baseline")
    rsi = ind.get("rsi")
    adx = ind.get("adx")

    # PBEMA target based on trade type
    if trade_type == "LONG":
        pbema_target = ind.get("pb_ema_bot")
        pbema_label = "PBEMA Bot"
    else:
        pbema_target = ind.get("pb_ema_top")
        pbema_label = "PBEMA Top"

    # Format indicator values
    def fmt_ind(val):
        if val is None:
            return "N/A"
        return format_price(val)

    at_flat_str = "Yes ⚠️" if at_is_flat else "No"

    # Add check mark for AT direction match
    at_check = ""
    if trade_type == "LONG" and at_dominant == "BUYERS":
        at_check = " ✓"
    elif trade_type == "SHORT" and at_dominant == "SELLERS":
        at_check = " ✓"
    elif trade_type == "LONG" and at_dominant == "SELLERS":
        at_check = " ⚠️"
    elif trade_type == "SHORT" and at_dominant == "BUYERS":
        at_check = " ⚠️"

    # Build the trade box
    lines = []
    lines.append(f"  ┌─ Trade #{trade_num} " + "─" * (65 - len(f"Trade #{trade_num}")) + "┐")
    lines.append(f"  │  {status_icon:<74}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  Symbol:     {symbol}-{timeframe:<58}│")
    lines.append(f"  │  Type:       {trade_type:<62}│")
    lines.append(f"  │  Setup:      {setup[:58]:<58}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  Entry Time: {str(entry_time)[:50]:<62}│")
    lines.append(f"  │  Entry:      ${format_price(entry):<61}│")
    lines.append(f"  │  TP Target:  ${format_price(tp)} ({tp_pct:+.2f}%){' ' * (48 - len(f'{format_price(tp)} ({tp_pct:+.2f}%)'))}│")
    lines.append(f"  │  SL Target:  ${format_price(sl)} ({sl_pct:+.2f}%){' ' * (48 - len(f'{format_price(sl)} ({sl_pct:+.2f}%)'))}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  Exit Time:  {str(exit_time)[:50]:<62}│")
    lines.append(f"  │  Exit Price: ${format_price(close_price)} ({exit_status}){' ' * (47 - len(f'{format_price(close_price)} ({exit_status})'))}│")
    lines.append(f"  │  Duration:   {duration:<62}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  PnL:        ${pnl:+,.2f}{' ' * (61 - len(f'{pnl:+,.2f}'))}│")
    lines.append(f"  │  R-Multiple: {r_multiple:+.2f}{' ' * (62 - len(f'{r_multiple:+.2f}'))}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  ─── Indicators at Entry ───{' ' * 46}│")
    lines.append(f"  │  AT Buyers:     {fmt_ind(at_buyers):<59}│")
    lines.append(f"  │  AT Sellers:    {fmt_ind(at_sellers):<59}│")
    lines.append(f"  │  AT Dominant:   {at_dominant}{at_check}{' ' * (57 - len(at_dominant + at_check))}│")
    lines.append(f"  │  AT Flat:       {at_flat_str:<59}│")
    lines.append(f"  │  Baseline:      {fmt_ind(baseline):<59}│")
    lines.append(f"  │  {pbema_label}:     {fmt_ind(pbema_target):<59}│")
    lines.append(f"  │  RSI:           {fmt_ind(rsi):<59}│")
    lines.append(f"  │  ADX:           {fmt_ind(adx):<59}│")
    lines.append(f"  └" + "─" * 76 + "┘")

    return "\n".join(lines)


def format_window_summary(window: dict, window_trades: list) -> str:
    """Format window summary section."""
    window_id = window.get("window_id", 0)
    total = len(window_trades)
    wins = sum(1 for t in window_trades if float(t.get("pnl", 0)) > 0)
    losses = total - wins
    total_pnl = sum(float(t.get("pnl", 0)) for t in window_trades)

    win_pct = (wins / total * 100) if total > 0 else 0
    loss_pct = (losses / total * 100) if total > 0 else 0

    # Best and worst trades
    best_trade = max(window_trades, key=lambda t: float(t.get("pnl", 0)), default=None)
    worst_trade = min(window_trades, key=lambda t: float(t.get("pnl", 0)), default=None)

    best_str = "N/A"
    worst_str = "N/A"

    if best_trade:
        best_pnl = float(best_trade.get("pnl", 0))
        best_r = best_trade.get("r_multiple", 0)
        best_sym = best_trade.get("symbol", "?")
        best_tf = best_trade.get("timeframe", "?")
        best_type = best_trade.get("type", "?")
        best_str = f"{best_sym}-{best_tf} {best_type} +${best_pnl:.2f} (R:{best_r:+.2f})"

    if worst_trade:
        worst_pnl = float(worst_trade.get("pnl", 0))
        worst_r = worst_trade.get("r_multiple", 0)
        worst_sym = worst_trade.get("symbol", "?")
        worst_tf = worst_trade.get("timeframe", "?")
        worst_type = worst_trade.get("type", "?")
        worst_str = f"{worst_sym}-{worst_tf} {worst_type} ${worst_pnl:.2f} (R:{worst_r:+.2f})"

    # By symbol breakdown
    by_symbol = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    for t in window_trades:
        key = f"{t.get('symbol', '?')}-{t.get('timeframe', '?')}"
        by_symbol[key]["trades"] += 1
        by_symbol[key]["pnl"] += float(t.get("pnl", 0))

    lines = []
    lines.append(f"  ┌─ Window #{window_id} Summary " + "─" * (54 - len(f"Window #{window_id} Summary")) + "┐")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  Total Trades: {total:<60}│")
    lines.append(f"  │  Wins: {wins} ({win_pct:.1f}%)    Losses: {losses} ({loss_pct:.1f}%){' ' * (41 - len(f'Wins: {wins} ({win_pct:.1f}%)    Losses: {losses} ({loss_pct:.1f}%)'))}│")
    lines.append(f"  │  Total PnL: ${total_pnl:+,.2f}{' ' * (60 - len(f'${total_pnl:+,.2f}'))}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  Best Trade:  {best_str[:60]:<60}│")
    lines.append(f"  │  Worst Trade: {worst_str[:60]:<60}│")
    lines.append(f"  │{' ' * 76}│")
    lines.append(f"  │  By Symbol:{' ' * 64}│")

    for sym_key in sorted(by_symbol.keys()):
        data = by_symbol[sym_key]
        sym_line = f"{sym_key}: {data['trades']} trades, ${data['pnl']:+,.2f}"
        lines.append(f"  │    {sym_line:<71}│")

    lines.append(f"  └" + "─" * 76 + "┘")

    return "\n".join(lines)


def format_overall_summary(all_trades: list, metrics: dict, window_results: list) -> str:
    """Format overall summary section."""
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if float(t.get("pnl", 0)) > 0)
    losses = total_trades - wins

    win_pct = (wins / total_trades * 100) if total_trades > 0 else 0
    loss_pct = (losses / total_trades * 100) if total_trades > 0 else 0

    total_pnl = sum(float(t.get("pnl", 0)) for t in all_trades)

    # Avg win/loss
    winning_trades = [t for t in all_trades if float(t.get("pnl", 0)) > 0]
    losing_trades = [t for t in all_trades if float(t.get("pnl", 0)) <= 0]

    avg_win = sum(float(t.get("pnl", 0)) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(float(t.get("pnl", 0)) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    avg_win_r = sum(t.get("r_multiple", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_r = sum(t.get("r_multiple", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    # Window stats
    total_windows = len(window_results)
    positive_windows = sum(1 for w in window_results if w.get("pnl", 0) > 0)
    negative_windows = total_windows - positive_windows

    pos_pct = (positive_windows / total_windows * 100) if total_windows > 0 else 0
    neg_pct = (negative_windows / total_windows * 100) if total_windows > 0 else 0

    lines = []
    lines.append("\n")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                           OVERALL SUMMARY                                    ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                                                                              ║")
    lines.append(f"║  Total Windows: {total_windows:<62}║")
    lines.append(f"║  Positive Windows: {positive_windows} ({pos_pct:.1f}%){' ' * (54 - len(f'{positive_windows} ({pos_pct:.1f}%)'))}║")
    lines.append(f"║  Negative Windows: {negative_windows} ({neg_pct:.1f}%){' ' * (54 - len(f'{negative_windows} ({neg_pct:.1f}%)'))}║")
    lines.append("║                                                                              ║")
    lines.append(f"║  Total Trades: {total_trades:<63}║")
    lines.append(f"║  Wins: {wins} ({win_pct:.1f}%){' ' * (65 - len(f'{wins} ({win_pct:.1f}%)'))}║")
    lines.append(f"║  Losses: {losses} ({loss_pct:.1f}%){' ' * (63 - len(f'{losses} ({loss_pct:.1f}%)'))}║")
    lines.append("║                                                                              ║")
    lines.append(f"║  Total PnL: ${total_pnl:+,.2f}{' ' * (62 - len(f'${total_pnl:+,.2f}'))}║")
    lines.append(f"║  Avg Win: ${avg_win:+,.2f} (R:{avg_win_r:+.2f}){' ' * (51 - len(f'${avg_win:+,.2f} (R:{avg_win_r:+.2f})'))}║")
    lines.append(f"║  Avg Loss: ${avg_loss:+,.2f} (R:{avg_loss_r:+.2f}){' ' * (50 - len(f'${avg_loss:+,.2f} (R:{avg_loss_r:+.2f})'))}║")
    lines.append("║                                                                              ║")

    return "\n".join(lines)


def format_top_trades(all_trades: list) -> str:
    """Format top 5 best and worst trades."""
    sorted_trades = sorted(all_trades, key=lambda t: float(t.get("pnl", 0)))

    worst_5 = sorted_trades[:5]
    best_5 = sorted_trades[-5:][::-1]

    lines = []
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                         TOP 5 WORST TRADES                                   ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                                                                              ║")

    for i, trade in enumerate(worst_5, 1):
        pnl = float(trade.get("pnl", 0))
        r_mult = trade.get("r_multiple", 0)
        sym = trade.get("symbol", "?")
        tf = trade.get("timeframe", "?")
        t_type = trade.get("type", "?")
        entry_time = str(trade.get("open_time_utc", ""))[:10]

        trade_line = f"{i}. {sym}-{tf} {t_type} {entry_time} | ${pnl:+,.2f} | R:{r_mult:+.2f}"
        lines.append(f"║  {trade_line:<74}║")

        # Add potential issue detection
        ind = trade.get("indicators_at_entry", {})
        issues = []

        if ind.get("at_is_flat"):
            issues.append("AT Flat: YES ⚠️  (Trade alınmamalıydı!)")

        at_dom = ind.get("at_dominant", "")
        if t_type == "LONG" and at_dom == "SELLERS":
            issues.append("AT Sellers dominant ⚠️ (LONG için yanlış yön!)")
        elif t_type == "SHORT" and at_dom == "BUYERS":
            issues.append("AT Buyers dominant ⚠️ (SHORT için yanlış yön!)")

        rsi = ind.get("rsi")
        if rsi is not None:
            if t_type == "LONG" and rsi > 70:
                issues.append(f"RSI: {rsi:.0f} ⚠️ (Overbought'ta LONG alınmış!)")
            elif t_type == "SHORT" and rsi < 30:
                issues.append(f"RSI: {rsi:.0f} ⚠️ (Oversold'da SHORT alınmış!)")

        for issue in issues[:1]:  # Show first issue only to save space
            lines.append(f"║     → {issue:<69}║")

        lines.append("║                                                                              ║")

    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                         TOP 5 BEST TRADES                                    ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                                                                              ║")

    for i, trade in enumerate(best_5, 1):
        pnl = float(trade.get("pnl", 0))
        r_mult = trade.get("r_multiple", 0)
        sym = trade.get("symbol", "?")
        tf = trade.get("timeframe", "?")
        t_type = trade.get("type", "?")
        entry_time = str(trade.get("open_time_utc", ""))[:10]

        trade_line = f"{i}. {sym}-{tf} {t_type} {entry_time} | ${pnl:+,.2f} | R:{r_mult:+.2f}"
        lines.append(f"║  {trade_line:<74}║")

    lines.append("║                                                                              ║")

    return "\n".join(lines)


def format_issues_summary(all_trades: list) -> str:
    """Format potential issues detected section."""
    issues = {
        "at_flat_trades": 0,
        "wrong_at_direction": 0,
        "rsi_extreme": 0,
        "pbema_too_close": 0,
    }

    for trade in all_trades:
        ind = trade.get("indicators_at_entry", {})
        t_type = trade.get("type", "")

        # AT flat check
        if ind.get("at_is_flat"):
            issues["at_flat_trades"] += 1

        # Wrong AT direction
        at_dom = ind.get("at_dominant", "")
        if t_type == "LONG" and at_dom == "SELLERS":
            issues["wrong_at_direction"] += 1
        elif t_type == "SHORT" and at_dom == "BUYERS":
            issues["wrong_at_direction"] += 1

        # RSI extreme
        rsi = ind.get("rsi")
        if rsi is not None:
            if (t_type == "LONG" and rsi > 70) or (t_type == "SHORT" and rsi < 30):
                issues["rsi_extreme"] += 1

        # PBEMA too close
        entry = float(trade.get("entry", 0))
        if t_type == "LONG":
            pbema = ind.get("pb_ema_bot")
        else:
            pbema = ind.get("pb_ema_top")

        if pbema is not None and entry > 0:
            distance_pct = abs(pbema - entry) / entry * 100
            if distance_pct < 0.4:
                issues["pbema_too_close"] += 1

    lines = []
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                       POTENTIAL ISSUES DETECTED                              ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                                                                              ║")

    if issues["at_flat_trades"] > 0:
        line = f"⚠️ {issues['at_flat_trades']} trades taken when AT was flat (should be 0)"
        lines.append(f"║  {line:<74}║")

    if issues["wrong_at_direction"] > 0:
        line = f"⚠️ {issues['wrong_at_direction']} trades with wrong AT direction"
        lines.append(f"║  {line:<74}║")

    if issues["rsi_extreme"] > 0:
        line = f"⚠️ {issues['rsi_extreme']} trades with RSI in overbought/oversold"
        lines.append(f"║  {line:<74}║")

    if issues["pbema_too_close"] > 0:
        line = f"⚠️ {issues['pbema_too_close']} trades with PBEMA distance < 0.4%"
        lines.append(f"║  {line:<74}║")

    if all(v == 0 for v in issues.values()):
        lines.append("║  ✅ No major issues detected                                                  ║")

    lines.append("║                                                                              ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


def write_trade_log(result: dict, output_dir: str = None) -> str:
    """
    Write detailed trade log from rolling WF results.

    Args:
        result: Result dict from run_rolling_walkforward containing:
            - run_id: Test ID
            - mode: "fixed", "monthly", "weekly", "5day", or "triday"
            - config: Test configuration
            - metrics: Overall metrics
            - window_results: Per-window results
            - trades: All trades list

        output_dir: Output directory (default: from result)

    Returns:
        Path to the generated log file
    """
    run_id = result.get("run_id", "unknown")
    mode = result.get("mode", "unknown")
    config = result.get("config", {})
    metrics = result.get("metrics", {})
    window_results = result.get("window_results", [])
    all_trades = result.get("trades", [])

    start_date = config.get("start_date", "N/A")
    end_date = config.get("end_date", "N/A")
    total_trades = metrics.get("total_trades", len(all_trades))
    win_rate = metrics.get("win_rate", 0)
    total_pnl = metrics.get("total_pnl", 0)

    # Determine output directory - prefer from result, then parameter, then construct
    if output_dir is None:
        output_dir = result.get("output_dir")
    if output_dir is None:
        output_dir = os.path.join("data", "rolling_wf_runs", run_id)

    # Check if there are any trades to log
    if not all_trades:
        print(f"⚠️ No trades to log for run {run_id}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "trades_detailed.txt")

    # Build trade-to-window mapping
    # Assign trades to windows based on their open_time_utc
    window_trade_map = defaultdict(list)

    for trade in all_trades:
        trade_time = trade.get("open_time_utc", "")
        assigned = False

        for window in window_results:
            w_start = window.get("trade_start", "")
            w_end = window.get("trade_end", "")

            try:
                # Compare dates (simple string comparison works for YYYY-MM-DD format)
                trade_date = str(trade_time)[:10]
                if w_start <= trade_date < w_end:
                    window_trade_map[window.get("window_id", 0)].append(trade)
                    assigned = True
                    break
            except Exception:
                pass

        if not assigned and window_results:
            # Assign to last window if not matched
            window_trade_map[window_results[-1].get("window_id", 0)].append(trade)

    # Write the log
    with open(filepath, "w", encoding="utf-8") as f:
        # Header
        f.write(format_header(run_id, mode, start_date, end_date,
                             total_trades, win_rate, total_pnl))

        # Each window
        for window in window_results:
            window_id = window.get("window_id", 0)
            window_trades = window_trade_map.get(window_id, [])

            if window.get("skipped"):
                f.write(f"\n  [Window #{window_id} skipped - insufficient data]\n")
                continue

            # Window header
            f.write(format_window_header(window))

            # Individual trades
            for i, trade in enumerate(window_trades, 1):
                f.write("\n")
                f.write(format_trade_box(trade, i))

            # Window summary
            if window_trades:
                f.write("\n")
                f.write(format_window_summary(window, window_trades))

        # Overall summary
        f.write(format_overall_summary(all_trades, metrics, window_results))
        f.write(format_top_trades(all_trades))
        f.write(format_issues_summary(all_trades))

    print(f"📝 Trade log saved: {filepath}")
    return filepath

from desktop_bot_refactored_v2_base_v7 import (
    run_rolling_walkforward,
    compare_rolling_modes,
    BASELINE_CONFIG,
    SYMBOLS,
    TIMEFRAMES,
)


def run_quick_test():
    """Hızlı test - 3 ay, az sembol"""
    print("\n" + "="*70)
    print("🧪 HIZLI TEST (3 ay, 3 sembol)")
    print("="*70 + "\n")

    result = run_rolling_walkforward(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframes=["15m", "1h"],
        mode="monthly",
        lookback_days=60,
        forward_days=30,
        start_date="2025-09-01",
        end_date="2025-12-01",
        verbose=True,
    )

    # Write detailed trade log
    if result.get("trades"):
        write_trade_log(result)

    return result


def run_comparison_test(start_date: str = None, end_date: str = None):
    """Fixed vs Monthly vs Weekly vs 5day vs Triday karşılaştırma testi"""
    print("\n" + "="*70)
    print("🔬 ROLLING WALK-FORWARD KARŞILAŞTIRMA TESTİ")
    print("="*70 + "\n")

    # Use BASELINE_CONFIG for fixed mode
    result = compare_rolling_modes(
        symbols=SYMBOLS,  # Tüm semboller
        timeframes=TIMEFRAMES,  # Tüm timeframe'ler
        start_date=start_date or "2025-06-01",
        end_date=end_date or "2025-12-18",
        fixed_config=BASELINE_CONFIG,
        verbose=True,
    )

    # Write detailed trade logs for each mode
    # Note: compare_rolling_modes returns {"results": {...}, "comparison": {...}}
    mode_results = result.get("results", {})
    for mode in ["fixed", "monthly", "weekly", "5day", "triday"]:
        mode_result = mode_results.get(mode, {})
        if mode_result.get("trades"):
            write_trade_log(mode_result)

    return result


def run_full_year_test():
    """2025 tam yıl testi"""
    print("\n" + "="*70)
    print("📊 2025 TAM YIL TESTİ")
    print("="*70 + "\n")

    result = compare_rolling_modes(
        symbols=SYMBOLS,  # Tüm semboller
        timeframes=TIMEFRAMES,  # Tüm timeframe'ler
        start_date="2025-01-01",
        end_date="2025-12-18",
        fixed_config=BASELINE_CONFIG,
        verbose=True,
    )

    # Write detailed trade logs for each mode
    # Note: compare_rolling_modes returns {"results": {...}, "comparison": {...}}
    mode_results = result.get("results", {})
    for mode in ["fixed", "monthly", "weekly", "5day", "triday"]:
        mode_result = mode_results.get(mode, {})
        if mode_result.get("trades"):
            write_trade_log(mode_result)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Walk-Forward Test Script")
    parser.add_argument('--quick', action='store_true', help='Hızlı test (3 ay, az sembol)')
    parser.add_argument('--full-year', action='store_true', help='2025 tam yıl testi')
    parser.add_argument('--start-date', type=str, help='Başlangıç tarihi YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='Bitiş tarihi YYYY-MM-DD')

    args = parser.parse_args()

    if args.quick:
        result = run_quick_test()
    elif args.full_year:
        result = run_full_year_test()
    else:
        result = run_comparison_test(args.start_date, args.end_date)

    print("\n" + "="*70)
    print("✅ TEST TAMAMLANDI")
    print("="*70)

    # Print summary if comparison was run
    if "comparison" in result:
        comp = result["comparison"]
        print(f"\n🏆 EN İYİ MOD: {comp['best_mode'].upper()}")
        print(f"   PnL: Fixed=${comp['pnl']['fixed']:.2f}, Monthly=${comp['pnl']['monthly']:.2f}, Weekly=${comp['pnl']['weekly']:.2f}, 5day=${comp['pnl']['5day']:.2f}, Triday=${comp['pnl']['triday']:.2f}")


if __name__ == "__main__":
    main()
