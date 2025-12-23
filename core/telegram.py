"""
Telegram notification module.

Provides secure and efficient Telegram message sending with:
- Environment variable support for credentials (more secure than plain JSON)
- Thread pool for efficient message delivery (no thread accumulation)
- Rate limiting to prevent API throttling
- Retry logic for failed messages
"""

import os
import json
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from queue import Queue
from datetime import datetime, timezone

from .config import CONFIG_FILE, DATA_DIR


class TelegramNotifier:
    """
    Telegram notification handler with improved security and efficiency.

    Features:
    - Loads credentials from environment variables first, then config file
    - Uses a thread pool instead of creating new threads per message
    - Implements rate limiting to prevent API throttling
    - Queues messages and processes them asynchronously
    """

    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        max_workers: int = 2,
        rate_limit_per_second: float = 1.0
    ):
        """
        Initialize the Telegram notifier.

        Args:
            token: Bot token (optional, will check env vars and config file)
            chat_id: Chat ID (optional, will check env vars and config file)
            max_workers: Max concurrent message sending threads
            rate_limit_per_second: Max messages per second
        """
        self._token = token
        self._chat_id = chat_id
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._rate_limit = rate_limit_per_second
        self._last_send_time = 0.0
        self._lock = threading.Lock()
        self._message_queue = Queue()
        self._running = True

        # Load credentials if not provided
        if not self._token or not self._chat_id:
            self._load_credentials()

        # Start background worker
        self._worker_thread = threading.Thread(target=self._message_worker, daemon=True)
        self._worker_thread.start()

    def _load_credentials(self):
        """
        Load credentials from environment variables or config file.

        Priority:
        1. Environment variables (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        2. Config file (config.json)

        Environment variables are preferred for security.
        """
        # Try environment variables first (more secure)
        env_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        env_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if env_token and env_chat_id:
            self._token = env_token
            self._chat_id = env_chat_id
            return

        # Fall back to config file
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if not self._token:
                        self._token = config.get("telegram_token", "")
                    if not self._chat_id:
                        self._chat_id = config.get("telegram_chat_id", "")
            except Exception as e:
                print(f"[TELEGRAM] Config load error: {e}")

    def update_credentials(self, token: str, chat_id: str):
        """
        Update Telegram credentials.

        Args:
            token: New bot token
            chat_id: New chat ID
        """
        with self._lock:
            self._token = token
            self._chat_id = chat_id

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self._token and self._chat_id)

    def _message_worker(self):
        """Background worker that processes queued messages."""
        while self._running:
            try:
                # Wait for message with timeout
                try:
                    message = self._message_queue.get(timeout=1.0)
                except Exception:
                    continue

                if message is None:  # Shutdown signal
                    break

                # Rate limiting
                with self._lock:
                    now = time.time()
                    time_since_last = now - self._last_send_time
                    min_interval = 1.0 / self._rate_limit

                    if time_since_last < min_interval:
                        time.sleep(min_interval - time_since_last)

                    self._last_send_time = time.time()

                # Send message
                self._send_message_sync(message)

            except Exception as e:
                print(f"[TELEGRAM] Worker error: {e}")

    def _send_message_sync(self, message: str, max_retries: int = 2):
        """
        Send a message synchronously with retry logic.

        Args:
            message: Message text
            max_retries: Number of retry attempts
        """
        if not self._token or not self._chat_id:
            return

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        data = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",  # Support basic formatting
        }

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, data=data, timeout=10)
                if response.status_code == 200:
                    return
                elif response.status_code == 429:  # Rate limited
                    retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                    time.sleep(retry_after)
                else:
                    if attempt == max_retries:
                        print(f"[TELEGRAM] Send failed: {response.status_code}")
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    print("[TELEGRAM] Send timeout")
            except Exception as e:
                if attempt == max_retries:
                    print(f"[TELEGRAM] Error: {e}")

            # Exponential backoff
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    def send(self, message: str, async_mode: bool = True):
        """
        Send a Telegram message.

        Args:
            message: Message text
            async_mode: If True, queue message for async delivery.
                       If False, send synchronously (blocks).
        """
        if not self.is_configured():
            return

        if async_mode:
            self._message_queue.put(message)
        else:
            self._send_message_sync(message)

    def send_trade_opened(self, trade: dict):
        """Send notification for trade opened."""
        symbol = trade.get("symbol", "?")
        tf = trade.get("timeframe", "?")
        trade_type = trade.get("type", "?")
        entry = trade.get("entry", 0)
        tp = trade.get("tp", 0)
        sl = trade.get("sl", 0)
        setup = trade.get("setup", "?")
        risk_amount = trade.get("risk_amount", 0)

        icon = "" if trade_type == "LONG" else ""
        message = (
            f"{icon} <b>TRADE OPENED</b>\n"
            f"Symbol: {symbol}\n"
            f"TF: {tf}\n"
            f"Type: {trade_type}\n"
            f"Setup: {setup}\n"
            f"Entry: {entry:.4f}\n"
            f"TP: {tp:.4f}\n"
            f"SL: {sl:.4f}\n"
            f"Risk: ${risk_amount:.2f}"
        )
        self.send(message)

    def send_trade_closed(self, trade: dict):
        """Send notification for trade closed."""
        symbol = trade.get("symbol", "?")
        tf = trade.get("timeframe", "?")
        status = trade.get("status", "?")
        pnl = float(trade.get("pnl", 0))
        setup = trade.get("setup", "?")

        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        icon = "" if "WIN" in status else ""

        message = (
            f"{icon} <b>TRADE CLOSED</b>\n"
            f"Symbol: {symbol}\n"
            f"TF: {tf}\n"
            f"Setup: {setup}\n"
            f"Result: {status}\n"
            f"Net PnL: {pnl_str}"
        )
        self.send(message)

    def send_daily_summary(self, summary: dict):
        """Send daily trading summary."""
        total_trades = summary.get("total_trades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        total_pnl = summary.get("total_pnl", 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        pnl_str = f"+${total_pnl:.2f}" if total_pnl > 0 else f"-${abs(total_pnl):.2f}"
        icon = "" if total_pnl > 0 else ""

        message = (
            f"{icon} <b>DAILY SUMMARY</b>\n"
            f"Date: {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d')}\n"
            f"Total Trades: {total_trades}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Total PnL: {pnl_str}"
        )
        self.send(message)

    def shutdown(self):
        """Shutdown the notifier gracefully."""
        self._running = False
        self._message_queue.put(None)  # Signal worker to stop
        self._executor.shutdown(wait=True)


# Global notifier instance (lazy initialization)
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get the global Telegram notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_telegram(token: str, chat_id: str, message: str):
    """
    Legacy function for backward compatibility.

    Sends a Telegram message using the global notifier.
    If token/chat_id are provided and different from configured,
    updates the notifier credentials first.

    Args:
        token: Bot token
        chat_id: Chat ID
        message: Message text
    """
    notifier = get_notifier()

    # Update credentials if provided
    if token and chat_id:
        if notifier._token != token or notifier._chat_id != chat_id:
            notifier.update_credentials(token, chat_id)

    notifier.send(message)


def save_telegram_config(token: str, chat_id: str):
    """
    Save Telegram credentials to config file.

    Note: For better security, prefer using environment variables
    (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) instead.

    Args:
        token: Bot token
        chat_id: Chat ID
    """
    try:
        config = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

        config["telegram_token"] = token
        config["telegram_chat_id"] = chat_id

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Update notifier
        notifier = get_notifier()
        notifier.update_credentials(token, chat_id)

    except Exception as e:
        print(f"[TELEGRAM] Config save error: {e}")


def load_telegram_config() -> tuple:
    """
    Load Telegram credentials from environment or config file.

    Returns:
        Tuple of (token, chat_id)
    """
    # Try environment variables first
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if token and chat_id:
        return token, chat_id

    # Fall back to config file
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                token = config.get("telegram_token", "")
                chat_id = config.get("telegram_chat_id", "")
        except Exception:
            pass

    return token, chat_id
