"""
Telegram Alert for AI Trading System
=====================================
v1.0 - Trade Notifications & Daily Reports

Features:
- Trade entry/exit alerts
- Daily P&L summary
- Error notifications
- Recovery mode alerts
"""

import requests
from datetime import datetime
from typing import Dict, Optional
from loguru import logger


class TelegramAlert:
    """
    Telegram Bot for Trading Alerts
    
    Setup:
    1. Create bot with @BotFather
    2. Get token and chat_id
    3. Set in ai_agent/trading_config.py
    """
    
    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None,
        enabled: bool = True,
    ):
        # Load from config if not provided
        if bot_token is None or chat_id is None:
            try:
                from ai_agent.trading_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                bot_token = bot_token or TELEGRAM_BOT_TOKEN
                chat_id = chat_id or TELEGRAM_CHAT_ID
            except ImportError:
                pass
        
        self.bot_token = bot_token or ""
        self.chat_id = chat_id or ""
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Stats
        self.messages_sent = 0
        self.last_message_time: Optional[datetime] = None
        
        # Rate limiting
        self.min_interval_seconds = 5  # Don't spam
        
        if self.enabled:
            logger.info("TelegramAlert initialized")
        else:
            logger.info("TelegramAlert disabled (no token/chat_id)")
    
    def _can_send(self) -> bool:
        """Check rate limit"""
        if not self.enabled:
            return False
        
        if self.last_message_time:
            elapsed = (datetime.now() - self.last_message_time).total_seconds()
            if elapsed < self.min_interval_seconds:
                return False
        
        return True
    
    def _send(self, text: str) -> bool:
        """Send message to Telegram"""
        if not self._can_send():
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                self.messages_sent += 1
                self.last_message_time = datetime.now()
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    # ============================================
    # Trade Alerts
    # ============================================
    
    def alert_trade_entry(
        self,
        symbol: str,
        direction: str,
        price: float,
        lot: float,
        sl: float,
        tp: float,
        confidence: float,
    ):
        """Send trade entry alert"""
        emoji = "🟢" if direction == "LONG" else "🔴"
        
        msg = f"""
{emoji} <b>TRADE OPENED</b>

📊 <b>{symbol}</b> {direction}
💰 Entry: {price:.2f}
📦 Lot: {lot}
🛑 SL: {sl:.2f}
🎯 TP: {tp:.2f}
💪 Confidence: {confidence:.0%}
🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_trade_exit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
    ):
        """Send trade exit alert"""
        emoji = "✅" if pnl > 0 else "❌"
        
        msg = f"""
{emoji} <b>TRADE CLOSED</b>

📊 <b>{symbol}</b> {direction}
📥 Entry: {entry_price:.2f}
📤 Exit: {exit_price:.2f}
{'💰' if pnl > 0 else '💸'} P&L: ${pnl:+.2f}
📝 Reason: {reason}
🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        self._send(msg.strip())
    
    # ============================================
    # Daily Reports
    # ============================================
    
    def send_daily_summary(
        self,
        trades: int,
        wins: int,
        total_pnl: float,
        equity: float,
        drawdown: float,
    ):
        """Send daily P&L summary"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        emoji = "📈" if total_pnl > 0 else "📉"
        
        msg = f"""
{emoji} <b>DAILY SUMMARY</b>
📅 {datetime.now().strftime('%Y-%m-%d')}

📊 Trades: {trades}
✅ Wins: {wins} ({win_rate:.0f}%)
{'💰' if total_pnl > 0 else '💸'} P&L: ${total_pnl:+.2f}
💼 Equity: ${equity:,.2f}
📉 Drawdown: {drawdown:.1%}
"""
        self._send(msg.strip())
    
    # ============================================
    # System Alerts
    # ============================================
    
    def alert_error(self, error_msg: str):
        """Send error alert"""
        msg = f"""
🚨 <b>ERROR</b>

{error_msg}

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_recovery_mode(self, is_activated: bool, drawdown: float):
        """Alert when recovery mode changes"""
        if is_activated:
            msg = f"""
⚠️ <b>RECOVERY MODE ACTIVATED</b>

📉 Drawdown: {drawdown:.1%}
📦 Position size reduced to 50%

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        else:
            msg = f"""
✅ <b>RECOVERY MODE DEACTIVATED</b>

📈 System back to normal
📦 Position size restored to 100%

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_daily_limit(self, daily_loss: float, limit: float):
        """Alert when daily loss limit hit"""
        msg = f"""
🛑 <b>DAILY LOSS LIMIT HIT</b>

💸 Daily Loss: ${daily_loss:.2f}
📊 Limit: {limit:.0%}
⏸️ Trading paused until tomorrow

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        self._send(msg.strip())
    
    def send_custom(self, message: str):
        """Send custom message"""
        self._send(message)


# ============================================
# Singleton
# ============================================

_telegram: Optional[TelegramAlert] = None

def get_telegram(token: str = None, chat_id: str = None) -> TelegramAlert:
    """Get singleton TelegramAlert instance"""
    global _telegram
    if _telegram is None:
        _telegram = TelegramAlert(token, chat_id)
    return _telegram


if __name__ == "__main__":
    # Test (will fail without real token)
    t = TelegramAlert()
    print(f"Enabled: {t.enabled}")
    print("Set BOT_TOKEN and CHAT_ID in .env to enable alerts")
