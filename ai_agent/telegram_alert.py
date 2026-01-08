"""
Telegram Alert for AI Trading System
=====================================
v2.0 - การแจ้งเตือน Telegram ภาษาไทย

Features:
- แจ้งเตือนเปิดออเดอร์
- แจ้งเตือนปิดออเดอร์
- แจ้งเตือนโดน SL/TP
- สรุปผลประจำวัน
- แจ้งเตือน Error
- แจ้งเตือน Recovery Mode
"""

import requests
from datetime import datetime
from typing import Dict, Optional
from loguru import logger


class TelegramAlert:
    """
    Telegram Bot สำหรับแจ้งเตือนการเทรด
    
    วิธีตั้งค่า:
    1. สร้าง Bot ที่ @BotFather
    2. ได้รับ token และ chat_id
    3. ตั้งค่าใน ai_agent/trading_config.py
    """
    
    def __init__(
        self,
        bot_token: str = None,
        chat_id: str = None,
        enabled: bool = True,
    ):
        # โหลดจาก config ถ้าไม่ได้ระบุ
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
        
        # สถิติ
        self.messages_sent = 0
        self.last_message_time: Optional[datetime] = None
        
        # จำกัดความถี่
        self.min_interval_seconds = 3
        
        if self.enabled:
            logger.info("TelegramAlert เปิดใช้งานแล้ว")
        else:
            logger.info("TelegramAlert ปิดอยู่ (ไม่มี token/chat_id)")
    
    def _can_send(self) -> bool:
        """ตรวจสอบ rate limit"""
        if not self.enabled:
            return False
        
        if self.last_message_time:
            elapsed = (datetime.now() - self.last_message_time).total_seconds()
            if elapsed < self.min_interval_seconds:
                return False
        
        return True
    
    def _send(self, text: str) -> bool:
        """ส่งข้อความไป Telegram"""
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
                logger.warning(f"ส่ง Telegram ล้มเหลว: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    # ============================================
    # การแจ้งเตือนการเทรด
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
        ticket: int = None,
    ):
        """แจ้งเตือนเปิดออเดอร์"""
        emoji = "🟢" if direction in ["LONG", "BUY"] else "🔴"
        dir_th = "ซื้อ (LONG)" if direction in ["LONG", "BUY"] else "ขาย (SHORT)"
        
        msg = f"""
{emoji} <b>เปิดออเดอร์แล้ว</b>

📊 <b>{symbol}</b> {dir_th}
🎫 Ticket: #{ticket or 'N/A'}
💰 ราคาเข้า: {price:.2f}
📦 Lot: {lot}
🛑 Stop Loss: {sl:.2f}
🎯 Take Profit: {tp:.2f}
💪 ความมั่นใจ: {confidence:.0%}
🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
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
        ticket: int = None,
        lot: float = None,
    ):
        """แจ้งเตือนปิดออเดอร์"""
        emoji = "✅" if pnl > 0 else "❌"
        dir_th = "ซื้อ (LONG)" if direction in ["LONG", "BUY"] else "ขาย (SHORT)"
        
        # แปลง reason เป็นภาษาไทย
        reason_th = self._translate_close_reason(reason)
        
        pnl_emoji = "💰" if pnl > 0 else "💸"
        pnl_text = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        
        msg = f"""
{emoji} <b>ปิดออเดอร์แล้ว</b>

📊 <b>{symbol}</b> {dir_th}
🎫 Ticket: #{ticket or 'N/A'}
📦 Lot: {lot or 'N/A'}
📥 ราคาเข้า: {entry_price:.2f}
📤 ราคาออก: {exit_price:.2f}
{pnl_emoji} กำไร/ขาดทุน: {pnl_text}
📝 เหตุผล: {reason_th}
🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_sl_hit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        pnl: float,
        ticket: int = None,
        lot: float = None,
    ):
        """แจ้งเตือนโดน Stop Loss"""
        dir_th = "ซื้อ (LONG)" if direction in ["LONG", "BUY"] else "ขาย (SHORT)"
        
        msg = f"""
🛑 <b>โดน STOP LOSS</b>

📊 <b>{symbol}</b> {dir_th}
🎫 Ticket: #{ticket or 'N/A'}
📦 Lot: {lot or 'N/A'}
📥 ราคาเข้า: {entry_price:.2f}
🛑 ราคา SL: {sl_price:.2f}
💸 ขาดทุน: -${abs(pnl):.2f}
🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

⚠️ ขาดทุนครั้งนี้อยู่ในขีดจำกัดที่ตั้งไว้
"""
        self._send(msg.strip())
    
    def alert_tp_hit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        tp_price: float,
        pnl: float,
        ticket: int = None,
        lot: float = None,
    ):
        """แจ้งเตือนโดน Take Profit"""
        dir_th = "ซื้อ (LONG)" if direction in ["LONG", "BUY"] else "ขาย (SHORT)"
        
        msg = f"""
🎯 <b>ถึง TAKE PROFIT!</b>

📊 <b>{symbol}</b> {dir_th}
🎫 Ticket: #{ticket or 'N/A'}
📦 Lot: {lot or 'N/A'}
📥 ราคาเข้า: {entry_price:.2f}
🎯 ราคา TP: {tp_price:.2f}
💰 กำไร: +${pnl:.2f}
🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

✨ ยินดีด้วย! ออเดอร์นี้กำไร!
"""
        self._send(msg.strip())
    
    def alert_trailing_stop(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        ts_price: float,
        pnl: float,
        ticket: int = None,
    ):
        """แจ้งเตือนโดน Trailing Stop"""
        dir_th = "ซื้อ (LONG)" if direction in ["LONG", "BUY"] else "ขาย (SHORT)"
        emoji = "💰" if pnl > 0 else "💸"
        pnl_text = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        
        msg = f"""
🔄 <b>ปิดด้วย TRAILING STOP</b>

📊 <b>{symbol}</b> {dir_th}
🎫 Ticket: #{ticket or 'N/A'}
📥 ราคาเข้า: {entry_price:.2f}
🔄 ราคา TS: {ts_price:.2f}
{emoji} กำไร/ขาดทุน: {pnl_text}
🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

📈 Trailing Stop ช่วยล็อคกำไร!
"""
        self._send(msg.strip())
    
    def _translate_close_reason(self, reason: str) -> str:
        """แปลเหตุผลปิดออเดอร์เป็นภาษาไทย"""
        translations = {
            "sl": "โดน Stop Loss",
            "stop_loss": "โดน Stop Loss",
            "tp": "ถึง Take Profit",
            "take_profit": "ถึง Take Profit",
            "trailing_stop": "Trailing Stop",
            "manual": "ปิดด้วยตนเอง",
            "signal": "สัญญาณกลับตัว",
            "reverse_signal": "สัญญาณกลับตัว",
            "time_exit": "หมดเวลาถือ",
            "risk_limit": "ถึงขีดจำกัดความเสี่ยง",
            "daily_limit": "ถึงขีดจำกัดรายวัน",
            "Close by AI": "AI ปิดออเดอร์",
        }
        return translations.get(reason.lower() if isinstance(reason, str) else reason, reason)
    
    # ============================================
    # สรุปผลประจำวัน
    # ============================================
    
    def send_daily_summary(
        self,
        trades: int,
        wins: int,
        total_pnl: float,
        equity: float,
        drawdown: float,
    ):
        """ส่งสรุปผลประจำวัน"""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        losses = trades - wins
        emoji = "📈" if total_pnl > 0 else "📉"
        pnl_emoji = "💰" if total_pnl > 0 else "💸"
        pnl_text = f"+${total_pnl:.2f}" if total_pnl > 0 else f"-${abs(total_pnl):.2f}"
        
        msg = f"""
{emoji} <b>สรุปผลประจำวัน</b>
📅 วันที่: {datetime.now().strftime('%d/%m/%Y')}

📊 จำนวนเทรด: {trades} ครั้ง
✅ ชนะ: {wins} ครั้ง
❌ แพ้: {losses} ครั้ง
📈 Win Rate: {win_rate:.0f}%
{pnl_emoji} กำไร/ขาดทุนวันนี้: {pnl_text}
💼 ยอดทุนปัจจุบัน: ${equity:,.2f}
📉 Drawdown: {drawdown:.1%}

🤖 AI Trading System
"""
        self._send(msg.strip())
    
    # ============================================
    # แจ้งเตือนระบบ
    # ============================================
    
    def alert_error(self, error_msg: str):
        """แจ้งเตือน Error"""
        msg = f"""
🚨 <b>เกิดข้อผิดพลาด!</b>

{error_msg}

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

⚠️ กรุณาตรวจสอบระบบ
"""
        self._send(msg.strip())
    
    def alert_recovery_mode(self, is_activated: bool, drawdown: float):
        """แจ้งเตือน Recovery Mode"""
        if is_activated:
            msg = f"""
⚠️ <b>เข้าสู่โหมดกู้คืน</b>

📉 Drawdown: {drawdown:.1%}
📦 ลดขนาดล็อตเหลือ 50%
⏸️ ระบบจะเทรดระมัดระวังขึ้น

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        else:
            msg = f"""
✅ <b>ออกจากโหมดกู้คืน</b>

📈 ระบบกลับสู่ปกติ
📦 ขนาดล็อตกลับเป็น 100%
🚀 พร้อมเทรดเต็มกำลัง!

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_daily_limit(self, daily_loss: float, limit: float):
        """แจ้งเตือนถึงขีดจำกัดขาดทุนรายวัน"""
        msg = f"""
🛑 <b>ถึงขีดจำกัดขาดทุนรายวัน!</b>

💸 ขาดทุนวันนี้: -${abs(daily_loss):.2f}
📊 ขีดจำกัด: {limit:.0%}
⏸️ หยุดเทรดจนถึงพรุ่งนี้

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

⚠️ ระบบจะเริ่มเทรดใหม่อัตโนมัติในวันถัดไป
"""
        self._send(msg.strip())
    
    def alert_position_modified(
        self,
        symbol: str,
        ticket: int,
        old_sl: float,
        new_sl: float,
        old_tp: float = None,
        new_tp: float = None,
    ):
        """แจ้งเตือนแก้ไข SL/TP"""
        sl_change = "🔼" if new_sl > old_sl else "🔽" if new_sl < old_sl else "➡️"
        
        msg = f"""
🔧 <b>แก้ไขออเดอร์</b>

📊 <b>{symbol}</b>
🎫 Ticket: #{ticket}
🛑 SL: {old_sl:.2f} {sl_change} {new_sl:.2f}
"""
        if old_tp and new_tp:
            tp_change = "🔼" if new_tp > old_tp else "🔽" if new_tp < old_tp else "➡️"
            msg += f"🎯 TP: {old_tp:.2f} {tp_change} {new_tp:.2f}\n"
        
        msg += f"🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        
        self._send(msg.strip())
    
    def alert_system_start(self):
        """แจ้งเตือนระบบเริ่มทำงาน"""
        msg = f"""
🚀 <b>ระบบเริ่มทำงาน</b>

🤖 AI Trading System Online
⚡ พร้อมเทรดอัตโนมัติ
📊 กำลังติดตามตลาด...

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
"""
        self._send(msg.strip())
    
    def alert_system_stop(self, reason: str = "Manual"):
        """แจ้งเตือนระบบหยุดทำงาน"""
        reason_th = {
            "Manual": "หยุดด้วยตนเอง",
            "Error": "เกิดข้อผิดพลาด",
            "Risk Limit": "ถึงขีดจำกัดความเสี่ยง",
            "Maintenance": "บำรุงรักษา",
        }.get(reason, reason)
        
        msg = f"""
⏹️ <b>ระบบหยุดทำงาน</b>

📝 เหตุผล: {reason_th}

🕐 เวลา: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

⚠️ ระบบจะไม่เทรดจนกว่าจะเริ่มใหม่
"""
        self._send(msg.strip())
    
    def send_custom(self, message: str):
        """ส่งข้อความกำหนดเอง"""
        self._send(message)


# ============================================
# Singleton
# ============================================

_telegram: Optional[TelegramAlert] = None

def get_telegram(token: str = None, chat_id: str = None) -> TelegramAlert:
    """รับ TelegramAlert instance (Singleton)"""
    global _telegram
    if _telegram is None:
        _telegram = TelegramAlert(token, chat_id)
    return _telegram


if __name__ == "__main__":
    # ทดสอบ (จะล้มเหลวถ้าไม่มี token จริง)
    t = TelegramAlert()
    print(f"เปิดใช้งาน: {t.enabled}")
    print("ตั้งค่า BOT_TOKEN และ CHAT_ID ใน trading_config.py เพื่อเปิดใช้งาน")
