"""
MT5 Trade Executor - Elite Version
===================================
Advanced trade execution with professional risk management

Features:
- 🎯 Market order execution (BUY/SELL)
- 🛡️ Advanced risk management
- 📊 Position management
- ⚡ Smart order routing
- 🔄 Order modification
- 📈 Slippage tracking
- 🔍 Execution quality monitoring

Author: AI Trading System
Version: 2.0.0 - Elite Production
"""

import MetaTrader5 as mt5
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
import time

# Import Telegram Alert
try:
    from ai_agent.telegram_alert import get_telegram, TelegramAlert
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("TelegramAlert ไม่พร้อมใช้งาน")


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[int] = None
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: float = 0.0
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    slippage: float = 0.0
    execution_time: float = 0.0


class MT5TradeExecutor:
    """
    Elite MT5 Trade Executor

    Handles all trading operations with professional-grade
    risk management and execution quality monitoring
    """

    def __init__(self, config=None, master_brain=None):
        """
        Initialize trade executor

        Args:
            config: Configuration manager (optional)
            master_brain: MasterBrain instance for learning (optional)
        """
        self.config = config
        self.max_slippage = 50  # Maximum allowed slippage in points
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Telegram Alert
        self.telegram = get_telegram() if TELEGRAM_AVAILABLE else None
        
        # MasterBrain for learning
        self.master_brain = master_brain
        
        # Track positions for SL/TP detection
        self._tracked_positions: Dict[int, Dict] = {}
        
        # Track market state for learning
        self._position_market_states: Dict[int, Dict] = {}

        logger.info("="*70)
        logger.info("  ELITE MT5 TRADE EXECUTOR v2.1.0")
        logger.info("="*70)
        logger.info(f"  Max Slippage: {self.max_slippage} points")
        logger.info(f"  Max Retries: {self.max_retries}")
        logger.info(f"  Telegram: {'เปิด' if self.telegram and self.telegram.enabled else 'ปิด'}")
        logger.info(f"  MasterBrain Learning: {'เปิด' if self.master_brain else 'ปิด'}")
        logger.info("="*70)
    
    def set_master_brain(self, master_brain):
        """ตั้งค่า MasterBrain สำหรับ learning"""
        self.master_brain = master_brain
        logger.info("MasterBrain connected for auto-learning")

    def execute_market_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "AI Trading System",
        magic: int = 234000
    ) -> TradeResult:
        """
        Execute market order with advanced error handling

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            order_type: "BUY" or "SELL"
            volume: Position size in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            comment: Order comment
            magic: Magic number for order identification

        Returns:
            TradeResult with execution details
        """

        start_time = time.time()

        logger.info(f"Executing {order_type} order: {volume} lots of {symbol}")

        # Validate inputs
        if not self._validate_order_params(symbol, order_type, volume):
            return TradeResult(
                success=False,
                error_message="Invalid order parameters"
            )

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return TradeResult(
                success=False,
                error_message=f"Symbol {symbol} not found"
            )

        # Enable symbol if needed
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return TradeResult(
                    success=False,
                    error_message=f"Failed to enable symbol {symbol}"
                )

        # Prepare request
        request = self._prepare_order_request(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
            magic=magic
        )

        if not request:
            return TradeResult(
                success=False,
                error_message="Failed to prepare order request"
            )

        # Execute with retries
        result = self._execute_with_retry(request)

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        if result.success:
            logger.success(
                f"✓ Order executed: Ticket={result.ticket}, "
                f"Price={result.price:.2f}, Time={execution_time:.2f}s"
            )
        else:
            logger.error(
                f"✗ Order failed: {result.error_message} "
                f"(Code: {result.error_code})"
            )

        return result

    def _validate_order_params(
        self,
        symbol: str,
        order_type: str,
        volume: float
    ) -> bool:
        """Validate order parameters"""

        if order_type not in ["BUY", "SELL"]:
            logger.error(f"Invalid order type: {order_type}")
            return False

        if volume <= 0:
            logger.error(f"Invalid volume: {volume}")
            return False

        # Check symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol not found: {symbol}")
            return False

        # Check volume limits
        if volume < symbol_info.volume_min:
            logger.error(
                f"Volume {volume} below minimum {symbol_info.volume_min}"
            )
            return False

        if volume > symbol_info.volume_max:
            logger.error(
                f"Volume {volume} above maximum {symbol_info.volume_max}"
            )
            return False

        return True

    def _prepare_order_request(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        comment: str,
        magic: int
    ) -> Optional[Dict]:
        """Prepare MT5 order request"""

        # Get current prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("Failed to get current tick")
            return None

        # Determine action and price
        if order_type == "BUY":
            action = mt5.TRADE_ACTION_DEAL
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:  # SELL
            action = mt5.TRADE_ACTION_DEAL
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Get symbol info for point value
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point
        digits = symbol_info.digits

        # Prepare request
        request = {
            "action": action,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "deviation": self.max_slippage,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
        }

        # Add SL/TP if provided
        if stop_loss is not None:
            request["sl"] = round(stop_loss, digits)

        if take_profit is not None:
            request["tp"] = round(take_profit, digits)

        return request

    def _execute_with_retry(self, request: Dict) -> TradeResult:
        """Execute order with retry logic"""

        last_error = None

        for attempt in range(self.max_retries):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)

            # Send order
            result = mt5.order_send(request)

            if result is None:
                last_error = "order_send returned None"
                continue

            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success!
                executed_price = result.price
                requested_price = request["price"]
                slippage = abs(executed_price - requested_price)

                return TradeResult(
                    success=True,
                    order_id=result.order,
                    ticket=result.order,
                    price=executed_price,
                    volume=result.volume,
                    slippage=slippage
                )

            # Handle specific error codes
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                # Price changed, update and retry
                logger.warning("Requote received, updating price...")
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    if request["type"] == mt5.ORDER_TYPE_BUY:
                        request["price"] = tick.ask
                    else:
                        request["price"] = tick.bid
                last_error = f"Requote (retcode={result.retcode})"

            elif result.retcode == mt5.TRADE_RETCODE_PRICE_OFF:
                logger.warning("Price off, retrying...")
                last_error = f"Price off (retcode={result.retcode})"

            elif result.retcode == mt5.TRADE_RETCODE_TIMEOUT:
                logger.warning("Timeout, retrying...")
                last_error = f"Timeout (retcode={result.retcode})"

            else:
                # Other error, don't retry
                error_msg = self._get_error_message(result.retcode)
                return TradeResult(
                    success=False,
                    error_code=result.retcode,
                    error_message=error_msg
                )

        # All retries failed
        return TradeResult(
            success=False,
            error_message=f"Max retries exceeded: {last_error}"
        )

    def _get_error_message(self, retcode: int) -> str:
        """Get human-readable error message"""

        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Partially filled",
            mt5.TRADE_RETCODE_ERROR: "Request error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Not enough money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Price off",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
            mt5.TRADE_RETCODE_NO_CHANGES: "No changes",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Auto trading disabled by server",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Auto trading disabled by client",
            mt5.TRADE_RETCODE_LOCKED: "Request locked",
            mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
            mt5.TRADE_RETCODE_INVALID_FILL: "Invalid fill type",
            mt5.TRADE_RETCODE_CONNECTION: "No connection",
            mt5.TRADE_RETCODE_ONLY_REAL: "Only real accounts allowed",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
        }

        return error_messages.get(retcode, f"Unknown error (retcode={retcode})")

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of position dictionaries
        """

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'magic': pos.magic,
                'comment': pos.comment,
                'time': datetime.fromtimestamp(pos.time)
            })

        return result

    def close_position(
        self,
        ticket: int,
        comment: str = "Close by AI"
    ) -> TradeResult:
        """
        Close an open position

        Args:
            ticket: Position ticket number
            comment: Close comment

        Returns:
            TradeResult
        """

        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return TradeResult(
                success=False,
                error_message=f"Position {ticket} not found"
            )

        position = position[0]

        # Prepare close request
        symbol = position.symbol
        volume = position.volume

        # Opposite order type
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.max_slippage,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Execute
        result = self._execute_with_retry(request)

        if result.success:
            logger.success(f"✓ Position {ticket} closed at {result.price:.2f}")
        else:
            logger.error(f"✗ Failed to close position {ticket}: {result.error_message}")

        return result

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeResult:
        """
        Modify position SL/TP

        Args:
            ticket: Position ticket
            stop_loss: New stop loss (optional)
            take_profit: New take profit (optional)

        Returns:
            TradeResult
        """

        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return TradeResult(
                success=False,
                error_message=f"Position {ticket} not found"
            )

        position = position[0]
        symbol = position.symbol

        # Get symbol info for digits
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
        }

        if stop_loss is not None:
            request["sl"] = round(stop_loss, digits)
        else:
            request["sl"] = position.sl

        if take_profit is not None:
            request["tp"] = round(take_profit, digits)
        else:
            request["tp"] = position.tp

        # Send request
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.success(f"✓ Position {ticket} modified")
            return TradeResult(success=True, ticket=ticket)
        else:
            error_msg = self._get_error_message(result.retcode) if result else "Unknown error"
            logger.error(f"✗ Failed to modify position {ticket}: {error_msg}")
            return TradeResult(
                success=False,
                error_code=result.retcode if result else None,
                error_message=error_msg
            )

    def calculate_position_size(
        self,
        symbol: str,
        risk_percent: float,
        stop_loss_pips: float,
        account_balance: float
    ) -> float:
        """
        Calculate position size based on risk

        Args:
            symbol: Trading symbol
            risk_percent: Risk percentage (e.g., 0.02 for 2%)
            stop_loss_pips: Stop loss distance in pips
            account_balance: Account balance

        Returns:
            Position size in lots
        """

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found")
            return 0.0

        # Calculate risk amount
        risk_amount = account_balance * risk_percent

        # Get point value
        point = symbol_info.point
        contract_size = symbol_info.trade_contract_size

        # Calculate position size
        stop_loss_points = stop_loss_pips * 10  # Convert pips to points
        risk_per_lot = stop_loss_points * point * contract_size

        if risk_per_lot == 0:
            return 0.0

        position_size = risk_amount / risk_per_lot

        # Round to step
        step = symbol_info.volume_step
        position_size = round(position_size / step) * step

        # Apply limits
        position_size = max(symbol_info.volume_min, position_size)
        position_size = min(symbol_info.volume_max, position_size)

        logger.info(
            f"Position size calculated: {position_size} lots "
            f"(Risk: ${risk_amount:.2f}, SL: {stop_loss_pips} pips)"
        )

        return position_size

    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information
        
        Returns:
            Account info dictionary or None
        """
        account = mt5.account_info()
        if not account:
            return None

        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'profit': account.profit,
            'leverage': account.leverage,
            'currency': account.currency
        }

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of position dictionaries
        """
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'magic': pos.magic,
                'comment': pos.comment,
                'time': pos.time
            })

        return result
    
    # ============================================
    # Position Tracking & SL/TP Detection
    # ============================================
    
    def track_position(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        volume: float,
        sl: float,
        tp: float,
    ):
        """
        เริ่มติดตาม Position เพื่อตรวจจับ SL/TP
        
        Args:
            ticket: หมายเลข ticket
            symbol: สัญลักษณ์
            direction: BUY หรือ SELL
            entry_price: ราคาเข้า
            volume: Lot size
            sl: Stop Loss
            tp: Take Profit
        """
        self._tracked_positions[ticket] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'volume': volume,
            'sl': sl,
            'tp': tp,
            'time_opened': datetime.now(),
        }
        logger.info(f"เริ่มติดตาม Position #{ticket}")
    
    def check_closed_positions_and_notify(self) -> List[Dict]:
        """
        ตรวจสอบ Positions ที่ถูกปิดและส่ง Telegram แจ้งเตือน
        
        Returns:
            รายการ positions ที่ถูกปิด
        """
        if not self._tracked_positions:
            return []
        
        closed_positions = []
        current_tickets = set()
        
        # ดึง positions ปัจจุบัน
        positions = mt5.positions_get()
        if positions:
            current_tickets = {pos.ticket for pos in positions}
        
        # ตรวจสอบว่า position ไหนถูกปิด
        for ticket, info in list(self._tracked_positions.items()):
            if ticket not in current_tickets:
                # Position ถูกปิดแล้ว!
                closed_info = self._get_closed_position_details(ticket, info)
                
                if closed_info:
                    closed_positions.append(closed_info)
                    self._send_close_notification(closed_info)
                
                # ลบออกจาก tracking
                del self._tracked_positions[ticket]
        
        return closed_positions
    
    def _get_closed_position_details(self, ticket: int, tracked_info: Dict) -> Optional[Dict]:
        """ดึงรายละเอียดของ position ที่ถูกปิด"""
        try:
            # ดึงจาก history
            from_date = tracked_info.get('time_opened', datetime.now())
            deals = mt5.history_deals_get(
                from_date,
                datetime.now(),
                position=ticket
            )
            
            if not deals:
                return None
            
            # หา deal ปิด (ที่ไม่ใช่ entry)
            close_deal = None
            for deal in deals:
                if deal.entry == 1:  # Entry out = ปิด position
                    close_deal = deal
                    break
            
            if not close_deal:
                return None
            
            # คำนวณข้อมูล
            entry_price = tracked_info['entry_price']
            exit_price = close_deal.price
            pnl = close_deal.profit
            sl = tracked_info['sl']
            tp = tracked_info['tp']
            direction = tracked_info['direction']
            
            # ตรวจสอบว่าโดน SL หรือ TP หรือปิดปกติ
            close_reason = self._detect_close_reason(
                direction, entry_price, exit_price, sl, tp, close_deal.comment
            )
            
            return {
                'ticket': ticket,
                'symbol': tracked_info['symbol'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'volume': tracked_info['volume'],
                'pnl': pnl,
                'sl': sl,
                'tp': tp,
                'close_reason': close_reason,
                'close_time': datetime.now(),
            }
            
        except Exception as e:
            logger.error(f"Error getting closed position details: {e}")
            return None
    
    def _detect_close_reason(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        sl: float,
        tp: float,
        comment: str = ""
    ) -> str:
        """ตรวจจับเหตุผลปิด Position"""
        # ตรวจสอบจาก comment ก่อน
        comment_lower = comment.lower() if comment else ""
        if "sl" in comment_lower or "stop" in comment_lower:
            return "sl"
        if "tp" in comment_lower or "profit" in comment_lower:
            return "tp"
        if "trailing" in comment_lower:
            return "trailing_stop"
        
        # ตรวจสอบจากราคา
        tolerance = 0.5  # tolerance in price units
        
        if sl and abs(exit_price - sl) < tolerance:
            return "sl"
        if tp and abs(exit_price - tp) < tolerance:
            return "tp"
        
        # ตรวจสอบจากทิศทางและกำไร/ขาดทุน
        if direction in ["BUY", "LONG"]:
            if exit_price <= sl if sl else exit_price < entry_price * 0.99:
                return "sl"
            if tp and exit_price >= tp:
                return "tp"
        else:  # SELL/SHORT
            if sl and exit_price >= sl:
                return "sl"
            if tp and exit_price <= tp:
                return "tp"
        
        return "manual"
    
    def _send_close_notification(self, close_info: Dict):
        """ส่ง Telegram แจ้งเตือนเมื่อปิด Position"""
        if not self.telegram or not self.telegram.enabled:
            return
        
        reason = close_info['close_reason']
        
        try:
            if reason == "sl":
                self.telegram.alert_sl_hit(
                    symbol=close_info['symbol'],
                    direction=close_info['direction'],
                    entry_price=close_info['entry_price'],
                    sl_price=close_info['exit_price'],
                    pnl=close_info['pnl'],
                    ticket=close_info['ticket'],
                    lot=close_info['volume'],
                )
                logger.info(f"📱 ส่ง Telegram แจ้งเตือน SL สำหรับ #{close_info['ticket']}")
                
            elif reason == "tp":
                self.telegram.alert_tp_hit(
                    symbol=close_info['symbol'],
                    direction=close_info['direction'],
                    entry_price=close_info['entry_price'],
                    tp_price=close_info['exit_price'],
                    pnl=close_info['pnl'],
                    ticket=close_info['ticket'],
                    lot=close_info['volume'],
                )
                logger.info(f"📱 ส่ง Telegram แจ้งเตือน TP สำหรับ #{close_info['ticket']}")
                
            elif reason == "trailing_stop":
                self.telegram.alert_trailing_stop(
                    symbol=close_info['symbol'],
                    direction=close_info['direction'],
                    entry_price=close_info['entry_price'],
                    ts_price=close_info['exit_price'],
                    pnl=close_info['pnl'],
                    ticket=close_info['ticket'],
                )
                logger.info(f"📱 ส่ง Telegram แจ้งเตือน Trailing Stop สำหรับ #{close_info['ticket']}")
                
            else:
                # ปิดปกติ
                self.telegram.alert_trade_exit(
                    symbol=close_info['symbol'],
                    direction=close_info['direction'],
                    entry_price=close_info['entry_price'],
                    exit_price=close_info['exit_price'],
                    pnl=close_info['pnl'],
                    reason=reason,
                    ticket=close_info['ticket'],
                    lot=close_info['volume'],
                )
                logger.info(f"📱 ส่ง Telegram แจ้งเตือนปิดออเดอร์สำหรับ #{close_info['ticket']}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
        
        # ============================================
        # Auto-record to MasterBrain for learning
        # ============================================
        self._auto_record_to_master_brain(close_info)
    
    def _auto_record_to_master_brain(self, close_info: Dict):
        """
        บันทึกผล trade ไปยัง MasterBrain อัตโนมัติ
        
        ทำให้ MasterBrain เรียนรู้จากทุก trade โดยอัตโนมัติ
        """
        if not self.master_brain:
            return
        
        ticket = close_info['ticket']
        pnl = close_info['pnl']
        is_win = pnl > 0
        
        # Get market state when position was opened
        market_state = self._position_market_states.get(ticket, {})
        if not market_state:
            # Default market state if not tracked
            market_state = {
                'regime': 'unknown',
                'volatility': 0.0,
                'trend': 0.0,
                'price': close_info['entry_price'],
            }
        
        # Determine action
        action = "LONG" if close_info['direction'] in ["BUY", "LONG"] else "SHORT"
        result = "win" if is_win else "loss"
        
        try:
            # Record to MasterBrain's trade result
            self.master_brain.record_trade_result(
                market_state=market_state,
                action=action,
                result=result,
                pnl=pnl,
                ticket=ticket,
            )
            
            # Record for intelligence learning
            if hasattr(self.master_brain, 'record_trade_for_learning'):
                current_hour = datetime.now().hour
                self.master_brain.record_trade_for_learning(
                    regime=market_state.get('regime', 'unknown'),
                    volatility=market_state.get('volatility', 0.0),
                    hour=current_hour,
                    is_win=is_win,
                    pnl=pnl,
                )
            
            logger.info(f"🧠 Auto-recorded trade #{ticket} to MasterBrain: {result} ${pnl:.2f}")
            
            # Clean up market state
            if ticket in self._position_market_states:
                del self._position_market_states[ticket]
                
        except Exception as e:
            logger.error(f"Error recording to MasterBrain: {e}")
    
    def record_market_state_for_position(self, ticket: int, market_state: Dict):
        """
        บันทึก market state สำหรับ position ที่เปิด
        
        เรียกฟังก์ชันนี้เมื่อเปิด position ใหม่
        """
        self._position_market_states[ticket] = {
            **market_state,
            'open_time': datetime.now(),
        }
        logger.debug(f"Recorded market state for position #{ticket}")
    
    def send_entry_notification(
        self,
        symbol: str,
        direction: str,
        price: float,
        volume: float,
        sl: float,
        tp: float,
        confidence: float,
        ticket: int = None,
    ):
        """ส่ง Telegram แจ้งเตือนเปิดออเดอร์"""
        if not self.telegram or not self.telegram.enabled:
            return
        
        try:
            self.telegram.alert_trade_entry(
                symbol=symbol,
                direction=direction,
                price=price,
                lot=volume,
                sl=sl,
                tp=tp,
                confidence=confidence,
                ticket=ticket,
            )
            logger.info(f"📱 ส่ง Telegram แจ้งเตือนเปิดออเดอร์สำหรับ #{ticket}")
        except Exception as e:
            logger.error(f"Error sending entry notification: {e}")


# Quick test
if __name__ == "__main__":
    print("="*70)
    print("  ELITE MT5 TRADE EXECUTOR v2.0.0")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Market order execution")
    print("  ✓ Advanced risk management")
    print("  ✓ Position management")
    print("  ✓ Smart order routing")
    print("  ✓ Slippage tracking")
    print("  ✓ Execution quality monitoring")
    print("  ✓ Telegram แจ้งเตือนภาษาไทย (SL/TP/Entry/Exit)")
    print("\nUsage:")
    print("  from mt5_trade_executor import MT5TradeExecutor")
    print("  executor = MT5TradeExecutor()")
    print("  result = executor.execute_market_order('XAUUSD', 'BUY', 0.01)")
    print("="*70)
