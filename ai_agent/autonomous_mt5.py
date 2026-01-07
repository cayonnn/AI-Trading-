"""
Autonomous MT5 Trading Integration
====================================
เชื่อมต่อ AutonomousAI กับ MetaTrader 5 สำหรับเทรดจริง

Features:
1. Real-time Market Data - รับข้อมูลจาก MT5
2. Live Execution - ส่งคำสั่งซื้อขาย
3. Position Management - จัดการ positions
4. Risk Control - ควบคุมความเสี่ยง
5. 24/7 Operation - ทำงานต่อเนื่อง
"""

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

# Import AutonomousAI
from ai_agent.autonomous_ai import AutonomousAI, create_autonomous_ai, TradeDecision, MarketState

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed. Install with: pip install MetaTrader5")

# v3.3: Import new modules
from ai_agent.telegram_alert import get_telegram
from ai_agent.model_monitor import get_model_monitor
from ai_agent.news_filter import get_news_filter


class AutonomousMT5Trader:
    """
    Autonomous MT5 Trading System
    
    ความสามารถ:
    1. เชื่อมต่อกับ MT5
    2. รับข้อมูลตลาด real-time
    3. ส่งคำสั่งซื้อขายอัตโนมัติ
    4. จัดการ positions และ risk
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "H1",
        capital: float = 10000.0,
        lot_size: float = 0.01,
        max_lot: float = 0.1,
        magic_number: int = 123456,
        paper_trading: bool = True,  # Default to paper trading
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        self.lot_size = lot_size
        self.max_lot = max_lot
        self.magic_number = magic_number
        self.paper_trading = paper_trading
        
        # Initialize AI
        self.ai = create_autonomous_ai(capital=capital)
        
        # MT5 Timeframe mapping
        self.tf_map = {
            "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
            "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
            "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
            "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
            "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
            "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
            "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
        }
        
        # State
        self.is_running = False
        self.trade_count = 0
        self.daily_pnl = 0.0
        self.last_check = None
        
        # Paper trading simulation
        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_pnl = 0.0
        
        # v3.3: Telegram & Monitoring
        self.telegram = get_telegram()
        self.model_monitor = get_model_monitor()
        
        logger.info(f"AutonomousMT5Trader initialized")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Paper Trading: {paper_trading}")
    
    def connect(self) -> bool:
        """เชื่อมต่อ MT5"""
        
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, using paper trading mode")
            self.paper_trading = True
            return True
        
        # Initialize with specific path
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        if not mt5.initialize(path=mt5_path):
            # Try without path as fallback
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
        
        # Get account info
        account = mt5.account_info()
        if account:
            logger.info(f"Connected to MT5 account: {account.login}")
            logger.info(f"  Balance: ${account.balance:.2f}")
            logger.info(f"  Leverage: 1:{account.leverage}")
        
        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                return False
        
        logger.info(f"Symbol {self.symbol} ready")
        return True
    
    def disconnect(self):
        """ปิดการเชื่อมต่อ"""
        if MT5_AVAILABLE:
            mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_market_data(self, bars: int = 500) -> pd.DataFrame:
        """ดึงข้อมูลตลาดจาก MT5"""
        
        if self.paper_trading or not MT5_AVAILABLE:
            # Generate synthetic data for testing
            return self._generate_test_data(bars)
        
        tf = self.tf_map.get(self.timeframe, mt5.TIMEFRAME_H1)
        
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.error("Failed to get market data")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    
    def _generate_test_data(self, bars: int) -> pd.DataFrame:
        """สร้างข้อมูลทดสอบ"""
        
        dates = pd.date_range(end=datetime.now(), periods=bars, freq="h")
        prices = 2000 + np.cumsum(np.random.randn(bars) * 3)
        
        return pd.DataFrame({
            "datetime": dates,
            "open": prices - np.random.rand(bars) * 3,
            "high": prices + np.random.rand(bars) * 5,
            "low": prices - np.random.rand(bars) * 5,
            "close": prices,
            "tick_volume": np.random.randint(1000, 5000, bars),
        })
    
    def get_current_price(self) -> Tuple[float, float]:
        """ดึงราคาปัจจุบัน (bid, ask)"""
        
        if self.paper_trading or not MT5_AVAILABLE:
            data = self.get_market_data(1)
            price = data['close'].iloc[-1] if len(data) > 0 else 2000
            return price - 0.5, price + 0.5
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return tick.bid, tick.ask
        
        return 0.0, 0.0
    
    def calculate_lot_size(self, decision: TradeDecision) -> float:
        """คำนวณ lot size จาก decision"""
        
        # Use position optimizer's recommendation
        position_value = decision.position_size
        
        # Estimate lot size (simplified)
        price = self.get_current_price()[0]
        if price > 0:
            lot = position_value / (price * 100)  # Gold: 100 units per lot
        else:
            lot = self.lot_size
        
        # Apply limits
        lot = max(self.lot_size, min(lot, self.max_lot))
        
        # Round to standard lot sizes
        lot = round(lot, 2)
        
        return lot
    
    def open_trade(
        self,
        order_type: str,
        lot: float,
        sl: float,
        tp: float,
    ) -> Dict:
        """เปิด trade"""
        
        bid, ask = self.get_current_price()
        
        if self.paper_trading:
            # Paper trading simulation
            self.paper_position = 1 if order_type == "BUY" else -1
            self.paper_entry_price = ask if order_type == "BUY" else bid
            
            logger.info(f"📝 PAPER {order_type} @ {self.paper_entry_price:.2f}")
            
            return {
                "success": True,
                "order_id": f"PAPER_{datetime.now().strftime('%H%M%S')}",
                "price": self.paper_entry_price,
            }
        
        if not MT5_AVAILABLE:
            return {"success": False, "error": "MT5 not available"}
        
        # Real MT5 order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": ask if order_type == "BUY" else bid,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "AutonomousAI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}")
            return {"success": False, "error": result.comment}
        
        logger.info(f"✅ {order_type} executed @ {result.price}")
        
        # v3.3: Send telegram alert
        try:
            self.telegram.alert_trade_entry(
                symbol=self.symbol,
                direction=order_type,
                price=result.price,
                lot=lot,
                sl=sl,
                tp=tp,
                confidence=0.7,
            )
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")
        
        return {
            "success": True,
            "order_id": result.order,
            "price": result.price,
        }
    
    def close_trade(self) -> Dict:
        """ปิด trade ที่เปิดอยู่"""
        
        bid, ask = self.get_current_price()
        
        if self.paper_trading:
            if self.paper_position == 0:
                return {"success": False, "error": "No position"}
            
            exit_price = bid if self.paper_position == 1 else ask
            pnl_pct = (exit_price - self.paper_entry_price) / self.paper_entry_price
            if self.paper_position == -1:
                pnl_pct = -pnl_pct
            
            pnl = pnl_pct * self.capital * 0.02  # Simplified
            self.paper_pnl += pnl
            
            logger.info(f"📝 PAPER CLOSE @ {exit_price:.2f} | P&L: ${pnl:.2f}")
            
            self.paper_position = 0
            self.paper_entry_price = 0
            
            return {
                "success": True,
                "price": exit_price,
                "pnl": pnl,
            }
        
        if not MT5_AVAILABLE:
            return {"success": False, "error": "MT5 not available"}
        
        # Get open position
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic_number)
        
        if not positions:
            return {"success": False, "error": "No open position"}
        
        position = positions[0]
        
        # Close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": bid if position.type == 0 else ask,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "AutonomousAI Close",
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.retcode}")
            return {"success": False, "error": result.comment}
        
        pnl = position.profit
        
        logger.info(f"✅ CLOSED @ {result.price} | P&L: ${pnl:.2f}")
        
        # v3.3: Send telegram alert
        try:
            direction = "LONG" if position.type == 0 else "SHORT"
            self.telegram.alert_trade_exit(
                symbol=self.symbol,
                direction=direction,
                entry_price=position.price_open,
                exit_price=result.price,
                pnl=pnl,
                reason="AI Exit",
            )
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")
        
        return {
            "success": True,
            "price": result.price,
            "pnl": pnl,
        }
    
    def check_and_execute(self) -> Dict:
        """
        ตรวจสอบตลาดและ execute ตาม AI decision
        
        Flow:
        1. ดึงข้อมูลตลาด
        2. ให้ AI ตัดสินใจ
        3. Execute ถ้าจำเป็น
        """
        
        # Get market data
        data = self.get_market_data(500)
        
        if len(data) < 50:
            return {"action": "skip", "reason": "Insufficient data"}
        
        # Get AI decision
        decision = self.ai.make_decision(data)
        
        result = {
            "timestamp": datetime.now(),
            "decision": decision.action,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "executed": False,
        }
        
        # Execute based on decision
        if decision.action == "LONG" and self.paper_position == 0:
            lot = self.calculate_lot_size(decision)
            
            exec_result = self.open_trade(
                "BUY",
                lot,
                decision.stop_loss,
                decision.take_profit,
            )
            
            if exec_result["success"]:
                self.trade_count += 1
                result["executed"] = True
                result["entry_price"] = exec_result["price"]
                result["lot"] = lot
        
        elif decision.action == "CLOSE" and self.paper_position != 0:
            exec_result = self.close_trade()
            
            if exec_result["success"]:
                result["executed"] = True
                result["exit_price"] = exec_result["price"]
                result["pnl"] = exec_result.get("pnl", 0)
                self.daily_pnl += result["pnl"]
        
        self.last_check = datetime.now()
        
        return result
    
    def _check_trailing_stop(self):
        """
        ตรวจสอบและขยับ trailing stop สำหรับ positions ที่เปิดอยู่
        ใช้ MasterBrain ในการตัดสินใจด้วย S/R-based trailing
        """
        try:
            # Get current price
            bid, ask = self.get_current_price()
            current_price = (bid + ask) / 2
            
            # Get market data with OHLC for S/R calculation
            data = self.get_market_data(100)
            if len(data) < 50:
                return
            
            atr = data['atr'].iloc[-1] if 'atr' in data.columns else 15.0
            
            # Get OHLC for S/R levels
            highs = data['high'].tolist()
            lows = data['low'].tolist()
            closes = data['close'].tolist()
            
            # Let MasterBrain manage positions with S/R-based trailing
            self.ai.master_brain.manage_positions_with_sr(
                current_price=current_price,
                highs=highs,
                lows=lows,
                closes=closes,
                atr=atr,
            )
            
        except Exception as e:
            pass  # Silent fail - don't break main loop
    
    def run(
        self,
        interval_seconds: int = 120,  # 2 minutes (changed from 5)
        max_iterations: int = None,
    ):
        """
        รัน autonomous trading loop
        
        Args:
            interval_seconds: ความถี่ในการตรวจสอบ
            max_iterations: จำนวนรอบสูงสุด (None = ไม่จำกัด)
        """
        
        logger.info("="*60)
        logger.info("   AUTONOMOUS MT5 TRADING - STARTING")
        logger.info("="*60)
        
        if not self.paper_trading:
            if not self.connect():
                logger.error("Failed to connect to MT5")
                return
        
        # v3.3: Send telegram startup alert
        try:
            self.telegram.send_custom(f"🚀 AI Trading Started\n\n📊 {self.symbol}\n⏰ TF: {self.timeframe}\n💼 Mode: {'LIVE' if not self.paper_trading else 'PAPER'}")
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")
        
        self.is_running = True
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                
                logger.info(f"\n--- Iteration {iteration} | {datetime.now().strftime('%H:%M:%S')} ---")
                
                # Check and execute
                result = self.check_and_execute()
                
                logger.info(
                    f"Decision: {result['decision']} | "
                    f"Confidence: {result['confidence']:.1%} | "
                    f"Executed: {result['executed']}"
                )
                
                # Show AI's human-like reasoning
                if 'reason' in result:
                    logger.info(f"🧠 AI Reasoning: {result['reason']}")
                
                if result['executed']:
                    if 'pnl' in result:
                        logger.info(f"P&L: ${result['pnl']:.2f}")
                
                # v3.1: Check and update trailing stop for open positions
                self._check_trailing_stop()
                
                # Check AI evolution periodically
                if iteration % 10 == 0:
                    self.ai.check_and_evolve()
                
                # Check max iterations
                if max_iterations and iteration >= max_iterations:
                    logger.info("Max iterations reached")
                    break
                
                # Wait for next check
                if self.is_running:
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("\nStopping by user request...")
        
        finally:
            self.stop()
    
    def stop(self):
        """หยุด trading"""
        
        self.is_running = False
        
        # Close any open positions
        if self.paper_position != 0:
            self.close_trade()
        
        # v3.3: Save MasterBrain state to database
        try:
            self.ai.master_brain._save_model()
            logger.info("💾 MasterBrain state saved to database")
        except Exception as e:
            logger.warning(f"Could not save MasterBrain: {e}")
        
        # v3.3: Send telegram shutdown alert
        try:
            self.telegram.send_custom(
                f"🛑 AI Trading Stopped\n\n"
                f"📊 Trades: {self.trade_count}\n"
                f"💰 P&L: ${self.daily_pnl:.2f}\n"
                f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            )
        except Exception as e:
            logger.debug(f"Telegram failed: {e}")
        
        self.disconnect()
        
        logger.info("="*60)
        logger.info("   TRADING SESSION ENDED")
        logger.info(f"   Total Trades: {self.trade_count}")
        logger.info(f"   Daily P&L: ${self.daily_pnl:.2f}")
        logger.info("="*60)
    
    def get_status(self) -> Dict:
        """ดึงสถานะ"""
        
        return {
            "is_running": self.is_running,
            "paper_trading": self.paper_trading,
            "current_position": self.paper_position if self.paper_trading else "check_mt5",
            "trade_count": self.trade_count,
            "daily_pnl": self.daily_pnl + self.paper_pnl,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "ai_status": self.ai.get_status(),
        }


def create_autonomous_mt5(
    symbol: str = "XAUUSD",
    capital: float = 10000.0,
    paper_trading: bool = True,
) -> AutonomousMT5Trader:
    """สร้าง AutonomousMT5Trader"""
    return AutonomousMT5Trader(
        symbol=symbol,
        capital=capital,
        paper_trading=paper_trading,
    )


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print()
    print("="*60)
    print("   AUTONOMOUS MT5 TRADING TEST")
    print("="*60)
    print()
    
    # Create trader in paper trading mode
    trader = create_autonomous_mt5(
        symbol="XAUUSD",
        capital=10000,
        paper_trading=True,
    )
    
    # Run for a few iterations
    print("\nRunning 5 iterations in paper trading mode...\n")
    trader.run(interval_seconds=1, max_iterations=5)
    
    # Status
    print("\nFinal Status:")
    status = trader.get_status()
    print(f"  Trade Count: {status['trade_count']}")
    print(f"  P&L: ${status['daily_pnl']:.2f}")
