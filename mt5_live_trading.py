"""
MT5 Live Trading with AI
=========================
เทรดจริงบน MetaTrader 5 ด้วย AI

Usage:
    python mt5_live_trading.py --symbol XAUUSD --lot 0.01 --mode auto
    python mt5_live_trading.py --symbol XAUUSD --lot 0.01 --mode manual
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import argparse
from loguru import logger

from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment, TradingState
from ai_agent.online_learning import create_online_learner
from ai_agent.ai_full_control import AIFullController, MarketRegime, TradingStrategy


class MT5LiveTrader:
    """
    MT5 Live Trader with AI
    ========================
    เทรดจริงด้วย AI บน MT5
    """
    
    def __init__(
        self,
        symbol: str = "XAUUSD",
        lot_size: float = 0.01,
        sl_pips: float = 200,  # Stop Loss in pips
        tp_pips: float = 600,  # Take Profit in pips (1:3 R:R)
        magic: int = 123456,
    ):
        self.symbol = symbol
        self.lot_size = lot_size
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.magic = magic
        
        # Load AI
        self.agent = None
        self.learner = None
        self.ai_controller = None  # AI Full Control
        self._load_ai()
        
        # State tracking
        self.current_position = None
        self.entry_price = 0
        self.entry_state = None
        self.entry_time = None
        
        logger.info(f"MT5LiveTrader initialized: {symbol}, Lot: {lot_size}")
    
    def _load_ai(self):
        """Load AI model"""
        logger.info("Loading AI model...")
        
        state_dim = 8 + 3
        self.agent = PPOAgentWalkForward(state_dim=state_dim)
        loaded = self.agent.load("best_wf")
        
        if not loaded:
            logger.error("Failed to load AI model!")
            raise Exception("AI model not found")
        
        # Online learner for continuous learning
        self.learner = create_online_learner()
        
        logger.info(f"AI loaded: {self.agent.training_episodes} episodes")
        
        # AI Full Controller for autonomous trading
        account = mt5.account_info() if mt5.initialize() else None
        balance = account.balance if account else 1000
        self.ai_controller = AIFullController(initial_balance=balance)
        mt5.shutdown()
    
    def connect(self) -> bool:
        """Connect to MT5"""
        logger.info("Connecting to MetaTrader 5...")
        
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account = mt5.account_info()
        if account is None:
            logger.error("Failed to get account info")
            return False
        
        logger.info(f"Connected to MT5!")
        logger.info(f"   Account: {account.login}")
        logger.info(f"   Server: {account.server}")
        logger.info(f"   Balance: ${account.balance:.2f}")
        logger.info(f"   Equity: ${account.equity:.2f}")
        
        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
        
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Point: {symbol_info.point}")
        
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_market_data(self, bars: int = 500) -> pd.DataFrame:
        """Get market data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.error("Failed to get market data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'})
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def get_ai_signal(self, df: pd.DataFrame) -> tuple:
        """Get trading signal from AI"""
        try:
            env = TradingEnvironment(df)
            state = env.reset()
            
            # Move to latest bar
            while env.current_step < len(df) - 2:
                env.current_step += 1
            
            state = env._get_state()
            action, log_prob, value = self.agent.select_action(state)
            
            action_names = {0: "WAIT", 1: "BUY", 2: "CLOSE"}
            confidence = abs(value)
            
            return action, action_names[action], confidence, state
            
        except Exception as e:
            logger.error(f"AI signal error: {e}")
            return 0, "WAIT", 0, None
    
    def get_current_position(self):
        """Get current open position"""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        
        if positions and len(positions) > 0:
            return positions[0]
        return None
    
    def calculate_dynamic_sltp(self, df: pd.DataFrame, position=None) -> dict:
        """
        คำนวณ SL/TP แบบ Dynamic ตามสภาพตลาด
        
        AI จะปรับ SL/TP ตาม:
        1. Volatility - ตลาดผันผวนมาก = SL กว้างขึ้น
        2. Trend strength - Trend แรง = TP ไกลขึ้น
        3. Current profit - กำไรอยู่ = Trail SL
        """
        # Calculate market conditions
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
        
        # Trend calculation
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        trend_strength = abs(ma20 - ma50) / ma50 * 100  # % difference
        trend_direction = 1 if ma20 > ma50 else -1
        
        # ATR for volatility-based SL
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # Base SL/TP in price terms
        symbol_info = mt5.symbol_info(self.symbol)
        point = symbol_info.point if symbol_info else 0.01
        
        # Dynamic SL based on ATR (typically 1.5-2x ATR)
        dynamic_sl = atr * 2.0
        
        # Dynamic TP based on trend strength (2-4x SL)
        if trend_strength > 0.5:  # Strong trend
            tp_multiplier = 4.0
        elif trend_strength > 0.2:  # Medium trend
            tp_multiplier = 3.0
        else:  # Weak trend
            tp_multiplier = 2.0
        
        dynamic_tp = dynamic_sl * tp_multiplier
        
        # Convert to pips
        sl_pips = dynamic_sl / point
        tp_pips = dynamic_tp / point
        
        # Clamp to reasonable values
        sl_pips = max(100, min(1000, sl_pips))
        tp_pips = max(200, min(3000, tp_pips))
        
        result = {
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'sl_price': dynamic_sl,
            'tp_price': dynamic_tp,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'atr': atr,
            'rr_ratio': tp_pips / sl_pips,
        }
        
        logger.debug(f"Dynamic SL/TP: SL={sl_pips:.0f} TP={tp_pips:.0f} R:R=1:{result['rr_ratio']:.1f}")
        
        return result
    
    def manage_position(self, position, df: pd.DataFrame):
        """
        จัดการ position ที่เปิดอยู่
        
        Features:
        1. Trailing Stop - เลื่อน SL ตามกำไร
        2. Break-even - ย้าย SL ไป entry เมื่อกำไรพอ
        3. Dynamic TP adjustment - ปรับ TP ตาม trend
        """
        if position is None:
            return
        
        tick = mt5.symbol_info_tick(self.symbol)
        symbol_info = mt5.symbol_info(self.symbol)
        
        if not tick or not symbol_info:
            return
        
        point = symbol_info.point
        current_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
        
        # Calculate profit in pips
        if position.type == mt5.POSITION_TYPE_BUY:
            profit_pips = (current_price - position.price_open) / point
        else:
            profit_pips = (position.price_open - current_price) / point
        
        # Get dynamic analysis
        dynamic = self.calculate_dynamic_sltp(df, position)
        
        new_sl = None
        new_tp = None
        
        # 1. Break-even: Move SL to entry + small buffer when profit > 50% of original SL
        break_even_trigger = self.sl_pips * 0.5
        if profit_pips > break_even_trigger and position.sl < position.price_open:
            if position.type == mt5.POSITION_TYPE_BUY:
                new_sl = position.price_open + 10 * point  # Entry + 10 pips buffer
            else:
                new_sl = position.price_open - 10 * point
            logger.info(f"📈 Moving SL to break-even: {new_sl:.2f}")
        
        # 2. Trailing Stop: Move SL up when profit > original SL
        trailing_trigger = self.sl_pips
        trailing_distance = dynamic['atr'] * 1.5
        
        if profit_pips > trailing_trigger:
            if position.type == mt5.POSITION_TYPE_BUY:
                trail_sl = current_price - trailing_distance
                if trail_sl > position.sl:
                    new_sl = trail_sl
                    logger.info(f"📈 Trailing SL up to: {new_sl:.2f}")
            else:
                trail_sl = current_price + trailing_distance
                if trail_sl < position.sl:
                    new_sl = trail_sl
                    logger.info(f"📉 Trailing SL down to: {new_sl:.2f}")
        
        # 3. Extend TP if strong trend continues
        if dynamic['trend_strength'] > 0.5 and profit_pips > self.sl_pips * 0.3:
            extended_tp = dynamic['tp_price'] * 1.5
            if position.type == mt5.POSITION_TYPE_BUY:
                new_potential_tp = position.price_open + extended_tp
                if new_potential_tp > position.tp:
                    new_tp = new_potential_tp
                    logger.info(f"🎯 Extending TP to: {new_tp:.2f} (strong trend)")
            else:
                new_potential_tp = position.price_open - extended_tp
                if new_potential_tp < position.tp:
                    new_tp = new_potential_tp
                    logger.info(f"🎯 Extending TP to: {new_tp:.2f} (strong trend)")
        
        # Apply modifications if any
        if new_sl or new_tp:
            self._modify_position(position, new_sl, new_tp)
    
    def _modify_position(self, position, new_sl=None, new_tp=None):
        """Modify position SL/TP"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": new_sl if new_sl else position.sl,
            "tp": new_tp if new_tp else position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed: {result.comment}")
            return False
        
        logger.info(f"✅ Position modified: SL={request['sl']:.2f}, TP={request['tp']:.2f}")
        return True
    
    def open_trade(self, action: str = "BUY"):
        """Open a new trade"""
        symbol_info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)
        
        if symbol_info is None or tick is None:
            logger.error("Failed to get symbol/tick info")
            return False
        
        point = symbol_info.point
        
        if action == "BUY":
            price = tick.ask
            sl = price - self.sl_pips * point
            tp = price + self.tp_pips * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + self.sl_pips * point
            tp = price - self.tp_pips * point
            order_type = mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": self.magic,
            "comment": "AI Sniper Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            return False
        
        logger.info(f"✅ OPENED {action}: {self.lot_size} lots @ {price:.2f}")
        logger.info(f"   SL: {sl:.2f}, TP: {tp:.2f}")
        
        self.entry_price = price
        self.entry_time = datetime.now()
        
        return True
    
    def close_trade(self, position):
        """Close an open trade"""
        tick = mt5.symbol_info_tick(self.symbol)
        
        if position.type == mt5.POSITION_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "magic": self.magic,
            "comment": "AI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment}")
            return False
        
        profit = position.profit
        logger.info(f"✅ CLOSED: P&L = ${profit:.2f}")
        
        # Record trade for online learning
        if self.learner and self.entry_state is not None:
            df = self.get_market_data(100)
            if df is not None:
                env = TradingEnvironment(df)
                exit_state = env._get_state()
                
                self.learner.record_trade(
                    trade_id=f"MT5_{position.ticket}",
                    symbol=self.symbol,
                    entry_state=self.entry_state.price_features,
                    entry_price=self.entry_price,
                    exit_state=exit_state.price_features,
                    exit_price=price,
                    action=2,
                    holding_bars=int((datetime.now() - self.entry_time).total_seconds() / 3600) if self.entry_time else 0,
                )
        
        self.entry_price = 0
        self.entry_state = None
        self.entry_time = None
        
        return True
    
    def run_once(self):
        """Run one trading cycle"""
        # Get market data
        df = self.get_market_data()
        if df is None:
            return
        
        # Get AI signal
        action, signal, confidence, state = self.get_ai_signal(df)
        
        # Get dynamic SL/TP
        dynamic = self.calculate_dynamic_sltp(df)
        
        # Get current position
        position = self.get_current_position()
        
        tick = mt5.symbol_info_tick(self.symbol)
        current_price = tick.bid if tick else 0
        
        # Display status
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.2f} | Signal: {signal} | Confidence: {confidence:.2f}")
        logger.info(f"   📊 Dynamic: SL={dynamic['sl_pips']:.0f} TP={dynamic['tp_pips']:.0f} R:R=1:{dynamic['rr_ratio']:.1f} | Volatility: {dynamic['volatility']:.2f}%")
        
        # Manage existing position (trailing, break-even, etc.)
        if position is not None:
            profit = position.profit
            logger.info(f"   💰 Open Position: P&L=${profit:.2f}")
            self.manage_position(position, df)
        
        # Execute based on signal
        if action == 1 and position is None:  # BUY signal, no position
            # Use dynamic SL/TP
            self.sl_pips = dynamic['sl_pips']
            self.tp_pips = dynamic['tp_pips']
            logger.info(f"🎯 AI says BUY! Opening trade with Dynamic SL={self.sl_pips:.0f} TP={self.tp_pips:.0f}...")
            self.entry_state = state
            self.open_trade("BUY")
            
        elif action == 2 and position is not None:  # CLOSE signal, has position
            logger.info("🎯 AI says CLOSE! Closing trade...")
            self.close_trade(position)
    
    def run_auto(self, interval_minutes: int = 60):
        """Run auto trading loop"""
        logger.info(f"Starting AUTO trading mode (check every {interval_minutes} min)...")
        logger.info(f"Symbol: {self.symbol}, Lot: {self.lot_size}")
        logger.info(f"SL: {self.sl_pips} pips, TP: {self.tp_pips} pips")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                self.run_once()
                
                # Wait
                logger.info(f"Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("\n⏹ Stopped by user")
    
    def run_realtime(self, signal_interval: int = 300, manage_interval: int = 30):
        """
        Run REAL-TIME trading mode
        
        Parameters:
        - signal_interval: วินาที ระหว่างการวิเคราะห์สัญญาณ AI (default 300 = 5 นาที)
        - manage_interval: วินาที ระหว่างการจัดการ position (default 30)
        
        Features:
        - ตรวจสอบและจัดการ position ทุก 30 วินาที
        - วิเคราะห์สัญญาณ AI ทุก 5 นาที
        - Trailing SL/TP แบบ real-time
        """
        logger.info("="*50)
        logger.info("🚀 REAL-TIME TRADING MODE")
        logger.info("="*50)
        logger.info(f"Symbol: {self.symbol}, Lot: {self.lot_size}")
        logger.info(f"Signal Check: Every {signal_interval}s ({signal_interval/60:.1f} min)")
        logger.info(f"Position Manage: Every {manage_interval}s")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*50)
        
        last_signal_check = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Get market data
                df = self.get_market_data()
                if df is None:
                    time.sleep(manage_interval)
                    continue
                
                # Get current position
                position = self.get_current_position()
                
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 0
                
                # Always manage existing position (real-time)
                if position is not None:
                    profit = position.profit
                    profit_pips = (current_price - position.price_open) / mt5.symbol_info(self.symbol).point
                    
                    # Dynamic display
                    dynamic = self.calculate_dynamic_sltp(df)
                    
                    status = "🟢" if profit > 0 else "🔴"
                    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] {status} Price: {current_price:.2f} | P&L: ${profit:.2f} ({profit_pips:.0f} pips)")
                    
                    # Manage position (trailing, break-even, etc.)
                    self.manage_position(position, df)
                else:
                    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.2f} | No position")
                
                # Check for new signals at intervals
                if current_time - last_signal_check >= signal_interval:
                    logger.info("-" * 40)
                    logger.info("🔍 Checking AI Signal...")
                    
                    action, signal, confidence, state = self.get_ai_signal(df)
                    dynamic = self.calculate_dynamic_sltp(df)
                    
                    logger.info(f"   Signal: {signal} | Confidence: {confidence:.2f}")
                    logger.info(f"   Dynamic: SL={dynamic['sl_pips']:.0f} TP={dynamic['tp_pips']:.0f} R:R=1:{dynamic['rr_ratio']:.1f}")
                    
                    # Execute trade if signal
                    if action == 1 and position is None:  # BUY
                        self.sl_pips = dynamic['sl_pips']
                        self.tp_pips = dynamic['tp_pips']
                        logger.info(f"🎯 Opening BUY with SL={self.sl_pips:.0f} TP={self.tp_pips:.0f}")
                        self.entry_state = state
                        self.open_trade("BUY")
                        
                    elif action == 2 and position is not None:  # CLOSE
                        logger.info("🎯 AI says CLOSE!")
                        self.close_trade(position)
                    
                    last_signal_check = current_time
                    logger.info("-" * 40)
                
                # Wait before next check
                time.sleep(manage_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹ Stopped by user")
    
    def run_full_control(self, check_interval: int = 60):
        """
        Run FULL AI CONTROL mode
        ========================
        AI ควบคุมทุกอย่าง:
        - Strategy selection (sniper/scalp/swing/trend/breakout)
        - Lot size (based on risk %)
        - SL/TP (based on ATR and market regime)
        - Position management (trailing, break-even)
        
        Parameters:
        - check_interval: วินาที ระหว่างการตรวจสอบ (default 60)
        """
        logger.info("="*60)
        logger.info("🤖 AI FULL CONTROL MODE")
        logger.info("="*60)
        logger.info("   AI controls EVERYTHING:")
        logger.info("   - Strategy Selection")
        logger.info("   - Lot Size (Risk-based)")
        logger.info("   - SL/TP (Dynamic)")
        logger.info("   - Position Management")
        logger.info("="*60)
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Check Interval: {check_interval}s")
        logger.info("   Press Ctrl+C to stop")
        logger.info("="*60)
        
        try:
            while True:
                # Get account info
                account = mt5.account_info()
                if not account:
                    logger.error("Failed to get account info")
                    time.sleep(check_interval)
                    continue
                
                balance = account.balance
                equity = account.equity
                
                # Get market data
                df = self.get_market_data()
                if df is None:
                    time.sleep(check_interval)
                    continue
                
                # Get current position
                position = self.get_current_position()
                
                # Get symbol info
                symbol_info = mt5.symbol_info(self.symbol)
                point = symbol_info.point if symbol_info else 0.01
                
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 0
                
                # Get AI signal
                ai_action, signal, ai_confidence, state = self.get_ai_signal(df)
                
                # AI Full Control analysis
                decision = self.ai_controller.analyze(
                    df=df,
                    balance=balance,
                    point=point,
                    ai_action=ai_action,
                    ai_confidence=ai_confidence,
                )
                
                # Display status
                logger.info("")
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] 💰 Balance: ${balance:.2f} | Equity: ${equity:.2f}")
                logger.info(f"   📊 Price: {current_price:.2f} | Regime: {decision['regime'].value.upper()}")
                logger.info(f"   🎯 Strategy: {decision['strategy'].value.upper()} | Confidence: {decision['confidence']:.1%}")
                logger.info(f"   📈 SL: {decision['sl_pips']:.0f} TP: {decision['tp_pips']:.0f} R:R: 1:{decision['rr_ratio']:.1f}")
                logger.info(f"   🎲 AI Signal: {signal} | Action: {decision['action']}")
                
                # Handle existing position
                if position is not None:
                    profit = position.profit
                    status = "🟢" if profit > 0 else "🔴"
                    logger.info(f"   {status} Open Position: P&L=${profit:.2f}")
                    
                    # Manage position (trailing, break-even)
                    self.manage_position(position, df)
                    
                    # Check for CLOSE signal
                    if decision['action'] == 'CLOSE':
                        logger.info("🎯 AI Full Control: CLOSING position!")
                        self.close_trade(position)
                        self.ai_controller.risk_manager.record_trade(profit)
                
                # Open new position if AI decides
                elif decision['should_trade'] and decision['action'] == 'BUY':
                    # Update parameters from AI decision
                    self.lot_size = decision['lot_size']
                    self.sl_pips = decision['sl_pips']
                    self.tp_pips = decision['tp_pips']
                    
                    logger.info(f"🎯 AI Full Control: Opening BUY!")
                    logger.info(f"   Lot: {self.lot_size} | SL: {self.sl_pips:.0f} | TP: {self.tp_pips:.0f}")
                    logger.info(f"   Reason: {decision['reason']}")
                    
                    self.entry_state = state
                    self.open_trade("BUY")
                
                else:
                    logger.info(f"   ⏸️ {decision['reason']}")
                
                # Wait
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹ Stopped by user")
    
    def run_manual(self):
        """Run manual mode - get signals but don't execute"""
        logger.info("Starting MANUAL mode (signals only, no auto-execution)")
        
        while True:
            df = self.get_market_data()
            if df is None:
                time.sleep(60)
                continue
            
            action, signal, confidence, state = self.get_ai_signal(df)
            position = self.get_current_position()
            
            tick = mt5.symbol_info_tick(self.symbol)
            current_price = tick.bid if tick else 0
            
            print(f"\n{'='*50}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Price: {current_price:.2f}")
            print(f"Signal: {signal}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Position: {'OPEN' if position else 'NONE'}")
            print(f"{'='*50}")
            
            if action == 1 and position is None:
                print("⚠️ AI recommends: OPEN BUY")
            elif action == 2 and position is not None:
                print("⚠️ AI recommends: CLOSE POSITION")
            
            cmd = input("\nAction (b=buy, c=close, q=quit, Enter=wait): ").strip().lower()
            
            if cmd == 'b' and position is None:
                self.entry_state = state
                self.open_trade("BUY")
            elif cmd == 'c' and position is not None:
                self.close_trade(position)
            elif cmd == 'q':
                break
            
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='MT5 Live Trading with AI')
    parser.add_argument('--symbol', default='XAUUSD', help='Trading symbol')
    parser.add_argument('--lot', type=float, default=0.01, help='Lot size (used as max for fullcontrol)')
    parser.add_argument('--sl', type=float, default=200, help='Stop loss in pips')
    parser.add_argument('--tp', type=float, default=600, help='Take profit in pips')
    parser.add_argument('--mode', choices=['auto', 'manual', 'realtime', 'fullcontrol'], default='manual', help='Trading mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in minutes (auto mode) or seconds (fullcontrol)')
    parser.add_argument('--signal-interval', type=int, default=300, help='Signal check interval in seconds (realtime mode)')
    parser.add_argument('--manage-interval', type=int, default=30, help='Position manage interval in seconds (realtime mode)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    logger.add("logs/mt5_trading.log", rotation="1 day")
    
    print("="*60)
    print("   AI SNIPER TRADING - MT5 LIVE")
    print("="*60)
    print(f"   Symbol: {args.symbol}")
    if args.mode != 'fullcontrol':
        print(f"   Lot Size: {args.lot}")
        print(f"   SL/TP: {args.sl}/{args.tp} pips (R:R 1:{args.tp/args.sl:.1f})")
    else:
        print("   ⚡ AI FULL CONTROL - AI decides lot, SL, TP, strategy")
    print(f"   Mode: {args.mode.upper()}")
    if args.mode == 'realtime':
        print(f"   Signal Interval: {args.signal_interval}s")
        print(f"   Manage Interval: {args.manage_interval}s")
    elif args.mode == 'fullcontrol':
        print(f"   Check Interval: {args.interval}s")
    print("="*60)
    
    # Safety confirmation
    if args.mode in ['auto', 'realtime', 'fullcontrol']:
        confirm = input(f"\n⚠️ {args.mode.upper()} MODE will trade with REAL money!\nType 'YES' to confirm: ")
        if confirm.upper() != 'YES':
            print("Cancelled.")
            return
    
    # Create trader
    trader = MT5LiveTrader(
        symbol=args.symbol,
        lot_size=args.lot,
        sl_pips=args.sl,
        tp_pips=args.tp,
    )
    
    # Connect
    if not trader.connect():
        print("Failed to connect to MT5")
        return
    
    try:
        if args.mode == 'auto':
            trader.run_auto(interval_minutes=args.interval)
        elif args.mode == 'realtime':
            trader.run_realtime(
                signal_interval=args.signal_interval,
                manage_interval=args.manage_interval
            )
        elif args.mode == 'fullcontrol':
            trader.run_full_control(check_interval=args.interval)
        else:
            trader.run_manual()
    finally:
        trader.disconnect()


if __name__ == "__main__":
    main()
