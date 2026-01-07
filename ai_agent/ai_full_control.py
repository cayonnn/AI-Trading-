"""
AI Full Control Trading System
===============================
AI ควบคุมทุกอย่าง: กลยุทธ์, Lot Size, SL/TP

Components:
1. Risk Manager - จัดการความเสี่ยงและ lot size
2. Regime Detector - ตรวจจับสภาพตลาด
3. Strategy Selector - เลือก strategy ที่เหมาะสม
4. AI Controller - รวมทุกอย่างเข้าด้วยกัน
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """สภาพตลาด"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class TradingStrategy(Enum):
    """กลยุทธ์การเทรด"""
    SNIPER = "sniper"          # High R:R, ลด SL เล็ก TP ใหญ่
    SCALP = "scalp"            # Quick trades, small targets
    SWING = "swing"            # Hold longer, bigger moves
    TREND_FOLLOW = "trend"     # Follow the trend
    BREAKOUT = "breakout"      # Trade breakouts
    WAIT = "wait"              # ไม่เทรด รอจังหวะ


@dataclass
class RiskParameters:
    """พารามิเตอร์ความเสี่ยง"""
    lot_size: float
    sl_pips: float
    tp_pips: float
    rr_ratio: float
    max_drawdown_pct: float
    confidence: float
    risk_per_trade_pct: float


class RiskManager:
    """
    Risk Manager
    =============
    จัดการความเสี่ยงและคำนวณ lot size
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 0.02,  # 2% per trade
        max_daily_risk: float = 0.06,       # 6% per day
        max_drawdown: float = 0.10,         # 10% max drawdown
        min_lot: float = 0.01,
        max_lot: float = 1.0,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown
        self.min_lot = min_lot
        self.max_lot = max_lot
        
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.trades_today = 0
    
    def calculate_lot_size(
        self,
        balance: float,
        sl_pips: float,
        pip_value: float = 1.0,  # $ per pip per lot
        confidence: float = 1.0,
    ) -> float:
        """
        คำนวณ lot size ตาม risk
        
        Formula: Lot = (Balance * Risk%) / (SL_pips * Pip_Value)
        """
        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        
        # Check drawdown
        current_drawdown = (self.peak_balance - balance) / self.peak_balance
        if current_drawdown >= self.max_drawdown:
            logger.warning(f"⚠️ Max drawdown reached ({current_drawdown:.1%}). Reducing risk.")
            risk_multiplier = 0.5
        else:
            risk_multiplier = 1.0
        
        # Adjust risk based on confidence
        adjusted_risk = self.max_risk_per_trade * confidence * risk_multiplier
        
        # Calculate dollar risk
        dollar_risk = balance * adjusted_risk
        
        # Calculate lot size
        if sl_pips > 0 and pip_value > 0:
            lot_size = dollar_risk / (sl_pips * pip_value)
        else:
            lot_size = self.min_lot
        
        # Clamp to min/max
        lot_size = max(self.min_lot, min(self.max_lot, lot_size))
        
        # Round to 2 decimal places
        lot_size = round(lot_size, 2)
        
        logger.debug(f"Lot Size: {lot_size} (Risk: {adjusted_risk:.1%}, SL: {sl_pips} pips)")
        
        return lot_size
    
    def can_trade(self, balance: float) -> Tuple[bool, str]:
        """ตรวจสอบว่าสามารถเทรดได้หรือไม่"""
        
        # Check daily risk limit
        daily_pnl_pct = self.daily_pnl / balance if balance > 0 else 0
        if daily_pnl_pct <= -self.max_daily_risk:
            return False, f"Daily loss limit reached ({daily_pnl_pct:.1%})"
        
        # Check drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - balance) / self.peak_balance
            if current_drawdown >= self.max_drawdown:
                return False, f"Max drawdown reached ({current_drawdown:.1%})"
        
        return True, "OK"
    
    def record_trade(self, pnl: float):
        """บันทึก trade"""
        self.daily_pnl += pnl
        self.trades_today += 1
    
    def reset_daily(self):
        """Reset สถิติประจำวัน"""
        self.daily_pnl = 0.0
        self.trades_today = 0


class RegimeDetector:
    """
    Market Regime Detector
    =======================
    ตรวจจับสภาพตลาด
    """
    
    def detect(self, df: pd.DataFrame) -> Tuple[MarketRegime, Dict]:
        """ตรวจจับสภาพตลาด"""
        
        if len(df) < 50:
            return MarketRegime.RANGING, {}
        
        # Calculate indicators
        close = df['close']
        
        # Trend: MA crossover
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        current_price = close.iloc[-1]
        
        ma20_val = ma20.iloc[-1]
        ma50_val = ma50.iloc[-1]
        
        # Trend strength
        trend_strength = (ma20_val - ma50_val) / ma50_val * 100
        
        # Volatility: ATR
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        atr_pct = atr / current_price * 100
        
        # Historical volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * 100
        avg_volatility = returns.rolling(100).std().mean() * 100
        
        # ADX approximation (trend strength)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        tr = high_low.rolling(14).mean()
        plus_di = (plus_dm.rolling(14).mean() / tr * 100).iloc[-1]
        minus_di = (minus_dm.rolling(14).mean() / tr * 100).iloc[-1]
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        
        # RSI for momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Determine regime
        details = {
            'trend_strength': trend_strength,
            'atr_pct': atr_pct,
            'volatility': volatility,
            'avg_volatility': avg_volatility,
            'rsi': rsi,
            'dx': dx,
            'ma20': ma20_val,
            'ma50': ma50_val,
        }
        
        # High volatility check
        if volatility > avg_volatility * 1.5:
            return MarketRegime.HIGH_VOLATILITY, details
        
        if volatility < avg_volatility * 0.5:
            return MarketRegime.LOW_VOLATILITY, details
        
        # Trend detection
        if trend_strength > 1.0:
            if trend_strength > 2.0:
                return MarketRegime.STRONG_UPTREND, details
            return MarketRegime.WEAK_UPTREND, details
        
        if trend_strength < -1.0:
            if trend_strength < -2.0:
                return MarketRegime.STRONG_DOWNTREND, details
            return MarketRegime.WEAK_DOWNTREND, details
        
        return MarketRegime.RANGING, details


class StrategySelector:
    """
    Strategy Selector
    ==================
    เลือก strategy ที่เหมาะกับสภาพตลาด
    """
    
    # Strategy parameters for each regime
    STRATEGY_MAP = {
        MarketRegime.STRONG_UPTREND: {
            'strategy': TradingStrategy.TREND_FOLLOW,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 4.0,
            'confidence_threshold': 0.6,
            'trade_direction': 'long',
        },
        MarketRegime.WEAK_UPTREND: {
            'strategy': TradingStrategy.SNIPER,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 6.0,
            'confidence_threshold': 0.7,
            'trade_direction': 'long',
        },
        MarketRegime.RANGING: {
            'strategy': TradingStrategy.SCALP,
            'sl_atr_mult': 1.0,
            'tp_atr_mult': 1.5,
            'confidence_threshold': 0.8,
            'trade_direction': 'both',
        },
        MarketRegime.WEAK_DOWNTREND: {
            'strategy': TradingStrategy.WAIT,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 4.0,
            'confidence_threshold': 0.9,
            'trade_direction': 'short',
        },
        MarketRegime.STRONG_DOWNTREND: {
            'strategy': TradingStrategy.WAIT,  # Only short for now
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 3.0,
            'confidence_threshold': 0.9,
            'trade_direction': 'short',
        },
        MarketRegime.HIGH_VOLATILITY: {
            'strategy': TradingStrategy.BREAKOUT,
            'sl_atr_mult': 2.5,
            'tp_atr_mult': 5.0,
            'confidence_threshold': 0.75,
            'trade_direction': 'both',
        },
        MarketRegime.LOW_VOLATILITY: {
            'strategy': TradingStrategy.WAIT,
            'sl_atr_mult': 1.0,
            'tp_atr_mult': 2.0,
            'confidence_threshold': 0.9,
            'trade_direction': 'both',
        },
    }
    
    def select(
        self,
        regime: MarketRegime,
        regime_details: Dict,
        atr: float,
        point: float,
    ) -> Tuple[TradingStrategy, RiskParameters]:
        """เลือก strategy และคำนวณ parameters"""
        
        config = self.STRATEGY_MAP.get(regime, self.STRATEGY_MAP[MarketRegime.RANGING])
        
        # Calculate SL/TP in pips
        sl_price = atr * config['sl_atr_mult']
        tp_price = atr * config['tp_atr_mult']
        
        sl_pips = sl_price / point
        tp_pips = tp_price / point
        
        # Clamp values
        sl_pips = max(50, min(500, sl_pips))
        tp_pips = max(100, min(2000, tp_pips))
        
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 1.0
        
        # Confidence from regime details
        trend_confidence = min(1.0, abs(regime_details.get('trend_strength', 0)) / 3.0)
        rsi = regime_details.get('rsi', 50)
        
        # Higher confidence when RSI is not extreme
        if 30 < rsi < 70:
            rsi_confidence = 1.0
        else:
            rsi_confidence = 0.7
        
        confidence = (trend_confidence + rsi_confidence) / 2
        confidence = max(0.3, min(1.0, confidence))
        
        params = RiskParameters(
            lot_size=0.01,  # Will be calculated by RiskManager
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            rr_ratio=rr_ratio,
            max_drawdown_pct=0.1,
            confidence=confidence,
            risk_per_trade_pct=0.02 * confidence,
        )
        
        logger.info(f"📊 Strategy: {config['strategy'].value.upper()}")
        logger.info(f"   Regime: {regime.value}")
        logger.info(f"   SL: {sl_pips:.0f} pips, TP: {tp_pips:.0f} pips, R:R: 1:{rr_ratio:.1f}")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        return config['strategy'], params


class AIFullController:
    """
    AI Full Control Trading Controller
    ====================================
    AI ควบคุมทุกอย่าง + Advanced Features
    
    Features:
    - Risk Management (lot sizing, drawdown protection)
    - Regime Detection (trend/range/volatile)
    - Strategy Selection (sniper/scalp/swing/trend/breakout)
    - Multi-timeframe Analysis (M15/H1/H4/D1)
    - News/Event Filter
    - Session Analysis
    - Correlation Analysis
    """
    
    def __init__(self, initial_balance: float = 1000.0, symbol: str = "GOLD"):
        self.risk_manager = RiskManager()
        self.regime_detector = RegimeDetector()
        self.strategy_selector = StrategySelector()
        
        # Import advanced features
        try:
            from ai_agent.advanced_features import AdvancedSignalGenerator
            self.advanced = AdvancedSignalGenerator(symbol)
            self.use_advanced = True
            logger.info("✅ Advanced Features enabled")
        except Exception as e:
            self.advanced = None
            self.use_advanced = False
            logger.warning(f"⚠️ Advanced Features not available: {e}")
        
        self.initial_balance = initial_balance
        self.current_regime = None
        self.current_strategy = None
        self.current_params = None
        self.market_context = None
        
        logger.info("🤖 AI Full Controller initialized")
    
    def analyze(
        self,
        df: pd.DataFrame,
        balance: float,
        point: float = 0.01,
        ai_action: int = 0,
        ai_confidence: float = 0.5,
    ) -> Dict:
        """
        วิเคราะห์ตลาดและตัดสินใจ
        
        Returns:
        {
            'should_trade': bool,
            'action': 'BUY' / 'SELL' / 'CLOSE' / 'WAIT',
            'lot_size': float,
            'sl_pips': float,
            'tp_pips': float,
            'strategy': TradingStrategy,
            'regime': MarketRegime,
            'confidence': float,
            'reason': str,
        }
        """
        
        # 1. Detect market regime
        regime, regime_details = self.regime_detector.detect(df)
        self.current_regime = regime
        
        # 2. Advanced Features Check (Multi-timeframe, News, Session)
        advanced_block = False
        advanced_reason = ""
        confidence_multiplier = 1.0
        
        if self.use_advanced and self.advanced:
            try:
                context = self.advanced.get_full_context(df)
                self.market_context = context
                
                # Check if trading is allowed by advanced analysis
                if not context.trade_allowed:
                    advanced_block = True
                    advanced_reason = context.recommended_action
                
                # Get confidence multiplier from context
                confidence_multiplier = context.confidence_multiplier
                
                # Log advanced analysis
                logger.info(f"📊 MTF: M15={context.trend_m15} H1={context.trend_h1} H4={context.trend_h4} D1={context.trend_d1}")
                logger.info(f"   Alignment: {context.trend_alignment:.0%} | Session: {context.current_session.value}")
                if context.has_high_impact_news:
                    logger.warning(f"⚠️ High-impact news nearby!")
                    
            except Exception as e:
                logger.warning(f"Advanced analysis error: {e}")
        
        # 3. Check if we can trade (Risk Manager)
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            return {
                'should_trade': False,
                'action': 'WAIT',
                'reason': reason,
                'regime': regime,
                'strategy': TradingStrategy.WAIT,
                'lot_size': 0.01,
                'sl_pips': 200,
                'tp_pips': 600,
                'rr_ratio': 3.0,
                'confidence': 0,
            }
        
        # Check advanced block
        if advanced_block:
            return {
                'should_trade': False,
                'action': 'WAIT',
                'reason': advanced_reason,
                'regime': regime,
                'strategy': TradingStrategy.WAIT,
                'lot_size': 0.01,
                'sl_pips': 200,
                'tp_pips': 600,
                'rr_ratio': 3.0,
                'confidence': confidence_multiplier,
            }
        
        # 4. Get ATR for calculations
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # 5. Select strategy based on regime
        strategy, params = self.strategy_selector.select(regime, regime_details, atr, point)
        self.current_strategy = strategy
        self.current_params = params
        
        # 6. Combine AI confidence with regime confidence AND advanced multiplier
        combined_confidence = (ai_confidence + params.confidence) / 2 * confidence_multiplier
        
        # 7. Get strategy config for trade direction
        config = self.strategy_selector.STRATEGY_MAP.get(regime, {})
        trade_direction = config.get('trade_direction', 'long')
        confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # 7. Determine action
        action = 'WAIT'
        should_trade = False
        
        if strategy == TradingStrategy.WAIT:
            action = 'WAIT'
            reason = f"Strategy is WAIT for {regime.value}"
            
        elif ai_action == 1:  # AI says BUY
            if trade_direction in ['long', 'both']:
                if combined_confidence >= confidence_threshold:
                    action = 'BUY'
                    should_trade = True
                    reason = f"AI BUY signal with {combined_confidence:.1%} confidence"
                else:
                    action = 'WAIT'
                    reason = f"Confidence too low ({combined_confidence:.1%} < {confidence_threshold:.1%})"
            else:
                action = 'WAIT'
                reason = f"Downtrend regime - no LONG trades"
                
        elif ai_action == 2:  # AI says CLOSE
            action = 'CLOSE'
            should_trade = True
            reason = "AI CLOSE signal"
        
        # 8. Calculate lot size if trading
        lot_size = 0.01
        if should_trade and action == 'BUY':
            # Pip value estimation (for gold, approximately $1 per pip per 0.01 lot)
            pip_value = 1.0
            lot_size = self.risk_manager.calculate_lot_size(
                balance=balance,
                sl_pips=params.sl_pips,
                pip_value=pip_value,
                confidence=combined_confidence,
            )
        
        return {
            'should_trade': should_trade,
            'action': action,
            'lot_size': lot_size,
            'sl_pips': params.sl_pips,
            'tp_pips': params.tp_pips,
            'rr_ratio': params.rr_ratio,
            'strategy': strategy,
            'regime': regime,
            'regime_details': regime_details,
            'confidence': combined_confidence,
            'reason': reason,
        }
    
    def get_status(self) -> Dict:
        """สถานะปัจจุบัน"""
        return {
            'regime': self.current_regime.value if self.current_regime else None,
            'strategy': self.current_strategy.value if self.current_strategy else None,
            'risk_manager': {
                'daily_pnl': self.risk_manager.daily_pnl,
                'trades_today': self.risk_manager.trades_today,
                'peak_balance': self.risk_manager.peak_balance,
            }
        }


# Test
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("="*60)
    print("   AI FULL CONTROL - TEST")
    print("="*60)
    
    # Load test data
    df = pd.read_csv("data/training/GOLD_H1.csv")
    df.columns = [c.lower() for c in df.columns]
    
    # Create controller
    controller = AIFullController(initial_balance=1000)
    
    # Test analysis
    result = controller.analyze(
        df=df.tail(500),
        balance=1000,
        point=0.01,
        ai_action=1,  # BUY signal
        ai_confidence=0.7,
    )
    
    print(f"\n📊 Analysis Result:")
    print(f"   Should Trade: {result['should_trade']}")
    print(f"   Action: {result['action']}")
    print(f"   Lot Size: {result['lot_size']}")
    print(f"   SL/TP: {result['sl_pips']:.0f}/{result['tp_pips']:.0f} pips")
    print(f"   R:R: 1:{result['rr_ratio']:.1f}")
    print(f"   Strategy: {result['strategy'].value}")
    print(f"   Regime: {result['regime'].value}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Reason: {result['reason']}")
    
    print("\n" + "="*60)
    print("   TEST COMPLETE!")
    print("="*60)
