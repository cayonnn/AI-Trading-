"""
Smart Timing Module
====================
จับจังหวะ Entry/Exit ที่แม่นยำที่สุด

Features:
1. Pullback Detection - รอ pullback ก่อน entry
2. Breakout Confirmation - ยืนยัน breakout
3. Exhaustion Detection - ตรวจจับ trend exhaustion
4. Optimal Entry Zones - หา zones ที่ดีที่สุด
5. Exit Optimization - หาจุด exit ที่ดีที่สุด
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from loguru import logger


class EntryType(Enum):
    """ประเภท Entry"""
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class ExitType(Enum):
    """ประเภท Exit"""
    TP_HIT = "take_profit"
    SL_HIT = "stop_loss"
    TRAILING = "trailing_stop"
    SIGNAL = "signal_exit"
    TIME = "time_exit"


@dataclass
class EntrySignal:
    """สัญญาณ Entry"""
    should_enter: bool
    entry_type: EntryType
    entry_price: float
    optimal_sl: float
    optimal_tp: float
    timing_score: float  # 0-100
    reason: str


@dataclass
class ExitSignal:
    """สัญญาณ Exit"""
    should_exit: bool
    exit_type: ExitType
    exit_price: float
    reason: str
    urgency: str  # 'low', 'medium', 'high', 'immediate'


class SmartTiming:
    """
    Optimal Entry/Exit Timing System
    
    ความสามารถ:
    1. หาจุด entry ที่ดีที่สุด
    2. ยืนยัน breakout/pullback
    3. ตรวจจับ trend exhaustion
    4. หาจุด exit ที่ดีที่สุด
    """
    
    def __init__(
        self,
        pullback_threshold: float = 0.382,  # Fibonacci
        breakout_buffer: float = 0.002,  # 0.2% buffer
    ):
        self.pullback_threshold = pullback_threshold
        self.breakout_buffer = breakout_buffer
        
        # Tracking
        self.recent_signals: List[Dict] = []
        
        logger.info("SmartTiming initialized")
    
    def analyze_entry(
        self,
        data: pd.DataFrame,
        direction: int,  # 1 = long, -1 = short
        current_price: float = None,
    ) -> EntrySignal:
        """
        วิเคราะห์จังหวะ Entry
        
        Args:
            data: OHLC data
            direction: ทิศทางที่ต้องการ
            current_price: ราคาปัจจุบัน
            
        Returns:
            EntrySignal
        """
        
        if len(data) < 30:
            return EntrySignal(
                should_enter=False,
                entry_type=EntryType.BREAKOUT,
                entry_price=0,
                optimal_sl=0,
                optimal_tp=0,
                timing_score=0,
                reason="Insufficient data",
            )
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        if current_price is None:
            current_price = close[-1]
        
        # 1. Check for breakout
        is_breakout, breakout_level = self._check_breakout(high, low, close, direction)
        
        # 2. Check for pullback
        is_pullback, pullback_level = self._check_pullback(high, low, close, direction)
        
        # 3. Check for exhaustion (avoid entry)
        is_exhausted = self._check_exhaustion(close, direction)
        
        # 4. Calculate entry zones
        entry_zone = self._calculate_entry_zone(high, low, close, direction)
        
        # 5. Determine entry type and timing
        if is_exhausted:
            return EntrySignal(
                should_enter=False,
                entry_type=EntryType.BREAKOUT,
                entry_price=current_price,
                optimal_sl=0,
                optimal_tp=0,
                timing_score=20,
                reason="Trend exhaustion detected - wait",
            )
        
        if is_breakout:
            entry_type = EntryType.BREAKOUT
            sl, tp = self._calculate_sl_tp(current_price, direction, high, low, "breakout")
            timing_score = 80 if not is_exhausted else 40
            reason = f"Breakout above {breakout_level:.2f}"
        elif is_pullback:
            entry_type = EntryType.PULLBACK
            sl, tp = self._calculate_sl_tp(current_price, direction, high, low, "pullback")
            timing_score = 90  # Pullback entries are often best
            reason = f"Pullback to {pullback_level:.2f}"
        else:
            entry_type = EntryType.CONTINUATION
            sl, tp = self._calculate_sl_tp(current_price, direction, high, low, "normal")
            timing_score = 60
            reason = "Continuation setup"
        
        # Check if price is in entry zone
        in_zone = entry_zone[0] <= current_price <= entry_zone[1]
        
        should_enter = timing_score >= 60 and not is_exhausted and in_zone
        
        if not in_zone:
            timing_score *= 0.7
            reason += f" (outside optimal zone: {entry_zone[0]:.2f}-{entry_zone[1]:.2f})"
        
        return EntrySignal(
            should_enter=should_enter,
            entry_type=entry_type,
            entry_price=current_price,
            optimal_sl=sl,
            optimal_tp=tp,
            timing_score=timing_score,
            reason=reason,
        )
    
    def analyze_exit(
        self,
        data: pd.DataFrame,
        entry_price: float,
        direction: int,
        current_sl: float,
        current_tp: float,
    ) -> ExitSignal:
        """
        วิเคราะห์จังหวะ Exit
        """
        
        if len(data) < 10:
            return ExitSignal(
                should_exit=False,
                exit_type=ExitType.SIGNAL,
                exit_price=data['close'].iloc[-1],
                reason="Holding",
                urgency="low",
            )
        
        current_price = data['close'].iloc[-1]
        
        # 1. Check SL/TP
        if direction == 1:
            if current_price <= current_sl:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.SL_HIT,
                    exit_price=current_price,
                    reason="Stop loss hit",
                    urgency="immediate",
                )
            if current_price >= current_tp:
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.TP_HIT,
                    exit_price=current_price,
                    reason="Take profit hit",
                    urgency="immediate",
                )
        
        # 2. Check for reversal signals
        is_reversing = self._check_reversal(data, direction)
        
        if is_reversing:
            pnl_pct = (current_price - entry_price) / entry_price * direction
            
            if pnl_pct > 0.01:  # 1% profit
                return ExitSignal(
                    should_exit=True,
                    exit_type=ExitType.SIGNAL,
                    exit_price=current_price,
                    reason="Reversal signal with profit",
                    urgency="high",
                )
        
        # 3. Check for trailing stop opportunity (S/R based)
        should_trail, new_sl = self._check_trailing_stop(
            data, entry_price, direction, current_price
        )
        
        if should_trail:
            return ExitSignal(
                should_exit=True,
                exit_type=ExitType.TRAILING,
                exit_price=current_price,
                reason="Trailing stop triggered",
                urgency="medium",
            )
        
        # Store new_sl for use by caller (to update SL)
        if new_sl > 0:
            # Return hold signal but with updated SL info in reason
            return ExitSignal(
                should_exit=False,
                exit_type=ExitType.TRAILING,
                exit_price=new_sl,  # Use exit_price field to return new SL
                reason=f"Move SL to S/R level: {new_sl:.2f}",
                urgency="low",
            )
        
        return ExitSignal(
            should_exit=False,
            exit_type=ExitType.SIGNAL,
            exit_price=current_price,
            reason="Holding position",
            urgency="low",
        )
    
    def _check_breakout(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        direction: int,
    ) -> Tuple[bool, float]:
        """ตรวจจับ Breakout"""
        
        # Look for break of recent range
        lookback = 20
        
        if direction == 1:
            recent_high = max(high[-lookback:-1])
            is_breakout = close[-1] > recent_high * (1 + self.breakout_buffer)
            return is_breakout, recent_high
        else:
            recent_low = min(low[-lookback:-1])
            is_breakout = close[-1] < recent_low * (1 - self.breakout_buffer)
            return is_breakout, recent_low
    
    def _check_pullback(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        direction: int,
    ) -> Tuple[bool, float]:
        """ตรวจจับ Pullback"""
        
        if len(close) < 20:
            return False, 0
        
        # Check if in trend
        trend = (close[-1] - close[-20]) / close[-20]
        
        if direction == 1 and trend > 0.01:
            # Look for pullback in uptrend
            swing_high = max(high[-15:])
            swing_low = min(low[-5:])
            
            # Fibonacci retracement
            fib_level = swing_high - (swing_high - swing_low) * self.pullback_threshold
            
            is_pullback = close[-1] <= fib_level and close[-1] > swing_low
            return is_pullback, fib_level
        
        elif direction == -1 and trend < -0.01:
            # Look for pullback in downtrend
            swing_low = min(low[-15:])
            swing_high = max(high[-5:])
            
            fib_level = swing_low + (swing_high - swing_low) * self.pullback_threshold
            
            is_pullback = close[-1] >= fib_level and close[-1] < swing_high
            return is_pullback, fib_level
        
        return False, 0
    
    def _check_exhaustion(
        self,
        close: np.ndarray,
        direction: int,
    ) -> bool:
        """ตรวจจับ Trend Exhaustion"""
        
        if len(close) < 20:
            return False
        
        # Calculate momentum
        returns = np.diff(close) / close[:-1]
        recent_returns = returns[-5:]
        past_returns = returns[-20:-5]
        
        # Check for slowing momentum
        recent_momentum = np.mean(recent_returns)
        past_momentum = np.mean(past_returns)
        
        if direction == 1:
            # In uptrend, check if momentum is slowing
            is_exhausted = (
                recent_momentum < past_momentum * 0.5 and
                past_momentum > 0
            )
        else:
            # In downtrend, check if momentum is slowing
            is_exhausted = (
                recent_momentum > past_momentum * 0.5 and
                past_momentum < 0
            )
        
        return is_exhausted
    
    def _check_reversal(
        self,
        data: pd.DataFrame,
        direction: int,
    ) -> bool:
        """ตรวจจับสัญญาณ Reversal"""
        
        close = data['close'].values
        
        if len(close) < 10:
            return False
        
        # Check for divergence-like pattern
        price_change = (close[-1] - close[-5]) / close[-5]
        
        if direction == 1 and price_change < -0.01:
            return True
        elif direction == -1 and price_change > 0.01:
            return True
        
        return False
    
    def _check_trailing_stop(
        self,
        data: pd.DataFrame,
        entry_price: float,
        direction: int,
        current_price: float,
    ) -> Tuple[bool, float]:
        """
        ตรวจสอบ Trailing Stop แบบ Support/Resistance
        
        Logic:
        1. หาแนวรับ/ต้านจาก swing points
        2. เมื่อราคาเบรคและยืนเหนือแนวได้ → เลื่อน SL มาที่แนวนั้น
        3. ทำซ้ำเป็น step เมื่อเบรคแนวถัดไป
        
        Returns:
            Tuple[should_exit, new_sl_price]
        """
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # คำนวณ P&L
        pnl_pct = (current_price - entry_price) / entry_price * direction
        
        # ยังไม่มีกำไร ไม่เลื่อน SL
        if pnl_pct < 0.005:  # < 0.5% profit
            return False, 0
        
        # หาแนวรับ/ต้าน (Swing Points)
        sr_levels = self._find_sr_levels(high, low, close)
        
        if not sr_levels:
            # Fallback เป็น trailing ปกติ
            if pnl_pct > 0.02:
                if direction == 1:
                    high_since = max(high[-20:])
                    pullback = (high_since - current_price) / high_since
                    return pullback > 0.01, entry_price + (high_since - entry_price) * 0.5
                else:
                    low_since = min(low[-20:])
                    pullback = (current_price - low_since) / low_since
                    return pullback > 0.01, entry_price - (entry_price - low_since) * 0.5
            return False, 0
        
        # หา S/R level ที่ราคายืนเหนือ/ใต้ได้แล้ว
        if direction == 1:  # LONG
            # หาแนวรับที่ราคา breakout ไปแล้ว (ราคาอยู่เหนือ)
            valid_levels = [lvl for lvl in sr_levels if current_price > lvl and lvl > entry_price]
            
            if valid_levels:
                # เอาแนวล่าสุดที่ใกล้ราคาที่สุด
                new_sl = max(valid_levels)
                
                # ตรวจสอบว่าราคายืนเหนือแนวได้ (ไม่ใช่แค่ spike)
                # ราคาต้องอยู่เหนือแนว 2 แท่งขึ้นไป
                candles_above = sum(1 for c in close[-5:] if c > new_sl)
                
                if candles_above >= 2:
                    # ตรวจสอบว่าราคากลับลงมาเทสแนวแล้ว
                    tested = any(low[-5:] <= new_sl * 1.002)  # +-0.2%
                    
                    if tested:
                        # ราคาเทสแนวแล้วและยืนได้ → เลื่อน SL มาแนวนี้
                        logger.debug(f"Trailing SL to S/R: {new_sl:.2f}")
                        return False, new_sl  # ไม่ exit แต่ return new SL
        
        else:  # SHORT
            # หาแนวต้านที่ราคา breakout ไปแล้ว (ราคาอยู่ใต้)
            valid_levels = [lvl for lvl in sr_levels if current_price < lvl and lvl < entry_price]
            
            if valid_levels:
                new_sl = min(valid_levels)
                
                candles_below = sum(1 for c in close[-5:] if c < new_sl)
                
                if candles_below >= 2:
                    tested = any(high[-5:] >= new_sl * 0.998)
                    
                    if tested:
                        logger.debug(f"Trailing SL to S/R: {new_sl:.2f}")
                        return False, new_sl
        
        return False, 0
    
    def _find_sr_levels(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 50,
    ) -> List[float]:
        """
        หาแนวรับ/ต้านจาก Swing Points
        
        Returns:
            List of S/R price levels
        """
        
        if len(close) < lookback:
            lookback = len(close) - 1
        
        if lookback < 10:
            return []
        
        levels = []
        
        # หา Swing Highs (แนวต้าน)
        for i in range(5, lookback - 5):
            idx = -lookback + i
            if high[idx] > max(high[idx-5:idx]) and high[idx] > max(high[idx+1:idx+6]):
                levels.append(high[idx])
        
        # หา Swing Lows (แนวรับ)
        for i in range(5, lookback - 5):
            idx = -lookback + i
            if low[idx] < min(low[idx-5:idx]) and low[idx] < min(low[idx+1:idx+6]):
                levels.append(low[idx])
        
        # รวม levels ที่ใกล้กัน (within 0.5%)
        if levels:
            levels = sorted(set(levels))
            merged = [levels[0]]
            for lvl in levels[1:]:
                if abs(lvl - merged[-1]) / merged[-1] > 0.005:
                    merged.append(lvl)
                else:
                    merged[-1] = (merged[-1] + lvl) / 2
            return merged
        
        return []
    
    def _calculate_entry_zone(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        direction: int,
    ) -> Tuple[float, float]:
        """คำนวณ Entry Zone"""
        
        recent_high = max(high[-10:])
        recent_low = min(low[-10:])
        range_size = recent_high - recent_low
        
        if direction == 1:
            # For long: entry zone in lower half of range
            zone_low = recent_low
            zone_high = recent_low + range_size * 0.5
        else:
            # For short: entry zone in upper half of range
            zone_low = recent_high - range_size * 0.5
            zone_high = recent_high
        
        return zone_low, zone_high
    
    def _calculate_sl_tp(
        self,
        entry_price: float,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        entry_type: str,
    ) -> Tuple[float, float]:
        """คำนวณ SL และ TP"""
        
        atr = np.mean(high[-14:] - low[-14:])
        
        if entry_type == "breakout":
            sl_mult = 1.5
            tp_mult = 3.0
        elif entry_type == "pullback":
            sl_mult = 1.0
            tp_mult = 3.5
        else:
            sl_mult = 1.5
            tp_mult = 3.0
        
        if direction == 1:
            sl = entry_price - atr * sl_mult
            tp = entry_price + atr * tp_mult
        else:
            sl = entry_price + atr * sl_mult
            tp = entry_price - atr * tp_mult
        
        return sl, tp
    
    def get_timing_score(
        self,
        data: pd.DataFrame,
        direction: int,
    ) -> float:
        """ดึง Timing Score (0-100)"""
        
        entry_signal = self.analyze_entry(data, direction)
        return entry_signal.timing_score


def create_smart_timing() -> SmartTiming:
    """สร้าง SmartTiming"""
    return SmartTiming()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   SMART TIMING TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data with trend
    n = 100
    trend = np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    prices = 2000 + trend
    
    data = pd.DataFrame({
        "open": prices - np.random.rand(n) * 2,
        "high": prices + np.random.rand(n) * 4,
        "low": prices - np.random.rand(n) * 4,
        "close": prices,
    })
    
    # Analyze entry
    st = create_smart_timing()
    entry = st.analyze_entry(data, direction=1)
    
    print(f"\nEntry Analysis (LONG):")
    print(f"  Should Enter: {entry.should_enter}")
    print(f"  Entry Type: {entry.entry_type.value}")
    print(f"  Price: {entry.entry_price:.2f}")
    print(f"  SL: {entry.optimal_sl:.2f}")
    print(f"  TP: {entry.optimal_tp:.2f}")
    print(f"  Timing Score: {entry.timing_score:.0f}/100")
    print(f"  Reason: {entry.reason}")
    
    # Analyze exit
    exit_signal = st.analyze_exit(
        data,
        entry_price=prices[-20],
        direction=1,
        current_sl=prices[-1] - 10,
        current_tp=prices[-1] + 30,
    )
    
    print(f"\nExit Analysis:")
    print(f"  Should Exit: {exit_signal.should_exit}")
    print(f"  Exit Type: {exit_signal.exit_type.value}")
    print(f"  Urgency: {exit_signal.urgency}")
    print(f"  Reason: {exit_signal.reason}")
