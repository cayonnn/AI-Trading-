"""
Adaptive Confidence Module
===========================
ปรับ Confidence แบบ Dynamic ตามสถานการณ์

Features:
1. Performance-Based Adjustment - ปรับตาม win/lose history
2. Regime-Specific Confidence - ปรับตาม market regime
3. Time-of-Day Adjustment - ปรับตามช่วงเวลา
4. Volatility Scaling - ปรับตาม volatility
5. Streak Adjustment - ปรับตาม winning/losing streak
"""

import numpy as np
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from loguru import logger


@dataclass
class ConfidenceFactors:
    """Factors ที่มีผลต่อ Confidence"""
    base_confidence: float = 0.70
    performance_adj: float = 0.0
    regime_adj: float = 0.0
    time_adj: float = 0.0
    volatility_adj: float = 0.0
    streak_adj: float = 0.0
    
    @property
    def final_confidence(self) -> float:
        total = (
            self.base_confidence +
            self.performance_adj +
            self.regime_adj +
            self.time_adj +
            self.volatility_adj +
            self.streak_adj
        )
        return np.clip(total, 0.3, 0.95)


class AdaptiveConfidence:
    """
    Dynamic Confidence Adjustment
    
    ความสามารถ:
    1. เรียนรู้จาก performance ที่ผ่านมา
    2. ปรับตาม market conditions
    3. ปรับตามช่วงเวลา trading sessions
    4. ปรับตาม recent streak
    """
    
    def __init__(
        self,
        base_confidence: float = 0.70,
        history_size: int = 50,
    ):
        self.base_confidence = base_confidence
        self.history_size = history_size
        
        # Performance tracking
        self.trade_results: deque = deque(maxlen=history_size)
        
        # Regime performance
        self.regime_performance: Dict[str, Dict] = {
            "trending_up": {"wins": 0, "total": 0},
            "trending_down": {"wins": 0, "total": 0},
            "ranging": {"wins": 0, "total": 0},
            "volatile": {"wins": 0, "total": 0},
        }
        
        # Time-based performance
        self.session_performance: Dict[str, Dict] = {
            "asian": {"wins": 0, "total": 0},  # 0-8 GMT
            "london": {"wins": 0, "total": 0},  # 7-16 GMT
            "new_york": {"wins": 0, "total": 0},  # 13-22 GMT
            "off_hours": {"wins": 0, "total": 0},
        }
        
        # Streak tracking
        self.current_streak = 0  # Positive = winning, Negative = losing
        
        # Stats
        self.adjustments_made = 0
        
        logger.info("AdaptiveConfidence initialized")
    
    def calculate_confidence(
        self,
        raw_confidence: float,
        regime: str,
        volatility: float,
        current_time: datetime = None,
    ) -> Tuple[float, ConfidenceFactors]:
        """
        คำนวณ Confidence ที่ปรับแล้ว
        
        Args:
            raw_confidence: Confidence จาก AI/model
            regime: Current market regime
            volatility: Current volatility
            current_time: Current datetime
            
        Returns:
            Tuple of (adjusted_confidence, factors)
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        factors = ConfidenceFactors(base_confidence=raw_confidence)
        
        # 1. Performance-based adjustment
        factors.performance_adj = self._performance_adjustment()
        
        # 2. Regime-based adjustment
        factors.regime_adj = self._regime_adjustment(regime)
        
        # 3. Time-based adjustment
        factors.time_adj = self._time_adjustment(current_time)
        
        # 4. Volatility-based adjustment
        factors.volatility_adj = self._volatility_adjustment(volatility)
        
        # 5. Streak-based adjustment
        factors.streak_adj = self._streak_adjustment()
        
        self.adjustments_made += 1
        
        return factors.final_confidence, factors
    
    def _performance_adjustment(self) -> float:
        """ปรับตาม recent performance"""
        
        if len(self.trade_results) < 10:
            return 0.0
        
        recent = list(self.trade_results)[-20:]
        win_rate = sum(1 for r in recent if r > 0) / len(recent)
        
        # Baseline is 0.5 (50% win rate)
        if win_rate > 0.6:
            return 0.05  # Boost confidence
        elif win_rate > 0.55:
            return 0.02
        elif win_rate < 0.4:
            return -0.10  # Reduce confidence significantly
        elif win_rate < 0.45:
            return -0.05
        
        return 0.0
    
    def _regime_adjustment(self, regime: str) -> float:
        """ปรับตาม regime performance"""
        
        regime_key = regime.lower().replace(" ", "_")
        
        if regime_key not in self.regime_performance:
            regime_key = "ranging"
        
        stats = self.regime_performance[regime_key]
        
        if stats["total"] < 5:
            # Not enough data, be conservative
            if regime_key == "volatile":
                return -0.05
            return 0.0
        
        win_rate = stats["wins"] / stats["total"]
        
        if win_rate > 0.6:
            return 0.05
        elif win_rate < 0.4:
            return -0.08
        
        return 0.0
    
    def _time_adjustment(self, current_time: datetime) -> float:
        """ปรับตาม trading session"""
        
        hour = current_time.hour
        
        # Determine session (GMT-based, adjust as needed)
        if 0 <= hour < 8:
            session = "asian"
        elif 7 <= hour < 16:
            session = "london"
        elif 13 <= hour < 22:
            session = "new_york"
        else:
            session = "off_hours"
        
        stats = self.session_performance[session]
        
        if stats["total"] < 5:
            # Default adjustments
            if session == "off_hours":
                return -0.05
            elif session == "london":
                return 0.02  # Usually good volatility
            return 0.0
        
        win_rate = stats["wins"] / stats["total"]
        
        if win_rate > 0.6:
            return 0.03
        elif win_rate < 0.4:
            return -0.05
        
        return 0.0
    
    def _volatility_adjustment(self, volatility: float) -> float:
        """ปรับตาม volatility"""
        
        # Base volatility assumed 1.5%
        base_vol = 0.015
        
        if volatility <= 0:
            return 0.0
        
        vol_ratio = volatility / base_vol
        
        if vol_ratio > 2.0:
            # High volatility - reduce confidence
            return -0.08
        elif vol_ratio > 1.5:
            return -0.04
        elif vol_ratio < 0.5:
            # Very low volatility - also reduce (may be choppy)
            return -0.03
        elif vol_ratio < 0.8:
            return 0.02  # Optimal lower volatility
        
        return 0.0
    
    def _streak_adjustment(self) -> float:
        """ปรับตาม winning/losing streak"""
        
        if self.current_streak >= 5:
            # Long winning streak - be cautious (reversion to mean)
            return -0.03
        elif self.current_streak >= 3:
            return 0.02  # Slight boost
        elif self.current_streak <= -5:
            # Long losing streak - reduce significantly
            return -0.15
        elif self.current_streak <= -3:
            return -0.08
        elif self.current_streak <= -2:
            return -0.04
        
        return 0.0
    
    def record_trade(
        self,
        pnl: float,
        regime: str = "unknown",
        trade_time: datetime = None,
    ):
        """บันทึกผล trade เพื่อ update adjustments"""
        
        is_win = pnl > 0
        
        # Record result
        self.trade_results.append(pnl)
        
        # Update streak
        if is_win:
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
        else:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
        
        # Update regime performance
        regime_key = regime.lower().replace(" ", "_")
        if regime_key in self.regime_performance:
            self.regime_performance[regime_key]["total"] += 1
            if is_win:
                self.regime_performance[regime_key]["wins"] += 1
        
        # Update session performance
        if trade_time:
            hour = trade_time.hour
            if 0 <= hour < 8:
                session = "asian"
            elif 7 <= hour < 16:
                session = "london"
            elif 13 <= hour < 22:
                session = "new_york"
            else:
                session = "off_hours"
            
            self.session_performance[session]["total"] += 1
            if is_win:
                self.session_performance[session]["wins"] += 1
    
    def get_minimum_confidence(self) -> float:
        """
        คำนวณ minimum confidence threshold ที่ควรใช้
        ปรับตาม recent performance
        """
        
        if len(self.trade_results) < 20:
            return 0.70
        
        recent = list(self.trade_results)[-20:]
        win_rate = sum(1 for r in recent if r > 0) / len(recent)
        
        if win_rate < 0.4:
            # Poor performance - raise threshold
            return 0.80
        elif win_rate < 0.5:
            return 0.75
        elif win_rate > 0.6:
            # Good performance - can lower slightly
            return 0.65
        
        return 0.70
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        
        total_trades = len(self.trade_results)
        
        if total_trades > 0:
            wins = sum(1 for r in self.trade_results if r > 0)
            win_rate = wins / total_trades
        else:
            win_rate = 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "current_streak": self.current_streak,
            "adjustments_made": self.adjustments_made,
            "regime_performance": {
                k: v["wins"] / v["total"] if v["total"] > 0 else 0
                for k, v in self.regime_performance.items()
            },
            "session_performance": {
                k: v["wins"] / v["total"] if v["total"] > 0 else 0
                for k, v in self.session_performance.items()
            },
        }


def create_adaptive_confidence() -> AdaptiveConfidence:
    """สร้าง AdaptiveConfidence"""
    return AdaptiveConfidence()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ADAPTIVE CONFIDENCE TEST")
    print("="*60)
    
    np.random.seed(42)
    
    ac = create_adaptive_confidence()
    
    # Simulate some trades
    print("\nSimulating trades...")
    for i in range(30):
        pnl = np.random.randn() * 100
        regime = np.random.choice(["trending_up", "ranging", "volatile"])
        ac.record_trade(pnl, regime, datetime.now())
    
    # Test confidence calculation
    print("\nConfidence calculations:")
    
    test_cases = [
        {"raw": 0.75, "regime": "trending_up", "vol": 0.015},
        {"raw": 0.70, "regime": "ranging", "vol": 0.025},
        {"raw": 0.80, "regime": "volatile", "vol": 0.035},
    ]
    
    for case in test_cases:
        conf, factors = ac.calculate_confidence(
            raw_confidence=case["raw"],
            regime=case["regime"],
            volatility=case["vol"],
        )
        
        print(f"\nRegime: {case['regime']}, Vol: {case['vol']:.1%}")
        print(f"  Raw: {case['raw']:.1%} -> Adjusted: {conf:.1%}")
        print(f"  Adjustments: Perf={factors.performance_adj:+.2f}, "
              f"Regime={factors.regime_adj:+.2f}, "
              f"Vol={factors.volatility_adj:+.2f}")
    
    # Stats
    print("\nStats:")
    stats = ac.get_stats()
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Current Streak: {stats['current_streak']}")
    print(f"  Min Confidence Threshold: {ac.get_minimum_confidence():.1%}")
