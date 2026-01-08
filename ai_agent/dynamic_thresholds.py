"""
Dynamic Regime Thresholds v1.0
===============================
ปรับ thresholds อัตโนมัติตาม market regime

Features:
1. Regime-specific confidence thresholds
2. Volatility-adaptive position sizing
3. Session-aware trading rules
4. Auto-learning from trade history
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


@dataclass
class RegimeConfig:
    """Configuration for each market regime"""
    min_confidence: float = 0.55
    position_multiplier: float = 1.0
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 3.0
    max_trades_per_session: int = 5
    avoid_hours: List[int] = field(default_factory=list)
    require_consensus: bool = False
    

class DynamicRegimeThresholds:
    """
    ปรับ thresholds อัตโนมัติตาม regime ปัจจุบัน
    
    Capabilities:
    1. Different thresholds for each regime
    2. Learn optimal thresholds from history
    3. Adapt in real-time
    4. Session-aware rules
    """
    
    def __init__(self):
        # Default configs for each regime
        self.regime_configs: Dict[str, RegimeConfig] = {
            "trending_up": RegimeConfig(
                min_confidence=0.55,
                position_multiplier=1.2,
                sl_atr_multiplier=1.5,
                tp_atr_multiplier=4.0,
                require_consensus=False,
            ),
            "trending_down": RegimeConfig(
                min_confidence=0.55,
                position_multiplier=1.0,
                sl_atr_multiplier=1.5,
                tp_atr_multiplier=4.0,
                require_consensus=False,
            ),
            "ranging": RegimeConfig(
                min_confidence=0.60,  # Higher for ranging
                position_multiplier=0.7,  # Smaller positions
                sl_atr_multiplier=1.0,  # Tighter SL
                tp_atr_multiplier=2.0,  # Smaller TP
                require_consensus=True,  # Need model consensus
            ),
            "volatile": RegimeConfig(
                min_confidence=0.70,  # Much higher
                position_multiplier=0.5,  # Half size
                sl_atr_multiplier=2.0,  # Wider SL
                tp_atr_multiplier=3.0,
                avoid_hours=[0, 1, 2, 3, 4, 5],  # Avoid Asian low liquidity
                require_consensus=True,
            ),
            "unknown": RegimeConfig(
                min_confidence=0.65,
                position_multiplier=0.5,
                sl_atr_multiplier=1.5,
                tp_atr_multiplier=2.0,
                require_consensus=True,
            ),
        }
        
        # Performance tracking per regime
        self.regime_performance: Dict[str, Dict] = {
            regime: {"trades": 0, "wins": 0, "pnl": 0.0}
            for regime in self.regime_configs
        }
        
        # Learned adjustments
        self.learned_adjustments: Dict[str, float] = {}
        
        # History for learning
        self.trade_history: deque = deque(maxlen=500)
        
        logger.info("DynamicRegimeThresholds initialized with 5 regimes")
    
    def get_thresholds(
        self,
        regime: str,
        volatility: float = 0.5,
        hour: int = None,
    ) -> Dict:
        """
        Get optimal thresholds for current regime
        
        Returns:
            Dict with min_confidence, position_mult, sl_mult, tp_mult, etc.
        """
        # Get base config
        if regime not in self.regime_configs:
            regime = "unknown"
        
        config = self.regime_configs[regime]
        
        # Get current hour
        if hour is None:
            hour = datetime.now().hour
        
        # Build thresholds
        thresholds = {
            "min_confidence": config.min_confidence,
            "position_multiplier": config.position_multiplier,
            "sl_atr_multiplier": config.sl_atr_multiplier,
            "tp_atr_multiplier": config.tp_atr_multiplier,
            "require_consensus": config.require_consensus,
            "can_trade": True,
            "reason": "OK",
        }
        
        # Apply volatility adjustment
        if volatility > 0.7:  # High volatility
            thresholds["min_confidence"] = min(0.80, thresholds["min_confidence"] + 0.10)
            thresholds["position_multiplier"] *= 0.6
            thresholds["sl_atr_multiplier"] *= 1.3
        elif volatility < 0.3:  # Low volatility
            thresholds["min_confidence"] = max(0.50, thresholds["min_confidence"] - 0.05)
            thresholds["position_multiplier"] *= 1.1
        
        # Check avoid hours
        if hour in config.avoid_hours:
            thresholds["can_trade"] = False
            thresholds["reason"] = f"Avoid hour {hour} for {regime} regime"
        
        # Apply learned adjustments
        if regime in self.learned_adjustments:
            adj = self.learned_adjustments[regime]
            thresholds["min_confidence"] = max(0.45, min(0.85, 
                thresholds["min_confidence"] + adj))
        
        # Adjust based on regime performance
        perf = self.regime_performance.get(regime, {})
        if perf.get("trades", 0) >= 10:
            win_rate = perf["wins"] / perf["trades"]
            
            if win_rate < 0.4:  # Poor performance in this regime
                thresholds["min_confidence"] = min(0.80, thresholds["min_confidence"] + 0.10)
                thresholds["position_multiplier"] *= 0.7
                thresholds["reason"] = f"Regime {regime} has low win rate ({win_rate:.0%})"
            elif win_rate > 0.6:  # Good performance
                thresholds["min_confidence"] = max(0.50, thresholds["min_confidence"] - 0.05)
                thresholds["position_multiplier"] *= 1.1
        
        return thresholds
    
    def record_trade(
        self,
        regime: str,
        is_win: bool,
        pnl: float,
        entry_confidence: float,
        volatility: float,
    ):
        """Record trade result for learning"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {"trades": 0, "wins": 0, "pnl": 0.0}
        
        self.regime_performance[regime]["trades"] += 1
        if is_win:
            self.regime_performance[regime]["wins"] += 1
        self.regime_performance[regime]["pnl"] += pnl
        
        # Store for learning
        self.trade_history.append({
            "timestamp": datetime.now(),
            "regime": regime,
            "is_win": is_win,
            "pnl": pnl,
            "confidence": entry_confidence,
            "volatility": volatility,
        })
        
        # Update learned adjustments
        self._update_learned_adjustments()
    
    def _update_learned_adjustments(self):
        """Learn optimal confidence adjustments from history"""
        if len(self.trade_history) < 20:
            return
        
        # Group by regime
        regime_trades: Dict[str, List] = {}
        for trade in self.trade_history:
            regime = trade["regime"]
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        # Learn adjustment for each regime
        for regime, trades in regime_trades.items():
            if len(trades) < 10:
                continue
            
            # Find optimal confidence for this regime
            winning_confs = [t["confidence"] for t in trades if t["is_win"]]
            losing_confs = [t["confidence"] for t in trades if not t["is_win"]]
            
            if winning_confs and losing_confs:
                avg_winning = np.mean(winning_confs)
                avg_losing = np.mean(losing_confs)
                
                # If winning trades had higher confidence, increase threshold
                if avg_winning > avg_losing + 0.05:
                    self.learned_adjustments[regime] = 0.05  # Raise threshold
                elif avg_losing > avg_winning + 0.05:
                    self.learned_adjustments[regime] = -0.03  # Lower threshold slightly
                else:
                    self.learned_adjustments[regime] = 0.0
    
    def get_regime_stats(self) -> Dict:
        """Get performance stats for each regime"""
        stats = {}
        for regime, perf in self.regime_performance.items():
            if perf["trades"] > 0:
                stats[regime] = {
                    "trades": perf["trades"],
                    "win_rate": perf["wins"] / perf["trades"],
                    "total_pnl": perf["pnl"],
                    "avg_pnl": perf["pnl"] / perf["trades"],
                }
        return stats
    
    def should_trade_regime(
        self,
        regime: str,
        confidence: float,
        volatility: float = 0.5,
    ) -> Tuple[bool, str]:
        """
        Quick check if we should trade in this regime
        
        Returns:
            (should_trade, reason)
        """
        thresholds = self.get_thresholds(regime, volatility)
        
        if not thresholds["can_trade"]:
            return False, thresholds["reason"]
        
        if confidence < thresholds["min_confidence"]:
            return False, f"Confidence {confidence:.0%} < {thresholds['min_confidence']:.0%} for {regime}"
        
        return True, "OK"
    
    def get_position_size_multiplier(
        self,
        regime: str,
        volatility: float,
        confidence: float,
    ) -> float:
        """Get position size multiplier for current conditions"""
        thresholds = self.get_thresholds(regime, volatility)
        base_mult = thresholds["position_multiplier"]
        
        # Boost for high confidence
        if confidence > 0.75:
            base_mult *= 1.15
        
        # Reduce for low confidence
        if confidence < 0.60:
            base_mult *= 0.80
        
        return base_mult


def create_dynamic_thresholds() -> DynamicRegimeThresholds:
    """Factory function"""
    return DynamicRegimeThresholds()


# Test
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("=" * 60)
    print("   DYNAMIC REGIME THRESHOLDS TEST")
    print("=" * 60)
    
    drt = create_dynamic_thresholds()
    
    # Test different regimes
    for regime in ["trending_up", "ranging", "volatile"]:
        th = drt.get_thresholds(regime, volatility=0.5)
        print(f"\n{regime}:")
        print(f"  Min Confidence: {th['min_confidence']:.0%}")
        print(f"  Position Mult: {th['position_multiplier']:.1f}x")
        print(f"  SL Mult: {th['sl_atr_multiplier']:.1f}x ATR")
        print(f"  TP Mult: {th['tp_atr_multiplier']:.1f}x ATR")
    
    # Simulate trades
    print("\n\nSimulating trades...")
    for i in range(20):
        is_win = np.random.random() > 0.5
        drt.record_trade(
            regime="trending_up" if i % 2 == 0 else "ranging",
            is_win=is_win,
            pnl=50 if is_win else -30,
            entry_confidence=0.65,
            volatility=0.5,
        )
    
    print("\nRegime Stats:")
    for regime, stats in drt.get_regime_stats().items():
        print(f"  {regime}: {stats['trades']} trades, {stats['win_rate']:.0%} WR, ${stats['total_pnl']:.0f}")
