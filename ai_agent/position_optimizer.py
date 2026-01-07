"""
Position Optimizer Module
==========================
คำนวณขนาด position อัตโนมัติโดยใช้ Kelly Criterion และ Risk Management

Features:
1. Kelly Criterion - คำนวณ optimal position size
2. Fractional Kelly - ลดความเสี่ยงด้วย Kelly fraction
3. Volatility Adjustment - ปรับตาม volatility
4. Drawdown Protection - ลดขนาดเมื่อขาดทุน
5. Confidence Scaling - ปรับตาม AI confidence
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


@dataclass
class PositionConfig:
    """การตั้งค่า Position Sizing"""
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # ใช้ 25% ของ Kelly (more conservative)
    min_position_pct: float = 0.005  # 0.5% minimum
    max_position_pct: float = 0.05  # 5% maximum
    
    # Risk parameters
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_daily_risk: float = 0.05  # 5% max daily risk
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    
    # Scaling factors
    confidence_weight: float = 0.5  # Weight for confidence scaling
    volatility_weight: float = 0.3  # Weight for volatility scaling
    
    # Drawdown protection
    drawdown_reduction_start: float = 0.05  # Start reducing at 5% DD
    drawdown_reduction_rate: float = 0.5  # Reduce by 50% per 5% DD


@dataclass  
class TradeResult:
    """ผลลัพธ์ของ trade สำหรับคำนวณ Kelly"""
    pnl: float
    pnl_pct: float
    is_win: bool
    risk_taken: float


class PositionOptimizer:
    """
    Position Size Optimizer
    
    ความสามารถ:
    1. คำนวณ Kelly Criterion จากประวัติ trades
    2. ปรับขนาดตาม confidence และ volatility
    3. จำกัดความเสี่ยงตาม drawdown
    4. Track และ optimize อัตโนมัติ
    """
    
    def __init__(
        self,
        config: PositionConfig = None,
        data_dir: str = "ai_agent/data",
    ):
        self.config = config or PositionConfig()
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Trade history for Kelly calculation
        self.trade_history: List[TradeResult] = []
        self.max_history = 100  # Last 100 trades
        
        # Current state
        self.current_kelly: float = 0.02  # Default 2%
        self.current_drawdown: float = 0.0
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        
        # Stats
        self.positions_calculated: int = 0
        self.avg_position_size: float = 0.0
        
        # Load existing
        self._load()
        
        logger.info("PositionOptimizer initialized")
    
    def calculate_kelly(self) -> float:
        """
        คำนวณ Kelly Criterion
        
        Kelly = W - (1-W)/R
        
        Where:
        - W = Win probability
        - R = Win/Loss ratio (average win / average loss)
        
        Returns:
            Optimal position size as fraction of capital
        """
        
        if len(self.trade_history) < 10:
            return self.config.max_position_pct * 0.5  # Conservative default
        
        wins = [t for t in self.trade_history if t.is_win]
        losses = [t for t in self.trade_history if not t.is_win]
        
        if not wins or not losses:
            return self.config.max_position_pct * 0.5
        
        # Win probability
        win_prob = len(wins) / len(self.trade_history)
        
        # Average win and loss
        avg_win = np.mean([t.pnl_pct for t in wins])
        avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
        
        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly = win_prob - ((1 - win_prob) / win_loss_ratio)
        
        # Apply fraction (conservative)
        fractional_kelly = kelly * self.config.kelly_fraction
        
        # Clip to reasonable range
        kelly_final = np.clip(
            fractional_kelly,
            self.config.min_position_pct,
            self.config.max_position_pct
        )
        
        self.current_kelly = kelly_final
        
        logger.debug(
            f"Kelly calc: W={win_prob:.1%}, R={win_loss_ratio:.2f}, "
            f"Full={kelly:.2%}, Frac={kelly_final:.2%}"
        )
        
        return kelly_final
    
    def get_position_size(
        self,
        capital: float,
        confidence: float,
        volatility: float,
        entry_price: float,
        stop_loss: float,
        current_equity: float = None,
    ) -> Dict[str, Any]:
        """
        คำนวณขนาด position ที่เหมาะสม
        
        Args:
            capital: เงินทุนทั้งหมด
            confidence: AI confidence (0-1)
            volatility: Current market volatility
            entry_price: ราคา entry
            stop_loss: ราคา stop loss
            current_equity: Equity ปัจจุบัน (สำหรับ drawdown calc)
            
        Returns:
            Dict with position details
        """
        
        self.positions_calculated += 1
        
        # Calculate Kelly
        kelly_pct = self.calculate_kelly()
        
        # Base position from Kelly
        base_position = capital * kelly_pct
        
        # 1. Adjust for confidence
        confidence_factor = self._confidence_scale(confidence)
        
        # 2. Adjust for volatility
        volatility_factor = self._volatility_scale(volatility)
        
        # 3. Adjust for drawdown
        if current_equity and self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        drawdown_factor = self._drawdown_scale(self.current_drawdown)
        
        # Apply all factors
        adjusted_position = base_position * confidence_factor * volatility_factor * drawdown_factor
        
        # 4. Risk-based limit
        if stop_loss > 0 and entry_price > 0:
            risk_per_unit = abs(entry_price - stop_loss) / entry_price
            max_from_risk = (capital * self.config.max_risk_per_trade) / (risk_per_unit + 1e-8)
            adjusted_position = min(adjusted_position, max_from_risk)
        
        # 5. Daily risk limit
        if self.daily_pnl < -(capital * self.config.max_daily_risk):
            adjusted_position = 0  # Stop trading for today
            logger.warning("Daily risk limit reached, no trading")
        
        # Final clamp
        final_position = np.clip(
            adjusted_position,
            capital * self.config.min_position_pct,
            capital * self.config.max_position_pct
        )
        
        # Calculate number of units/lots
        if entry_price > 0:
            units = final_position / entry_price
        else:
            units = 0
        
        # Update average
        self.avg_position_size = (
            self.avg_position_size * 0.95 + 
            (final_position / capital) * 0.05
        )
        
        result = {
            "position_value": final_position,
            "position_pct": final_position / capital,
            "units": units,
            "kelly_pct": kelly_pct,
            "confidence_factor": confidence_factor,
            "volatility_factor": volatility_factor,
            "drawdown_factor": drawdown_factor,
            "current_drawdown": self.current_drawdown,
        }
        
        logger.debug(
            f"Position: ${final_position:.2f} ({final_position/capital:.2%}), "
            f"Factors: C={confidence_factor:.2f}, V={volatility_factor:.2f}, DD={drawdown_factor:.2f}"
        )
        
        return result
    
    def _confidence_scale(self, confidence: float) -> float:
        """ปรับขนาดตาม confidence"""
        
        # High confidence = bigger position
        # confidence 0.5 -> factor 0.5
        # confidence 0.8 -> factor 1.0
        # confidence 0.95 -> factor 1.2
        
        if confidence >= 0.9:
            factor = 1.2
        elif confidence >= 0.8:
            factor = 1.0
        elif confidence >= 0.7:
            factor = 0.8
        elif confidence >= 0.6:
            factor = 0.6
        else:
            factor = 0.4
        
        # Blend with weight
        return 1.0 + (factor - 1.0) * self.config.confidence_weight
    
    def _volatility_scale(self, volatility: float) -> float:
        """ปรับขนาดตาม volatility (inverse)"""
        
        # High volatility = smaller position
        # Base volatility assumed to be 1.5%
        base_vol = 0.015
        
        if volatility <= 0:
            return 1.0
        
        factor = base_vol / (volatility + 1e-8)
        factor = np.clip(factor, 0.3, 1.5)
        
        # Blend with weight
        return 1.0 + (factor - 1.0) * self.config.volatility_weight
    
    def _drawdown_scale(self, drawdown: float) -> float:
        """ลดขนาดเมื่อ drawdown"""
        
        if drawdown <= self.config.drawdown_reduction_start:
            return 1.0
        
        # Reduce progressively
        excess_dd = drawdown - self.config.drawdown_reduction_start
        reduction_steps = excess_dd / 0.05  # Per 5% drawdown
        
        factor = max(
            0.1,  # Minimum 10%
            1.0 - (reduction_steps * self.config.drawdown_reduction_rate)
        )
        
        # Stop if at max drawdown
        if drawdown >= self.config.max_drawdown_limit:
            logger.warning(f"Max drawdown {drawdown:.1%} reached, stopping trades")
            return 0.0
        
        return factor
    
    def record_trade(
        self,
        pnl: float,
        pnl_pct: float,
        risk_taken: float = 0.02,
        current_equity: float = None,
    ):
        """บันทึกผล trade เพื่อ update Kelly"""
        
        result = TradeResult(
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_win=pnl > 0,
            risk_taken=risk_taken,
        )
        
        self.trade_history.append(result)
        
        # Keep only recent history
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history:]
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update peak equity
        if current_equity:
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Recalculate Kelly
        self.calculate_kelly()
        
        self._save()
    
    def reset_daily(self):
        """Reset daily tracking (เรียกทุกวัน)"""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        
        if self.trade_history:
            wins = [t for t in self.trade_history if t.is_win]
            return {
                "total_trades": len(self.trade_history),
                "win_rate": len(wins) / len(self.trade_history),
                "current_kelly": self.current_kelly,
                "current_drawdown": self.current_drawdown,
                "avg_position_size": self.avg_position_size,
                "positions_calculated": self.positions_calculated,
            }
        return {
            "total_trades": 0,
            "current_kelly": self.config.max_position_pct * 0.5,
        }
    
    def _save(self):
        """บันทึก state"""
        
        state = {
            "trade_history": [asdict(t) for t in self.trade_history],
            "current_kelly": self.current_kelly,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "positions_calculated": self.positions_calculated,
            "config": asdict(self.config),
        }
        
        path = os.path.join(self.data_dir, "position_optimizer.json")
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลด state"""
        
        path = os.path.join(self.data_dir, "position_optimizer.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                self.trade_history = [
                    TradeResult(**t) for t in state.get("trade_history", [])
                ]
                self.current_kelly = state.get("current_kelly", 0.02)
                self.current_drawdown = state.get("current_drawdown", 0.0)
                self.peak_equity = state.get("peak_equity", 0.0)
                self.positions_calculated = state.get("positions_calculated", 0)
                
                logger.info(f"Loaded position optimizer with {len(self.trade_history)} trades")
                
            except Exception as e:
                logger.warning(f"Failed to load position optimizer: {e}")


def create_position_optimizer() -> PositionOptimizer:
    """สร้าง PositionOptimizer"""
    return PositionOptimizer()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   POSITION OPTIMIZER TEST")
    print("="*60)
    
    np.random.seed(42)
    
    optimizer = create_position_optimizer()
    
    # Simulate some trades
    print("\nSimulating trades...")
    capital = 10000
    equity = capital
    
    for i in range(30):
        # Random trade result
        is_win = np.random.rand() > 0.4  # 60% win rate
        pnl_pct = np.random.uniform(0.01, 0.03) if is_win else -np.random.uniform(0.005, 0.02)
        pnl = capital * 0.02 * pnl_pct / abs(pnl_pct)
        
        equity += pnl
        
        optimizer.record_trade(
            pnl=pnl,
            pnl_pct=pnl_pct,
            current_equity=equity,
        )
    
    print(f"\nKelly calculated: {optimizer.current_kelly:.2%}")
    
    # Test position sizing
    print("\nPosition sizing tests:")
    
    test_cases = [
        {"confidence": 0.6, "volatility": 0.01, "desc": "Low conf, low vol"},
        {"confidence": 0.8, "volatility": 0.015, "desc": "Normal"},
        {"confidence": 0.9, "volatility": 0.02, "desc": "High conf, high vol"},
        {"confidence": 0.95, "volatility": 0.01, "desc": "Very high conf"},
    ]
    
    for case in test_cases:
        result = optimizer.get_position_size(
            capital=capital,
            confidence=case["confidence"],
            volatility=case["volatility"],
            entry_price=2000,
            stop_loss=1980,
            current_equity=equity,
        )
        
        print(f"  {case['desc']}: ${result['position_value']:.0f} ({result['position_pct']:.2%})")
    
    # Stats
    print("\nStats:")
    stats = optimizer.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
