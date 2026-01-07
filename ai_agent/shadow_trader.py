"""
Shadow Trader Module
=====================
A/B Testing สำหรับกลยุทธ์ใหม่โดยไม่มีความเสี่ยง

Features:
1. Shadow Trading - ทดสอบกลยุทธ์ใหม่แบบ paper trading ควบคู่กับ live
2. Strategy Comparison - เปรียบเทียบผลลัพธ์
3. Gradual Rollout - ค่อยๆ เปลี่ยนไปใช้กลยุทธ์ใหม่
4. Performance Tracking - ติดตามผลทั้งสองกลยุทธ์
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from loguru import logger


@dataclass
class ShadowTrade:
    """บันทึก shadow trade"""
    trade_id: str
    timestamp: datetime
    strategy_id: str  # 'production' or 'shadow'
    
    # Trade details
    action: str  # 'LONG', 'WAIT', 'CLOSE'
    entry_price: float
    exit_price: float = 0.0
    
    # Results (filled when closed)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_closed: bool = False
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class StrategyPerformance:
    """สถิติของกลยุทธ์"""
    strategy_id: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0
    
    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0


class ShadowTrader:
    """
    Shadow Trading System
    
    ความสามารถ:
    1. รัน 2 กลยุทธ์พร้อมกัน (production + shadow)
    2. เปรียบเทียบผลลัพธ์
    3. ตัดสินใจว่าควรเปลี่ยนไปใช้กลยุทธ์ใหม่หรือไม่
    4. Gradual rollout - ค่อยๆ เพิ่มน้ำหนัก
    """
    
    def __init__(
        self,
        min_trades_for_comparison: int = 20,
        confidence_threshold: float = 0.95,
        data_dir: str = "ai_agent/data",
    ):
        self.min_trades = min_trades_for_comparison
        self.confidence_threshold = confidence_threshold
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Track strategies
        self.production_strategy = "production"
        self.shadow_strategy: Optional[str] = None
        
        # Trade tracking
        self.production_trades: List[ShadowTrade] = []
        self.shadow_trades: List[ShadowTrade] = []
        
        # Performance
        self.production_perf = StrategyPerformance("production")
        self.shadow_perf: Optional[StrategyPerformance] = None
        
        # Rollout state
        self.shadow_weight: float = 0.0  # 0 = all production, 1 = all shadow
        self.rollout_step: float = 0.1  # Increase by 10% each time
        
        self._load()
        
        logger.info("ShadowTrader initialized")
    
    def start_shadow_test(self, strategy_id: str):
        """เริ่มทดสอบกลยุทธ์ใหม่แบบ shadow"""
        
        self.shadow_strategy = strategy_id
        self.shadow_trades = []
        self.shadow_perf = StrategyPerformance(strategy_id)
        self.shadow_weight = 0.0
        
        logger.info(f"Started shadow testing: {strategy_id}")
    
    def record_trade_decision(
        self,
        strategy_id: str,
        action: str,
        price: float,
    ) -> str:
        """บันทึกการตัดสินใจเทรด"""
        
        trade = ShadowTrade(
            trade_id=f"{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            action=action,
            entry_price=price if action == "LONG" else 0.0,
        )
        
        if strategy_id == self.production_strategy:
            self.production_trades.append(trade)
        elif strategy_id == self.shadow_strategy:
            self.shadow_trades.append(trade)
        
        return trade.trade_id
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
    ):
        """ปิด trade และบันทึกผล"""
        
        # Find trade
        trade = None
        trades_list = None
        perf = None
        
        for t in self.production_trades:
            if t.trade_id == trade_id:
                trade = t
                trades_list = self.production_trades
                perf = self.production_perf
                break
        
        if not trade and self.shadow_trades:
            for t in self.shadow_trades:
                if t.trade_id == trade_id:
                    trade = t
                    trades_list = self.shadow_trades
                    perf = self.shadow_perf
                    break
        
        if not trade or not perf:
            return
        
        # Calculate P&L
        if trade.entry_price > 0:
            trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
            trade.pnl = trade.pnl_pct * 1000  # Assume $1000 base
        
        trade.exit_price = exit_price
        trade.is_closed = True
        
        # Update performance
        perf.total_trades += 1
        perf.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            perf.wins += 1
        else:
            perf.losses += 1
        
        self._save()
    
    def get_active_strategy(self) -> str:
        """
        เลือกกลยุทธ์ที่จะใช้ในการเทรดจริง
        
        ใช้ shadow_weight เพื่อค่อยๆ rollout
        """
        
        if not self.shadow_strategy or self.shadow_weight <= 0:
            return self.production_strategy
        
        if np.random.rand() < self.shadow_weight:
            return self.shadow_strategy
        
        return self.production_strategy
    
    def compare_strategies(self) -> Dict[str, Any]:
        """
        เปรียบเทียบผลลัพธ์ระหว่าง production และ shadow
        """
        
        if not self.shadow_perf:
            return {"status": "no_shadow_strategy"}
        
        result = {
            "production": {
                "trades": self.production_perf.total_trades,
                "win_rate": self.production_perf.win_rate,
                "total_pnl": self.production_perf.total_pnl,
                "avg_pnl": self.production_perf.avg_pnl,
            },
            "shadow": {
                "trades": self.shadow_perf.total_trades,
                "win_rate": self.shadow_perf.win_rate,
                "total_pnl": self.shadow_perf.total_pnl,
                "avg_pnl": self.shadow_perf.avg_pnl,
            }
        }
        
        # Calculate comparison
        if self.shadow_perf.total_trades >= self.min_trades:
            wr_diff = self.shadow_perf.win_rate - self.production_perf.win_rate
            pnl_diff = self.shadow_perf.avg_pnl - self.production_perf.avg_pnl
            
            result["comparison"] = {
                "win_rate_diff": wr_diff,
                "avg_pnl_diff": pnl_diff,
                "shadow_is_better": wr_diff > 0 and pnl_diff > 0,
                "significant": abs(wr_diff) > 0.05 or abs(pnl_diff) > 10,
            }
        else:
            result["comparison"] = {
                "status": "insufficient_data",
                "trades_needed": self.min_trades - self.shadow_perf.total_trades,
            }
        
        return result
    
    def should_promote_shadow(self) -> Tuple[bool, str]:
        """
        ตัดสินใจว่าควรเลื่อน shadow เป็น production หรือไม่
        """
        
        if not self.shadow_perf:
            return False, "No shadow strategy"
        
        if self.shadow_perf.total_trades < self.min_trades:
            return False, f"Not enough trades ({self.shadow_perf.total_trades}/{self.min_trades})"
        
        # Statistical comparison
        prod_wr = self.production_perf.win_rate
        shadow_wr = self.shadow_perf.win_rate
        
        # Simple comparison (can be made more sophisticated with statistical tests)
        if shadow_wr > prod_wr * 1.1:  # 10% better win rate
            if self.shadow_perf.avg_pnl > self.production_perf.avg_pnl:
                return True, f"Shadow better: WR {shadow_wr:.1%} vs {prod_wr:.1%}"
        
        if self.shadow_perf.total_pnl > self.production_perf.total_pnl * 1.2:
            return True, f"Shadow 20%+ better P&L"
        
        return False, "Shadow not significantly better"
    
    def increase_shadow_weight(self) -> float:
        """เพิ่มน้ำหนักให้ shadow (gradual rollout)"""
        
        should_promote, reason = self.should_promote_shadow()
        
        if should_promote:
            self.shadow_weight = min(1.0, self.shadow_weight + self.rollout_step)
            logger.info(f"Increased shadow weight to {self.shadow_weight:.0%}: {reason}")
            self._save()
        
        return self.shadow_weight
    
    def promote_shadow_to_production(self):
        """เลื่อน shadow เป็น production"""
        
        if not self.shadow_strategy:
            return False
        
        old_production = self.production_strategy
        
        self.production_strategy = self.shadow_strategy
        self.production_perf = self.shadow_perf
        self.production_trades = self.shadow_trades
        
        self.shadow_strategy = None
        self.shadow_perf = None
        self.shadow_trades = []
        self.shadow_weight = 0.0
        
        logger.info(f"Promoted {self.production_strategy} to production (was: {old_production})")
        
        self._save()
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะ"""
        
        return {
            "production_strategy": self.production_strategy,
            "shadow_strategy": self.shadow_strategy,
            "shadow_weight": self.shadow_weight,
            "production_trades": self.production_perf.total_trades,
            "shadow_trades": self.shadow_perf.total_trades if self.shadow_perf else 0,
            "comparison": self.compare_strategies(),
        }
    
    def _save(self):
        """บันทึก state"""
        
        state = {
            "production_strategy": self.production_strategy,
            "shadow_strategy": self.shadow_strategy,
            "shadow_weight": self.shadow_weight,
            "production_perf": asdict(self.production_perf),
            "shadow_perf": asdict(self.shadow_perf) if self.shadow_perf else None,
        }
        
        path = os.path.join(self.data_dir, "shadow_trader.json")
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลด state"""
        
        path = os.path.join(self.data_dir, "shadow_trader.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                self.production_strategy = state.get("production_strategy", "production")
                self.shadow_strategy = state.get("shadow_strategy")
                self.shadow_weight = state.get("shadow_weight", 0.0)
                
                if state.get("production_perf"):
                    self.production_perf = StrategyPerformance(**state["production_perf"])
                
                if state.get("shadow_perf"):
                    self.shadow_perf = StrategyPerformance(**state["shadow_perf"])
                
            except Exception as e:
                logger.warning(f"Failed to load shadow trader: {e}")


def create_shadow_trader() -> ShadowTrader:
    """สร้าง ShadowTrader"""
    return ShadowTrader()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   SHADOW TRADER TEST")
    print("="*60)
    
    np.random.seed(42)
    
    trader = create_shadow_trader()
    
    # Start shadow test
    trader.start_shadow_test("new_strategy_v2")
    
    # Simulate trades
    print("\nSimulating trades...")
    
    for i in range(30):
        price = 2000 + np.random.randn() * 10
        
        # Production trade
        prod_id = trader.record_trade_decision("production", "LONG", price)
        exit_price = price + np.random.randn() * 20
        trader.close_trade(prod_id, exit_price)
        
        # Shadow trade (slightly better)
        shadow_id = trader.record_trade_decision("new_strategy_v2", "LONG", price)
        exit_price = price + np.random.randn() * 20 + 5  # Bias toward profit
        trader.close_trade(shadow_id, exit_price)
    
    # Compare
    print("\nComparison:")
    comparison = trader.compare_strategies()
    print(f"  Production WR: {comparison['production']['win_rate']:.1%}")
    print(f"  Shadow WR: {comparison['shadow']['win_rate']:.1%}")
    print(f"  Production P&L: ${comparison['production']['total_pnl']:.2f}")
    print(f"  Shadow P&L: ${comparison['shadow']['total_pnl']:.2f}")
    
    # Check promotion
    should, reason = trader.should_promote_shadow()
    print(f"\nShould promote: {should} ({reason})")
    
    # Gradual rollout
    if should:
        for _ in range(5):
            weight = trader.increase_shadow_weight()
            print(f"  Shadow weight: {weight:.0%}")
