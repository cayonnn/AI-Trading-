"""
Self-Corrector Module
=====================
ปรับพฤติกรรมการเทรดอัตโนมัติเพื่อไม่ให้ผิดซ้ำ

Features:
1. Auto-Correction - ปรับพฤติกรรมตาม patterns ที่ตรวจพบ
2. Adaptive Parameters - ปรับค่า parameters อัตโนมัติ
3. Learning Integration - เชื่อมกับ Online Learning
4. Performance Tracking - ติดตามผลการแก้ไข
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ai_agent.error_analyzer import ErrorAnalyzer, ErrorPattern


@dataclass
class CorrectionResult:
    """ผลลัพธ์จากการแก้ไข"""
    correction_id: str
    pattern_id: str
    applied_at: datetime
    action_taken: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    trades_since_correction: int = 0
    wins_since_correction: int = 0
    losses_since_correction: int = 0
    
    @property
    def win_rate_improvement(self) -> float:
        if self.trades_since_correction == 0:
            return 0.0
        return self.wins_since_correction / self.trades_since_correction


@dataclass
class AdaptiveParameters:
    """Parameters ที่ปรับได้อัตโนมัติ"""
    # Confidence thresholds (lowered for calibrated confidence)
    min_confidence: float = 0.50  # Lowered from 0.60 for calibrated confidence
    high_confidence: float = 0.80
    
    # Position sizing
    base_position_pct: float = 0.02  # 2% of capital
    max_position_pct: float = 0.05  # Max 5%
    volatility_scale: float = 1.0
    
    # Risk management
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    consecutive_loss_limit: int = 3
    
    # Trade filters
    min_rr_ratio: float = 2.0
    min_trend_strength: float = 0.005
    max_volatility: float = 0.025
    
    # Time-based
    avoid_high_impact_news: bool = True
    max_holding_bars: int = 48
    
    # Regime-specific
    ranging_allowed: bool = True
    ranging_size_multiplier: float = 0.5


class SelfCorrector:
    """
    ระบบแก้ไขตัวเองอัตโนมัติ
    
    ความสามารถ:
    1. รับข้อมูลจาก ErrorAnalyzer
    2. ปรับ parameters ตาม patterns ที่พบ
    3. ติดตามผลการแก้ไข
    4. Rollback ถ้าแก้ไขแล้วแย่ลง
    """
    
    def __init__(
        self,
        error_analyzer: ErrorAnalyzer = None,
        data_dir: str = "ai_agent/data",
    ):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.error_analyzer = error_analyzer or ErrorAnalyzer()
        
        # Current parameters
        self.params = AdaptiveParameters()
        self.default_params = AdaptiveParameters()  # For rollback
        
        # Correction history
        self.corrections: List[CorrectionResult] = []
        self.active_corrections: Dict[str, CorrectionResult] = {}
        
        # Performance tracking
        self.trades_before_correction = []
        self.trades_after_correction = []
        
        # Load existing state
        self._load()
        
        logger.info("SelfCorrector initialized")
    
    def analyze_and_correct(self) -> List[str]:
        """
        วิเคราะห์ patterns และปรับ parameters อัตโนมัติ
        
        Returns:
            List of corrections applied
        """
        corrections_applied = []
        
        # Get patterns from error analyzer
        rules = self.error_analyzer.get_correction_rules()
        
        for pattern_id, rule in rules.items():
            if pattern_id in self.active_corrections:
                continue  # Already corrected
            
            correction = self._apply_correction(pattern_id, rule)
            if correction:
                corrections_applied.append(f"{pattern_id}: {correction}")
        
        if corrections_applied:
            self._save()
            logger.info(f"Applied {len(corrections_applied)} corrections")
        
        return corrections_applied
    
    def _apply_correction(self, pattern_id: str, rule: Dict) -> Optional[str]:
        """ปรับ parameters ตาม rule"""
        
        action = rule.get("action", "")
        params_before = asdict(self.params)
        
        if action == "skip_trade":
            # Update filter conditions
            if "regime" in rule.get("condition", ""):
                self.params.ranging_allowed = False
                logger.info("Correction: Disabled ranging trades")
            if "rsi" in rule.get("condition", ""):
                # Already handled in should_trade check
                pass
            if "volatility" in rule.get("condition", ""):
                self.params.max_volatility = 0.02  # More strict
            
        elif action == "reduce_size":
            multiplier = rule.get("size_multiplier", 0.5)
            self.params.volatility_scale *= multiplier
            logger.info(f"Correction: Reduced size by {(1-multiplier)*100:.0f}%")
            
        elif action == "increase_threshold":
            new_conf = rule.get("new_min_confidence", 0.75)
            self.params.min_confidence = max(self.params.min_confidence, new_conf)
            logger.info(f"Correction: Increased min confidence to {self.params.min_confidence}")
            
        elif action == "cap_size":
            max_pct = rule.get("max_position_pct", 0.03)
            self.params.max_position_pct = min(self.params.max_position_pct, max_pct)
            logger.info(f"Correction: Capped position size to {max_pct*100:.0f}%")
            
        elif action == "add_time_stop":
            max_bars = rule.get("max_bars", 36)
            self.params.max_holding_bars = min(self.params.max_holding_bars, max_bars)
            logger.info(f"Correction: Added time stop at {max_bars} bars")
            
        elif action == "require_trend_alignment":
            min_trend = rule.get("min_trend", 0.005)
            self.params.min_trend_strength = max(self.params.min_trend_strength, min_trend)
            logger.info(f"Correction: Required min trend strength {min_trend}")
            
        else:
            return None
        
        # Record correction
        params_after = asdict(self.params)
        
        result = CorrectionResult(
            correction_id=f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_id=pattern_id,
            applied_at=datetime.now(),
            action_taken=action,
            parameters_before=params_before,
            parameters_after=params_after,
        )
        
        self.corrections.append(result)
        self.active_corrections[pattern_id] = result
        
        # Mark pattern as corrected
        self.error_analyzer.mark_pattern_corrected(pattern_id)
        
        return rule.get("description", action)
    
    def should_trade(
        self,
        confidence: float,
        volatility: float,
        trend: float,
        regime: str,
        rsi: float,
    ) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าควรเทรดหรือไม่ โดยใช้ parameters ที่ปรับแล้ว
        
        Returns:
            (should_trade, reason)
        """
        
        # Check confidence
        if confidence < self.params.min_confidence:
            return False, f"Confidence {confidence:.1%} < {self.params.min_confidence:.1%}"
        
        # Check volatility
        if volatility > self.params.max_volatility:
            return False, f"Volatility {volatility:.2%} > {self.params.max_volatility:.2%}"
        
        # Check regime
        if not self.params.ranging_allowed and regime == "ranging":
            return False, "Ranging market not allowed"
        
        # Check trend strength
        if abs(trend) < self.params.min_trend_strength:
            return False, f"Trend {trend:.3f} too weak"
        
        # Check from error analyzer as well
        skip, reason = self.error_analyzer.should_skip_trade(
            regime, trend, volatility, rsi, confidence
        )
        if skip:
            return False, reason
        
        return True, "OK"
    
    def get_position_size(
        self,
        capital: float,
        volatility: float,
        confidence: float,
        recent_losses: int = 0,
    ) -> float:
        """
        คำนวณขนาด position โดยใช้ parameters ที่ปรับแล้ว
        
        Returns:
            Position size in dollars
        """
        
        base_size = capital * self.params.base_position_pct
        
        # Scale by volatility
        if volatility > 0.015:
            base_size *= self.params.volatility_scale
        
        # Scale by confidence
        if confidence >= self.params.high_confidence:
            base_size *= 1.2  # Boost for high confidence
        elif confidence < 0.7:
            base_size *= 0.8
        
        # Get multiplier from error analyzer
        multiplier = self.error_analyzer.get_position_multiplier(
            volatility, recent_losses
        )
        base_size *= multiplier
        
        # Apply caps
        max_size = capital * self.params.max_position_pct
        
        # Reduce after consecutive losses
        if recent_losses >= self.params.consecutive_loss_limit:
            return 0  # Don't trade
        elif recent_losses >= 2:
            max_size *= 0.5
        
        return min(base_size, max_size)
    
    def record_trade_result(self, is_win: bool, pnl: float):
        """บันทึกผลการเทรดเพื่อติดตามประสิทธิผลการแก้ไข"""
        
        for result in self.active_corrections.values():
            result.trades_since_correction += 1
            if is_win:
                result.wins_since_correction += 1
            else:
                result.losses_since_correction += 1
        
        self._save()
    
    def evaluate_corrections(self) -> Dict[str, Any]:
        """
        ประเมินผลการแก้ไข
        
        Returns:
            Evaluation report
        """
        
        evaluations = {}
        
        for pattern_id, result in self.active_corrections.items():
            if result.trades_since_correction < 5:
                status = "insufficient_data"
            elif result.win_rate_improvement > 0.5:
                status = "effective"
            elif result.win_rate_improvement > 0.4:
                status = "moderate"
            else:
                status = "ineffective"
            
            evaluations[pattern_id] = {
                "status": status,
                "trades": result.trades_since_correction,
                "win_rate": result.win_rate_improvement,
                "applied_at": result.applied_at.isoformat(),
            }
        
        return evaluations
    
    def rollback_correction(self, pattern_id: str):
        """ย้อนกลับการแก้ไขที่ไม่ได้ผล"""
        
        if pattern_id not in self.active_corrections:
            return False
        
        result = self.active_corrections[pattern_id]
        
        # Restore previous parameters
        for key, value in result.parameters_before.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        
        del self.active_corrections[pattern_id]
        
        logger.info(f"Rolled back correction for {pattern_id}")
        
        self._save()
        return True
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """ดึง parameters ปัจจุบัน"""
        return asdict(self.params)
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะระบบ"""
        return {
            "active_corrections": len(self.active_corrections),
            "total_corrections": len(self.corrections),
            "current_params": {
                "min_confidence": self.params.min_confidence,
                "max_volatility": self.params.max_volatility,
                "max_position_pct": self.params.max_position_pct,
                "ranging_allowed": self.params.ranging_allowed,
            },
            "evaluations": self.evaluate_corrections(),
        }
    
    def _save(self):
        """บันทึกสถานะ"""
        state = {
            "params": asdict(self.params),
            "active_corrections": {
                k: {
                    **asdict(v),
                    "applied_at": v.applied_at.isoformat(),
                }
                for k, v in self.active_corrections.items()
            },
        }
        
        path = os.path.join(self.data_dir, "self_corrector_state.json")
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลดสถานะ"""
        path = os.path.join(self.data_dir, "self_corrector_state.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                # Restore params
                for key, value in state.get("params", {}).items():
                    if hasattr(self.params, key):
                        setattr(self.params, key, value)
                
                # Restore active corrections
                for k, v in state.get("active_corrections", {}).items():
                    v["applied_at"] = datetime.fromisoformat(v["applied_at"])
                    self.active_corrections[k] = CorrectionResult(**v)
                    
                logger.info(f"Loaded {len(self.active_corrections)} active corrections")
                
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")


def create_self_corrector(error_analyzer: ErrorAnalyzer = None) -> SelfCorrector:
    """สร้าง SelfCorrector"""
    return SelfCorrector(error_analyzer=error_analyzer)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   SELF-CORRECTOR TEST")
    print("="*60)
    
    from ai_agent.error_analyzer import create_error_analyzer
    
    analyzer = create_error_analyzer()
    corrector = create_self_corrector(analyzer)
    
    # Simulate some losing trades
    np.random.seed(42)
    
    for i in range(15):
        entry = 2000 + np.random.randn() * 10
        exit = entry - np.random.rand() * 30
        
        analyzer.record_error(
            trade_id=f"TEST_{i}",
            entry_price=entry,
            exit_price=exit,
            pnl=-(entry - exit),
            holding_bars=np.random.randint(1, 60),
            volatility=np.random.rand() * 0.03,
            trend=np.random.randn() * 0.02,
            regime=np.random.choice(["trending_up", "trending_down", "ranging"]),
            rsi=np.random.rand() * 100,
            confidence=np.random.rand() * 0.5 + 0.5,
        )
    
    # Apply corrections
    print("\nApplying corrections...")
    corrections = corrector.analyze_and_correct()
    for c in corrections:
        print(f"  - {c}")
    
    # Test should_trade
    print("\nTesting should_trade:")
    for conf in [0.65, 0.75, 0.85]:
        can_trade, reason = corrector.should_trade(
            confidence=conf,
            volatility=0.015,
            trend=0.01,
            regime="trending_up",
            rsi=55,
        )
        print(f"  Conf={conf:.0%}: {can_trade} ({reason})")
    
    # Get status
    print("\nStatus:")
    status = corrector.get_status()
    print(f"  Active corrections: {status['active_corrections']}")
    print(f"  Min confidence: {status['current_params']['min_confidence']:.0%}")
