"""
Decision Brain Module
======================
สมองกลางสำหรับตัดสินใจเทรดอย่างชาญฉลาด

Features:
1. Multi-Layer Decision - หลายชั้นของการตัดสินใจ
2. Risk-Reward Optimization - หาจุดเข้าที่ดีที่สุด
3. Entry Timing - จับจังหวะ entry ที่แม่นยำ
4. Dynamic SL/TP - ปรับ SL/TP ตามสภาวะตลาด
5. Trade Quality Scoring - ให้คะแนน quality ของ trade
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


@dataclass
class TradeSetup:
    """โครงสร้าง Trade Setup"""
    action: str  # 'LONG', 'SHORT', 'WAIT'
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Quality metrics
    quality_score: float = 0.0  # 0-100
    risk_reward: float = 0.0
    confidence: float = 0.0
    
    # Reasons
    entry_reason: str = ""
    exit_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """ตรวจสอบว่า setup valid หรือไม่"""
        return (
            self.action in ['LONG', 'SHORT'] and
            self.quality_score >= 60 and
            self.risk_reward >= 2.0 and
            self.confidence >= 0.65
        )


class DecisionBrain:
    """
    Smart Decision Engine
    
    ความสามารถ:
    1. รวบรวม signals จากทุกแหล่ง
    2. คำนวณ optimal entry/exit
    3. ให้คะแนน quality ของ trade
    4. ตัดสินใจสุดท้าย
    """
    
    def __init__(
        self,
        min_quality: float = 45.0,  # Lowered from 60 to allow more trades
        min_rr: float = 1.5,  # Lowered from 1.9 for Gold trading
        min_confidence: float = 0.50,  # Lowered from 0.60
    ):
        self.min_quality = min_quality
        self.min_rr = min_rr
        self.min_confidence = min_confidence
        
        # Decision weights
        self.weights = {
            "market_intelligence": 0.30,
            "trend_alignment": 0.20,
            "momentum": 0.15,
            "structure": 0.15,
            "timing": 0.10,
            "volume": 0.10,
        }
        
        # Stats
        self.total_decisions = 0
        self.approved_trades = 0
        
        logger.info("DecisionBrain initialized")
    
    def analyze_setup(
        self,
        current_price: float,
        market_intel: Dict,
        atr: float,
        additional_signals: Dict = None,
    ) -> TradeSetup:
        """
        วิเคราะห์และสร้าง Trade Setup
        
        Args:
            current_price: ราคาปัจจุบัน
            market_intel: ผลจาก MarketIntelligence
            atr: Average True Range
            additional_signals: signals เพิ่มเติม
            
        Returns:
            TradeSetup with all details
        """
        
        self.total_decisions += 1
        
        # 1. Determine direction
        direction = self._determine_direction(market_intel)
        
        if direction == 0:
            return TradeSetup(
                action="WAIT",
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                quality_score=0,
                entry_reason="No clear direction",
            )
        
        # 2. Calculate entry, SL, TP
        entry, sl, tp = self._calculate_levels(
            current_price, direction, atr, market_intel
        )
        
        # 3. Calculate risk-reward
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0
        
        # 4. Calculate quality score
        quality = self._calculate_quality(
            market_intel, rr, additional_signals
        )
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            market_intel, quality, direction
        )
        
        # 6. Generate warnings
        warnings = self._generate_warnings(market_intel, rr, quality)
        
        # 7. Create setup
        action = "LONG" if direction > 0 else "SHORT"
        
        setup = TradeSetup(
            action=action,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            quality_score=quality,
            risk_reward=rr,
            confidence=confidence,
            entry_reason=self._get_entry_reason(market_intel, direction),
            exit_reason=f"TP at {rr:.1f}R or SL",
            warnings=warnings,
        )
        
        if setup.is_valid():
            self.approved_trades += 1
        
        return setup
    
    def _determine_direction(self, intel: Dict) -> int:
        """กำหนดทิศทาง: 1=LONG, -1=SHORT, 0=NEUTRAL"""
        
        bias_score = intel.get("bias_score", 0)
        trade_score = intel.get("trade_score", 0)
        
        # Strong signals (lowered thresholds for more trades)
        if bias_score >= 1 and trade_score >= 0.35:
            return 1
        elif bias_score <= -1 and trade_score >= 0.35:
            return -1
        
        # Medium signals (lowered from 0.4 to 0.25)
        if bias_score > 0 and trade_score >= 0.25:
            return 1
        elif bias_score < 0 and trade_score >= 0.25:
            return -1
        
        # Weak but valid signals
        if abs(bias_score) >= 0.5 and trade_score >= 0.15:
            return 1 if bias_score > 0 else -1
        
        return 0
    
    def _calculate_levels(
        self,
        price: float,
        direction: int,
        atr: float,
        intel: Dict,
    ) -> Tuple[float, float, float]:
        """คำนวณ Entry, SL, TP"""
        
        levels = intel.get("levels", {})
        
        if direction > 0:  # LONG
            # Entry at current or pullback
            entry = price
            
            # SL below nearest support or 1.5 ATR
            nearest_support = levels.get("nearest_support", price - atr * 1.5)
            sl = min(nearest_support - atr * 0.2, price - atr * 1.5)
            
            # TP at nearest resistance or 3 ATR
            nearest_resistance = levels.get("nearest_resistance", price + atr * 3)
            tp = max(nearest_resistance, price + atr * 3)
            
        else:  # SHORT
            entry = price
            
            nearest_resistance = levels.get("nearest_resistance", price + atr * 1.5)
            sl = max(nearest_resistance + atr * 0.2, price + atr * 1.5)
            
            nearest_support = levels.get("nearest_support", price - atr * 3)
            tp = min(nearest_support, price - atr * 3)
        
        return entry, sl, tp
    
    def _calculate_quality(
        self,
        intel: Dict,
        rr: float,
        additional: Dict = None,
    ) -> float:
        """คำนวณ Quality Score (0-100)"""
        
        scores = {}
        
        # Trade score from intelligence (40 points max)
        trade_score = intel.get("trade_score", 0)
        scores["intelligence"] = trade_score * 40
        
        # Risk-reward (20 points max)
        rr_score = min(rr / 4, 1) * 20  # Max at 4R
        scores["risk_reward"] = rr_score
        
        # Trend alignment (15 points max)
        trend = intel.get("trend", {})
        trend_score = abs(trend.get("score", 0)) * 15
        scores["trend"] = trend_score
        
        # Structure (15 points max)
        structure = intel.get("structure")
        if structure:
            if structure.trend in ["uptrend", "downtrend"]:
                scores["structure"] = 15
            elif structure.trend == "ranging":
                scores["structure"] = 5
            else:
                scores["structure"] = 0
        else:
            scores["structure"] = 0
        
        # Momentum (10 points max)
        momentum = intel.get("momentum", {})
        mom_score = abs(momentum.get("score", 0)) * 10
        scores["momentum"] = mom_score
        
        total = sum(scores.values())
        
        return min(100, total)
    
    def _calculate_confidence(
        self,
        intel: Dict,
        quality: float,
        direction: int,
    ) -> float:
        """คำนวณ Confidence (0-1)"""
        
        base_confidence = quality / 100
        
        # Boost for aligned factors
        factors = intel.get("confluence", {})
        if hasattr(factors, 'factors'):
            aligned = sum(1 for v in factors.factors.values() 
                         if (v > 0) == (direction > 0))
            boost = aligned * 0.05
        else:
            boost = 0
        
        # Penalty for warnings
        momentum = intel.get("momentum", {})
        if momentum.get("rsi_zone") in ["oversold", "overbought"]:
            if direction > 0 and momentum.get("rsi", 50) > 70:
                boost -= 0.1
            elif direction < 0 and momentum.get("rsi", 50) < 30:
                boost -= 0.1
        
        confidence = base_confidence + boost
        
        return np.clip(confidence, 0, 0.95)
    
    def _generate_warnings(
        self,
        intel: Dict,
        rr: float,
        quality: float,
    ) -> List[str]:
        """สร้าง warnings"""
        
        warnings = []
        
        if rr < 2:
            warnings.append(f"Low R:R ratio ({rr:.1f})")
        
        if quality < 60:
            warnings.append(f"Low quality score ({quality:.0f})")
        
        momentum = intel.get("momentum", {})
        if momentum.get("rsi_zone") == "overbought":
            warnings.append("RSI overbought")
        elif momentum.get("rsi_zone") == "oversold":
            warnings.append("RSI oversold")
        
        structure = intel.get("structure")
        if structure and structure.structure_shift:
            warnings.append("Recent structure shift (CHoCH)")
        
        return warnings
    
    def _get_entry_reason(self, intel: Dict, direction: int) -> str:
        """สร้างเหตุผล entry"""
        
        reasons = []
        
        # Bias
        bias = intel.get("bias", "NEUTRAL")
        if "BULLISH" in bias and direction > 0:
            reasons.append(f"Market bias: {bias}")
        elif "BEARISH" in bias and direction < 0:
            reasons.append(f"Market bias: {bias}")
        
        # Trend
        trend = intel.get("trend", {})
        if trend.get("direction", "").startswith("strong"):
            reasons.append(f"Strong trend: {trend['direction']}")
        
        # Confluence
        confluence = intel.get("confluence")
        if hasattr(confluence, 'description'):
            reasons.append(confluence.description)
        
        return "; ".join(reasons) if reasons else "Multiple factors aligned"
    
    def should_enter(self, setup: TradeSetup) -> Tuple[bool, str]:
        """ตัดสินใจสุดท้ายว่าควร enter หรือไม่"""
        
        if setup.action == "WAIT":
            return False, "No setup"
        
        if setup.quality_score < self.min_quality:
            return False, f"Quality {setup.quality_score:.0f} < {self.min_quality}"
        
        if setup.risk_reward < self.min_rr:
            return False, f"R:R {setup.risk_reward:.1f} < min {self.min_rr}"
        
        if setup.confidence < self.min_confidence:
            return False, f"Confidence {setup.confidence:.1%} < {self.min_confidence:.1%}"
        
        if len(setup.warnings) > 2:
            return False, f"Too many warnings: {len(setup.warnings)}"
        
        return True, f"Quality: {setup.quality_score:.0f}, R:R: {setup.risk_reward:.1f}"
    
    def get_stats(self) -> Dict:
        """ดึงสถิติ"""
        
        approval_rate = (
            self.approved_trades / self.total_decisions
            if self.total_decisions > 0 else 0
        )
        
        return {
            "total_decisions": self.total_decisions,
            "approved_trades": self.approved_trades,
            "approval_rate": approval_rate,
        }


def create_decision_brain() -> DecisionBrain:
    """สร้าง DecisionBrain"""
    return DecisionBrain()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   DECISION BRAIN TEST")
    print("="*60)
    
    # Mock market intelligence
    mock_intel = {
        "bias": "BULLISH",
        "bias_score": 1,
        "trade_score": 0.65,
        "trend": {"direction": "strong_up", "score": 0.8},
        "momentum": {"rsi": 55, "rsi_zone": "neutral", "score": 0.3},
        "structure": type('obj', (object,), {'trend': 'uptrend', 'structure_shift': False})(),
        "confluence": type('obj', (object,), {
            'description': 'Strong bullish: trend, structure',
            'factors': {'trend': 0.8, 'structure': 0.6}
        })(),
        "levels": {
            "nearest_support": 1990,
            "nearest_resistance": 2020,
        },
    }
    
    brain = create_decision_brain()
    
    # Analyze setup
    setup = brain.analyze_setup(
        current_price=2000,
        market_intel=mock_intel,
        atr=10,
    )
    
    print(f"\nAction: {setup.action}")
    print(f"Entry: {setup.entry_price:.2f}")
    print(f"SL: {setup.stop_loss:.2f}")
    print(f"TP: {setup.take_profit:.2f}")
    print(f"R:R: {setup.risk_reward:.2f}")
    print(f"Quality: {setup.quality_score:.0f}")
    print(f"Confidence: {setup.confidence:.1%}")
    print(f"Reason: {setup.entry_reason}")
    print(f"Warnings: {setup.warnings}")
    
    should, reason = brain.should_enter(setup)
    print(f"\nShould Enter: {should} ({reason})")
