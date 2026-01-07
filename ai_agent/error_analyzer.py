"""
Error Analyzer Module
=====================
วิเคราะห์สาเหตุที่ขาดทุนและค้นหา patterns ที่ผิดซ้ำ

Features:
1. Error Pattern Detection - หา patterns ที่ทำให้ขาดทุน
2. Market Condition Analysis - วิเคราะห์สภาพตลาดที่ขาดทุนบ่อย
3. Self-Correction Suggestions - แนะนำการแก้ไข
4. Learning Integration - เชื่อมกับระบบ Online Learning
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger


@dataclass
class ErrorPattern:
    """รูปแบบการขาดทุนที่ตรวจพบ"""
    pattern_id: str
    pattern_type: str  # 'regime', 'volatility', 'timing', 'size', 'consecutive'
    description: str
    frequency: int  # จำนวนครั้งที่เกิด
    avg_loss: float
    total_loss: float
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    correction_applied: bool = False
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['first_seen'] = self.first_seen.isoformat()
        d['last_seen'] = self.last_seen.isoformat()
        return d
    
    @staticmethod
    def from_dict(d: Dict) -> 'ErrorPattern':
        d['first_seen'] = datetime.fromisoformat(d['first_seen'])
        d['last_seen'] = datetime.fromisoformat(d['last_seen'])
        return ErrorPattern(**d)


@dataclass 
class TradeError:
    """บันทึกข้อผิดพลาดจาก trade"""
    trade_id: str
    timestamp: datetime
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    
    # Market conditions at entry
    volatility: float
    trend: float
    regime: str  # 'trending_up', 'trending_down', 'ranging'
    rsi: float
    
    # Trade context
    confidence: float
    position_size: float
    
    # Error classification
    error_type: str = ""  # จะถูก classify โดย analyzer
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'


class ErrorAnalyzer:
    """
    วิเคราะห์ข้อผิดพลาดจากการเทรด
    
    ความสามารถ:
    1. จำแนกประเภทข้อผิดพลาด
    2. หารูปแบบที่ขาดทุนซ้ำ
    3. สร้างคำแนะนำแก้ไข
    4. ติดตามความคืบหน้าการแก้ไข
    """
    
    def __init__(
        self,
        data_dir: str = "ai_agent/data",
        min_samples_for_pattern: int = 3,
    ):
        self.data_dir = data_dir
        self.min_samples = min_samples_for_pattern
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Storage
        self.errors: List[TradeError] = []
        self.patterns: Dict[str, ErrorPattern] = {}
        self.corrections: Dict[str, Dict] = {}  # pattern_id -> correction rule
        
        # Load existing data
        self._load()
        
        logger.info(f"ErrorAnalyzer initialized with {len(self.errors)} historical errors")
    
    def record_error(
        self,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        holding_bars: int,
        volatility: float = 0.0,
        trend: float = 0.0,
        regime: str = "unknown",
        rsi: float = 50.0,
        confidence: float = 0.5,
        position_size: float = 0.01,
    ) -> TradeError:
        """บันทึก trade ที่ขาดทุน"""
        
        if pnl >= 0:
            return None  # ไม่ใช่ error ถ้าไม่ขาดทุน
        
        pnl_pct = (exit_price - entry_price) / entry_price
        
        error = TradeError(
            trade_id=trade_id,
            timestamp=datetime.now(),
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_bars=holding_bars,
            volatility=volatility,
            trend=trend,
            regime=regime,
            rsi=rsi,
            confidence=confidence,
            position_size=position_size,
        )
        
        # Classify error
        error = self._classify_error(error)
        
        self.errors.append(error)
        self._save()
        
        # Check for patterns
        self._detect_patterns()
        
        logger.info(f"Recorded error: {trade_id} | Type: {error.error_type} | Severity: {error.severity}")
        
        return error
    
    def _classify_error(self, error: TradeError) -> TradeError:
        """จำแนกประเภทข้อผิดพลาด"""
        
        # Check severity first
        if error.pnl_pct < -0.03:  # > 3% loss
            error.severity = "critical"
        elif error.pnl_pct < -0.02:  # > 2% loss
            error.severity = "high"
        elif error.pnl_pct < -0.01:  # > 1% loss
            error.severity = "medium"
        else:
            error.severity = "low"
        
        # Classify error type
        error_types = []
        
        # 1. Wrong regime
        if error.regime == "ranging" and abs(error.trend) < 0.01:
            error_types.append("wrong_regime")
        
        # 2. High volatility entry
        if error.volatility > 0.02:  # > 2% volatility
            error_types.append("high_volatility")
        
        # 3. Overbought/Oversold ignored
        if error.rsi > 70 or error.rsi < 30:
            error_types.append("extreme_rsi")
        
        # 4. Low confidence entry
        if error.confidence < 0.6:
            error_types.append("low_confidence")
        
        # 5. Position too big
        if error.position_size > 0.05 and error.pnl < -50:
            error_types.append("oversized")
        
        # 6. Held too long
        if error.holding_bars > 48:  # > 48 hours for H1
            error_types.append("held_too_long")
        
        # 7. Against trend
        if (error.trend > 0.02 and error.pnl < 0) or (error.trend < -0.02 and error.pnl < 0):
            if error.trend < 0:  # Trend was down
                error_types.append("against_trend")
        
        error.error_type = "|".join(error_types) if error_types else "unknown"
        
        return error
    
    def _detect_patterns(self):
        """ตรวจหา patterns ที่ขาดทุนซ้ำ"""
        
        if len(self.errors) < self.min_samples:
            return
        
        recent_errors = self.errors[-100:]  # Last 100 errors
        
        # Group by error type
        type_groups = defaultdict(list)
        for error in recent_errors:
            for err_type in error.error_type.split("|"):
                if err_type:
                    type_groups[err_type].append(error)
        
        # Detect patterns
        for err_type, errors in type_groups.items():
            if len(errors) >= self.min_samples:
                pattern_id = f"pattern_{err_type}"
                
                avg_loss = np.mean([e.pnl for e in errors])
                total_loss = sum(e.pnl for e in errors)
                
                # Aggregate market conditions
                conditions = {
                    "avg_volatility": np.mean([e.volatility for e in errors]),
                    "avg_rsi": np.mean([e.rsi for e in errors]),
                    "common_regime": max(set([e.regime for e in errors]), 
                                        key=lambda x: [e.regime for e in errors].count(x)),
                }
                
                pattern = ErrorPattern(
                    pattern_id=pattern_id,
                    pattern_type=err_type,
                    description=self._get_pattern_description(err_type),
                    frequency=len(errors),
                    avg_loss=avg_loss,
                    total_loss=total_loss,
                    market_conditions=conditions,
                    last_seen=errors[-1].timestamp,
                )
                
                if pattern_id in self.patterns:
                    # Update existing
                    old = self.patterns[pattern_id]
                    pattern.first_seen = old.first_seen
                    pattern.correction_applied = old.correction_applied
                
                self.patterns[pattern_id] = pattern
                
                if not pattern.correction_applied:
                    logger.warning(f"Pattern detected: {err_type} ({len(errors)} times, ${total_loss:.2f} loss)")
    
    def _get_pattern_description(self, err_type: str) -> str:
        """สร้างคำอธิบาย pattern"""
        descriptions = {
            "wrong_regime": "เปิด trade ในช่วงตลาด sideway/ranging ซึ่งไม่มี trend ชัดเจน",
            "high_volatility": "เปิด trade ในช่วงที่ตลาดมีความผันผวนสูง",
            "extreme_rsi": "เปิด trade เมื่อ RSI อยู่ในโซน overbought/oversold",
            "low_confidence": "เปิด trade ถึงแม้ว่า AI confidence ต่ำ",
            "oversized": "ขนาด position ใหญ่เกินไปเทียบกับความเสี่ยง",
            "held_too_long": "ถือ position นานเกินไป",
            "against_trend": "เปิด trade สวนทาง trend หลัก",
            "unknown": "ยังไม่สามารถระบุสาเหตุได้ชัดเจน",
        }
        return descriptions.get(err_type, "รูปแบบข้อผิดพลาดที่ยังไม่รู้จัก")
    
    def get_correction_rules(self) -> Dict[str, Dict]:
        """สร้างกฎการแก้ไขจาก patterns ที่พบ"""
        
        rules = {}
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.frequency >= self.min_samples:
                rule = self._create_correction_rule(pattern)
                if rule:
                    rules[pattern_id] = rule
        
        self.corrections = rules
        return rules
    
    def _create_correction_rule(self, pattern: ErrorPattern) -> Dict:
        """สร้างกฎแก้ไขสำหรับ pattern"""
        
        pt = pattern.pattern_type
        
        rules = {
            "wrong_regime": {
                "action": "skip_trade",
                "condition": "regime == 'ranging' and abs(trend) < 0.01",
                "description": "ไม่เทรดในช่วง sideway",
                "priority": 1,
            },
            "high_volatility": {
                "action": "reduce_size",
                "condition": "volatility > 0.02",
                "size_multiplier": 0.5,
                "description": "ลดขนาด position ครึ่งนึงเมื่อ volatility สูง",
                "priority": 2,
            },
            "extreme_rsi": {
                "action": "skip_trade",
                "condition": "rsi > 75 or rsi < 25",
                "description": "ไม่เทรดเมื่อ RSI อยู่ในโซน extreme",
                "priority": 2,
            },
            "low_confidence": {
                "action": "increase_threshold",
                "new_min_confidence": 0.75,
                "description": "เพิ่ม minimum confidence จาก 0.6 เป็น 0.75",
                "priority": 1,
            },
            "oversized": {
                "action": "cap_size",
                "max_position_pct": 0.03,
                "description": "จำกัดขนาด position ไม่เกิน 3%",
                "priority": 1,
            },
            "held_too_long": {
                "action": "add_time_stop",
                "max_bars": 36,  # 36 hours for H1
                "description": "เพิ่ม time-based exit หลังจาก 36 bars",
                "priority": 3,
            },
            "against_trend": {
                "action": "require_trend_alignment",
                "min_trend": 0.005,
                "description": "ต้อง trade ตาม trend เท่านั้น",
                "priority": 1,
            },
        }
        
        return rules.get(pt, {
            "action": "log_and_learn",
            "description": f"บันทึกและเรียนรู้จาก {pt}",
            "priority": 5,
        })
    
    def should_skip_trade(
        self,
        regime: str,
        trend: float,
        volatility: float,
        rsi: float,
        confidence: float,
    ) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าควร skip trade นี้หรือไม่
        
        Returns:
            (should_skip, reason)
        """
        
        corrections = self.get_correction_rules()
        
        # Check each rule
        if "pattern_wrong_regime" in corrections:
            if regime == "ranging" and abs(trend) < 0.01:
                return True, "Skip: ตลาด sideway (pattern detected)"
        
        if "pattern_high_volatility" in corrections:
            if volatility > 0.025:  # Stricter after detecting pattern
                return True, "Skip: Volatility สูงเกินไป"
        
        if "pattern_extreme_rsi" in corrections:
            if rsi > 75 or rsi < 25:
                return True, "Skip: RSI อยู่ในโซน extreme"
        
        if "pattern_low_confidence" in corrections:
            if confidence < 0.75:  # Increased from 0.6
                return True, "Skip: Confidence ต่ำกว่า threshold ใหม่"
        
        if "pattern_against_trend" in corrections:
            if trend < -0.01:  # Against uptrend
                return True, "Skip: สวนทาง trend"
        
        return False, ""
    
    def get_position_multiplier(
        self,
        volatility: float,
        recent_losses: int,
    ) -> float:
        """
        คำนวณ multiplier สำหรับขนาด position
        
        Returns:
            multiplier (0.1 - 1.0)
        """
        multiplier = 1.0
        
        corrections = self.get_correction_rules()
        
        # Reduce for high volatility
        if "pattern_high_volatility" in corrections:
            if volatility > 0.015:
                multiplier *= 0.5
        
        # Reduce after consecutive losses
        if recent_losses >= 2:
            multiplier *= 0.5
        if recent_losses >= 3:
            multiplier *= 0.5
        
        return max(0.1, min(1.0, multiplier))
    
    def get_analysis_report(self) -> Dict:
        """สร้างรายงานวิเคราะห์ข้อผิดพลาด"""
        
        if not self.errors:
            return {"status": "no_data", "message": "ยังไม่มีข้อมูลข้อผิดพลาด"}
        
        total_loss = sum(e.pnl for e in self.errors)
        avg_loss = np.mean([e.pnl for e in self.errors])
        
        # Group by error type
        by_type = defaultdict(list)
        for error in self.errors:
            for err_type in error.error_type.split("|"):
                if err_type:
                    by_type[err_type].append(error)
        
        type_stats = {}
        for err_type, errors in by_type.items():
            type_stats[err_type] = {
                "count": len(errors),
                "total_loss": sum(e.pnl for e in errors),
                "avg_loss": np.mean([e.pnl for e in errors]),
                "pct_of_total": len(errors) / len(self.errors) * 100,
            }
        
        # Sort by total loss
        worst_types = sorted(type_stats.items(), key=lambda x: x[1]["total_loss"])
        
        return {
            "total_errors": len(self.errors),
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "patterns_detected": len(self.patterns),
            "corrections_active": sum(1 for p in self.patterns.values() if p.correction_applied),
            "by_error_type": type_stats,
            "worst_types": worst_types[:5],  # Top 5 worst
            "recommendations": [
                self._get_pattern_description(t[0]) 
                for t in worst_types[:3]
            ],
        }
    
    def mark_pattern_corrected(self, pattern_id: str):
        """ทำเครื่องหมายว่า pattern ได้รับการแก้ไขแล้ว"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].correction_applied = True
            self._save()
            logger.info(f"Pattern {pattern_id} marked as corrected")
    
    def _save(self):
        """บันทึกข้อมูล"""
        # Save errors
        errors_path = os.path.join(self.data_dir, "trade_errors.json")
        with open(errors_path, 'w') as f:
            json.dump([asdict(e) for e in self.errors[-1000:]], f, indent=2, default=str)
        
        # Save patterns
        patterns_path = os.path.join(self.data_dir, "error_patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.patterns.items()}, f, indent=2)
    
    def _load(self):
        """โหลดข้อมูล"""
        # Load errors
        errors_path = os.path.join(self.data_dir, "trade_errors.json")
        if os.path.exists(errors_path):
            try:
                with open(errors_path, 'r') as f:
                    data = json.load(f)
                # Convert back to TradeError objects
                for d in data:
                    d['timestamp'] = datetime.fromisoformat(d['timestamp'])
                    self.errors.append(TradeError(**d))
            except Exception as e:
                logger.warning(f"Failed to load errors: {e}")
        
        # Load patterns
        patterns_path = os.path.join(self.data_dir, "error_patterns.json")
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path, 'r') as f:
                    data = json.load(f)
                self.patterns = {k: ErrorPattern.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")


# Convenience function
def create_error_analyzer() -> ErrorAnalyzer:
    """สร้าง ErrorAnalyzer"""
    return ErrorAnalyzer()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ERROR ANALYZER TEST")
    print("="*60)
    
    analyzer = create_error_analyzer()
    
    # Simulate some losing trades
    np.random.seed(42)
    
    for i in range(10):
        entry = 2000 + np.random.randn() * 10
        exit = entry - np.random.rand() * 30  # Ensure loss
        
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
            confidence=np.random.rand(),
        )
    
    print("\n" + "="*60)
    print("   ANALYSIS REPORT")
    print("="*60)
    
    report = analyzer.get_analysis_report()
    print(f"Total Errors: {report['total_errors']}")
    print(f"Total Loss: ${report['total_loss']:.2f}")
    print(f"Patterns Detected: {report['patterns_detected']}")
    
    print("\nWorst Error Types:")
    for i, (err_type, stats) in enumerate(report['worst_types'][:3], 1):
        print(f"  {i}. {err_type}: ${stats['total_loss']:.2f} ({stats['count']} times)")
    
    print("\nCorrection Rules:")
    rules = analyzer.get_correction_rules()
    for pattern_id, rule in rules.items():
        print(f"  - {pattern_id}: {rule.get('description', 'N/A')}")
