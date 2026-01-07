"""
Knowledge Base Module
=====================
Knowledge Distillation และ Long-term Memory สำหรับ AI

Features:
1. Market Pattern Memory - จำ patterns ที่ทำกำไร
2. Regime Knowledge - ความรู้เฉพาะ regime
3. Trade Rules - กฎจากประสบการณ์
4. Strategy Repository - เก็บกลยุทธ์ที่ดี
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
class MarketPattern:
    """รูปแบบตลาดที่จดจำ"""
    pattern_id: str
    name: str
    description: str
    
    # Pattern conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    times_seen: int = 0
    times_profitable: int = 0
    avg_return: float = 0.0
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        return self.times_profitable / self.times_seen if self.times_seen > 0 else 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['discovered_at'] = self.discovered_at.isoformat()
        d['last_seen'] = self.last_seen.isoformat()
        return d


@dataclass
class TradeRule:
    """กฎการเทรดที่เรียนรู้"""
    rule_id: str
    rule_type: str  # 'entry', 'exit', 'sizing', 'filter'
    condition: str  # Human-readable condition
    action: str  # What to do
    
    # Confidence
    confidence: float = 0.5
    times_applied: int = 0
    times_correct: int = 0
    
    # Source
    source: str = "learned"  # 'hardcoded', 'learned', 'evolved'
    
    @property
    def accuracy(self) -> float:
        return self.times_correct / self.times_applied if self.times_applied > 0 else 0.5


@dataclass
class RegimeKnowledge:
    """ความรู้เฉพาะ regime"""
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    
    # Best parameters for this regime
    optimal_confidence: float = 0.60  # Lowered from 0.75
    optimal_position_size: float = 0.02
    optimal_stop_atr: float = 1.5
    optimal_target_atr: float = 3.0
    
    # Statistics
    trades: int = 0
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    
    # Learned filters
    filters: List[str] = field(default_factory=list)


class KnowledgeBase:
    """
    AI Knowledge Base
    
    ความสามารถ:
    1. บันทึกและเรียกคืน patterns
    2. จัดการ trade rules
    3. เก็บความรู้เฉพาะ regime
    4. Knowledge distillation
    """
    
    def __init__(
        self,
        data_dir: str = "ai_agent/data",
    ):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Storage
        self.patterns: Dict[str, MarketPattern] = {}
        self.rules: Dict[str, TradeRule] = {}
        self.regime_knowledge: Dict[str, RegimeKnowledge] = {}
        self.strategy_archive: List[Dict] = []
        
        # Initialize default regime knowledge
        for regime in ['trending_up', 'trending_down', 'ranging', 'volatile']:
            self.regime_knowledge[regime] = RegimeKnowledge(regime=regime)
        
        self._load()
        
        logger.info(f"KnowledgeBase initialized with {len(self.patterns)} patterns, {len(self.rules)} rules")
    
    def add_pattern(
        self,
        name: str,
        description: str,
        conditions: Dict[str, Any],
        was_profitable: bool,
        return_pct: float = 0.0,
    ) -> str:
        """เพิ่มหรืออัพเดต pattern"""
        
        # Create pattern ID from conditions
        pattern_id = f"pat_{hash(json.dumps(conditions, sort_keys=True)) % 100000}"
        
        if pattern_id in self.patterns:
            # Update existing
            pattern = self.patterns[pattern_id]
            pattern.times_seen += 1
            if was_profitable:
                pattern.times_profitable += 1
            # Moving average of returns
            pattern.avg_return = (
                pattern.avg_return * 0.9 + return_pct * 0.1
            )
            pattern.last_seen = datetime.now()
        else:
            # Create new
            pattern = MarketPattern(
                pattern_id=pattern_id,
                name=name,
                description=description,
                conditions=conditions,
                times_seen=1,
                times_profitable=1 if was_profitable else 0,
                avg_return=return_pct,
            )
            self.patterns[pattern_id] = pattern
            logger.info(f"New pattern discovered: {name}")
        
        self._save()
        return pattern_id
    
    def find_matching_patterns(
        self,
        current_conditions: Dict[str, float],
        min_success_rate: float = 0.5,
    ) -> List[MarketPattern]:
        """ค้นหา patterns ที่ตรงกับสภาวะปัจจุบัน"""
        
        matches = []
        
        for pattern in self.patterns.values():
            if pattern.times_seen < 5:
                continue
            
            if pattern.success_rate < min_success_rate:
                continue
            
            # Check if conditions match
            match_score = self._calculate_match_score(
                pattern.conditions, current_conditions
            )
            
            if match_score > 0.7:
                matches.append(pattern)
        
        # Sort by success rate
        matches.sort(key=lambda x: x.success_rate, reverse=True)
        
        return matches[:5]  # Top 5
    
    def _calculate_match_score(
        self,
        pattern_conditions: Dict,
        current_conditions: Dict,
    ) -> float:
        """คำนวณว่า conditions ตรงกันแค่ไหน"""
        
        if not pattern_conditions or not current_conditions:
            return 0.0
        
        matches = 0
        total = 0
        
        for key, pattern_value in pattern_conditions.items():
            if key not in current_conditions:
                continue
            
            current_value = current_conditions[key]
            
            if isinstance(pattern_value, (int, float)):
                # Numeric comparison (within 20%)
                if pattern_value != 0:
                    diff = abs(current_value - pattern_value) / abs(pattern_value)
                    if diff < 0.2:
                        matches += 1
            else:
                # String comparison
                if current_value == pattern_value:
                    matches += 1
            
            total += 1
        
        return matches / total if total > 0 else 0.0
    
    def add_rule(
        self,
        rule_type: str,
        condition: str,
        action: str,
        source: str = "learned",
    ) -> str:
        """เพิ่ม trade rule"""
        
        rule_id = f"rule_{len(self.rules) + 1}"
        
        rule = TradeRule(
            rule_id=rule_id,
            rule_type=rule_type,
            condition=condition,
            action=action,
            source=source,
        )
        
        self.rules[rule_id] = rule
        
        logger.info(f"Added rule: {condition} -> {action}")
        
        self._save()
        return rule_id
    
    def update_rule(self, rule_id: str, was_correct: bool):
        """อัพเดตความถูกต้องของ rule"""
        
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        rule.times_applied += 1
        
        if was_correct:
            rule.times_correct += 1
        
        # Update confidence
        rule.confidence = rule.accuracy
        
        self._save()
    
    def get_rules_for_type(
        self,
        rule_type: str,
        min_confidence: float = 0.5,
    ) -> List[TradeRule]:
        """ดึง rules ตามประเภท"""
        
        return [
            rule for rule in self.rules.values()
            if rule.rule_type == rule_type and rule.confidence >= min_confidence
        ]
    
    def update_regime_knowledge(
        self,
        regime: str,
        trade_result: Dict,
    ):
        """อัพเดตความรู้ regime จากผล trade"""
        
        if regime not in self.regime_knowledge:
            self.regime_knowledge[regime] = RegimeKnowledge(regime=regime)
        
        rk = self.regime_knowledge[regime]
        
        # Update stats
        rk.trades += 1
        
        pnl = trade_result.get('pnl', 0)
        is_win = pnl > 0
        
        # Moving average update
        alpha = 0.1
        rk.win_rate = rk.win_rate * (1 - alpha) + (1 if is_win else 0) * alpha
        rk.avg_pnl = rk.avg_pnl * (1 - alpha) + pnl * alpha
        
        # Update optimal parameters based on winning trades
        if is_win:
            confidence = trade_result.get('confidence', 0.75)
            position = trade_result.get('position_pct', 0.02)
            
            rk.optimal_confidence = rk.optimal_confidence * 0.9 + confidence * 0.1
            rk.optimal_position_size = rk.optimal_position_size * 0.9 + position * 0.1
        
        self._save()
    
    def get_regime_parameters(self, regime: str) -> Dict[str, float]:
        """ดึง optimal parameters สำหรับ regime"""
        
        if regime not in self.regime_knowledge:
            return {
                'optimal_confidence': 0.60,  # Lowered from 0.75
                'optimal_position_size': 0.02,
                'optimal_stop_atr': 1.5,
                'optimal_target_atr': 3.0,
            }
        
        rk = self.regime_knowledge[regime]
        return {
            'optimal_confidence': rk.optimal_confidence,
            'optimal_position_size': rk.optimal_position_size,
            'optimal_stop_atr': rk.optimal_stop_atr,
            'optimal_target_atr': rk.optimal_target_atr,
            'historical_win_rate': rk.win_rate,
            'historical_avg_pnl': rk.avg_pnl,
        }
    
    def archive_strategy(
        self,
        strategy_id: str,
        params: Dict,
        performance: Dict,
    ):
        """เก็บกลยุทธ์ที่ดีไว้ใน archive"""
        
        self.strategy_archive.append({
            'strategy_id': strategy_id,
            'archived_at': datetime.now().isoformat(),
            'params': params,
            'performance': performance,
        })
        
        # Keep only top 100
        self.strategy_archive = sorted(
            self.strategy_archive,
            key=lambda x: x['performance'].get('sharpe', 0),
            reverse=True
        )[:100]
        
        self._save()
    
    def get_best_strategies(self, n: int = 5) -> List[Dict]:
        """ดึงกลยุทธ์ที่ดีที่สุด"""
        return self.strategy_archive[:n]
    
    def distill_knowledge(self) -> Dict[str, Any]:
        """สรุปความรู้ทั้งหมด"""
        
        # Top patterns
        top_patterns = sorted(
            [p for p in self.patterns.values() if p.times_seen >= 5],
            key=lambda x: x.success_rate * x.avg_return,
            reverse=True
        )[:10]
        
        # Top rules
        top_rules = sorted(
            [r for r in self.rules.values() if r.times_applied >= 5],
            key=lambda x: x.accuracy,
            reverse=True
        )[:10]
        
        # Best regime
        best_regime = max(
            self.regime_knowledge.values(),
            key=lambda x: x.win_rate * (1 + x.avg_pnl / 100),
            default=None
        )
        
        return {
            'top_patterns': [p.to_dict() for p in top_patterns],
            'top_rules': [asdict(r) for r in top_rules],
            'best_regime': best_regime.regime if best_regime else 'unknown',
            'regime_stats': {
                r: {'win_rate': rk.win_rate, 'trades': rk.trades}
                for r, rk in self.regime_knowledge.items()
            },
            'strategies_archived': len(self.strategy_archive),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        return {
            'total_patterns': len(self.patterns),
            'total_rules': len(self.rules),
            'strategies_archived': len(self.strategy_archive),
            'regimes_tracked': len(self.regime_knowledge),
        }
    
    def _save(self):
        """บันทึก knowledge"""
        
        state = {
            'patterns': {k: v.to_dict() for k, v in self.patterns.items()},
            'rules': {k: asdict(v) for k, v in self.rules.items()},
            'regime_knowledge': {k: asdict(v) for k, v in self.regime_knowledge.items()},
            'strategy_archive': self.strategy_archive,
        }
        
        path = os.path.join(self.data_dir, 'knowledge_base.json')
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลด knowledge"""
        
        path = os.path.join(self.data_dir, 'knowledge_base.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                # Load patterns
                for k, v in state.get('patterns', {}).items():
                    v['discovered_at'] = datetime.fromisoformat(v['discovered_at'])
                    v['last_seen'] = datetime.fromisoformat(v['last_seen'])
                    self.patterns[k] = MarketPattern(**v)
                
                # Load rules
                for k, v in state.get('rules', {}).items():
                    self.rules[k] = TradeRule(**v)
                
                # Load regime knowledge
                for k, v in state.get('regime_knowledge', {}).items():
                    self.regime_knowledge[k] = RegimeKnowledge(**v)
                
                self.strategy_archive = state.get('strategy_archive', [])
                
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")


def create_knowledge_base() -> KnowledgeBase:
    """สร้าง KnowledgeBase"""
    return KnowledgeBase()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   KNOWLEDGE BASE TEST")
    print("="*60)
    
    kb = create_knowledge_base()
    
    # Add patterns
    kb.add_pattern(
        name="RSI Divergence Bullish",
        description="Price makes lower low but RSI makes higher low",
        conditions={'rsi': 35, 'trend': -0.01, 'divergence': True},
        was_profitable=True,
        return_pct=0.02,
    )
    
    kb.add_pattern(
        name="Breakout Above Resistance",
        description="Price breaks above key resistance level",
        conditions={'breakout': True, 'volume_surge': True},
        was_profitable=True,
        return_pct=0.03,
    )
    
    # Add rules
    kb.add_rule(
        rule_type='entry',
        condition='RSI < 30 and trend_aligned',
        action='LONG with 70% confidence',
    )
    
    kb.add_rule(
        rule_type='filter',
        condition='volatility > 0.03',
        action='reduce position by 50%',
    )
    
    # Update regime knowledge
    for _ in range(10):
        kb.update_regime_knowledge('trending_up', {
            'pnl': np.random.uniform(-100, 200),
            'confidence': 0.8,
            'position_pct': 0.02,
        })
    
    # Stats
    print("\nStats:")
    stats = kb.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Distill
    print("\nDistilled Knowledge:")
    knowledge = kb.distill_knowledge()
    print(f"  Top patterns: {len(knowledge['top_patterns'])}")
    print(f"  Top rules: {len(knowledge['top_rules'])}")
    print(f"  Best regime: {knowledge['best_regime']}")
