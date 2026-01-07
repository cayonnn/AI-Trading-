"""
Unified Ensemble Module
========================
รวม LSTM + XGBoost + PPO เข้าด้วยกัน

Features:
1. Multi-Model Voting - รวม predictions จาก 3 models
2. Dynamic Weighting - ปรับน้ำหนักตาม performance
3. Confidence Aggregation - รวม confidence scores
4. Consensus Detection - ตรวจจับว่า models เห็นตรงกัน
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class UnifiedPrediction:
    """ผลการทำนายจาก Unified Ensemble"""
    action: str  # 'LONG', 'SHORT', 'WAIT'
    confidence: float
    direction: int  # 1 = long, -1 = short, 0 = wait
    
    # Individual predictions
    lstm_action: Optional[int] = None
    lstm_confidence: float = 0.0
    xgb_action: Optional[int] = None
    xgb_confidence: float = 0.0
    ppo_action: Optional[int] = None
    ppo_confidence: float = 0.0
    
    # Consensus
    consensus: bool = False
    consensus_count: int = 0
    
    # Reasoning
    reasoning: str = ""


class UnifiedEnsemble:
    """
    รวม LSTM + XGBoost + PPO predictions
    
    ความสามารถ:
    1. Weighted voting จาก 3 models
    2. ปรับน้ำหนักตาม historical performance
    3. ตรวจจับ consensus
    4. Dynamic confidence adjustment
    """
    
    def __init__(
        self,
        lstm_weight: float = 0.35,  # 35% - Increased (good at trends)
        xgb_weight: float = 0.30,   # 30% - Reduced (needs re-training)
        ppo_weight: float = 0.45,   # 45% - RL agent (best for decisions)
        min_consensus: float = 0.66,  # 2/3 models agree
        min_confidence: float = 0.55,
    ):
        self.weights = {
            'lstm': lstm_weight,
            'xgb': xgb_weight,
            'ppo': ppo_weight,
        }
        self.min_consensus = min_consensus
        self.min_confidence = min_confidence
        
        # Performance tracking
        self.performance: Dict[str, deque] = {
            'lstm': deque(maxlen=100),
            'xgb': deque(maxlen=100),
            'ppo': deque(maxlen=100),
        }
        
        # Recent predictions for analysis
        self.recent_predictions: deque = deque(maxlen=50)
        
        logger.info("UnifiedEnsemble initialized")
    
    def predict(
        self,
        lstm_prediction: Tuple[Optional[int], float],
        xgb_prediction: Tuple[Optional[int], float],
        ppo_prediction: Tuple[int, float],
    ) -> UnifiedPrediction:
        """
        รวม predictions จากทุก models
        
        Args:
            lstm_prediction: (action, confidence) from LSTM
            xgb_prediction: (action, confidence) from XGBoost
            ppo_prediction: (action, confidence) from PPO
            
        Returns:
            UnifiedPrediction
        """
        
        # Extract predictions
        lstm_action, lstm_conf = lstm_prediction
        xgb_action, xgb_conf = xgb_prediction
        ppo_action, ppo_conf = ppo_prediction
        
        # Collect valid predictions
        predictions = []
        weights = []
        
        if lstm_action is not None:
            predictions.append(('lstm', lstm_action, lstm_conf))
            weights.append(self.weights['lstm'])
        
        if xgb_action is not None:
            predictions.append(('xgb', xgb_action, xgb_conf))
            weights.append(self.weights['xgb'])
        
        if ppo_action is not None:
            predictions.append(('ppo', ppo_action, ppo_conf))
            weights.append(self.weights['ppo'])
        
        if not predictions:
            return UnifiedPrediction(
                action="WAIT",
                confidence=0.0,
                direction=0,
                reasoning="No valid predictions",
            )
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted voting
        vote_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # 0=wait, 1=long, 2=short
        
        for (name, action, conf), weight in zip(predictions, weights):
            # Handle action values
            if action is None:
                continue
            action_key = min(action, 2)  # Clamp to 0, 1, 2
            vote_scores[action_key] += weight * conf
        
        # Determine final action - winner takes all
        if vote_scores[1] > vote_scores[0]:
            final_action = 1
            final_confidence = vote_scores[1]
        else:
            final_action = 0
            final_confidence = vote_scores[0]
        
        # Count consensus (how many models agree with final action)
        consensus_count = sum(1 for _, action, _ in predictions if action == final_action)
        consensus = consensus_count >= len(predictions) * self.min_consensus
        
        # Professional confidence adjustment
        if consensus:
            # All 3 models agree - strong signal
            final_confidence = min(0.90, final_confidence * 1.20)
        elif consensus_count >= 2:
            # 2 of 3 models agree - still good signal, light boost
            final_confidence = min(0.85, final_confidence * 1.10)
        else:
            # Models disagree - light penalty (not too harsh)
            final_confidence *= 0.92
        
        # Determine action string
        if final_confidence < self.min_confidence:
            action_str = "WAIT"
            direction = 0
        elif final_action == 1:
            action_str = "LONG"
            direction = 1
        else:
            action_str = "WAIT"  # No short for now
            direction = 0
        
        # Generate reasoning
        reasoning_parts = []
        if lstm_action is not None:
            reasoning_parts.append(f"LSTM:{'↑' if lstm_action == 1 else '↓'}({lstm_conf:.0%})")
        if xgb_action is not None:
            reasoning_parts.append(f"XGB:{'↑' if xgb_action == 1 else '↓'}({xgb_conf:.0%})")
        if ppo_action is not None:
            reasoning_parts.append(f"PPO:{'↑' if ppo_action == 1 else '↓'}({ppo_conf:.0%})")
        
        reasoning = " | ".join(reasoning_parts)
        if consensus:
            reasoning += " [CONSENSUS]"
        
        return UnifiedPrediction(
            action=action_str,
            confidence=final_confidence,
            direction=direction,
            lstm_action=lstm_action,
            lstm_confidence=lstm_conf,
            xgb_action=xgb_action,
            xgb_confidence=xgb_conf,
            ppo_action=ppo_action,
            ppo_confidence=ppo_conf,
            consensus=consensus,
            consensus_count=consensus_count,
            reasoning=reasoning,
        )
    
    def update_performance(
        self,
        model_name: str,
        was_correct: bool,
    ):
        """อัพเดต performance ของ model"""
        
        if model_name in self.performance:
            self.performance[model_name].append(1 if was_correct else 0)
            self._update_weights()
    
    def _update_weights(self):
        """ปรับน้ำหนักตาม performance"""
        
        accuracies = {}
        
        for name, history in self.performance.items():
            if len(history) >= 20:
                accuracies[name] = sum(history) / len(history)
            else:
                accuracies[name] = 0.5
        
        # Normalize to weights
        total = sum(accuracies.values())
        if total > 0:
            for name in self.weights:
                if name in accuracies:
                    # Blend: 50% base weight + 50% performance-based
                    base = self.weights[name]
                    perf = accuracies[name] / total
                    self.weights[name] = base * 0.5 + perf * 0.5
    
    def get_best_model(self) -> str:
        """หา model ที่ดีที่สุด"""
        
        best_acc = 0
        best_model = 'xgb'
        
        for name, history in self.performance.items():
            if len(history) >= 10:
                acc = sum(history) / len(history)
                if acc > best_acc:
                    best_acc = acc
                    best_model = name
        
        return best_model
    
    def should_trade(self, prediction: UnifiedPrediction) -> Tuple[bool, str]:
        """ตัดสินใจว่าควรเทรดหรือไม่"""
        
        if prediction.action == "WAIT":
            return False, "No signal"
        
        if prediction.confidence < self.min_confidence:
            return False, f"Low confidence ({prediction.confidence:.0%})"
        
        if not prediction.consensus:
            return False, "No consensus"
        
        return True, f"Consensus {prediction.consensus_count}/3, Conf: {prediction.confidence:.0%}"
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        
        return {
            "weights": self.weights.copy(),
            "performance": {
                name: sum(h) / len(h) if h else 0
                for name, h in self.performance.items()
            },
            "best_model": self.get_best_model(),
        }


def create_unified_ensemble() -> UnifiedEnsemble:
    """สร้าง UnifiedEnsemble"""
    return UnifiedEnsemble()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   UNIFIED ENSEMBLE TEST")
    print("="*60)
    
    ensemble = create_unified_ensemble()
    
    # Test predictions
    test_cases = [
        # All agree on LONG
        ((1, 0.7), (1, 0.75), (1, 0.6)),
        # Mixed signals
        ((1, 0.6), (0, 0.55), (1, 0.65)),
        # All agree on DOWN/WAIT
        ((0, 0.7), (0, 0.8), (0, 0.65)),
        # LSTM and XGB agree, PPO disagrees
        ((1, 0.8), (1, 0.85), (0, 0.5)),
    ]
    
    print("\nTest Results:")
    for i, (lstm, xgb, ppo) in enumerate(test_cases):
        result = ensemble.predict(lstm, xgb, ppo)
        should, reason = ensemble.should_trade(result)
        print(f"\nCase {i+1}:")
        print(f"  Action: {result.action}, Confidence: {result.confidence:.1%}")
        print(f"  Consensus: {result.consensus} ({result.consensus_count}/3)")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Should Trade: {should} - {reason}")
