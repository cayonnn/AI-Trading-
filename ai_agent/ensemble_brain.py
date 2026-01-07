"""
Ensemble Brain Module
======================
รวม predictions จากหลาย models เพื่อความแม่นยำสูงสุด

Features:
1. Multi-Model Ensemble - รวมหลาย models
2. Weighted Voting - ให้น้ำหนักตาม performance
3. Confidence Calibration - ปรับ confidence ให้แม่นยำ
4. Dynamic Model Selection - เลือก model ที่ดีที่สุดตาม regime
5. Consensus Detection - ตรวจจับว่า models เห็นตรงกันหรือไม่
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from loguru import logger


@dataclass
class ModelPrediction:
    """การทำนายจาก model"""
    model_name: str
    action: int  # 0=wait, 1=long, 2=short
    confidence: float
    features_used: List[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """ผลลัพธ์จาก Ensemble"""
    final_action: int
    final_confidence: float
    consensus_level: float  # 0-1 ว่า models เห็นตรงกันแค่ไหน
    individual_predictions: List[ModelPrediction] = field(default_factory=list)
    reasoning: str = ""


class TrendModel(nn.Module):
    """Model สำหรับวิเคราะห์ Trend"""
    
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # wait, long, short
        )
    
    def forward(self, x):
        return self.net(x)


class MomentumModel(nn.Module):
    """Model สำหรับวิเคราะห์ Momentum"""
    
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 3),
        )
    
    def forward(self, x):
        return self.net(x)


class ReversalModel(nn.Module):
    """Model สำหรับตรวจจับ Reversal"""
    
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )
    
    def forward(self, x):
        return self.net(x)


class EnsembleBrain:
    """
    Advanced Ensemble Prediction System
    
    ความสามารถ:
    1. รวม predictions จากหลาย models
    2. ปรับน้ำหนักตาม performance
    3. ตรวจจับ consensus
    4. เลือก model ที่ดีที่สุดตาม regime
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        min_consensus: float = 0.6,
    ):
        self.input_dim = input_dim
        self.min_consensus = min_consensus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.models = {
            "trend": TrendModel(input_dim).to(self.device),
            "momentum": MomentumModel(input_dim).to(self.device),
            "reversal": ReversalModel(input_dim).to(self.device),
        }
        
        # Set to eval mode
        for model in self.models.values():
            model.eval()
        
        # Model weights (dynamic)
        self.model_weights = {
            "trend": 0.4,
            "momentum": 0.35,
            "reversal": 0.25,
        }
        
        # Performance tracking
        self.model_performance: Dict[str, deque] = {
            name: deque(maxlen=100) for name in self.models
        }
        
        # Regime-specific weights
        self.regime_weights = {
            "trending_up": {"trend": 0.5, "momentum": 0.35, "reversal": 0.15},
            "trending_down": {"trend": 0.5, "momentum": 0.35, "reversal": 0.15},
            "ranging": {"trend": 0.2, "momentum": 0.3, "reversal": 0.5},
            "volatile": {"trend": 0.3, "momentum": 0.4, "reversal": 0.3},
        }
        
        logger.info(f"EnsembleBrain initialized with {len(self.models)} models")
    
    def predict(
        self,
        features: np.ndarray,
        regime: str = "unknown",
    ) -> EnsembleResult:
        """
        ทำนายด้วย Ensemble
        
        Args:
            features: Feature vector
            regime: Current market regime
            
        Returns:
            EnsembleResult
        """
        
        # Prepare input
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]
        
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get predictions from all models
        predictions = []
        
        for name, model in self.models.items():
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                action = probs.argmax().item()
                confidence = probs.max().item()
            
            predictions.append(ModelPrediction(
                model_name=name,
                action=action,
                confidence=confidence,
            ))
        
        # Get weights for current regime
        weights = self.regime_weights.get(regime, self.model_weights)
        
        # Weighted voting
        vote_scores = [0.0, 0.0, 0.0]  # wait, long, short
        total_weight = 0
        
        for pred in predictions:
            w = weights.get(pred.model_name, 0.33)
            vote_scores[pred.action] += w * pred.confidence
            total_weight += w
        
        # Normalize
        vote_scores = [s / total_weight for s in vote_scores]
        
        # Final decision
        final_action = np.argmax(vote_scores)
        final_confidence = vote_scores[final_action]
        
        # Calculate consensus
        agreeing_models = sum(1 for p in predictions if p.action == final_action)
        consensus_level = agreeing_models / len(predictions)
        
        # Lower confidence if low consensus
        if consensus_level < self.min_consensus:
            final_confidence *= 0.7
        
        # Generate reasoning
        action_names = ["WAIT", "LONG", "SHORT"]
        model_votes = [f"{p.model_name}:{action_names[p.action]}" for p in predictions]
        reasoning = f"Ensemble: {', '.join(model_votes)} → {action_names[final_action]}"
        
        return EnsembleResult(
            final_action=final_action,
            final_confidence=final_confidence,
            consensus_level=consensus_level,
            individual_predictions=predictions,
            reasoning=reasoning,
        )
    
    def update_performance(
        self,
        model_name: str,
        was_correct: bool,
    ):
        """อัพเดต performance ของ model"""
        
        if model_name in self.model_performance:
            self.model_performance[model_name].append(1 if was_correct else 0)
            
            # Update weights based on performance
            self._update_weights()
    
    def _update_weights(self):
        """ปรับน้ำหนักตาม performance"""
        
        performances = {}
        
        for name, history in self.model_performance.items():
            if len(history) >= 10:
                performances[name] = sum(history) / len(history)
            else:
                performances[name] = 0.5
        
        total = sum(performances.values())
        
        if total > 0:
            for name in self.model_weights:
                self.model_weights[name] = performances.get(name, 0.5) / total
    
    def get_best_model(self, regime: str = "unknown") -> str:
        """หา model ที่ดีที่สุดสำหรับ regime"""
        
        weights = self.regime_weights.get(regime, self.model_weights)
        return max(weights, key=weights.get)
    
    def get_consensus_signal(
        self,
        features: np.ndarray,
        regime: str = "unknown",
    ) -> Tuple[int, float, bool]:
        """
        ดึงสัญญาณเฉพาะเมื่อ models เห็นตรงกัน
        
        Returns:
            (action, confidence, has_consensus)
        """
        
        result = self.predict(features, regime)
        
        has_consensus = result.consensus_level >= self.min_consensus
        
        return result.final_action, result.final_confidence, has_consensus
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        
        return {
            "models": list(self.models.keys()),
            "weights": self.model_weights.copy(),
            "performance": {
                name: sum(h) / len(h) if h else 0
                for name, h in self.model_performance.items()
            },
        }


def create_ensemble_brain() -> EnsembleBrain:
    """สร้าง EnsembleBrain"""
    return EnsembleBrain()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ENSEMBLE BRAIN TEST")
    print("="*60)
    
    np.random.seed(42)
    
    eb = create_ensemble_brain()
    
    # Test prediction
    features = np.random.randn(20)
    result = eb.predict(features, regime="trending_up")
    
    print(f"\nEnsemble Result:")
    print(f"  Action: {['WAIT', 'LONG', 'SHORT'][result.final_action]}")
    print(f"  Confidence: {result.final_confidence:.1%}")
    print(f"  Consensus: {result.consensus_level:.1%}")
    print(f"  Reasoning: {result.reasoning}")
    
    print(f"\nIndividual Predictions:")
    for pred in result.individual_predictions:
        print(f"  {pred.model_name}: {['WAIT', 'LONG', 'SHORT'][pred.action]} ({pred.confidence:.1%})")
    
    # Consensus signal
    action, conf, has_consensus = eb.get_consensus_signal(features, "trending_up")
    print(f"\nConsensus: {'YES' if has_consensus else 'NO'} ({['WAIT', 'LONG', 'SHORT'][action]})")
