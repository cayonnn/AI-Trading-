"""
Model Monitor for AI Trading System
====================================
v1.0 - Model Drift Detection & Accuracy Tracking

Features:
- Track rolling model accuracy
- Detect performance degradation
- Alert when retraining needed
- Accuracy comparison across models
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger
import json


class ModelMonitor:
    """
    Model Performance Monitor
    
    Tracks accuracy over time and detects drift
    """
    
    def __init__(
        self,
        window_size: int = 50,  # Rolling window for accuracy
        drift_threshold: float = 0.20,  # 20% accuracy drop triggers alert
        min_trades_for_alert: int = 20,  # Minimum trades before alerting
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_trades = min_trades_for_alert
        
        # Model tracking
        self.models = {
            'lstm': ModelStats('lstm', window_size),
            'xgboost': ModelStats('xgboost', window_size),
            'ppo': ModelStats('ppo', window_size),
            'ensemble': ModelStats('ensemble', window_size),
        }
        
        # Baseline accuracy (from training)
        self.baseline_accuracy = {
            'lstm': 0.55,
            'xgboost': 0.58,
            'ppo': 0.52,
            'ensemble': 0.60,
        }
        
        # Alert state
        self.drift_alerts: Dict[str, bool] = {m: False for m in self.models}
        
        logger.info("ModelMonitor initialized")
    
    def record_prediction(
        self,
        model_name: str,
        prediction: str,
        actual: str,
        confidence: float = 0.0,
    ):
        """Record a model prediction and actual outcome"""
        if model_name not in self.models:
            return
        
        is_correct = (prediction == actual)
        self.models[model_name].add_prediction(is_correct, confidence)
        
        # Check for drift
        self._check_drift(model_name)
    
    def _check_drift(self, model_name: str) -> Optional[str]:
        """Check if model accuracy has drifted"""
        stats = self.models[model_name]
        
        if stats.total_predictions < self.min_trades:
            return None
        
        current_acc = stats.get_accuracy()
        baseline_acc = self.baseline_accuracy.get(model_name, 0.5)
        
        # Calculate drift
        acc_drop = baseline_acc - current_acc
        
        if acc_drop >= self.drift_threshold and not self.drift_alerts[model_name]:
            self.drift_alerts[model_name] = True
            msg = f"⚠️ MODEL DRIFT: {model_name} accuracy dropped {acc_drop:.0%} (now {current_acc:.0%})"
            logger.warning(msg)
            return msg
        
        # Reset alert if accuracy recovers
        if current_acc >= baseline_acc * 0.9:
            self.drift_alerts[model_name] = False
        
        return None
    
    def get_model_status(self, model_name: str) -> Dict:
        """Get status for a specific model"""
        if model_name not in self.models:
            return {}
        
        stats = self.models[model_name]
        baseline = self.baseline_accuracy.get(model_name, 0.5)
        current = stats.get_accuracy()
        
        return {
            'name': model_name,
            'accuracy': current,
            'baseline': baseline,
            'drift': baseline - current,
            'predictions': stats.total_predictions,
            'avg_confidence': stats.get_avg_confidence(),
            'needs_retrain': self.drift_alerts.get(model_name, False),
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all models"""
        return {m: self.get_model_status(m) for m in self.models}
    
    def get_best_model(self) -> Tuple[str, float]:
        """Get the current best performing model"""
        best_model = 'ensemble'
        best_acc = 0.0
        
        for name, stats in self.models.items():
            if stats.total_predictions >= self.min_trades:
                acc = stats.get_accuracy()
                if acc > best_acc:
                    best_acc = acc
                    best_model = name
        
        return best_model, best_acc
    
    def needs_retraining(self) -> List[str]:
        """Get list of models that need retraining"""
        return [m for m, drifted in self.drift_alerts.items() if drifted]
    
    def set_baseline(self, model_name: str, accuracy: float):
        """Set baseline accuracy for a model (after training)"""
        self.baseline_accuracy[model_name] = accuracy
        logger.info(f"📊 {model_name} baseline set to {accuracy:.1%}")
    
    def reset(self):
        """Reset all tracking"""
        for stats in self.models.values():
            stats.reset()
        self.drift_alerts = {m: False for m in self.models}
        logger.info("ModelMonitor reset")


class ModelStats:
    """Statistics tracker for a single model"""
    
    def __init__(self, name: str, window_size: int = 50):
        self.name = name
        self.window_size = window_size
        
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)
        
        self.total_predictions = 0
        self.total_correct = 0
    
    def add_prediction(self, is_correct: bool, confidence: float):
        """Add a prediction result"""
        self.predictions.append(1 if is_correct else 0)
        self.confidences.append(confidence)
        
        self.total_predictions += 1
        if is_correct:
            self.total_correct += 1
    
    def get_accuracy(self) -> float:
        """Get rolling window accuracy"""
        if not self.predictions:
            return 0.5  # Default
        return sum(self.predictions) / len(self.predictions)
    
    def get_lifetime_accuracy(self) -> float:
        """Get all-time accuracy"""
        if self.total_predictions == 0:
            return 0.5
        return self.total_correct / self.total_predictions
    
    def get_avg_confidence(self) -> float:
        """Get average confidence"""
        if not self.confidences:
            return 0.5
        return sum(self.confidences) / len(self.confidences)
    
    def reset(self):
        """Reset stats"""
        self.predictions.clear()
        self.confidences.clear()
        self.total_predictions = 0
        self.total_correct = 0


# ============================================
# Singleton
# ============================================

_monitor: Optional[ModelMonitor] = None

def get_model_monitor() -> ModelMonitor:
    """Get singleton ModelMonitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor


if __name__ == "__main__":
    # Test
    monitor = ModelMonitor()
    
    # Simulate predictions
    import random
    
    for i in range(100):
        is_correct = random.random() < 0.58  # ~58% accuracy
        prediction = "LONG" if random.random() > 0.5 else "SHORT"
        actual = prediction if is_correct else ("SHORT" if prediction == "LONG" else "LONG")
        
        monitor.record_prediction("xgboost", prediction, actual, random.uniform(0.5, 0.8))
    
    # Check status
    status = monitor.get_all_status()
    for model, data in status.items():
        print(f"{model}: Acc={data['accuracy']:.1%}, Drift={data['drift']:+.1%}")
    
    best, acc = monitor.get_best_model()
    print(f"\nBest model: {best} ({acc:.1%})")
