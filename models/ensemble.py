"""
Ensemble Model Module
=====================
Production-grade ensemble combining LSTM, CNN, and XGBoost.

Features:
- Multiple ensemble strategies (voting, weighted, stacking)
- Confidence scoring
- Dynamic weight adjustment
- Model disagreement detection
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    AVERAGE = "average"


@dataclass
class ModelPrediction:
    """Container for individual model predictions."""
    name: str
    prediction: np.ndarray
    probability: Optional[np.ndarray] = None
    confidence: Optional[float] = None


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    prediction: np.ndarray
    probability: np.ndarray
    confidence: np.ndarray
    model_agreement: np.ndarray
    individual_predictions: Dict[str, np.ndarray]


class EnsembleModel:
    """
    Ensemble model combining predictions from multiple models.
    
    Supports:
    - LSTM (deep learning)
    - XGBoost (gradient boosting)
    - CNN (pattern recognition) - optional
    
    Usage:
        ensemble = EnsembleModel(method='weighted_voting')
        ensemble.add_model('lstm', lstm_predictor)
        ensemble.add_model('xgboost', xgb_model)
        
        result = ensemble.predict(X_lstm=X_seq, X_xgb=X_flat)
    """
    
    DEFAULT_WEIGHTS = {
        'lstm': 0.4,
        'xgboost': 0.4,
        'cnn': 0.2
    }
    
    def __init__(
        self,
        method: str = 'weighted_voting',
        weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.6,
        require_agreement: bool = False
    ):
        """
        Initialize ensemble model.
        
        Args:
            method: Ensemble method ('voting', 'weighted_voting', 'stacking')
            weights: Model weights (must sum to 1.0)
            min_confidence: Minimum confidence threshold for signals
            require_agreement: Require model agreement for signal
        """
        self.method = EnsembleMethod(method)
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.min_confidence = min_confidence
        self.require_agreement = require_agreement
        
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(
            f"EnsembleModel initialized: method={method}, "
            f"weights={self.weights}, min_confidence={min_confidence}"
        )
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def add_model(
        self,
        name: str,
        model: Any,
        weight: Optional[float] = None
    ) -> 'EnsembleModel':
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model instance (must have predict and predict_proba methods)
            weight: Optional weight override
            
        Returns:
            self
        """
        self.models[name] = model
        
        if weight is not None:
            self.weights[name] = weight
            self._normalize_weights()
        
        logger.info(f"Added model: {name} (weight={self.weights.get(name, 0):.3f})")
        
        return self
    
    def remove_model(self, name: str) -> 'EnsembleModel':
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            if name in self.weights:
                del self.weights[name]
            self._normalize_weights()
            logger.info(f"Removed model: {name}")
        
        return self
    
    def update_weights(
        self,
        weights: Dict[str, float]
    ) -> 'EnsembleModel':
        """Update model weights."""
        self.weights.update(weights)
        self._normalize_weights()
        logger.info(f"Updated weights: {self.weights}")
        return self
    
    def _get_model_prediction(
        self,
        name: str,
        X: np.ndarray
    ) -> ModelPrediction:
        """Get prediction from a single model."""
        model = self.models[name]
        
        # Get predictions
        try:
            prediction = model.predict(X)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(X)
                if len(probability.shape) > 1:
                    probability = probability[:, 1] if probability.shape[1] == 2 else probability
            else:
                probability = prediction.astype(float)
            
            # Calculate confidence
            confidence = np.abs(probability - 0.5).mean() * 2  # Scale to 0-1
            
            return ModelPrediction(
                name=name,
                prediction=prediction,
                probability=probability,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error getting prediction from {name}: {e}")
            raise
    
    def predict(
        self,
        X_lstm: Optional[np.ndarray] = None,
        X_xgb: Optional[np.ndarray] = None,
        X_cnn: Optional[np.ndarray] = None,
        return_details: bool = False
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction.
        
        Args:
            X_lstm: Input for LSTM (sequences)
            X_xgb: Input for XGBoost (flat features)
            X_cnn: Input for CNN (optional)
            return_details: Include individual model predictions
            
        Returns:
            EnsemblePrediction with results
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Collect predictions from each model
        predictions: Dict[str, ModelPrediction] = {}
        
        if 'lstm' in self.models and X_lstm is not None:
            predictions['lstm'] = self._get_model_prediction('lstm', X_lstm)
        
        if 'xgboost' in self.models and X_xgb is not None:
            predictions['xgboost'] = self._get_model_prediction('xgboost', X_xgb)
        
        if 'cnn' in self.models and X_cnn is not None:
            predictions['cnn'] = self._get_model_prediction('cnn', X_cnn)
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Combine predictions
        result = self._combine_predictions(predictions)
        
        return result
    
    def _combine_predictions(
        self,
        predictions: Dict[str, ModelPrediction]
    ) -> EnsemblePrediction:
        """Combine predictions based on ensemble method."""
        
        # Get sample size from first prediction
        n_samples = len(next(iter(predictions.values())).prediction)
        
        if self.method == EnsembleMethod.VOTING:
            return self._voting(predictions, n_samples)
        elif self.method == EnsembleMethod.WEIGHTED_VOTING:
            return self._weighted_voting(predictions, n_samples)
        elif self.method == EnsembleMethod.AVERAGE:
            return self._average(predictions, n_samples)
        else:
            return self._weighted_voting(predictions, n_samples)
    
    def _voting(
        self,
        predictions: Dict[str, ModelPrediction],
        n_samples: int
    ) -> EnsemblePrediction:
        """Simple majority voting."""
        vote_sum = np.zeros(n_samples)
        individual = {}
        
        for name, pred in predictions.items():
            vote_sum += pred.prediction.flatten()
            individual[name] = pred.prediction.flatten()
        
        n_models = len(predictions)
        final_prediction = (vote_sum >= n_models / 2).astype(int)
        probability = vote_sum / n_models
        
        # Calculate agreement (how many models agree)
        agreement = np.maximum(vote_sum, n_models - vote_sum) / n_models
        
        # Confidence is based on agreement
        confidence = agreement
        
        return EnsemblePrediction(
            prediction=final_prediction,
            probability=probability,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=individual
        )
    
    def _weighted_voting(
        self,
        predictions: Dict[str, ModelPrediction],
        n_samples: int
    ) -> EnsemblePrediction:
        """Weighted voting based on model weights."""
        weighted_prob = np.zeros(n_samples)
        total_weight = 0
        individual = {}
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0 / len(predictions))
            
            if pred.probability is not None:
                prob = pred.probability.flatten()
            else:
                prob = pred.prediction.astype(float).flatten()
            
            weighted_prob += weight * prob
            total_weight += weight
            individual[name] = pred.prediction.flatten()
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_prob /= total_weight
        
        final_prediction = (weighted_prob >= 0.5).astype(int)
        
        # Calculate model agreement
        preds_array = np.array([p.flatten() for p in individual.values()])
        agreement = np.mean(preds_array == final_prediction.reshape(1, -1), axis=0)
        
        # Confidence based on probability distance from 0.5 and agreement
        prob_confidence = np.abs(weighted_prob - 0.5) * 2
        confidence = (prob_confidence + agreement) / 2
        
        return EnsemblePrediction(
            prediction=final_prediction,
            probability=weighted_prob,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=individual
        )
    
    def _average(
        self,
        predictions: Dict[str, ModelPrediction],
        n_samples: int
    ) -> EnsemblePrediction:
        """Simple average of probabilities."""
        prob_sum = np.zeros(n_samples)
        individual = {}
        
        for name, pred in predictions.items():
            prob = pred.probability if pred.probability is not None else pred.prediction.astype(float)
            prob_sum += prob.flatten()
            individual[name] = pred.prediction.flatten()
        
        probability = prob_sum / len(predictions)
        final_prediction = (probability >= 0.5).astype(int)
        
        # Agreement
        preds_array = np.array([p.flatten() for p in individual.values()])
        agreement = np.mean(preds_array == final_prediction.reshape(1, -1), axis=0)
        
        confidence = np.abs(probability - 0.5) * 2
        
        return EnsemblePrediction(
            prediction=final_prediction,
            probability=probability,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=individual
        )
    
    def get_trading_signals(
        self,
        ensemble_result: EnsemblePrediction,
        prices: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Convert ensemble predictions to trading signals.
        
        Args:
            ensemble_result: Ensemble prediction results
            prices: Current prices (optional)
            
        Returns:
            DataFrame with signals
        """
        n_samples = len(ensemble_result.prediction)
        
        signals = pd.DataFrame({
            'raw_prediction': ensemble_result.prediction,
            'probability': ensemble_result.probability,
            'confidence': ensemble_result.confidence,
            'model_agreement': ensemble_result.model_agreement
        })
        
        # Apply confidence filter
        signals['signal'] = 0  # No trade
        
        # Long signal: prediction=1 and confidence >= threshold
        long_mask = (
            (signals['raw_prediction'] == 1) &
            (signals['confidence'] >= self.min_confidence)
        )
        signals.loc[long_mask, 'signal'] = 1
        
        # Short signal: prediction=0 and confidence >= threshold
        short_mask = (
            (signals['raw_prediction'] == 0) &
            (signals['confidence'] >= self.min_confidence)
        )
        signals.loc[short_mask, 'signal'] = -1
        
        # Apply agreement filter if required
        if self.require_agreement:
            disagreement_mask = signals['model_agreement'] < 0.67  # At least 2/3 agree
            signals.loc[disagreement_mask, 'signal'] = 0
        
        # Add price if provided
        if prices is not None:
            signals['price'] = prices
        
        # Add individual model predictions
        for name, preds in ensemble_result.individual_predictions.items():
            signals[f'{name}_pred'] = preds
        
        # Signal strength (for position sizing)
        signals['signal_strength'] = signals['confidence'] * signals['model_agreement']
        
        return signals
    
    def update_performance(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Update model performance metrics.
        
        Args:
            name: Model name
            y_true: True labels
            y_pred: Predicted labels
        """
        accuracy = (y_true == y_pred).mean()
        
        self.model_performance[name] = {
            'accuracy': accuracy,
            'n_samples': len(y_true)
        }
        
        logger.info(f"Updated {name} performance: accuracy={accuracy:.4f}")
    
    def adapt_weights(
        self,
        lookback: int = 100
    ) -> None:
        """
        Adapt weights based on recent performance.
        
        Args:
            lookback: Number of recent predictions to consider
        """
        if not self.model_performance:
            logger.warning("No performance data to adapt weights")
            return
        
        # Calculate new weights based on accuracy
        total_accuracy = sum(p['accuracy'] for p in self.model_performance.values())
        
        if total_accuracy > 0:
            new_weights = {
                name: perf['accuracy'] / total_accuracy
                for name, perf in self.model_performance.items()
            }
            self.update_weights(new_weights)
            logger.info(f"Adapted weights based on performance: {self.weights}")
    
    def save(self, filepath: str) -> None:
        """Save ensemble configuration."""
        config = {
            'method': self.method.value,
            'weights': self.weights,
            'min_confidence': self.min_confidence,
            'require_agreement': self.require_agreement,
            'model_performance': self.model_performance
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Ensemble config saved to {filepath}")
    
    def load(self, filepath: str) -> 'EnsembleModel':
        """Load ensemble configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.method = EnsembleMethod(config['method'])
        self.weights = config['weights']
        self.min_confidence = config['min_confidence']
        self.require_agreement = config['require_agreement']
        self.model_performance = config.get('model_performance', {})
        
        logger.info(f"Ensemble config loaded from {filepath}")
        return self


def create_ensemble(
    lstm_model=None,
    xgboost_model=None,
    cnn_model=None,
    weights: Optional[Dict[str, float]] = None
) -> EnsembleModel:
    """
    Convenience function to create an ensemble model.
    
    Args:
        lstm_model: LSTM predictor instance
        xgboost_model: XGBoost model instance
        cnn_model: CNN model instance (optional)
        weights: Custom weights
        
    Returns:
        Configured EnsembleModel
    """
    ensemble = EnsembleModel(method='weighted_voting', weights=weights)
    
    if lstm_model is not None:
        ensemble.add_model('lstm', lstm_model)
    
    if xgboost_model is not None:
        ensemble.add_model('xgboost', xgboost_model)
    
    if cnn_model is not None:
        ensemble.add_model('cnn', cnn_model)
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble model
    print("=== Testing Ensemble Model ===")
    
    # Create mock models
    class MockModel:
        def __init__(self, bias=0.5):
            self.bias = bias
        
        def predict(self, X):
            return (np.random.rand(len(X)) > self.bias).astype(int)
        
        def predict_proba(self, X):
            return np.random.rand(len(X)) * 0.5 + 0.25
    
    # Create ensemble
    ensemble = EnsembleModel(method='weighted_voting')
    ensemble.add_model('lstm', MockModel(0.4))
    ensemble.add_model('xgboost', MockModel(0.5))
    
    # Make predictions
    X_test = np.random.randn(100, 10)
    result = ensemble.predict(X_lstm=X_test, X_xgb=X_test)
    
    print(f"Predictions: {result.prediction[:10]}")
    print(f"Probabilities: {result.probability[:10]}")
    print(f"Confidence: {result.confidence[:10]}")
    print(f"Agreement: {result.model_agreement[:10]}")
    
    # Get trading signals
    signals = ensemble.get_trading_signals(result)
    print(f"\nTrading signals summary:")
    print(signals['signal'].value_counts())
