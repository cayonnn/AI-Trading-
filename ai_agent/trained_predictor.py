"""
Trained Model Predictor
========================
ใช้ models ที่ train แล้ว (LSTM, XGBoost) สำหรับ prediction

Features:
1. Load trained LSTM and XGBoost
2. Ensemble prediction from both models
3. Feature extraction for real-time use
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Import models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMPredictor
from models.xgboost_model import XGBoostModel
from data.indicators import calculate_indicators


@dataclass
class TrainedPrediction:
    """ผลการทำนายจาก trained models"""
    action: str  # 'LONG', 'SHORT', 'WAIT'
    confidence: float
    lstm_prediction: Optional[int] = None
    lstm_confidence: float = 0.0
    xgb_prediction: Optional[int] = None
    xgb_confidence: float = 0.0
    consensus: bool = False


class TrainedModelPredictor:
    """
    Predictor ที่ใช้ trained models จริง
    
    ความสามารถ:
    1. Load LSTM และ XGBoost ที่ train แล้ว
    2. Ensemble prediction
    3. Feature extraction
    """
    
    def __init__(
        self,
        lstm_path: str = "models/checkpoints/lstm_best.pt",
        xgb_path: str = "models/checkpoints/xgboost_best.json",
        features_path: str = "models/checkpoints/selected_features.json",
        sequence_length: int = 120,  # UPGRADED: 60 → 120
    ):
        self.lstm_path = lstm_path
        self.xgb_path = xgb_path
        self.features_path = features_path
        self.sequence_length = sequence_length
        
        # Load models
        self.lstm_model = None
        self.xgb_model = None
        self.selected_features = None  # SHAP selected features
        
        self._load_models()
        self._load_selected_features()
        
        # Weights
        self.lstm_weight = 0.4  # 40%
        self.xgb_weight = 0.6   # 60% (XGBoost performed better)
        
        logger.info("TrainedModelPredictor initialized")
    
    def _load_selected_features(self):
        """โหลด SHAP selected features"""
        import json
        if os.path.exists(self.features_path):
            try:
                with open(self.features_path, 'r') as f:
                    data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    self.selected_features = data
                elif isinstance(data, dict):
                    self.selected_features = data.get('selected_features', [])
                else:
                    self.selected_features = []
                logger.info(f"✓ Loaded {len(self.selected_features)} selected features")
            except Exception as e:
                logger.warning(f"Could not load selected features: {e}")
    
    def _load_models(self):
        """โหลด trained models"""
        
        # Load LSTM
        if os.path.exists(self.lstm_path):
            try:
                import torch
                checkpoint = torch.load(self.lstm_path, map_location='cpu', weights_only=False)
                config = checkpoint.get('config')
                
                # Create predictor with detected input size
                if config:
                    self.lstm_model = LSTMPredictor(
                        input_size=config.input_size,
                        hidden_size=config.hidden_size,
                        num_layers=config.num_layers,
                        task=config.task,
                    )
                    self.lstm_model.load(self.lstm_path)
                    logger.info(f"✓ LSTM loaded from {self.lstm_path}")
            except Exception as e:
                logger.warning(f"Could not load LSTM: {e}")
        else:
            logger.warning(f"LSTM not found at {self.lstm_path}")
        
        # Load XGBoost
        xgb_json_path = self.xgb_path.replace('.json', '') + '.json'
        xgb_meta_path = self.xgb_path.replace('.json', '') + '.meta.json'
        
        if os.path.exists(xgb_json_path):
            try:
                self.xgb_model = XGBoostModel(task='classification')
                self.xgb_model.load(xgb_json_path)
                logger.info(f"✓ XGBoost loaded from {xgb_json_path}")
            except Exception as e:
                logger.warning(f"Could not load XGBoost: {e}")
        else:
            logger.warning(f"XGBoost not found at {xgb_json_path}")
    
    def predict(self, data: pd.DataFrame) -> TrainedPrediction:
        """
        ทำนายด้วย trained models
        
        Args:
            data: OHLCV data with at least 100 rows
            
        Returns:
            TrainedPrediction
        """
        
        if len(data) < 100:
            return TrainedPrediction(
                action="WAIT",
                confidence=0.0,
            )
        
        # 1. Calculate features (including advanced features)
        try:
            df = calculate_indicators(data.copy(), 'all')
            
            # Add cross-asset features (DXY/VIX proxy)
            try:
                from data.cross_asset_features import add_cross_asset_features
                df = add_cross_asset_features(df)
            except Exception as e:
                logger.debug(f"Cross-asset features skipped: {e}")
            
            # Add session features (hour_sin, hour_cos, etc.)
            try:
                from data.session_features import create_all_session_features
                df = create_all_session_features(df)
            except Exception as e:
                logger.debug(f"Session features skipped: {e}")
            
            # Fill NaN instead of drop to preserve rows
            df = df.fillna(0)
            
            if len(df) < self.sequence_length:
                return TrainedPrediction(action="WAIT", confidence=0.0)
        except Exception as e:
            logger.warning(f"Feature calculation failed: {e}")
            return TrainedPrediction(action="WAIT", confidence=0.0)
        
        # 2. Get predictions from each model
        lstm_pred, lstm_conf = self._predict_lstm(df)
        xgb_pred, xgb_conf = self._predict_xgboost(df)
        
        # 3. Ensemble
        final_action, final_conf, consensus = self._ensemble(
            lstm_pred, lstm_conf,
            xgb_pred, xgb_conf
        )
        
        return TrainedPrediction(
            action=final_action,
            confidence=final_conf,
            lstm_prediction=lstm_pred,
            lstm_confidence=lstm_conf,
            xgb_prediction=xgb_pred,
            xgb_confidence=xgb_conf,
            consensus=consensus,
        )
    
    def _predict_lstm(self, df: pd.DataFrame) -> Tuple[Optional[int], float]:
        """ทำนายด้วย LSTM"""
        
        if self.lstm_model is None:
            return None, 0.0
        
        try:
            # Get feature columns (exclude non-feature columns)
            exclude_cols = ['timestamp', 'datetime', 'time', 'date', 'tick_volume', 'volume', 'spread']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            # Get expected feature count from model config
            expected_features = self.lstm_model.config.input_size  # Should be 66
            
            # Trim to expected feature count (take first N features)
            if len(feature_cols) > expected_features:
                feature_cols = feature_cols[:expected_features]
            
            # Get last sequence
            features = df[feature_cols].values[-self.sequence_length:]
            
            # Pad if not enough features
            if features.shape[1] < expected_features:
                padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            
            # Reshape for LSTM: (1, seq_len, features)
            X = features.reshape(1, self.sequence_length, expected_features)
            X = torch.FloatTensor(X)
            
            # Predict
            with torch.no_grad():
                pred = self.lstm_model.predict(X)
                pred_class = int(pred[0])
                
                # Get probabilities if available
                if hasattr(self.lstm_model, 'predict_proba'):
                    probs = self.lstm_model.predict_proba(X)
                    confidence = float(probs.max())
                else:
                    confidence = 0.6  # Default
            
            return pred_class, confidence
            
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return None, 0.0
    
    def _predict_xgboost(self, df: pd.DataFrame) -> Tuple[Optional[int], float]:
        """ทำนายด้วย XGBoost"""
        
        if self.xgb_model is None:
            return None, 0.0
        
        try:
            # Use SHAP selected features if available
            expected_features = 50
            
            if self.selected_features:
                # Filter to only selected features that exist in df
                available_features = [f for f in self.selected_features if f in df.columns]
                if len(available_features) >= 10:
                    X = df[available_features].values[-1:].astype(np.float32)
                    # Pad if needed to match expected features
                    if X.shape[1] < expected_features:
                        padding = np.zeros((1, expected_features - X.shape[1]))
                        X = np.concatenate([X, padding], axis=1)
                else:
                    # Fallback to default
                    exclude_cols = ['timestamp', 'datetime', 'time', 'date', 'tick_volume', 'volume', 'spread']
                    feature_cols = [c for c in df.columns if c not in exclude_cols][:expected_features]
                    X = df[feature_cols].values[-1:].astype(np.float32)
            else:
                # Fallback to default features
                exclude_cols = ['timestamp', 'datetime', 'time', 'date', 'tick_volume', 'volume', 'spread']
                feature_cols = [c for c in df.columns if c not in exclude_cols][:expected_features]
                X = df[feature_cols].values[-1:].astype(np.float32)
            
            # Handle NaN
            X = np.nan_to_num(X, nan=0.0)
            
            # Predict
            pred = self.xgb_model.predict(X)
            pred_class = int(pred[0])
            
            # Get probabilities
            if hasattr(self.xgb_model, 'predict_proba'):
                probs = self.xgb_model.predict_proba(X)
                confidence = float(probs.max())
            else:
                confidence = 0.64  # Based on training accuracy
            
            return pred_class, confidence
            
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return None, 0.0
    
    def _ensemble(
        self,
        lstm_pred: Optional[int],
        lstm_conf: float,
        xgb_pred: Optional[int],
        xgb_conf: float,
    ) -> Tuple[str, float, bool]:
        """รวม predictions จาก models"""
        
        predictions = []
        weights = []
        
        if lstm_pred is not None:
            predictions.append(lstm_pred)
            weights.append(self.lstm_weight * lstm_conf)
        
        if xgb_pred is not None:
            predictions.append(xgb_pred)
            weights.append(self.xgb_weight * xgb_conf)
        
        if not predictions:
            return "WAIT", 0.0, False
        
        # Weighted voting
        if len(predictions) == 2:
            consensus = predictions[0] == predictions[1]
            
            if consensus:
                final_pred = predictions[0]
                final_conf = (lstm_conf * self.lstm_weight + xgb_conf * self.xgb_weight)
            else:
                # Use model with higher weighted confidence
                if weights[0] > weights[1]:
                    final_pred = predictions[0]
                    final_conf = lstm_conf * 0.8  # Reduce due to disagreement
                else:
                    final_pred = predictions[1]
                    final_conf = xgb_conf * 0.8
        else:
            final_pred = predictions[0]
            final_conf = weights[0]
            consensus = False
        
        # Convert to action string
        action_map = {0: "WAIT", 1: "LONG"}
        final_action = action_map.get(final_pred, "WAIT")
        
        return final_action, final_conf, consensus
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะ"""
        
        return {
            "lstm_loaded": self.lstm_model is not None,
            "xgb_loaded": self.xgb_model is not None,
            "lstm_weight": self.lstm_weight,
            "xgb_weight": self.xgb_weight,
            "online_learning": True,
            "lstm_online_samples": getattr(self, 'lstm_online_samples', 0),
            "xgb_online_samples": getattr(self, 'xgb_online_samples', 0),
        }
    
    # ============================================
    # Online Learning Methods (Learn from live trades)
    # ============================================
    
    def record_trade_result(
        self,
        features: np.ndarray,
        target: int,  # 1 = win, 0 = loss
        model_prediction: int,
        actual_direction: str,
    ):
        """
        บันทึกผลเทรดเพื่อ online learning
        
        Args:
            features: Input features ที่ใช้ predict
            target: ผลลัพธ์จริง (1=win, 0=loss)
            model_prediction: prediction ของ model
            actual_direction: ทิศทางจริง (LONG/SHORT)
        """
        
        # Initialize online learning buffers
        if not hasattr(self, 'online_buffer'):
            self.online_buffer = []
            self.lstm_online_samples = 0
            self.xgb_online_samples = 0
            self.lstm_correct = 0
            self.xgb_correct = 0
        
        # Store sample
        self.online_buffer.append({
            'features': features,
            'target': target,
            'prediction': model_prediction,
        })
        
        # Update accuracy tracking
        is_correct = (model_prediction == target)
        
        # Train when enough samples
        if len(self.online_buffer) >= 32:
            self._online_train_models()
    
    def _online_train_models(self):
        """
        Online training สำหรับ LSTM และ XGBoost
        """
        if not self.online_buffer:
            return
        
        # Prepare data
        features = np.array([s['features'] for s in self.online_buffer])
        targets = np.array([s['target'] for s in self.online_buffer])
        
        # Train XGBoost (incremental update)
        if self.xgb_model is not None:
            try:
                # XGBoost can do incremental training
                import xgboost as xgb
                
                dtrain = xgb.DMatrix(features, label=targets)
                
                # Incremental training with existing model
                self.xgb_model.model = xgb.train(
                    self.xgb_model.model.get_booster().attr('config') if hasattr(self.xgb_model, 'model') else {},
                    dtrain,
                    num_boost_round=5,  # Small update
                    xgb_model=self.xgb_model.model.get_booster() if hasattr(self.xgb_model, 'model') else None,
                )
                self.xgb_online_samples += len(self.online_buffer)
                logger.debug(f"📈 XGBoost online trained: +{len(self.online_buffer)} samples")
            except Exception as e:
                logger.warning(f"XGBoost online training failed: {e}")
        
        # Train LSTM (fine-tune)
        if self.lstm_model is not None:
            try:
                import torch
                import torch.nn as nn
                
                # Prepare sequence data
                if len(features) >= 10:
                    X = torch.FloatTensor(features).unsqueeze(0)  # Add batch dim
                    y = torch.LongTensor(targets)
                    
                    # Fine-tune with small learning rate
                    optimizer = torch.optim.Adam(
                        self.lstm_model.model.parameters(),
                        lr=0.0001  # Very small lr for fine-tuning
                    )
                    criterion = nn.CrossEntropyLoss()
                    
                    self.lstm_model.model.train()
                    for _ in range(3):  # Few epochs
                        optimizer.zero_grad()
                        output = self.lstm_model.model(X)
                        loss = criterion(output, y[-1:])  # Use last target
                        loss.backward()
                        optimizer.step()
                    
                    self.lstm_model.model.eval()
                    self.lstm_online_samples += len(self.online_buffer)
                    logger.debug(f"📈 LSTM online trained: +{len(self.online_buffer)} samples")
            except Exception as e:
                logger.warning(f"LSTM online training failed: {e}")
        
        # Update model weights based on performance
        self._update_weights()
        
        # Clear buffer
        self.online_buffer = []
    
    def _update_weights(self):
        """
        ปรับ weights ของ models ตาม performance
        """
        if not hasattr(self, 'lstm_correct') or not hasattr(self, 'xgb_correct'):
            return
        
        total_lstm = max(1, self.lstm_online_samples)
        total_xgb = max(1, self.xgb_online_samples)
        
        # Calculate accuracy
        lstm_acc = self.lstm_correct / total_lstm
        xgb_acc = self.xgb_correct / total_xgb
        
        # Adjust weights proportionally
        total_acc = lstm_acc + xgb_acc
        if total_acc > 0:
            self.lstm_weight = lstm_acc / total_acc
            self.xgb_weight = xgb_acc / total_acc
            
            # Ensure minimum weights
            self.lstm_weight = max(0.2, min(0.8, self.lstm_weight))
            self.xgb_weight = 1.0 - self.lstm_weight
            
            logger.debug(f"📊 Model weights updated: LSTM={self.lstm_weight:.1%}, XGB={self.xgb_weight:.1%}")
    
    def save_online_models(self):
        """
        บันทึก models ที่ train แล้วจาก online learning
        """
        try:
            if self.xgb_model is not None:
                self.xgb_model.save("models/checkpoints/xgboost_online.json")
                logger.info("💾 XGBoost online model saved")
            
            if self.lstm_model is not None:
                self.lstm_model.save("models/checkpoints/lstm_online.pt")
                logger.info("💾 LSTM online model saved")
        except Exception as e:
            logger.warning(f"Could not save online models: {e}")


def create_trained_predictor() -> TrainedModelPredictor:
    """สร้าง TrainedModelPredictor"""
    return TrainedModelPredictor()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   TRAINED MODEL PREDICTOR TEST")
    print("="*60)
    
    predictor = create_trained_predictor()
    print(f"\nStatus: {predictor.get_status()}")
    
    # Load sample data
    data_path = "data/training/GOLD_H1.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df.tail(200)  # Last 200 rows
        
        result = predictor.predict(df)
        
        print(f"\nPrediction Result:")
        print(f"  Action: {result.action}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  LSTM: {result.lstm_prediction} ({result.lstm_confidence:.1%})")
        print(f"  XGBoost: {result.xgb_prediction} ({result.xgb_confidence:.1%})")
        print(f"  Consensus: {result.consensus}")
