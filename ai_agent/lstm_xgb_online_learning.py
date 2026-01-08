"""
LSTM & XGBoost Online Learning Module
======================================
เรียนรู้จากการเทรดจริงสำหรับ LSTM และ XGBoost

Features:
1. Incremental Learning - เรียนรู้ทีละน้อยจาก trade ใหม่
2. Experience Buffer - เก็บประสบการณ์สูงสุด 1000 trades
3. Auto-retrain - Re-train อัตโนมัติเมื่อถึงเงื่อนไข
4. Learn from Losses - เรียนรู้จากความผิดพลาด
5. Feature Adaptation - ปรับ features ตามสภาพตลาด
"""

import numpy as np
import pandas as pd
import torch
import joblib
import os
import json
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger

# Try importing models
try:
    from models.lstm_model import LSTMPredictor, LSTMConfig
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("LSTM model not available")

try:
    from models.xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost model not available")


@dataclass
class LSTMXGBExperience:
    """ประสบการณ์สำหรับ LSTM/XGBoost learning"""
    trade_id: str
    timestamp: datetime
    symbol: str
    
    # Features (ที่ใช้ตอน entry)
    features: np.ndarray  # shape: (seq_len, n_features) for LSTM or (n_features,) for XGB
    
    # Entry/Exit
    entry_price: float
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    
    # Result
    pnl: float
    pnl_pct: float
    is_win: bool
    
    # Labels
    actual_label: int  # What actually happened (0=wait, 1=long, 2=short)
    predicted_label: int  # What model predicted
    was_correct: bool
    
    # Market conditions
    volatility: float = 0.0
    trend: float = 0.0
    regime: str = "unknown"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['features'] = self.features.tolist()
        return d
    
    @staticmethod
    def from_dict(d: Dict) -> 'LSTMXGBExperience':
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        d['features'] = np.array(d['features'])
        return LSTMXGBExperience(**d)


class LSTMXGBOnlineLearner:
    """
    Online Learning สำหรับ LSTM และ XGBoost
    =========================================
    เรียนรู้จากการเทรดจริงแบบ incremental
    
    Features:
    - Incremental LSTM update (fine-tuning)
    - XGBoost partial_fit simulation
    - Learn from both wins AND losses
    - Auto-retrain at milestones
    """
    
    def __init__(
        self,
        lstm_model: 'LSTMPredictor' = None,
        xgb_model: 'XGBoostModel' = None,
        scaler = None,
        selected_features: List[str] = None,
        buffer_size: int = 1000,
        min_samples_for_update: int = 20,
        update_frequency: int = 10,
        model_dir: str = "models/checkpoints",
        data_dir: str = "ai_agent/data",
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Models
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.selected_features = selected_features or []
        
        # Load models if not provided
        self._load_models()
        
        # Experience buffer
        self.experiences: deque = deque(maxlen=buffer_size)
        self.win_experiences: deque = deque(maxlen=buffer_size // 2)
        self.loss_experiences: deque = deque(maxlen=buffer_size // 2)
        
        # Learning parameters
        self.min_samples_for_update = min_samples_for_update
        self.update_frequency = update_frequency
        
        # Training buffer for incremental learning
        self.lstm_buffer: List[Tuple[np.ndarray, int]] = []
        self.xgb_buffer: List[Tuple[np.ndarray, int]] = []
        
        # Stats
        self.total_trades = 0
        self.lstm_updates = 0
        self.xgb_updates = 0
        self.last_update = None
        
        # Load existing experiences
        self._load_experiences()
        
        logger.info(f"LSTMXGBOnlineLearner initialized - Buffer: {len(self.experiences)} experiences")
    
    def _load_models(self):
        """โหลด LSTM และ XGBoost models"""
        # Load LSTM
        if self.lstm_model is None and LSTM_AVAILABLE:
            lstm_path = f"{self.model_dir}/lstm_best.pt"
            if os.path.exists(lstm_path):
                try:
                    # Load config first
                    checkpoint = torch.load(lstm_path, map_location='cpu', weights_only=False)
                    config = checkpoint.get('config')
                    if config:
                        self.lstm_model = LSTMPredictor(
                            input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            task='classification'
                        )
                        self.lstm_model.load(lstm_path)
                        logger.info("LSTM model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load LSTM: {e}")
        
        # Load XGBoost
        if self.xgb_model is None and XGBOOST_AVAILABLE:
            xgb_path = f"{self.model_dir}/xgboost_best.json"
            if os.path.exists(xgb_path):
                try:
                    self.xgb_model = XGBoostModel(task='classification')
                    self.xgb_model.load(xgb_path)
                    logger.info("XGBoost model loaded")
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost: {e}")
        
        # Load scaler
        if self.scaler is None:
            scaler_path = f"{self.model_dir}/scaler.joblib"
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Scaler loaded")
                except Exception as e:
                    logger.warning(f"Failed to load scaler: {e}")
        
        # Load selected features
        if not self.selected_features:
            features_path = f"{self.model_dir}/selected_features.json"
            if os.path.exists(features_path):
                try:
                    with open(features_path, 'r') as f:
                        data = json.load(f)
                    self.selected_features = data if isinstance(data, list) else data.get('selected_features', [])
                    logger.info(f"Loaded {len(self.selected_features)} selected features")
                except Exception as e:
                    logger.warning(f"Failed to load features: {e}")
    
    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        features: np.ndarray,
        entry_price: float,
        exit_price: float,
        direction: str,
        predicted_label: int,
        volatility: float = 0.0,
        trend: float = 0.0,
        regime: str = "unknown",
    ) -> Dict[str, Any]:
        """
        บันทึก trade และเรียนรู้
        
        Args:
            trade_id: รหัส trade
            symbol: สัญลักษณ์
            features: features ที่ใช้ตอน entry
            entry_price: ราคาเข้า
            exit_price: ราคาออก
            direction: "LONG" หรือ "SHORT"
            predicted_label: label ที่ model ทำนาย (0=wait, 1=long, 2=short)
            volatility: volatility ตอน entry
            trend: trend ตอน entry
            regime: market regime
            
        Returns:
            ผลการเรียนรู้
        """
        
        # Calculate results
        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl = pnl_pct * 1000  # Assume $1000 trade
        is_win = pnl > 0
        
        # Determine actual label (what would have been correct)
        if is_win:
            actual_label = 1 if direction == "LONG" else 2  # The predicted direction was correct
        else:
            actual_label = 0  # Should have waited
        
        was_correct = actual_label == predicted_label
        
        # Create experience
        exp = LSTMXGBExperience(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            features=features,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_win=is_win,
            actual_label=actual_label,
            predicted_label=predicted_label,
            was_correct=was_correct,
            volatility=volatility,
            trend=trend,
            regime=regime,
        )
        
        # Add to buffers
        self.experiences.append(exp)
        if is_win:
            self.win_experiences.append(exp)
        else:
            self.loss_experiences.append(exp)
        
        self.total_trades += 1
        
        # Add to training buffers (both wins AND losses)
        self._add_to_training_buffer(exp)
        
        # Check if should update models
        results = {
            "trade_id": trade_id,
            "is_win": is_win,
            "pnl": pnl,
            "was_correct": was_correct,
        }
        
        if len(self.lstm_buffer) >= self.min_samples_for_update:
            if self.total_trades % self.update_frequency == 0:
                update_result = self._incremental_update()
                results["learning"] = update_result
        
        # Save experiences periodically
        if self.total_trades % 10 == 0:
            self._save_experiences()
        
        logger.info(
            f"📚 Recorded trade {trade_id}: "
            f"{'✅ WIN' if is_win else '❌ LOSS'} ${pnl:.2f} | "
            f"Correct: {'✓' if was_correct else '✗'}"
        )
        
        return results
    
    def _add_to_training_buffer(self, exp: LSTMXGBExperience):
        """เพิ่ม experience เข้า training buffer"""
        
        # For LSTM: features should be (seq_len, n_features)
        if exp.features.ndim == 2:
            lstm_features = exp.features
        else:
            # If 1D, reshape for single timestep
            lstm_features = exp.features.reshape(1, -1)
        
        # For XGB: features should be 1D (flatten if needed)
        if exp.features.ndim == 2:
            xgb_features = exp.features[-1]  # Last timestep
        else:
            xgb_features = exp.features
        
        # Label assignment:
        # - For wins: use actual_label (what was correct)
        # - For losses: use 0 (WAIT) to teach model to avoid
        if exp.is_win:
            label = exp.actual_label
        else:
            # Learn from loss: in this situation, should have waited
            label = 0
        
        # Weight by importance
        # Losses are more important to learn from
        weight = 2.0 if not exp.is_win else 1.0
        
        # Add multiple copies for losses (prioritized learning)
        n_copies = int(weight)
        for _ in range(n_copies):
            self.lstm_buffer.append((lstm_features, label))
            self.xgb_buffer.append((xgb_features, label))
    
    def _incremental_update(self) -> Dict:
        """อัพเดต models แบบ incremental"""
        
        results = {"status": "started"}
        
        # ==============================
        # LSTM Incremental Update
        # ==============================
        if self.lstm_model is not None and len(self.lstm_buffer) >= self.min_samples_for_update:
            lstm_result = self._update_lstm()
            results["lstm"] = lstm_result
        
        # ==============================
        # XGBoost Incremental Update
        # ==============================
        if self.xgb_model is not None and len(self.xgb_buffer) >= self.min_samples_for_update:
            xgb_result = self._update_xgboost()
            results["xgboost"] = xgb_result
        
        self.last_update = datetime.now()
        
        return results
    
    def _update_lstm(self) -> Dict:
        """Fine-tune LSTM จาก experiences"""
        
        logger.info(f"🧠 Fine-tuning LSTM with {len(self.lstm_buffer)} samples...")
        
        try:
            # Prepare data
            X_list = [x[0] for x in self.lstm_buffer[-100:]]  # Last 100 samples
            y_list = [x[1] for x in self.lstm_buffer[-100:]]
            
            # Pad sequences to same length
            max_len = max(x.shape[0] for x in X_list)
            n_features = X_list[0].shape[1] if X_list[0].ndim > 1 else X_list[0].shape[0]
            
            X = np.zeros((len(X_list), max_len, n_features), dtype=np.float32)
            for i, x in enumerate(X_list):
                if x.ndim == 2:
                    X[i, :x.shape[0], :] = x
                else:
                    X[i, 0, :] = x
            
            y = np.array(y_list, dtype=np.float32)
            
            # Scale features if scaler available
            if self.scaler is not None:
                X_reshaped = X.reshape(-1, n_features)
                X_scaled = self.scaler.transform(X_reshaped)
                X = X_scaled.reshape(X.shape)
            
            # Fine-tune for a few epochs
            device = self.lstm_model.device
            self.lstm_model.model.train()
            
            # Create tensors
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
            
            # Train for a few iterations
            total_loss = 0
            n_iterations = 5
            
            for _ in range(n_iterations):
                self.lstm_model.optimizer.zero_grad()
                outputs = self.lstm_model.model(X_tensor)
                loss = self.lstm_model.criterion(outputs, y_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.model.parameters(), 0.5)
                self.lstm_model.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / n_iterations
            
            # Save updated model
            self.lstm_model.save(f"{self.model_dir}/lstm_online.pt")
            
            self.lstm_updates += 1
            
            logger.info(f"✅ LSTM updated #{self.lstm_updates} | Loss: {avg_loss:.4f}")
            
            return {
                "status": "success",
                "updates": self.lstm_updates,
                "loss": avg_loss,
                "samples": len(X_list),
            }
            
        except Exception as e:
            logger.error(f"LSTM update failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _update_xgboost(self) -> Dict:
        """อัพเดต XGBoost (retrain with new data)"""
        
        logger.info(f"🌳 Updating XGBoost with {len(self.xgb_buffer)} samples...")
        
        try:
            # XGBoost doesn't support true incremental learning
            # We'll retrain on recent experiences + sample from old data
            
            X_list = [x[0] for x in self.xgb_buffer[-200:]]  # Last 200 samples
            y_list = [x[1] for x in self.xgb_buffer[-200:]]
            
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
            # Scale if scaler available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Retrain XGBoost with new data
            # Note: This is simplified - in production you'd want to
            # combine with existing training data
            
            self.xgb_model.fit(
                X_train=X[:int(len(X)*0.8)],
                y_train=y[:int(len(y)*0.8)],
                X_val=X[int(len(X)*0.8):],
                y_val=y[int(len(y)*0.8):],
                verbose=False
            )
            
            # Save updated model
            self.xgb_model.save(f"{self.model_dir}/xgboost_online.json")
            
            self.xgb_updates += 1
            
            logger.info(f"✅ XGBoost updated #{self.xgb_updates}")
            
            return {
                "status": "success",
                "updates": self.xgb_updates,
                "samples": len(X_list),
            }
            
        except Exception as e:
            logger.error(f"XGBoost update failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def learn_from_loss(self, exp: LSTMXGBExperience):
        """
        เรียนรู้จากการขาดทุน (สอนว่าอะไรไม่ควรทำ)
        
        Logic:
        - ถ้าขาดทุน → สอนว่าในสถานการณ์นี้ควร WAIT (label = 0)
        - เพิ่ม weight ให้ loss experiences ในการ training
        """
        
        logger.info(f"📖 Learning from loss: {exp.trade_id}")
        
        # Add to buffer with label = 0 (WAIT)
        if exp.features.ndim == 2:
            lstm_features = exp.features
            xgb_features = exp.features[-1]
        else:
            lstm_features = exp.features.reshape(1, -1)
            xgb_features = exp.features
        
        # Add multiple times to emphasize
        for _ in range(3):  # Triple weight for losses
            self.lstm_buffer.append((lstm_features, 0))
            self.xgb_buffer.append((xgb_features, 0))
    
    def should_retrain(self) -> Tuple[bool, str]:
        """ตรวจสอบว่าควร retrain เต็มรูปแบบหรือไม่"""
        
        if len(self.experiences) < 50:
            return False, "Not enough experiences"
        
        # Check win rate
        recent = list(self.experiences)[-50:]
        wins = sum(1 for e in recent if e.is_win)
        win_rate = wins / len(recent)
        
        if win_rate < 0.35:
            return True, f"Win rate too low: {win_rate:.1%}"
        
        # Check accuracy
        correct = sum(1 for e in recent if e.was_correct)
        accuracy = correct / len(recent)
        
        if accuracy < 0.40:
            return True, f"Accuracy too low: {accuracy:.1%}"
        
        # Check time since last update
        if self.last_update:
            days_since = (datetime.now() - self.last_update).days
            if days_since >= 7:
                return True, f"{days_since} days since last update"
        
        return False, ""
    
    def get_stats(self) -> Dict:
        """ดึงสถิติการเรียนรู้"""
        
        if not self.experiences:
            return {"total_trades": 0}
        
        wins = sum(1 for e in self.experiences if e.is_win)
        correct = sum(1 for e in self.experiences if e.was_correct)
        
        return {
            "total_trades": self.total_trades,
            "total_experiences": len(self.experiences),
            "wins": wins,
            "losses": len(self.experiences) - wins,
            "win_rate": wins / len(self.experiences) if self.experiences else 0,
            "accuracy": correct / len(self.experiences) if self.experiences else 0,
            "lstm_updates": self.lstm_updates,
            "xgb_updates": self.xgb_updates,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "lstm_buffer_size": len(self.lstm_buffer),
            "xgb_buffer_size": len(self.xgb_buffer),
        }
    
    def _save_experiences(self):
        """บันทึก experiences"""
        
        data = [e.to_dict() for e in self.experiences]
        path = f"{self.data_dir}/lstm_xgb_experiences.json"
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_experiences(self):
        """โหลด experiences"""
        
        path = f"{self.data_dir}/lstm_xgb_experiences.json"
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                for d in data:
                    exp = LSTMXGBExperience.from_dict(d)
                    self.experiences.append(exp)
                    if exp.is_win:
                        self.win_experiences.append(exp)
                    else:
                        self.loss_experiences.append(exp)
                
                logger.info(f"Loaded {len(self.experiences)} experiences")
            except Exception as e:
                logger.warning(f"Failed to load experiences: {e}")


# Singleton instance
_lstm_xgb_learner: Optional[LSTMXGBOnlineLearner] = None


def get_lstm_xgb_learner() -> LSTMXGBOnlineLearner:
    """Get singleton LSTM/XGBoost online learner"""
    global _lstm_xgb_learner
    if _lstm_xgb_learner is None:
        _lstm_xgb_learner = LSTMXGBOnlineLearner()
    return _lstm_xgb_learner


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   LSTM/XGBoost Online Learning Test")
    print("="*60)
    
    learner = get_lstm_xgb_learner()
    
    # Simulate trades
    print("\nSimulating trades...")
    
    for i in range(25):
        # Random features
        features = np.random.randn(50, 50).astype(np.float32)  # (seq_len, n_features)
        entry = 2000 + np.random.randn() * 10
        exit = entry + np.random.randn() * 30
        
        result = learner.record_trade(
            trade_id=f"TEST_{i}",
            symbol="XAUUSD",
            features=features,
            entry_price=entry,
            exit_price=exit,
            direction="LONG" if np.random.rand() > 0.3 else "SHORT",
            predicted_label=np.random.randint(0, 3),
            volatility=np.random.rand() * 0.03,
            trend=np.random.randn() * 0.02,
            regime="trending_up",
        )
    
    print("\n" + "="*60)
    print("   LEARNING STATS")
    print("="*60)
    
    stats = learner.get_stats()
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Accuracy: {stats['accuracy']:.1%}")
    print(f"LSTM Updates: {stats['lstm_updates']}")
    print(f"XGBoost Updates: {stats['xgb_updates']}")
    
    should, reason = learner.should_retrain()
    if should:
        print(f"\n⚠️ Retrain recommended: {reason}")
    else:
        print("\n✓ Model is performing well")
