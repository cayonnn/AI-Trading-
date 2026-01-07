"""
Learning Engine - GPU Accelerated
==================================
เครื่องยนต์เรียนรู้ที่รองรับ GPU

Features:
- GPU Training (CUDA/MPS)
- Reinforcement Learning
- Pattern Recognition
- Auto-improvement from trades
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import json

# PyTorch with GPU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check GPU availability
def get_device():
    """ตรวจสอบและเลือก Device ที่ดีที่สุด"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Apple Silicon GPU (MPS) Detected")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ No GPU found, using CPU")
    return device

DEVICE = get_device()


@dataclass
class LearningConfig:
    """การตั้งค่า Learning Engine"""
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    hidden_units: List[int] = None
    dropout: float = 0.3
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [256, 128, 64]


class TradingNN(nn.Module):
    """
    Neural Network สำหรับ Trading
    
    ใช้ GPU สำหรับ Training และ Inference
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_units: List[int] = [256, 128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden in hidden_units:
            layers.extend([
                nn.Linear(prev_size, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden
        
        # Output layer - Confidence score (0-1)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class QLearningAgent:
    """
    Q-Learning Agent สำหรับ Trading
    
    ใช้ Deep Q-Network (DQN) บน GPU
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 2,  # LONG, WAIT
        hidden_units: List[int] = [256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,  # Discount factor
        epsilon: float = 1.0,  # Exploration rate
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = DEVICE
        
        # Q-Network
        self.q_network = self._build_network(hidden_units).to(self.device)
        self.target_network = self._build_network(hidden_units).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience Replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64
        
        logger.info(f"Q-Learning Agent initialized on {self.device}")
    
    def _build_network(self, hidden_units: List[int]) -> nn.Module:
        """สร้าง Q-Network"""
        layers = []
        prev_size = self.state_size
        
        for hidden in hidden_units:
            layers.extend([
                nn.Linear(prev_size, hidden),
                nn.ReLU(),
            ])
            prev_size = hidden
        
        layers.append(nn.Linear(prev_size, self.action_size))
        
        return nn.Sequential(*layers)
    
    def remember(self, state, action, reward, next_state, done):
        """เก็บประสบการณ์"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """เลือก Action"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self) -> float:
        """เรียนรู้จาก Experience Replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """อัพเดต Target Network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """บันทึก Model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """โหลด Model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        logger.info(f"Agent loaded from {path}")


class LearningEngine:
    """
    เครื่องยนต์เรียนรู้หลัก - GPU Accelerated
    
    ความสามารถ:
    - เทรนโมเดลจากข้อมูล
    - เรียนรู้จากผลเทรด
    - ปรับปรุงกลยุทธ์อัตโนมัติ
    - Q-Learning สำหรับ Decision Making
    """
    
    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        model_dir: str = "ai_agent/models",
    ):
        self.config = config or LearningConfig()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.device = DEVICE
        self.confidence_model: Optional[TradingNN] = None
        self.q_agent: Optional[QLearningAgent] = None
        
        # Training history
        self.training_history = []
        
        logger.info(f"LearningEngine initialized on {self.device}")
    
    def build_confidence_model(self, input_size: int):
        """สร้างโมเดล Confidence Prediction"""
        self.confidence_model = TradingNN(
            input_size=input_size,
            hidden_units=self.config.hidden_units,
            dropout=self.config.dropout,
        ).to(self.device)
        
        logger.info(f"Confidence model built: {sum(p.numel() for p in self.confidence_model.parameters())} params")
    
    def build_q_agent(self, state_size: int):
        """สร้าง Q-Learning Agent"""
        self.q_agent = QLearningAgent(
            state_size=state_size,
            action_size=2,  # LONG, WAIT
            hidden_units=[256, 128],
            learning_rate=self.config.learning_rate,
        )
        logger.info("Q-Learning Agent built")
    
    def train_confidence_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        เทรนโมเดล Confidence Prediction บน GPU
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) - 1 for win, 0 for loss
            
        Returns:
            Training history
        """
        if self.confidence_model is None:
            self.build_confidence_model(X.shape[1])
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]
        
        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val = torch.FloatTensor(y[val_idx]).unsqueeze(1).to(self.device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(
            self.confidence_model.parameters(),
            lr=self.config.learning_rate,
        )
        criterion = nn.BCELoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Training on {self.device} with {len(X_train)} samples...")
        
        for epoch in range(self.config.epochs):
            # Train
            self.confidence_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.confidence_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.confidence_model.eval()
            with torch.no_grad():
                val_outputs = self.confidence_model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_accuracy = ((val_outputs > 0.5) == y_val).float().mean().item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.confidence_model.state_dict(), 
                          f"{self.model_dir}/confidence_best.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2%}")
        
        # Load best model
        self.confidence_model.load_state_dict(
            torch.load(f"{self.model_dir}/confidence_best.pt", map_location=self.device)
        )
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': len(history['train_loss']),
            'final_val_accuracy': history['val_accuracy'][-1],
        })
        
        logger.info(f"Training complete. Best Val Accuracy: {max(history['val_accuracy']):.2%}")
        return history
    
    def predict_confidence(self, features: np.ndarray) -> float:
        """ทำนาย Confidence จาก Features"""
        if self.confidence_model is None:
            return 0.5  # Default
        
        self.confidence_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            confidence = self.confidence_model(x).item()
        
        return confidence
    
    def learn_from_trade(
        self,
        state: np.ndarray,
        action: int,  # 0=WAIT, 1=LONG
        reward: float,  # P&L based reward
        next_state: np.ndarray,
        done: bool = False,
    ):
        """เรียนรู้จากผลเทรด"""
        if self.q_agent is None:
            self.build_q_agent(len(state))
        
        self.q_agent.remember(state, action, reward, next_state, done)
        loss = self.q_agent.replay()
        
        return loss
    
    def decide_action(self, state: np.ndarray) -> int:
        """ตัดสินใจ Action จาก Q-Learning"""
        if self.q_agent is None:
            return 0  # WAIT
        
        return self.q_agent.act(state)
    
    def save_all(self, prefix: str = ""):
        """บันทึกโมเดลทั้งหมด"""
        if self.confidence_model:
            torch.save(
                self.confidence_model.state_dict(),
                f"{self.model_dir}/{prefix}confidence_model.pt"
            )
        
        if self.q_agent:
            self.q_agent.save(f"{self.model_dir}/{prefix}q_agent.pt")
        
        # Save config
        with open(f"{self.model_dir}/{prefix}config.json", 'w') as f:
            json.dump({
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'hidden_units': self.config.hidden_units,
                'training_history': self.training_history,
            }, f, indent=2)
        
        logger.info(f"All models saved to {self.model_dir}")
    
    def load_all(self, prefix: str = ""):
        """โหลดโมเดลทั้งหมด"""
        confidence_path = f"{self.model_dir}/{prefix}confidence_model.pt"
        if os.path.exists(confidence_path):
            # Infer input size from saved model
            state_dict = torch.load(confidence_path, map_location=self.device)
            input_size = state_dict['network.0.weight'].shape[1]
            self.build_confidence_model(input_size)
            self.confidence_model.load_state_dict(state_dict)
            logger.info(f"Confidence model loaded")
        
        q_agent_path = f"{self.model_dir}/{prefix}q_agent.pt"
        if os.path.exists(q_agent_path):
            checkpoint = torch.load(q_agent_path, map_location=self.device)
            state_size = checkpoint['q_network']['0.weight'].shape[1]
            self.build_q_agent(state_size)
            self.q_agent.load(q_agent_path)
            logger.info(f"Q-Agent loaded")
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะ Engine"""
        return {
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "confidence_model_loaded": self.confidence_model is not None,
            "q_agent_loaded": self.q_agent is not None,
            "training_count": len(self.training_history),
        }


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("\n" + "="*60)
    print("LEARNING ENGINE TEST - GPU ACCELERATED")
    print("="*60)
    
    # Test Learning Engine
    engine = LearningEngine()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.random.rand(n_samples) > 0.5).astype(np.float32)
    
    # Train
    history = engine.train_confidence_model(X, y)
    
    # Test prediction
    test_features = np.random.randn(n_features).astype(np.float32)
    confidence = engine.predict_confidence(test_features)
    print(f"\nTest Prediction Confidence: {confidence:.2%}")
    
    # Test Q-Learning
    engine.build_q_agent(n_features)
    action = engine.decide_action(test_features)
    print(f"Q-Learning Action: {'LONG' if action == 1 else 'WAIT'}")
    
    # Status
    print(f"\nEngine Status:")
    for k, v in engine.get_status().items():
        print(f"  {k}: {v}")
    
    # Save
    engine.save_all("test_")
    
    print("\n✅ Learning Engine Test Complete")
