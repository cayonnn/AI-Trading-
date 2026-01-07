"""
Curiosity Module
================
Curiosity-Driven Exploration สำหรับการค้นหารูปแบบใหม่

Features:
1. Intrinsic Reward - ให้ reward เมื่อค้นพบสถานการณ์ใหม่
2. Forward Model - ทำนายผลลัพธ์ของ action
3. Novelty Detection - ตรวจจับ states ที่ไม่เคยเจอ
4. Pattern Discovery - ค้นหารูปแบบการเทรดใหม่
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
import json
from loguru import logger


# Get device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


@dataclass
class NovelPattern:
    """รูปแบบใหม่ที่ค้นพบ"""
    pattern_id: str
    discovered_at: datetime
    state_features: np.ndarray
    predicted_outcome: str
    actual_outcome: str
    novelty_score: float
    trading_profitable: bool = False
    occurrences: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "discovered_at": self.discovered_at.isoformat(),
            "state_features": self.state_features.tolist(),
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "novelty_score": self.novelty_score,
            "trading_profitable": self.trading_profitable,
            "occurrences": self.occurrences,
        }


class ForwardModel(nn.Module):
    """
    Forward Dynamics Model
    
    ทำนาย next_state จาก (state, action)
    ใช้สำหรับคำนวณ prediction error เป็น intrinsic reward
    """
    
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        # Encode state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Embed action
        self.action_embed = nn.Embedding(action_dim, hidden_dim // 2)
        
        # Predict next state (from encoded state + action)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
    ) -> torch.Tensor:
        """ทำนาย next state"""
        state_enc = self.state_encoder(state)
        action_enc = self.action_embed(action)
        
        combined = torch.cat([state_enc, action_enc], dim=-1)
        next_state_pred = self.predictor(combined)
        
        return next_state_pred
    
    def prediction_error(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """คำนวณ prediction error (ใช้เป็น curiosity reward)"""
        pred = self.forward(state, action)
        error = torch.mean((pred - next_state) ** 2, dim=-1)
        return error


class InverseModel(nn.Module):
    """
    Inverse Dynamics Model
    
    ทำนาย action จาก (state, next_state)
    ช่วยให้ model focus บน features ที่ influenced by actions
    """
    
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """ทำนาย action จาก state transition"""
        combined = torch.cat([state, next_state], dim=-1)
        return self.model(combined)


class CuriosityModule:
    """
    Curiosity-Driven Exploration
    
    ความสามารถ:
    1. คำนวณ intrinsic reward จาก prediction error
    2. ตรวจจับ novel states
    3. บันทึก patterns ที่ค้นพบ
    4. ให้ bonus reward สำหรับ exploration
    """
    
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        curiosity_weight: float = 0.1,  # Weight for intrinsic reward
        learning_rate: float = 1e-3,
        novelty_threshold: float = 0.5,
        model_dir: str = "ai_agent/models",
        data_dir: str = "ai_agent/data",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.curiosity_weight = curiosity_weight
        self.novelty_threshold = novelty_threshold
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Models
        self.forward_model = ForwardModel(state_dim, action_dim).to(DEVICE)
        self.inverse_model = InverseModel(state_dim, action_dim).to(DEVICE)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.forward_model.parameters()) + 
            list(self.inverse_model.parameters()),
            lr=learning_rate,
        )
        
        # State memory for novelty detection
        self.state_memory: deque = deque(maxlen=5000)
        
        # Discovered patterns
        self.novel_patterns: List[NovelPattern] = []
        
        # Running stats for normalization
        self.error_mean = 0.0
        self.error_std = 1.0
        self.update_count = 0
        
        logger.info("CuriosityModule initialized")
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        คำนวณ intrinsic reward จาก prediction error
        
        Higher error = More novel = Higher intrinsic reward
        """
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action_t = torch.LongTensor([action]).to(DEVICE)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
            
            error = self.forward_model.prediction_error(
                state_t, action_t, next_state_t
            ).item()
        
        # Normalize error
        normalized_error = (error - self.error_mean) / (self.error_std + 1e-8)
        
        # Clip to reasonable range
        intrinsic_reward = np.clip(normalized_error, 0, 3) * self.curiosity_weight
        
        return intrinsic_reward
    
    def get_combined_reward(
        self,
        extrinsic_reward: float,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> Tuple[float, float]:
        """
        คำนวณ combined reward (extrinsic + intrinsic)
        
        Returns:
            (combined_reward, intrinsic_reward)
        """
        intrinsic = self.compute_intrinsic_reward(state, action, next_state)
        combined = extrinsic_reward + intrinsic
        
        return combined, intrinsic
    
    def detect_novelty(self, state: np.ndarray) -> Tuple[bool, float]:
        """
        ตรวจจับว่า state นี้เป็น novel หรือไม่
        
        Returns:
            (is_novel, novelty_score)
        """
        
        if len(self.state_memory) < 10:
            self.state_memory.append(state)
            return False, 0.0
        
        # Compare with recent states
        recent_states = np.array(list(self.state_memory)[-100:])
        
        # Calculate distance to nearest neighbor
        distances = np.sqrt(np.sum((recent_states - state) ** 2, axis=1))
        min_distance = np.min(distances)
        mean_distance = np.mean(distances)
        
        # Novelty score
        novelty_score = min_distance / (mean_distance + 1e-8)
        
        # Add to memory
        self.state_memory.append(state)
        
        is_novel = novelty_score > self.novelty_threshold
        
        return is_novel, novelty_score
    
    def record_experience(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        pnl: float,
    ):
        """
        บันทึกประสบการณ์และอัพเดต models
        """
        
        # Check for novelty
        is_novel, novelty_score = self.detect_novelty(state)
        
        if is_novel and novelty_score > 1.0:
            # Record novel pattern
            outcome = "profit" if pnl > 0 else "loss"
            
            pattern = NovelPattern(
                pattern_id=f"novel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                discovered_at=datetime.now(),
                state_features=state.copy(),
                predicted_outcome="unknown",
                actual_outcome=outcome,
                novelty_score=novelty_score,
                trading_profitable=pnl > 0,
            )
            
            self.novel_patterns.append(pattern)
            
            if pnl > 0:
                logger.info(f"Novel profitable pattern discovered! Score: {novelty_score:.2f}")
        
        # Update models
        self._update_models(state, action, next_state)
    
    def _update_models(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ):
        """อัพเดต forward และ inverse models"""
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action_t = torch.LongTensor([action]).to(DEVICE)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        
        # Forward model loss
        pred_next = self.forward_model(state_t, action_t)
        forward_loss = nn.MSELoss()(pred_next, next_state_t)
        
        # Inverse model loss
        pred_action = self.inverse_model(state_t, next_state_t)
        inverse_loss = nn.CrossEntropyLoss()(pred_action, action_t)
        
        # Total loss
        loss = forward_loss + 0.2 * inverse_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update running stats
        error = forward_loss.item()
        self.update_count += 1
        alpha = 0.01  # Smoothing factor
        self.error_mean = (1 - alpha) * self.error_mean + alpha * error
        self.error_std = (1 - alpha) * self.error_std + alpha * abs(error - self.error_mean)
    
    def get_exploration_bonus(self, state: np.ndarray) -> float:
        """
        คำนวณ bonus สำหรับ exploration
        
        States ที่เจอน้อย → higher bonus
        """
        
        is_novel, novelty_score = self.detect_novelty(state)
        
        if is_novel:
            return 0.1 * novelty_score
        return 0.0
    
    def get_novel_profitable_patterns(self) -> List[NovelPattern]:
        """ดึง patterns ที่ค้นพบและทำกำไร"""
        return [p for p in self.novel_patterns if p.trading_profitable]
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ"""
        
        profitable = sum(1 for p in self.novel_patterns if p.trading_profitable)
        
        return {
            "total_patterns": len(self.novel_patterns),
            "profitable_patterns": profitable,
            "states_in_memory": len(self.state_memory),
            "update_count": self.update_count,
            "error_mean": self.error_mean,
            "curiosity_weight": self.curiosity_weight,
        }
    
    def set_curiosity_weight(self, weight: float):
        """ปรับน้ำหนัก curiosity"""
        self.curiosity_weight = np.clip(weight, 0.01, 0.5)
        logger.info(f"Curiosity weight set to {self.curiosity_weight}")
    
    def save(self, name: str = "curiosity"):
        """บันทึก models"""
        
        path = f"{self.model_dir}/{name}_module.pt"
        torch.save({
            "forward_model": self.forward_model.state_dict(),
            "inverse_model": self.inverse_model.state_dict(),
            "error_mean": self.error_mean,
            "error_std": self.error_std,
            "update_count": self.update_count,
        }, path)
        
        # Save patterns
        patterns_path = f"{self.data_dir}/novel_patterns.json"
        with open(patterns_path, 'w') as f:
            json.dump([p.to_dict() for p in self.novel_patterns], f, indent=2)
        
        logger.info(f"Saved CuriosityModule to {path}")
    
    def load(self, name: str = "curiosity"):
        """โหลด models"""
        
        path = f"{self.model_dir}/{name}_module.pt"
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            self.forward_model.load_state_dict(checkpoint["forward_model"])
            self.inverse_model.load_state_dict(checkpoint["inverse_model"])
            self.error_mean = checkpoint.get("error_mean", 0.0)
            self.error_std = checkpoint.get("error_std", 1.0)
            self.update_count = checkpoint.get("update_count", 0)
            logger.info("Loaded CuriosityModule")
            return True
        return False


def create_curiosity_module(state_dim: int = 11) -> CuriosityModule:
    """สร้าง CuriosityModule"""
    return CuriosityModule(state_dim=state_dim)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   CURIOSITY MODULE TEST")
    print("="*60)
    
    np.random.seed(42)
    
    curiosity = create_curiosity_module(state_dim=11)
    
    # Simulate experiences
    print("\nSimulating experiences...")
    
    for i in range(50):
        state = np.random.randn(11).astype(np.float32)
        action = np.random.randint(3)
        next_state = state + np.random.randn(11).astype(np.float32) * 0.1
        pnl = np.random.randn() * 100
        
        curiosity.record_experience(state, action, next_state, pnl)
        
        # Get combined reward
        extrinsic = pnl / 1000
        combined, intrinsic = curiosity.get_combined_reward(
            extrinsic, state, action, next_state
        )
        
        if i % 10 == 0:
            print(f"  Step {i}: Extrinsic={extrinsic:.3f}, Intrinsic={intrinsic:.3f}")
    
    # Test novelty detection
    print("\nNovelty detection:")
    for i in range(3):
        test_state = np.random.randn(11).astype(np.float32) * 2  # Unusual state
        is_novel, score = curiosity.detect_novelty(test_state)
        print(f"  Test {i+1}: Novel={is_novel}, Score={score:.2f}")
    
    # Stats
    print("\nStats:")
    stats = curiosity.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Profitable patterns
    print(f"\nProfitable patterns: {len(curiosity.get_novel_profitable_patterns())}")
