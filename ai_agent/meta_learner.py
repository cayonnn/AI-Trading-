"""
Meta-Learner Module (MAML-based)
================================
ปรับตัวเข้ากับสภาพตลาดใหม่ภายใน 5-10 trades

Features:
1. Model-Agnostic Meta-Learning - เรียนรู้วิธีการเรียนรู้
2. Few-Shot Adaptation - ปรับตัวจาก trades น้อยๆ
3. Regime-Aware Learning - เรียนรู้แยกตาม market regime
4. Fast Adaptation - ปรับตัวเร็วเมื่อตลาดเปลี่ยน
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from dataclasses import dataclass, field
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
class TaskExperience:
    """ประสบการณ์สำหรับ meta-learning"""
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    timestamp: datetime
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool = False


class AdaptiveNetwork(nn.Module):
    """
    Neural Network ที่ปรับตัวได้เร็ว
    
    ใช้โครงสร้างที่ออกแบบมาสำหรับ few-shot learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Smaller network for faster adaptation
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value


class MetaLearner:
    """
    Meta-Learning Agent
    
    ความสามารถ:
    1. เก็บประสบการณ์แยกตาม regime
    2. สร้าง regime-specific policies
    3. ปรับตัวเร็วเมื่อ regime เปลี่ยน
    4. Transfer learning ระหว่าง regimes
    """
    
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        inner_lr: float = 0.01,  # Fast adaptation learning rate
        outer_lr: float = 0.001,  # Meta-learning rate
        adaptation_steps: int = 5,  # Steps for fast adaptation
        min_experiences_per_regime: int = 5,
        model_dir: str = "ai_agent/models",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps
        self.min_experiences = min_experiences_per_regime
        self.model_dir = model_dir
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Meta-model (shared initialization)
        self.meta_model = AdaptiveNetwork(state_dim, action_dim).to(DEVICE)
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=outer_lr)
        
        # Regime-specific models (adapted from meta-model)
        self.regime_models: Dict[str, AdaptiveNetwork] = {}
        self.regime_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Current regime tracking
        self.current_regime = "unknown"
        self.regime_history = deque(maxlen=50)
        
        # Learning stats
        self.adaptation_count = 0
        self.meta_updates = 0
        
        logger.info(f"MetaLearner initialized on {DEVICE}")
    
    def detect_regime(
        self,
        trend: float,
        volatility: float,
        momentum: float = 0.0,
    ) -> str:
        """ตรวจจับ market regime"""
        
        # High volatility
        if volatility > 0.02:
            return "volatile"
        
        # Trending up
        if trend > 0.01 and momentum > 0:
            return "trending_up"
        
        # Trending down
        if trend < -0.01 and momentum < 0:
            return "trending_down"
        
        # Ranging
        return "ranging"
    
    def update_regime(
        self,
        trend: float,
        volatility: float,
        momentum: float = 0.0,
    ):
        """อัพเดต regime และตรวจสอบการเปลี่ยนแปลง"""
        
        new_regime = self.detect_regime(trend, volatility, momentum)
        
        if new_regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = new_regime
            
            logger.info(f"Regime changed: {old_regime} -> {new_regime}")
            
            # Trigger adaptation
            self._adapt_to_new_regime(new_regime)
        
        self.regime_history.append(new_regime)
    
    def _adapt_to_new_regime(self, regime: str):
        """ปรับตัวเมื่อ regime เปลี่ยน"""
        
        if regime not in self.regime_models:
            # Create new model from meta-model
            self.regime_models[regime] = AdaptiveNetwork(
                self.state_dim, self.action_dim
            ).to(DEVICE)
            # Copy weights from meta-model
            self.regime_models[regime].load_state_dict(
                self.meta_model.state_dict()
            )
            logger.info(f"Created new model for regime: {regime}")
        
        # Fast adaptation if we have enough experience
        if len(self.regime_buffers[regime]) >= self.min_experiences:
            self._fast_adapt(regime)
    
    def _fast_adapt(self, regime: str):
        """Fast adaptation (inner loop) - เรียนรู้เร็วจาก few examples"""
        
        model = self.regime_models[regime]
        experiences = list(self.regime_buffers[regime])
        
        if len(experiences) < self.min_experiences:
            return
        
        # Use recent experiences
        recent = experiences[-20:]
        
        # Prepare data
        states = torch.FloatTensor(
            np.array([e.state for e in recent])
        ).to(DEVICE)
        actions = torch.LongTensor([e.action for e in recent]).to(DEVICE)
        rewards = torch.FloatTensor([e.reward for e in recent]).to(DEVICE)
        
        # Normalize rewards
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Fast gradient steps
        optimizer = optim.SGD(model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.adaptation_steps):
            policy, values = model(states)
            
            dist = torch.distributions.Categorical(policy)
            log_probs = dist.log_prob(actions)
            
            # Simple policy gradient
            policy_loss = -(log_probs * rewards).mean()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        
        self.adaptation_count += 1
        logger.info(f"Fast adaptation #{self.adaptation_count} for regime: {regime}")
    
    def record_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        trend: float = 0.0,
        volatility: float = 0.0,
    ):
        """บันทึกประสบการณ์และ update regime"""
        
        regime = self.detect_regime(trend, volatility)
        
        exp = TaskExperience(
            regime=regime,
            timestamp=datetime.now(),
            state=state.astype(np.float32),
            action=action,
            reward=reward,
            next_state=next_state.astype(np.float32),
        )
        
        self.regime_buffers[regime].append(exp)
        
        # Auto-adapt if we have enough new experiences
        if len(self.regime_buffers[regime]) % 5 == 0:
            self._fast_adapt(regime)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        เลือก action โดยใช้ regime-specific model
        
        Returns:
            (action, confidence)
        """
        
        # Use regime-specific model if available
        if self.current_regime in self.regime_models:
            model = self.regime_models[self.current_regime]
        else:
            model = self.meta_model
        
        model.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            policy, value = model(state_t)
            
            # Sample action
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            
            # Confidence based on policy entropy
            entropy = dist.entropy().item()
            max_entropy = np.log(self.action_dim)
            confidence = 1 - (entropy / max_entropy)  # Lower entropy = higher confidence
        
        return action.item(), confidence
    
    def meta_update(self):
        """
        Meta-update (outer loop) - อัพเดต meta-model จากประสบการณ์ทุก regime
        """
        
        # Need experiences from multiple regimes
        regimes_with_data = [r for r, buf in self.regime_buffers.items() 
                           if len(buf) >= self.min_experiences]
        
        if len(regimes_with_data) < 2:
            return  # Need at least 2 regimes
        
        total_loss = 0.0
        
        for regime in regimes_with_data:
            experiences = list(self.regime_buffers[regime])[-20:]
            
            # Sample support set (for adaptation) and query set (for meta-update)
            np.random.shuffle(experiences)
            split = len(experiences) // 2
            support_set = experiences[:split]
            query_set = experiences[split:]
            
            if len(support_set) < 3 or len(query_set) < 3:
                continue
            
            # Create adapted model
            adapted = AdaptiveNetwork(self.state_dim, self.action_dim).to(DEVICE)
            adapted.load_state_dict(self.meta_model.state_dict())
            
            # Adapt on support set (inner loop)
            support_states = torch.FloatTensor(
                np.array([e.state for e in support_set])
            ).to(DEVICE)
            support_actions = torch.LongTensor([e.action for e in support_set]).to(DEVICE)
            support_rewards = torch.FloatTensor([e.reward for e in support_set]).to(DEVICE)
            
            if support_rewards.std() > 0:
                support_rewards = (support_rewards - support_rewards.mean()) / (support_rewards.std() + 1e-8)
            
            optimizer = optim.SGD(adapted.parameters(), lr=self.inner_lr)
            
            for _ in range(self.adaptation_steps):
                policy, _ = adapted(support_states)
                dist = torch.distributions.Categorical(policy)
                log_probs = dist.log_prob(support_actions)
                loss = -(log_probs * support_rewards).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Evaluate on query set (outer loop loss)
            query_states = torch.FloatTensor(
                np.array([e.state for e in query_set])
            ).to(DEVICE)
            query_actions = torch.LongTensor([e.action for e in query_set]).to(DEVICE)
            query_rewards = torch.FloatTensor([e.reward for e in query_set]).to(DEVICE)
            
            if query_rewards.std() > 0:
                query_rewards = (query_rewards - query_rewards.mean()) / (query_rewards.std() + 1e-8)
            
            policy, _ = adapted(query_states)
            dist = torch.distributions.Categorical(policy)
            log_probs = dist.log_prob(query_actions)
            query_loss = -(log_probs * query_rewards).mean()
            
            total_loss += query_loss
        
        if total_loss > 0:
            # Update meta-model
            self.meta_optimizer.zero_grad()
            total_loss.backward()
            self.meta_optimizer.step()
            
            self.meta_updates += 1
            logger.info(f"Meta-update #{self.meta_updates}, loss: {total_loss.item():.4f}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """ดึงสถิติการเรียนรู้"""
        
        regime_stats = {}
        for regime, buffer in self.regime_buffers.items():
            if len(buffer) > 0:
                wins = sum(1 for e in buffer if e.reward > 0)
                regime_stats[regime] = {
                    "experiences": len(buffer),
                    "win_rate": wins / len(buffer) if buffer else 0,
                    "model_exists": regime in self.regime_models,
                }
        
        return {
            "current_regime": self.current_regime,
            "adaptation_count": self.adaptation_count,
            "meta_updates": self.meta_updates,
            "regimes": regime_stats,
        }
    
    def save(self, name: str = "meta_learner"):
        """บันทึก models"""
        
        # Save meta-model
        torch.save({
            "meta_model": self.meta_model.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "adaptation_count": self.adaptation_count,
            "meta_updates": self.meta_updates,
        }, f"{self.model_dir}/{name}_meta.pt")
        
        # Save regime models
        for regime, model in self.regime_models.items():
            torch.save(model.state_dict(), f"{self.model_dir}/{name}_{regime}.pt")
        
        logger.info(f"Saved MetaLearner to {self.model_dir}")
    
    def load(self, name: str = "meta_learner"):
        """โหลด models"""
        
        meta_path = f"{self.model_dir}/{name}_meta.pt"
        if os.path.exists(meta_path):
            checkpoint = torch.load(meta_path, map_location=DEVICE, weights_only=False)
            self.meta_model.load_state_dict(checkpoint["meta_model"])
            self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
            self.adaptation_count = checkpoint.get("adaptation_count", 0)
            self.meta_updates = checkpoint.get("meta_updates", 0)
            logger.info("Loaded MetaLearner")
            return True
        return False


def create_meta_learner(state_dim: int = 11) -> MetaLearner:
    """สร้าง MetaLearner"""
    return MetaLearner(state_dim=state_dim)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   META-LEARNER TEST")
    print("="*60)
    
    np.random.seed(42)
    
    meta = create_meta_learner(state_dim=11)
    
    # Simulate experiences in different regimes
    regimes = [
        ("trending_up", 0.02, 0.01),
        ("trending_down", -0.02, 0.01),
        ("ranging", 0.0, 0.01),
        ("volatile", 0.005, 0.03),
    ]
    
    for regime_name, trend, vol in regimes:
        print(f"\nSimulating {regime_name} regime...")
        
        for i in range(10):
            state = np.random.randn(11).astype(np.float32)
            action = np.random.randint(3)
            reward = np.random.randn() * 0.5
            next_state = np.random.randn(11).astype(np.float32)
            
            meta.record_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                trend=trend + np.random.randn() * 0.005,
                volatility=vol + np.random.rand() * 0.005,
            )
    
    # Meta-update
    print("\nPerforming meta-update...")
    meta.meta_update()
    
    # Stats
    print("\nLearning Stats:")
    stats = meta.get_learning_stats()
    print(f"  Adaptations: {stats['adaptation_count']}")
    print(f"  Meta-updates: {stats['meta_updates']}")
    print(f"  Regimes:")
    for regime, info in stats['regimes'].items():
        print(f"    {regime}: {info['experiences']} exp, {info['win_rate']:.1%} WR")
    
    # Test action selection
    print("\nAction selection test:")
    test_state = np.random.randn(11).astype(np.float32)
    for regime in ["trending_up", "ranging"]:
        meta.current_regime = regime
        action, conf = meta.select_action(test_state)
        print(f"  {regime}: action={action}, confidence={conf:.1%}")
