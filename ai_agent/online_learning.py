"""
Online Learning Module
=======================
เรียนรู้จากการเทรดจริงแบบ Real-time

Features:
1. Experience Buffer - เก็บประสบการณ์จาก trade จริง
2. Incremental Learning - อัพเดท model ทีละน้อย
3. Auto Re-train - Re-train อัตโนมัติเมื่อถึงเงื่อนไข
4. Market Adaptation - ปรับตัวตามสภาพตลาด
"""

import numpy as np
import pandas as pd
import torch
import os
import json
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from loguru import logger

from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingState, DEVICE
from ai_agent.trade_memory import TradeMemory


@dataclass
class TradeExperience:
    """ประสบการณ์จาก trade จริง"""
    trade_id: str
    timestamp: datetime
    symbol: str
    
    # State when entering
    entry_state: np.ndarray
    entry_price: float
    
    # State when exiting
    exit_state: np.ndarray
    exit_price: float
    
    # Results
    action: int  # 1=LONG, 2=CLOSE
    pnl: float
    pnl_pct: float
    holding_bars: int
    
    # Market conditions
    volatility: float
    trend: float
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['entry_state'] = self.entry_state.tolist()
        d['exit_state'] = self.exit_state.tolist()
        return d
    
    @staticmethod
    def from_dict(d):
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        d['entry_state'] = np.array(d['entry_state'])
        d['exit_state'] = np.array(d['exit_state'])
        return TradeExperience(**d)


class ExperienceBuffer:
    """Buffer สำหรับเก็บประสบการณ์จาก trade จริง"""
    
    def __init__(self, capacity: int = 1000, save_path: str = "ai_agent/data/experiences.json"):
        self.capacity = capacity
        self.save_path = save_path
        self.buffer: deque = deque(maxlen=capacity)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._load()
    
    def add(self, experience: TradeExperience):
        """เพิ่มประสบการณ์ใหม่"""
        self.buffer.append(experience)
        self._save()
        logger.info(f"Added experience: {experience.trade_id}, P&L: ${experience.pnl:.2f}")
    
    def get_recent(self, n: int = 100) -> List[TradeExperience]:
        """ดึงประสบการณ์ล่าสุด"""
        return list(self.buffer)[-n:]
    
    def get_all(self) -> List[TradeExperience]:
        """ดึงทั้งหมด"""
        return list(self.buffer)
    
    def get_winning_trades(self) -> List[TradeExperience]:
        """ดึงเฉพาะ trade ที่ชนะ"""
        return [e for e in self.buffer if e.pnl > 0]
    
    def get_losing_trades(self) -> List[TradeExperience]:
        """ดึงเฉพาะ trade ที่แพ้"""
        return [e for e in self.buffer if e.pnl <= 0]
    
    def get_stats(self) -> Dict:
        """สถิติของประสบการณ์"""
        if not self.buffer:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0}
        
        wins = [e for e in self.buffer if e.pnl > 0]
        losses = [e for e in self.buffer if e.pnl <= 0]
        
        return {
            "total": len(self.buffer),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.buffer) if self.buffer else 0,
            "avg_pnl": np.mean([e.pnl for e in self.buffer]),
            "total_pnl": sum(e.pnl for e in self.buffer),
            "avg_rr": abs(np.mean([e.pnl for e in wins])) / abs(np.mean([e.pnl for e in losses])) if losses and wins else 0,
        }
    
    def _save(self):
        """บันทึกลงไฟล์"""
        data = [e.to_dict() for e in self.buffer]
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """โหลดจากไฟล์"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.buffer = deque([TradeExperience.from_dict(d) for d in data], maxlen=self.capacity)
                logger.info(f"Loaded {len(self.buffer)} experiences from {self.save_path}")
            except Exception as e:
                logger.warning(f"Failed to load experiences: {e}")
    
    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """
    Online Learning System
    =======================
    เรียนรู้จากการเทรดจริงแบบ Real-time
    """
    
    def __init__(
        self,
        agent: PPOAgentWalkForward = None,
        experience_buffer: ExperienceBuffer = None,
        learning_rate: float = 1e-4,
        min_experiences: int = 10,
        update_frequency: int = 5,
        model_dir: str = "ai_agent/models",
    ):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or create agent
        if agent is None:
            state_dim = 8 + 3
            self.agent = PPOAgentWalkForward(state_dim=state_dim)
            self.agent.load("best_wf")
        else:
            self.agent = agent
        
        # Experience buffer
        self.experience_buffer = experience_buffer or ExperienceBuffer()
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.min_experiences = min_experiences
        self.update_frequency = update_frequency
        
        # Stats
        self.updates_count = 0
        self.last_update = None
        
        logger.info("OnlineLearner initialized")
    
    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        entry_state: np.ndarray,
        entry_price: float,
        exit_state: np.ndarray,
        exit_price: float,
        action: int,
        holding_bars: int,
        volatility: float = 0.0,
        trend: float = 0.0,
    ):
        """บันทึก trade และเรียนรู้"""
        
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl = pnl_pct * 1000  # Assume $1000 per trade
        
        experience = TradeExperience(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            entry_state=entry_state,
            entry_price=entry_price,
            exit_state=exit_state,
            exit_price=exit_price,
            action=action,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_bars=holding_bars,
            volatility=volatility,
            trend=trend,
        )
        
        self.experience_buffer.add(experience)
        
        # Check if should update
        if len(self.experience_buffer) >= self.min_experiences:
            if len(self.experience_buffer) % self.update_frequency == 0:
                self._incremental_update()
        
        return experience
    
    def _incremental_update(self):
        """อัพเดท model จากประสบการณ์ล่าสุด"""
        
        experiences = self.experience_buffer.get_recent(self.update_frequency * 2)
        
        if len(experiences) < self.min_experiences:
            return
        
        logger.info(f"Performing incremental update with {len(experiences)} experiences...")
        
        # Build training data from experiences
        states = []
        actions = []
        rewards = []
        
        for exp in experiences:
            states.append(exp.entry_state)
            actions.append(exp.action)
            
            # Calculate reward with sniper shaping
            reward = self._calculate_reward(exp)
            rewards.append(reward)
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        
        # Simple policy gradient update
        self.agent.policy.train()
        
        for _ in range(3):  # Few iterations
            action_probs, values = self.agent.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Normalize rewards
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Policy loss
            policy_loss = -(log_probs * rewards_norm).mean()
            
            # Value loss
            value_loss = torch.nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 0.5)
            self.agent.optimizer.step()
        
        self.updates_count += 1
        self.last_update = datetime.now()
        
        # Save updated model
        self.agent.save("online_updated")
        
        logger.info(f"Incremental update #{self.updates_count} complete. Loss: {loss.item():.4f}")
    
    def _calculate_reward(self, exp: TradeExperience) -> float:
        """คำนวณ reward แบบ sniper style"""
        reward = 0.0
        
        if exp.pnl > 0:
            # Win bonus
            if exp.pnl_pct > 0.02:  # >2% gain
                reward = exp.pnl_pct * 50
            elif exp.pnl_pct > 0.01:  # >1% gain
                reward = exp.pnl_pct * 20
            else:
                reward = exp.pnl_pct * 10
            
            # Quick win bonus
            if exp.holding_bars < 10 and exp.pnl_pct > 0.01:
                reward += 0.1
        else:
            # Loss penalty
            if exp.pnl_pct < -0.02:  # >2% loss
                reward = exp.pnl_pct * 100
            elif exp.pnl_pct < -0.01:  # 1-2% loss
                reward = exp.pnl_pct * 50
            else:
                reward = exp.pnl_pct * 20
        
        return reward
    
    def full_retrain(self, data: pd.DataFrame = None, episodes: int = 100):
        """Re-train เต็มรูปแบบจากข้อมูลใหม่"""
        
        logger.info("Starting full re-train...")
        
        # Combine historical data with experience data
        if data is None:
            data = pd.read_csv("data/training/GOLD_H1.csv")
            data.columns = [c.lower() for c in data.columns]
        
        # Re-train with walk-forward
        history = self.agent.walk_forward_train(
            data,
            n_folds=3,
            episodes_per_fold=episodes // 3,
        )
        
        # Save new model
        self.agent.save("retrained")
        
        logger.info("Full re-train complete!")
        
        return history
    
    def get_learning_stats(self) -> Dict:
        """สถิติการเรียนรู้"""
        exp_stats = self.experience_buffer.get_stats()
        
        return {
            "total_experiences": len(self.experience_buffer),
            "updates_count": self.updates_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "experience_stats": exp_stats,
        }
    
    def should_retrain(self) -> bool:
        """ตรวจสอบว่าควร re-train หรือไม่"""
        
        # Re-train if:
        # 1. มี experience มากกว่า 100
        # 2. Win rate ต่ำกว่า 40%
        # 3. ไม่มี update มานาน (7 วัน)
        
        stats = self.experience_buffer.get_stats()
        
        if stats['total'] >= 100:
            if stats['win_rate'] < 0.4:
                logger.info("Re-train recommended: Win rate below 40%")
                return True
        
        if self.last_update:
            days_since_update = (datetime.now() - self.last_update).days
            if days_since_update >= 7 and stats['total'] >= 50:
                logger.info(f"Re-train recommended: {days_since_update} days since last update")
                return True
        
        return False


# Convenience function
def create_online_learner() -> OnlineLearner:
    """สร้าง OnlineLearner"""
    return OnlineLearner()


if __name__ == "__main__":
    # Test
    logger.remove()
    logger.add(lambda m: print(m, end=""), format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ONLINE LEARNING TEST")
    print("="*60)
    
    learner = create_online_learner()
    
    # Simulate some trades
    for i in range(10):
        state = np.random.randn(11).astype(np.float32)
        entry = 2000 + np.random.randn() * 10
        exit = entry + np.random.randn() * 20
        
        learner.record_trade(
            trade_id=f"TEST_{i}",
            symbol="XAUUSD",
            entry_state=state,
            entry_price=entry,
            exit_state=state,
            exit_price=exit,
            action=1,
            holding_bars=np.random.randint(1, 50),
        )
    
    print("\n" + "="*60)
    print("   LEARNING STATS")
    print("="*60)
    
    stats = learner.get_learning_stats()
    print(f"Total Experiences: {stats['total_experiences']}")
    print(f"Updates: {stats['updates_count']}")
    print(f"Win Rate: {stats['experience_stats']['win_rate']:.1%}")
    print(f"Avg P&L: ${stats['experience_stats']['avg_pnl']:.2f}")
