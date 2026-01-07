"""
Enhanced Online Learning Module
================================
ระบบเรียนรู้แบบ Real-time ที่ปรับปรุงแล้ว

Features:
1. Prioritized Experience Replay - เรียนรู้จาก trades ที่ผิดพลาดมากกว่าปกติ
2. Integrated Error Analyzer - วิเคราะห์และแก้ไขข้อผิดพลาดอัตโนมัติ
3. Meta-Learning - ปรับตัวเข้ากับสภาพตลาดใหม่ภายใน 5-10 trades
4. Curiosity-Driven Exploration - ค้นหารูปแบบการเทรดใหม่
5. Self-Correction - ปรับพฤติกรรมไม่ให้ผิดซ้ำ
"""

import numpy as np
import pandas as pd
import torch
import os
import json
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger

# Import sub-modules
from ai_agent.error_analyzer import ErrorAnalyzer, create_error_analyzer
from ai_agent.self_corrector import SelfCorrector, create_self_corrector
from ai_agent.meta_learner import MetaLearner, create_meta_learner
from ai_agent.curiosity_module import CuriosityModule, create_curiosity_module


# Try to import PPO agent
try:
    from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingState, DEVICE
except ImportError:
    from ai_agent.ppo_agent import PPOAgent as PPOAgentWalkForward, TradingState, DEVICE


@dataclass
class PrioritizedExperience:
    """ประสบการณ์พร้อม priority"""
    trade_id: str
    timestamp: datetime
    symbol: str
    
    # State and action
    entry_state: np.ndarray
    exit_state: np.ndarray
    action: int
    
    # Results
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    
    # Market conditions
    volatility: float
    trend: float
    regime: str
    
    # Priority for replay
    priority: float = 1.0
    td_error: float = 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['entry_state'] = self.entry_state.tolist()
        d['exit_state'] = self.exit_state.tolist()
        return d
    
    @staticmethod
    def from_dict(d: Dict) -> 'PrioritizedExperience':
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        d['entry_state'] = np.array(d['entry_state'])
        d['exit_state'] = np.array(d['exit_state'])
        return PrioritizedExperience(**d)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    - เก็บ experiences พร้อม priority
    - Sample โดยให้น้ำหนักกับ experiences ที่มี TD error สูง
    - อัพเดต priority หลังจากเรียนรู้
    """
    
    def __init__(
        self,
        capacity: int = 2000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,  # Importance sampling exponent
        beta_increment: float = 0.001,
        save_path: str = "ai_agent/data/prioritized_experiences.json",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.save_path = save_path
        
        self.buffer: List[PrioritizedExperience] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._load()
    
    def add(self, experience: PrioritizedExperience):
        """เพิ่ม experience พร้อมคำนวณ priority"""
        
        # Higher priority for losses (to learn from mistakes)
        base_priority = 1.0
        
        if experience.pnl < 0:
            # Higher priority for losses
            base_priority = abs(experience.pnl_pct) * 10 + 1
        else:
            # Normal priority for wins
            base_priority = 1.0 + experience.pnl_pct * 2
        
        # Boost priority for unusual situations
        if experience.volatility > 0.02:
            base_priority *= 1.5
        
        experience.priority = base_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Update priority array
        max_priority = max(self.priorities.max(), base_priority)
        self.priorities[self.position] = max_priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
        
        self._save()
    
    def sample(self, batch_size: int) -> Tuple[List[PrioritizedExperience], List[int], np.ndarray]:
        """
        Sample experiences โดยใช้ priority weighting
        
        Returns:
            (experiences, indices, importance_weights)
        """
        
        if len(self.buffer) == 0:
            return [], [], np.array([])
        
        # Calculate sampling probabilities
        current_size = len(self.buffer)
        priorities = self.priorities[:current_size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        batch_size = min(batch_size, current_size)
        indices = np.random.choice(current_size, batch_size, p=probs, replace=False)
        
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (current_size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, list(indices), weights
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """อัพเดต priorities หลังจากเรียนรู้"""
        
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.buffer):
                self.buffer[idx].td_error = td_error
                self.priorities[idx] = (abs(td_error) + 0.01) ** self.alpha
    
    def get_losing_experiences(self, n: int = 50) -> List[PrioritizedExperience]:
        """ดึง experiences ที่ขาดทุน"""
        losers = [e for e in self.buffer if e.pnl < 0]
        return sorted(losers, key=lambda x: x.pnl)[:n]
    
    def get_winning_experiences(self, n: int = 50) -> List[PrioritizedExperience]:
        """ดึง experiences ที่กำไร"""
        winners = [e for e in self.buffer if e.pnl > 0]
        return sorted(winners, key=lambda x: -x.pnl)[:n]
    
    def get_stats(self) -> Dict:
        """ดึงสถิติ"""
        if not self.buffer:
            return {"total": 0}
        
        pnls = [e.pnl for e in self.buffer]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            "total": len(self.buffer),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.buffer) if self.buffer else 0,
            "avg_priority": np.mean(self.priorities[:len(self.buffer)]),
            "total_pnl": sum(pnls),
        }
    
    def _save(self):
        """บันทึก buffer"""
        data = [e.to_dict() for e in self.buffer]
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """โหลด buffer"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.buffer = [PrioritizedExperience.from_dict(d) for d in data]
                
                # Rebuild priority array
                for i, exp in enumerate(self.buffer):
                    self.priorities[i] = exp.priority ** self.alpha
                
                self.position = len(self.buffer) % self.capacity
                logger.info(f"Loaded {len(self.buffer)} prioritized experiences")
            except Exception as e:
                logger.warning(f"Failed to load experiences: {e}")
    
    def __len__(self):
        return len(self.buffer)


class EnhancedOnlineLearner:
    """
    Enhanced Online Learning System
    ================================
    ระบบเรียนรู้แบบ Real-time ที่รวม:
    - Prioritized Experience Replay
    - Error Analysis & Self-Correction
    - Meta-Learning (MAML)
    - Curiosity-Driven Exploration
    """
    
    def __init__(
        self,
        agent=None,
        state_dim: int = 11,
        learning_rate: float = 1e-4,
        min_experiences: int = 10,
        update_frequency: int = 5,
        model_dir: str = "ai_agent/models",
    ):
        self.state_dim = state_dim
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or create PPO agent
        if agent is None:
            self.agent = PPOAgentWalkForward(state_dim=state_dim)
            try:
                self.agent.load("best_wf")
            except:
                logger.info("No existing model found, using new agent")
        else:
            self.agent = agent
        
        # Prioritized Experience Buffer
        self.buffer = PrioritizedReplayBuffer()
        
        # Sub-modules
        self.error_analyzer = create_error_analyzer()
        self.self_corrector = create_self_corrector(self.error_analyzer)
        self.meta_learner = create_meta_learner(state_dim=state_dim)
        self.curiosity = create_curiosity_module(state_dim=state_dim)
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.min_experiences = min_experiences
        self.update_frequency = update_frequency
        
        # Stats
        self.updates_count = 0
        self.last_update = None
        self.total_trades = 0
        self.consecutive_losses = 0
        
        logger.info("EnhancedOnlineLearner initialized with all sub-modules")
    
    def should_trade(
        self,
        confidence: float,
        volatility: float,
        trend: float,
        regime: str,
        rsi: float = 50.0,
    ) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าควรเทรดหรือไม่ โดยใช้ self-corrector
        """
        return self.self_corrector.should_trade(
            confidence, volatility, trend, regime, rsi
        )
    
    def get_position_size(
        self,
        capital: float,
        volatility: float,
        confidence: float,
    ) -> float:
        """คำนวณขนาด position โดยใช้ self-corrector"""
        return self.self_corrector.get_position_size(
            capital, volatility, confidence, self.consecutive_losses
        )
    
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
        regime: str = "unknown",
        rsi: float = 50.0,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        บันทึก trade และเรียนรู้จากมัน
        
        Returns:
            Learning results and recommendations
        """
        
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl = pnl_pct * 1000  # Assume $1000 per trade base
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Create prioritized experience
        experience = PrioritizedExperience(
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            entry_state=entry_state.astype(np.float32),
            exit_state=exit_state.astype(np.float32),
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_bars=holding_bars,
            volatility=volatility,
            trend=trend,
            regime=regime,
        )
        
        self.buffer.add(experience)
        self.total_trades += 1
        
        # Record to error analyzer if loss
        if pnl < 0:
            self.error_analyzer.record_error(
                trade_id=trade_id,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                holding_bars=holding_bars,
                volatility=volatility,
                trend=trend,
                regime=regime,
                rsi=rsi,
                confidence=confidence,
            )
        
        # Update self-corrector
        self.self_corrector.record_trade_result(pnl > 0, pnl)
        
        # Record for meta-learning
        reward = self._calculate_reward(pnl, pnl_pct, holding_bars)
        self.meta_learner.record_experience(
            state=entry_state,
            action=action,
            reward=reward,
            next_state=exit_state,
            trend=trend,
            volatility=volatility,
        )
        
        # Update regime
        self.meta_learner.update_regime(trend, volatility)
        
        # Record for curiosity
        self.curiosity.record_experience(entry_state, action, exit_state, pnl)
        
        # Check if should do learning update
        results = {"action": "recorded"}
        
        if len(self.buffer) >= self.min_experiences:
            if len(self.buffer) % self.update_frequency == 0:
                update_result = self._prioritized_update()
                results["learning"] = update_result
        
        # Auto-correct if patterns detected
        if self.total_trades % 10 == 0:
            corrections = self.self_corrector.analyze_and_correct()
            if corrections:
                results["corrections"] = corrections
        
        # Meta-update periodically
        if self.total_trades % 20 == 0:
            self.meta_learner.meta_update()
            results["meta_update"] = True
        
        return results
    
    def _calculate_reward(
        self,
        pnl: float,
        pnl_pct: float,
        holding_bars: int,
    ) -> float:
        """คำนวณ reward แบบ sniper style + curiosity bonus"""
        
        reward = 0.0
        
        if pnl > 0:
            # Win bonus
            if pnl_pct > 0.02:  # > 2% gain
                reward = pnl_pct * 50
            elif pnl_pct > 0.01:  # > 1% gain
                reward = pnl_pct * 20
            else:
                reward = pnl_pct * 10
            
            # Quick win bonus
            if holding_bars < 10 and pnl_pct > 0.01:
                reward += 0.1
        else:
            # Loss penalty
            if pnl_pct < -0.02:  # > 2% loss
                reward = pnl_pct * 100
            elif pnl_pct < -0.01:  # 1-2% loss
                reward = pnl_pct * 50
            else:
                reward = pnl_pct * 20
        
        return reward
    
    def _prioritized_update(self) -> Dict:
        """อัพเดต model โดยใช้ prioritized replay"""
        
        # Sample with priority
        experiences, indices, weights = self.buffer.sample(
            batch_size=min(32, len(self.buffer))
        )
        
        if len(experiences) < self.min_experiences:
            return {"status": "insufficient_data"}
        
        logger.info(f"Performing prioritized update with {len(experiences)} experiences...")
        
        # Prepare data
        states = torch.FloatTensor(
            np.array([e.entry_state for e in experiences])
        ).to(DEVICE)
        actions = torch.LongTensor([e.action for e in experiences]).to(DEVICE)
        
        rewards = []
        for exp in experiences:
            base_reward = self._calculate_reward(exp.pnl, exp.pnl_pct, exp.holding_bars)
            
            # Add curiosity bonus
            intrinsic = self.curiosity.compute_intrinsic_reward(
                exp.entry_state, exp.action, exp.exit_state
            )
            
            rewards.append(base_reward + intrinsic)
        
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        weights_t = torch.FloatTensor(weights).to(DEVICE)
        
        # Train policy
        self.agent.policy.train()
        
        td_errors = []
        
        for _ in range(3):  # Few iterations
            action_probs, values = self.agent.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Normalize rewards
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Calculate TD errors for priority update
            with torch.no_grad():
                td_errors = (rewards - values.squeeze()).abs().tolist()
            
            # Weighted policy loss
            policy_loss = -(log_probs * rewards_norm * weights_t).mean()
            
            # Value loss
            value_loss = torch.nn.MSELoss()(values.squeeze(), rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 0.5)
            self.agent.optimizer.step()
        
        # Update priorities
        self.buffer.update_priorities(indices, td_errors)
        
        self.updates_count += 1
        self.last_update = datetime.now()
        
        # Save updated model
        self.agent.save("online_enhanced")
        
        logger.info(f"Prioritized update #{self.updates_count} complete. Loss: {loss.item():.4f}")
        
        return {
            "status": "success",
            "loss": loss.item(),
            "experiences": len(experiences),
            "update_number": self.updates_count,
        }
    
    def select_action(
        self,
        state: np.ndarray,
        volatility: float = 0.0,
        trend: float = 0.0,
    ) -> Tuple[int, float]:
        """
        เลือก action โดยใช้ meta-learner (regime-aware)
        
        Returns:
            (action, confidence)
        """
        # Update regime
        self.meta_learner.update_regime(trend, volatility)
        
        # Use meta-learner for regime-specific action
        action, confidence = self.meta_learner.select_action(state)
        
        # Add exploration bonus
        exploration_bonus = self.curiosity.get_exploration_bonus(state)
        
        # Sometimes explore more if in novel state
        if exploration_bonus > 0.05 and np.random.rand() < 0.1:
            action = np.random.randint(3)
            logger.debug("Exploration override due to novel state")
        
        return action, confidence
    
    def should_retrain(self) -> Tuple[bool, str]:
        """ตรวจสอบว่าควร retrain หรือไม่"""
        
        stats = self.buffer.get_stats()
        
        # Check win rate
        if stats['total'] >= 100:
            if stats['win_rate'] < 0.4:
                return True, "Win rate below 40%"
        
        # Check if too many consecutive losses
        if self.consecutive_losses >= 5:
            return True, f"{self.consecutive_losses} consecutive losses"
        
        # Check time since last update
        if self.last_update:
            days_since = (datetime.now() - self.last_update).days
            if days_since >= 7 and stats['total'] >= 50:
                return True, f"{days_since} days since last update"
        
        # Check error patterns
        report = self.error_analyzer.get_analysis_report()
        if report.get('patterns_detected', 0) >= 3:
            uncorrected = report['patterns_detected'] - report.get('corrections_active', 0)
            if uncorrected >= 2:
                return True, f"{uncorrected} uncorrected error patterns"
        
        return False, ""
    
    def get_learning_stats(self) -> Dict:
        """ดึงสถิติการเรียนรู้"""
        
        buffer_stats = self.buffer.get_stats()
        error_report = self.error_analyzer.get_analysis_report()
        meta_stats = self.meta_learner.get_learning_stats()
        curiosity_stats = self.curiosity.get_stats()
        corrector_status = self.self_corrector.get_status()
        
        return {
            "total_trades": self.total_trades,
            "updates_count": self.updates_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "consecutive_losses": self.consecutive_losses,
            "buffer": buffer_stats,
            "errors": {
                "total": error_report.get('total_errors', 0),
                "patterns": error_report.get('patterns_detected', 0),
            },
            "meta_learning": meta_stats,
            "curiosity": curiosity_stats,
            "self_correction": corrector_status,
        }
    
    def save_all(self, name: str = "enhanced"):
        """บันทึกทุก models"""
        
        self.agent.save(f"{name}_agent")
        self.meta_learner.save(f"{name}_meta")
        self.curiosity.save(f"{name}_curiosity")
        
        # Save stats
        stats_path = f"{self.model_dir}/{name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "total_trades": self.total_trades,
                "updates_count": self.updates_count,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "consecutive_losses": self.consecutive_losses,
            }, f, indent=2)
        
        logger.info(f"Saved all models with prefix '{name}'")
    
    def load_all(self, name: str = "enhanced"):
        """โหลดทุก models"""
        
        try:
            self.agent.load(f"{name}_agent")
        except:
            pass
        
        self.meta_learner.load(f"{name}_meta")
        self.curiosity.load(f"{name}_curiosity")
        
        stats_path = f"{self.model_dir}/{name}_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.total_trades = stats.get("total_trades", 0)
            self.updates_count = stats.get("updates_count", 0)
            if stats.get("last_update"):
                self.last_update = datetime.fromisoformat(stats["last_update"])


def create_enhanced_online_learner() -> EnhancedOnlineLearner:
    """สร้าง EnhancedOnlineLearner"""
    return EnhancedOnlineLearner()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ENHANCED ONLINE LEARNING TEST")
    print("="*60)
    
    np.random.seed(42)
    
    learner = create_enhanced_online_learner()
    
    # Simulate trades
    print("\nSimulating trades...")
    
    for i in range(20):
        state = np.random.randn(11).astype(np.float32)
        entry = 2000 + np.random.randn() * 10
        exit = entry + np.random.randn() * 30
        
        # Simulate market conditions
        volatility = np.random.rand() * 0.03
        trend = np.random.randn() * 0.02
        regime = np.random.choice(["trending_up", "trending_down", "ranging"])
        
        # Check if should trade
        can_trade, reason = learner.should_trade(
            confidence=0.7 + np.random.rand() * 0.2,
            volatility=volatility,
            trend=trend,
            regime=regime,
        )
        
        if can_trade or i < 10:  # Force some trades for testing
            result = learner.record_trade(
                trade_id=f"TEST_{i}",
                symbol="XAUUSD",
                entry_state=state,
                entry_price=entry,
                exit_state=state + np.random.randn(11).astype(np.float32) * 0.1,
                exit_price=exit,
                action=1,
                holding_bars=np.random.randint(1, 50),
                volatility=volatility,
                trend=trend,
                regime=regime,
            )
            
            if "corrections" in result:
                print(f"  Trade {i}: Corrections applied!")
    
    print("\n" + "="*60)
    print("   LEARNING STATS")
    print("="*60)
    
    stats = learner.get_learning_stats()
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Updates: {stats['updates_count']}")
    print(f"Buffer: {stats['buffer']['total']} experiences")
    print(f"Win Rate: {stats['buffer']['win_rate']:.1%}")
    print(f"Error Patterns: {stats['errors']['patterns']}")
    print(f"Meta Adaptations: {stats['meta_learning']['adaptation_count']}")
    print(f"Novel Patterns: {stats['curiosity']['total_patterns']}")
    
    # Check if should retrain
    should, reason = learner.should_retrain()
    if should:
        print(f"\n⚠️ Retrain recommended: {reason}")
    else:
        print("\n✓ Model is performing well")
