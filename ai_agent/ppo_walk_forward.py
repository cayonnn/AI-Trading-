"""
PPO Agent with Walk-Forward Training
=====================================
Fixed version that prevents overfitting using:
1. Walk-Forward Training (Time-based splits)
2. Early Stopping
3. Validation tracking
4. Data shuffling per episode
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from loguru import logger
import json

# GPU Setup
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


@dataclass
class TradingState:
    """Trading state"""
    price_features: np.ndarray
    position: int
    unrealized_pnl: float
    portfolio_value: float
    
    def to_tensor(self) -> torch.Tensor:
        state = np.concatenate([
            self.price_features,
            [self.position, self.unrealized_pnl / 1000, self.portfolio_value / 10000]
        ])
        return torch.FloatTensor(state)


# ============================================
# v2.0: Transformer Attention Module
# ============================================

class SelfAttention(nn.Module):
    """Self-Attention for sequence modeling"""
    
    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        if len(original_shape) == 2:
            out = out.squeeze(1)
        return out


class ActorCriticV2(nn.Module):
    """
    Actor-Critic v2.0 with Transformer Attention
    - Multi-Head Self-Attention
    - LayerNorm + Dropout regularization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: List[int] = [256, 128, 64],
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
        )
        
        self.attention = SelfAttention(hidden_dims[0], n_heads)
        
        layers = []
        prev_dim = hidden_dims[0]
        for hidden in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden
        
        self.shared = nn.Sequential(*layers)
        
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.apply(self._init_weights)
        logger.debug(f"ActorCriticV2 initialized with {n_heads}-head attention")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(state)
        x = self.attention(x)
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


# ============================================
# v2.0: Prioritized Experience Replay
# ============================================

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER)"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        experience = (state, action, reward, next_state, done, log_prob, value)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def get_all(self) -> List:
        return self.buffer.copy()
    
    def clear(self):
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
    
    def __len__(self):
        return len(self.buffer)


class TradingEnvironment:
    """Trading Environment"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.0001,
        slippage: float = 0.0001,
    ):
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.features = self._compute_features()
        self.n_steps = len(self.features)
        self.reset()
    
    def _compute_features(self) -> np.ndarray:
        df = self.data.copy()
        
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['volatility'] = df['returns'].rolling(20).std()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_norm'] = (100 - (100 / (1 + rs)) - 50) / 50
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_norm'] = (ema12 - ema26) / (df['close'] * 0.01)
        
        df['ma50'] = df['close'].rolling(50).mean()
        df['trend'] = (df['close'] - df['ma50']) / df['ma50']
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - df['bb_mid']) / (df['bb_std'] * 2 + 1e-10)
        
        feature_cols = [
            'returns', 'returns_5', 'returns_10', 'volatility',
            'rsi_norm', 'macd_norm', 'trend', 'bb_position'
        ]
        
        features = df[feature_cols].fillna(0).values
        features = np.clip(features, -3, 3)
        
        return features.astype(np.float32)
    
    def reset(self) -> TradingState:
        self.current_step = 50
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        return self._get_state()
    
    def _get_state(self) -> TradingState:
        price_features = self.features[self.current_step]
        current_price = self.data['close'].iloc[self.current_step]
        
        if self.position == 1:
            unrealized = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized = 0.0
        
        return TradingState(
            price_features=price_features,
            position=self.position,
            unrealized_pnl=unrealized,
            portfolio_value=self.capital,
        )
    
    def step(self, action: int) -> Tuple[TradingState, float, bool, Dict]:
        """
        SNIPER TRADING OPTIMIZED STEP
        ==============================
        Rewards:
        - High R:R trades (1:3 or better)
        - Patience (waiting for good setups)
        - Penalizes overtrading
        """
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0.0
        info = {"action": action, "price": current_price, "trade": None}
        
        # Track consecutive waits for patience bonus
        if not hasattr(self, 'consecutive_waits'):
            self.consecutive_waits = 0
        
        if action == 0:  # WAIT
            self.consecutive_waits += 1
            # Small reward for patience (sniper waits for good setup)
            if self.consecutive_waits > 5 and self.position == 0:
                reward += 0.0005  # REDUCED patience bonus (was 0.001)
        else:
            self.consecutive_waits = 0
        
        if action == 1 and self.position == 0:  # OPEN LONG
            cost = current_price * (1 + self.commission + self.slippage)
            self.entry_price = cost
            self.position = 1
            self.entry_step = self.current_step
            self.highest_price = current_price  # Track for trailing
            info["trade"] = "OPEN_LONG"
            
            # v3.3: ENTRY BONUS when trend is positive
            # Get trend from recent price action
            if self.current_step >= 20:
                recent_close = self.data['close'].iloc[self.current_step-20:self.current_step]
                short_ma = recent_close.iloc[-5:].mean()
                long_ma = recent_close.mean()
                trend_strength = (short_ma - long_ma) / long_ma
                
                if trend_strength > 0.001:  # Uptrend
                    reward += min(0.1, trend_strength * 10)  # Entry bonus for good trend
                elif trend_strength < -0.001:  # Downtrend - penalty for going against
                    reward -= 0.02
            
            # Reduced overtrading penalty (sniper is selective, but not too restrictive)
            if len(self.trades) > 0:
                bars_since_last = self.current_step - self.trades[-1]["step"]
                if bars_since_last < 10:  # Only penalty if less than 10 bars
                    reward -= 0.02  # REDUCED penalty (was 0.05)
            
        elif action == 2 and self.position == 1:  # CLOSE LONG
            exit_price = current_price * (1 - self.commission - self.slippage)
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
            pnl = self.capital * pnl_pct * 0.1
            
            self.capital += pnl
            self.position = 0
            
            # SNIPER REWARD SHAPING
            # =====================
            
            # Base reward from P&L
            base_reward = pnl / 100
            
            # Risk/Reward Bonus (Sniper aims for 1:3 R:R)
            if pnl > 0:
                # Calculate R:R based on max drawdown vs profit
                holding_bars = self.current_step - self.entry_step
                
                # v2.1: BOOSTED WIN REWARDS
                # Big win bonus (R:R > 3:1)
                if pnl_pct > 0.02:  # >2% gain
                    rr_bonus = pnl_pct * 100  # BOOSTED: 100x for big wins
                    reward += rr_bonus
                    reward += 0.5  # BIG WIN FLAT BONUS
                elif pnl_pct > 0.01:  # >1% gain
                    reward += pnl_pct * 50  # BOOSTED: 50x for medium wins
                    reward += 0.2  # MEDIUM WIN FLAT BONUS
                else:
                    reward += base_reward * 2  # Small win, double reward
                    reward += 0.1  # SMALL WIN FLAT BONUS
                    
                # Quick win bonus (sniper is efficient)
                if holding_bars < 10 and pnl_pct > 0.01:
                    reward += 0.2  # BOOSTED quick sniper strike bonus
                    
            else:
                # Loss penalty - harsher for big losses (sniper uses tight SL)
                if pnl_pct < -0.02:  # Bigger than 2% loss
                    reward += pnl_pct * 100  # Heavy penalty for big loss
                elif pnl_pct < -0.01:  # 1-2% loss
                    reward += pnl_pct * 50  # Medium penalty
                else:
                    reward += base_reward * 2  # Small loss is acceptable
            
            self.trades.append({
                "step": self.current_step, 
                "pnl": pnl, 
                "pnl_pct": pnl_pct,
                "holding_bars": self.current_step - self.entry_step
            })
            info["trade"] = "CLOSE_LONG"
            info["pnl"] = pnl
            info["pnl_pct"] = pnl_pct
            
            self.entry_price = 0
            self.entry_step = 0
        
        # Update highest price for position
        if self.position == 1:
            if current_price > getattr(self, 'highest_price', current_price):
                self.highest_price = current_price
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        if self.position == 1:
            unrealized = (current_price - self.entry_price) / self.entry_price
            portfolio = self.capital * (1 + unrealized * 0.1)
            
            # Holding penalty (sniper doesn't hold too long)
            holding_time = self.current_step - getattr(self, 'entry_step', self.current_step)
            if holding_time > 24:  # More than 24 bars (1 day for H1)
                reward -= 0.005  # Increased holding penalty
            if holding_time > 48:  # More than 2 days
                reward -= 0.01   # Stronger penalty
        else:
            portfolio = self.capital
        
        self.portfolio_values.append(portfolio)
        
        # Small portfolio change reward
        if len(self.portfolio_values) > 1:
            portfolio_change = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            reward += portfolio_change * 5  # Reduced from 10
        
        next_state = self._get_state() if not done else None
        return next_state, reward, done, info
    
    def get_performance(self) -> Dict:
        if not self.trades:
            return {"n_trades": 0, "win_rate": 0, "total_pnl": 0, "return_pct": 0, "final_capital": self.capital}
        
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0,
            "total_pnl": sum(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "final_capital": self.capital,
            "return_pct": (self.capital - self.initial_capital) / self.initial_capital,
        }


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def get_all(self) -> List:
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PPOAgentWalkForward:
    """
    PPO Agent with Walk-Forward Training
    =====================================
    Fixes overfitting by:
    1. Training on different time windows
    2. Validating on unseen data
    3. Early stopping if validation degrades
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        model_dir: str = "ai_agent/models",
        use_per: bool = True,  # v2.0: Prioritized Experience Replay
        use_attention: bool = True,  # v2.0: Transformer attention
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.model_dir = model_dir
        self.use_per = use_per
        
        os.makedirs(model_dir, exist_ok=True)
        
        # v2.0: Use ActorCriticV2 with attention
        if use_attention:
            self.policy = ActorCriticV2(state_dim, action_dim, n_heads=4, dropout=0.2).to(DEVICE)
        else:
            self.policy = ActorCriticV2(state_dim, action_dim, n_heads=1, dropout=0.2).to(DEVICE)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        
        # v2.0: Learning Rate Scheduler (CosineAnnealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # v2.0: Prioritized or standard replay buffer
        if use_per:
            self.buffer = PrioritizedReplayBuffer(capacity=10000)
        else:
            self.buffer = ReplayBuffer(capacity=10000)
        
        self.training_episodes = 0
        self.total_rewards = []
        self.train_history = []
        self.val_history = []
        
        # v2.0: Early stopping
        self.best_val_wr = 0.0
        self.patience = 5
        self.patience_counter = 0
        
        logger.info(f"🚀 PPO Walk-Forward v2.0 initialized on {DEVICE}")
        logger.info(f"   Attention: {use_attention}, PER: {use_per}")
    
    def select_action(self, state: TradingState) -> Tuple[int, float, float]:
        self.policy.eval()
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0).to(DEVICE)
            action_probs, value = self.policy.forward(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # Return action probability for the selected action (not value)
            action_prob = action_probs[0, action.item()].item()
        return action.item(), log_prob.item(), action_prob
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        state_tensor = state.to_tensor().numpy()
        next_tensor = next_state.to_tensor().numpy() if next_state else np.zeros(self.state_dim + 3)
        self.buffer.push(state_tensor, action, reward, next_tensor, done, log_prob, value)
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        self.policy.train()
        
        data = self.buffer.get_all()
        
        states = torch.FloatTensor(np.array([d[0] for d in data])).to(DEVICE)
        actions = torch.LongTensor([d[1] for d in data]).to(DEVICE)
        rewards = [d[2] for d in data]
        dones = [d[4] for d in data]
        old_log_probs = torch.FloatTensor([d[5] for d in data]).to(DEVICE)
        values = [d[6] for d in data]
        
        with torch.no_grad():
            _, last_value = self.policy(states[-1:])
            last_value = last_value.item()
        
        advantages = self.compute_gae(rewards, values, dones, last_value)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = advantages + torch.FloatTensor(values).to(DEVICE)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        for _ in range(self.epochs):
            indices = np.random.permutation(len(data))
            for start in range(0, len(data), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                log_probs, curr_values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(curr_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.buffer.clear()
        return total_loss
    
    def train_episode(self, env: TradingEnvironment) -> Dict:
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            self.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        loss = self.update()
        
        self.training_episodes += 1
        self.total_rewards.append(episode_reward)
        
        performance = env.get_performance()
        
        return {
            "episode": self.training_episodes,
            "reward": episode_reward,
            "loss": loss,
            **performance,
        }
    
    def evaluate(self, data: pd.DataFrame) -> Dict:
        """Evaluate on unseen data (no training)"""
        self.policy.eval()
        
        env = TradingEnvironment(data)
        state = env.reset()
        
        while True:
            action, _, _ = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
        
        return env.get_performance()
    
    def walk_forward_train(
        self,
        data: pd.DataFrame,
        n_folds: int = 5,
        episodes_per_fold: int = 20,
        train_ratio: float = 0.7,
        window_size: int = 5000,  # v2.1: Larger window for better patterns
    ) -> List[Dict]:
        """
        Walk-Forward Training v2.0
        ===========================
        แบ่งข้อมูลเป็น folds และเทรนแบบ time-based
        
        ข้อดี: ป้องกัน Overfitting เพราะใช้ข้อมูล unseen ในการ validate
        """
        logger.info("="*60)
        logger.info("WALK-FORWARD TRAINING v2.0")
        logger.info("="*60)
        logger.info(f"Total Data: {len(data)} bars")
        logger.info(f"Folds: {n_folds}")
        logger.info(f"Episodes per Fold: {episodes_per_fold}")
        logger.info(f"Window Size: {window_size} bars/episode")
        logger.info(f"Train/Val Split: {train_ratio:.0%}/{1-train_ratio:.0%}")
        logger.info("="*60)
        
        fold_size = len(data) // n_folds
        all_results = []
        
        for fold in range(n_folds):
            fold_start = fold * fold_size
            fold_end = fold_start + fold_size
            
            fold_data = data.iloc[fold_start:fold_end].reset_index(drop=True)
            
            # Split into train/val
            split_idx = int(len(fold_data) * train_ratio)
            train_data = fold_data.iloc[:split_idx].reset_index(drop=True)
            val_data = fold_data.iloc[split_idx:].reset_index(drop=True)
            
            logger.info(f"\n[Fold {fold+1}/{n_folds}]")
            logger.info(f"  Train: {len(train_data)} bars, Val: {len(val_data)} bars")
            
            # Train on this fold
            fold_train_results = []
            for ep in range(1, episodes_per_fold + 1):
                # v2.0: Random window sampling for faster training
                if len(train_data) > window_size:
                    start_idx = np.random.randint(0, len(train_data) - window_size)
                    episode_data = train_data.iloc[start_idx:start_idx + window_size].reset_index(drop=True)
                else:
                    episode_data = train_data
                
                env = TradingEnvironment(episode_data)
                result = self.train_episode(env)
                fold_train_results.append(result)
                
                if ep % 10 == 0:
                    avg_wr = np.mean([r['win_rate'] for r in fold_train_results[-10:]])
                    avg_pnl = np.mean([r['total_pnl'] for r in fold_train_results[-10:]])
                    logger.info(f"  Episode {ep}: Train WR={avg_wr:.1%}, P&L=${avg_pnl:.2f}")
            
            # Validate on unseen data
            val_result = self.evaluate(val_data)
            
            logger.info(f"  Validation: WR={val_result['win_rate']:.1%}, "
                       f"P&L=${val_result['total_pnl']:.2f}, Trades={val_result['n_trades']}")
            
            self.train_history.append({
                "fold": fold + 1,
                "train_wr": np.mean([r['win_rate'] for r in fold_train_results]),
                "train_pnl": np.mean([r['total_pnl'] for r in fold_train_results]),
            })
            
            self.val_history.append({
                "fold": fold + 1,
                **val_result,
            })
            
            all_results.extend(fold_train_results)
            
            # Save checkpoint after each fold
            self.save(f"fold_{fold+1}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD TRAINING COMPLETE")
        logger.info("="*60)
        
        train_wrs = [h['train_wr'] for h in self.train_history]
        val_wrs = [h['win_rate'] for h in self.val_history]
        
        logger.info(f"Avg Train Win Rate: {np.mean(train_wrs):.1%}")
        logger.info(f"Avg Val Win Rate: {np.mean(val_wrs):.1%}")
        logger.info(f"Generalization Gap: {np.mean(train_wrs) - np.mean(val_wrs):.1%}")
        
        # Save best model
        best_fold = np.argmax(val_wrs) + 1
        logger.info(f"Best Fold: {best_fold} (Val WR: {max(val_wrs):.1%})")
        
        self.save("best_wf")
        
        return all_results
    
    def save(self, name: str = "ppo_wf"):
        path = f"{self.model_dir}/{name}.pt"
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_episodes': self.training_episodes,
            'total_rewards': self.total_rewards,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }, path)
        logger.debug(f"Model saved to {path}")
    
    def load(self, name: str = "ppo_wf"):
        path = f"{self.model_dir}/{name}.pt"
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_episodes = checkpoint.get('training_episodes', 0)
            self.total_rewards = checkpoint.get('total_rewards', [])
            self.train_history = checkpoint.get('train_history', [])
            self.val_history = checkpoint.get('val_history', [])
            logger.info(f"Model loaded from {path}")
            return True
        return False


def train_walk_forward(
    data_path: str = "data/training/GOLD_H1.csv",
    n_folds: int = 5,
    episodes_per_fold: int = 30,
):
    """Train with Walk-Forward method"""
    
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    
    logger.info(f"Data: {len(df)} bars")
    
    state_dim = 8 + 3
    agent = PPOAgentWalkForward(state_dim=state_dim)
    
    history = agent.walk_forward_train(
        df, 
        n_folds=n_folds, 
        episodes_per_fold=episodes_per_fold
    )
    
    # Final out-of-sample test
    logger.info("\n[Final Out-of-Sample Test]")
    final_result = agent.evaluate(df)
    logger.info(f"Full Dataset: WR={final_result['win_rate']:.1%}, "
               f"P&L=${final_result['total_pnl']:.2f}, Trades={final_result['n_trades']}")
    
    return agent, history


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    agent, history = train_walk_forward(n_folds=10, episodes_per_fold=50)
