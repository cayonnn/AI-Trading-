"""
Master AI Brain - Human-like Thinking System (v3.0 - Transformer)
==================================================================
AI หลักที่คิดแบบมนุษย์ มีสิทธิ์ตัดสินใจสูงสุด
สามารถ override 3 models (LSTM, XGBoost, PPO) ได้เมื่อมั่นใจ

v3.0 Features:
1. TransformerBrain - Multi-Head Attention for better context understanding
2. VectorMemory - Embedding-based experience retrieval
3. MultiTFFusion - Multi-timeframe alignment
4. Human-like Reasoning - อ่านตลาด, ประเมินความเสี่ยง, กรองอารมณ์
5. Experience Memory - 500 trades
6. Override Capability - เลือกฟัง/ไม่ฟัง models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from loguru import logger
import os
import json
import math


@dataclass
class MasterThought:
    """ความคิดของ Master AI"""
    market_view: str  # "bullish", "bearish", "neutral", "dangerous"
    confidence: float  # 0-1
    reasoning: str  # เหตุผลแบบมนุษย์
    suggested_action: str  # "LONG", "SHORT", "WAIT", "CLOSE"
    override_models: bool  # ควร override models หรือไม่
    risk_level: str  # "low", "medium", "high", "extreme"
    ml_confidence: float = 0.0  # ML model's confidence
    tf_alignment: float = 0.0  # Multi-timeframe alignment score


@dataclass
class TradeMemory:
    """ความทรงจำจากการเทรด"""
    timestamp: datetime
    market_state: Dict[str, float]
    action: str
    result: str  # "win", "loss", "pending"
    pnl: float
    lesson: str  # บทเรียนที่ได้
    embedding: Optional[np.ndarray] = None  # Vector embedding


# ============================================================
#  Phase 1: Transformer Architecture
# ============================================================

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBrain(nn.Module):
    """
    Transformer-based Decision Network (v3.0)
    
    Features:
    - Multi-Head Self-Attention
    - Positional Encoding
    - Better context understanding
    
    Input: 12 market features (sequence)
    Output: 3 actions (WAIT, LONG, CLOSE)
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use last token's output
        x = x[:, -1, :]
        x = self.fc(x)
        
        return F.softmax(x, dim=-1)


# ============================================================
#  Phase 2: Vector Memory Search
# ============================================================

class VectorMemory:
    """
    Embedding-based experience memory
    
    Features:
    - Encode market states to embeddings
    - Cosine similarity search
    - Top-K similar experience retrieval
    """
    
    def __init__(self, embedding_dim: int = 32, device: str = 'cpu'):
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Simple encoder network
        self.encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        ).to(device)
        
        # Memory storage
        self.embeddings: List[torch.Tensor] = []
        self.memories: List[TradeMemory] = []
    
    def encode(self, features: np.ndarray) -> torch.Tensor:
        """Encode features to embedding"""
        x = torch.FloatTensor(features).to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.encoder(x)
        
        return F.normalize(embedding, dim=-1)
    
    def add_memory(self, memory: TradeMemory, embedding: torch.Tensor):
        """Add memory with embedding"""
        self.embeddings.append(embedding.cpu())
        self.memories.append(memory)
    
    def search_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[TradeMemory, float]]:
        """Find top-K similar experiences"""
        
        if not self.embeddings:
            return []
        
        # Stack all embeddings
        all_embeddings = torch.stack(self.embeddings)
        query = query_embedding.cpu()
        
        # Cosine similarity
        similarities = F.cosine_similarity(query, all_embeddings)
        
        # Get top-K
        k = min(top_k, len(self.memories))
        top_indices = torch.topk(similarities, k).indices.tolist()
        
        results = []
        for idx in top_indices:
            results.append((self.memories[idx], similarities[idx].item()))
        
        return results


# ============================================================
#  Phase 3: Multi-Timeframe Fusion
# ============================================================

class MultiTFFusion:
    """
    Multi-Timeframe Analysis Fusion
    
    Analyzes M15, H1, H4, D1 trends and calculates alignment score
    """
    
    def __init__(self):
        self.timeframes = ['M15', 'H1', 'H4', 'D1']
        self.weights = {
            'M15': 0.15,  # Short-term noise
            'H1': 0.30,   # Primary trading TF
            'H4': 0.30,   # Medium-term trend
            'D1': 0.25,   # Long-term direction
        }
    
    def analyze(
        self,
        tf_data: Dict[str, Dict[str, float]],
    ) -> Tuple[float, str, str]:
        """
        Analyze multi-timeframe alignment
        
        Args:
            tf_data: {'M15': {'trend': 0.5, 'momentum': 0.3}, ...}
            
        Returns:
            alignment_score: 0-1 (1 = all TFs aligned)
            direction: 'bullish', 'bearish', 'mixed'
            insight: Human-readable insight
        """
        
        if not tf_data:
            return 0.5, 'mixed', 'No multi-TF data'
        
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.weights.items():
            if tf in tf_data:
                trend = tf_data[tf].get('trend', 0)
                momentum = tf_data[tf].get('momentum', 0)
                
                # Combine trend and momentum
                direction_score = (trend + momentum) / 2
                
                if direction_score > 0:
                    bullish_score += direction_score * weight
                else:
                    bearish_score += abs(direction_score) * weight
                
                total_weight += weight
        
        if total_weight == 0:
            return 0.5, 'mixed', 'Insufficient TF data'
        
        # Normalize
        bullish_score /= total_weight
        bearish_score /= total_weight
        
        # Calculate alignment
        diff = abs(bullish_score - bearish_score)
        alignment = min(1.0, diff * 2)  # 0-1 scale
        
        # Determine direction
        if bullish_score > bearish_score + 0.1:
            direction = 'bullish'
            insight = f"TFs aligned bullish ({alignment:.0%})"
        elif bearish_score > bullish_score + 0.1:
            direction = 'bearish'
            insight = f"TFs aligned bearish ({alignment:.0%})"
        else:
            direction = 'mixed'
            insight = f"TFs mixed signals ({alignment:.0%})"
        
        return alignment, direction, insight
    
    def boost_confidence(
        self,
        base_confidence: float,
        alignment: float,
        direction: str,
        action: str,
    ) -> float:
        """
        Boost or reduce confidence based on TF alignment
        """
        
        # Check if action matches TF direction
        action_matches = (
            (action == 'LONG' and direction == 'bullish') or
            (action == 'CLOSE' and direction == 'bearish')
        )
        
        if action_matches:
            # Boost confidence by alignment score
            boost = alignment * 0.15  # Max 15% boost
            return min(0.99, base_confidence + boost)
        elif direction != 'mixed':
            # Reduce confidence if opposite direction
            reduction = alignment * 0.10  # Max 10% reduction
            return max(0.1, base_confidence - reduction)
        
        return base_confidence


# DecisionNetwork removed - replaced by TransformerBrain


class MasterBrain:
    """
    Master AI Brain - ผู้ตัดสินใจสูงสุด (v3.0 - Transformer)
    
    v3.0 Features:
    1. TransformerBrain - Multi-Head Attention
    2. VectorMemory - Embedding-based search
    3. MultiTFFusion - Multi-timeframe alignment
    4. Human-like Reasoning
    5. 500 trades memory
    
    มีสิทธิ์:
    - Override 3 models เมื่อมั่นใจ > 80%
    - หรือเลือกฟัง models เมื่อไม่แน่ใจ
    """
    
    def __init__(
        self,
        override_threshold: float = 0.80,
        risk_per_trade: float = 0.02,
        memory_size: int = 500,
        learning_rate: float = 0.001,
    ):
        self.override_threshold = override_threshold
        self.risk_per_trade = risk_per_trade
        self.memory_size = memory_size
        
        # Experience Memory
        self.trade_memory: deque = deque(maxlen=memory_size)
        self.pattern_memory: Dict[str, List[TradeMemory]] = {}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # v3.0: Transformer Brain (replaced DecisionNetwork)
        self.transformer_brain = TransformerBrain(input_dim=12).to(self.device)
        self.optimizer = optim.Adam(self.transformer_brain.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # v3.0: Vector Memory for similarity search
        self.vector_memory = VectorMemory(embedding_dim=32, device=self.device)
        
        # v3.0: Multi-Timeframe Fusion
        self.mtf_fusion = MultiTFFusion()
        
        # Training buffer for batch learning
        self.training_buffer: List[Tuple[np.ndarray, int]] = []
        self.batch_size = 32
        self.min_samples_to_train = 50
        
        # Expanded Market State Categories
        self.market_states = {
            'trending_up_low_vol': {'sl_mult': 1.2, 'win_rate': 0.5},
            'trending_up_high_vol': {'sl_mult': 1.5, 'win_rate': 0.5},
            'trending_down_low_vol': {'sl_mult': 1.2, 'win_rate': 0.5},
            'trending_down_high_vol': {'sl_mult': 1.5, 'win_rate': 0.5},
            'ranging_low_vol': {'sl_mult': 0.8, 'win_rate': 0.5},
            'ranging_high_vol': {'sl_mult': 1.0, 'win_rate': 0.5},
            'breakout_up': {'sl_mult': 1.3, 'win_rate': 0.5},
            'breakout_down': {'sl_mult': 1.3, 'win_rate': 0.5},
            'consolidation': {'sl_mult': 0.7, 'win_rate': 0.5},
            'reversal_top': {'sl_mult': 1.0, 'win_rate': 0.5},
            'reversal_bottom': {'sl_mult': 1.0, 'win_rate': 0.5},
            'unknown': {'sl_mult': 1.0, 'win_rate': 0.5},
        }
        
        # ============================================
        # v3.1: Adaptive Risk (เรียนรู้จากประสบการณ์จริง)
        # ============================================
        
        # Streak Tracking
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # LEARNED parameters (ไม่ hardcode, เรียนรู้จาก trade history)
        self.learned_params = {
            # เริ่มต้นค่าเฉลี่ย จะถูกปรับจากประสบการณ์
            'optimal_sl_atr': 1.0,      # SL = n × ATR
            'optimal_tp_atr': 2.0,      # TP = n × ATR
            'optimal_risk_pct': 0.01,   # Risk per trade
            'optimal_lot_mult': 1.0,    # Lot multiplier
            
            # ค่าที่เรียนรู้ตาม regime
            'regime_adjustments': {},
            
            # ค่าที่เรียนรู้ตาม streak
            'streak_multiplier': 1.0,
            
            # ค่าสำหรับ trailing
            'trailing_start_profit': 1.5,  # Start trail at 1.5× SL
            'trailing_distance': 0.5,      # Trail distance in ATR
        }
        
        # ============================================
        # v3.3: Ensemble Rebalancing
        # ============================================
        self.model_accuracy = {
            'lstm': {'correct': 0, 'total': 0, 'weight': 0.4},
            'xgboost': {'correct': 0, 'total': 0, 'weight': 0.4},
            'ppo': {'correct': 0, 'total': 0, 'weight': 0.2},
        }
        self.rebalance_interval = 20  # Rebalance every N trades
        self.trades_since_rebalance = 0
        
        # ============================================
        # v3.3: Time-of-Day Pattern Learning
        # ============================================
        self.hourly_performance = {h: {'pnl': 0.0, 'trades': 0, 'wins': 0} for h in range(24)}
        self.session_performance = {
            'asia': {'pnl': 0.0, 'trades': 0, 'wins': 0},    # 00:00-08:00 UTC
            'london': {'pnl': 0.0, 'trades': 0, 'wins': 0},  # 08:00-16:00 UTC
            'newyork': {'pnl': 0.0, 'trades': 0, 'wins': 0}, # 13:00-21:00 UTC
        }
        self.preferred_hours = list(range(24))  # Will be learned
        
        # ============================================
        # v3.3: Daily Loss Limit
        # ============================================
        self.daily_pnl = 0.0
        self.daily_loss_limit = 0.03  # 3% max daily loss
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.is_daily_limit_hit = False
        
        # ============================================
        # v3.3: Recovery Mode
        # ============================================
        self.in_recovery_mode = False
        self.recovery_wins_needed = 5
        self.recovery_win_streak = 0
        self.drawdown_threshold = 0.10  # 10% drawdown triggers recovery
        
        # Trade outcome tracking by SL/TP settings
        self.sl_tp_performance: Dict[str, Dict] = {}
        
        # Drawdown tracking (เรียนรู้ threshold)
        self.equity_peak = 0.0
        self.current_drawdown = 0.0
        self.learned_drawdown_threshold = 0.05  # จะถูกปรับตามประสบการณ์
        
        # Position management state
        self.active_positions: Dict[int, Dict] = {}  # ticket: {entry, sl, tp, partial_closed}
        
        # ============================================
        # v3.1: MT5 Direct Control
        # ============================================
        self.mt5_connected = False
        self._init_mt5()
        
        # Statistics
        self.total_decisions = 0
        self.override_count = 0
        self.override_win_rate = 0.5
        self.ml_accuracy = 0.0
        
        # Current state
        self.current_view: Optional[MasterThought] = None
        
        # Load saved model if exists
        self._load_model()
        
        # v3.2: Auto-save settings
        self.model_save_path = "ai_agent/models/master_brain_state.json"
        self.auto_save_interval = 10  # Save every N trades
        self.trades_since_save = 0
        
        logger.info(f"🧠 MasterBrain v3.2 initialized - Learned Risk | Trailing | MT5 Control | Auto-Save | Memory: {memory_size}")
    
    def _init_db(self):
        """v3.2: Initialize SQLite database connection"""
        import sqlite3
        self.db_path = "trade_memory.db"
        try:
            self.db_conn = sqlite3.connect(self.db_path)
            # Create master_brain_state table if not exists
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS master_brain_state (
                    id INTEGER PRIMARY KEY,
                    key TEXT UNIQUE,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db_conn.commit()
            logger.debug("MasterBrain: DB connected")
        except Exception as e:
            self.db_conn = None
            logger.warning(f"MasterBrain: DB not available - {e}")
    
    def _auto_save_check(self):
        """v3.2: Check if we should auto-save"""
        self.trades_since_save += 1
        if self.trades_since_save >= self.auto_save_interval:
            self._save_model()
            self.trades_since_save = 0
    
    def _init_mt5(self):
        """Initialize MT5 connection for direct control"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            self.mt5_connected = True
            logger.debug("MasterBrain: MT5 module loaded")
        except ImportError:
            self.mt5 = None
            self.mt5_connected = False
            logger.warning("MasterBrain: MT5 not available - position control disabled")
    
    # ============================================
    # v3.3: Ensemble Rebalancing Methods
    # ============================================
    
    def update_model_accuracy(self, model_name: str, prediction: str, actual: str):
        """Update accuracy tracking for a model"""
        if model_name not in self.model_accuracy:
            return
        
        self.model_accuracy[model_name]['total'] += 1
        if prediction == actual:
            self.model_accuracy[model_name]['correct'] += 1
        
        self.trades_since_rebalance += 1
        if self.trades_since_rebalance >= self.rebalance_interval:
            self._rebalance_ensemble()
            self.trades_since_rebalance = 0
    
    def _rebalance_ensemble(self):
        """Auto-adjust model weights based on accuracy"""
        accuracies = {}
        total_acc = 0
        
        for model, data in self.model_accuracy.items():
            if data['total'] > 0:
                acc = data['correct'] / data['total']
                accuracies[model] = acc
                total_acc += acc
            else:
                accuracies[model] = 0.33
                total_acc += 0.33
        
        if total_acc > 0:
            for model in self.model_accuracy:
                new_weight = accuracies[model] / total_acc
                new_weight = max(0.1, min(0.6, new_weight))  # Clamp 10-60%
                self.model_accuracy[model]['weight'] = new_weight
            
            logger.info(f"🔄 Ensemble rebalanced: LSTM={self.model_accuracy['lstm']['weight']:.0%}, "
                       f"XGB={self.model_accuracy['xgboost']['weight']:.0%}, "
                       f"PPO={self.model_accuracy['ppo']['weight']:.0%}")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return {m: d['weight'] for m, d in self.model_accuracy.items()}
    
    # ============================================
    # v3.3: Time-of-Day & Session Methods
    # ============================================
    
    def record_hourly_performance(self, hour: int, pnl: float, is_win: bool):
        """Record performance for an hour"""
        if 0 <= hour < 24:
            self.hourly_performance[hour]['pnl'] += pnl
            self.hourly_performance[hour]['trades'] += 1
            if is_win:
                self.hourly_performance[hour]['wins'] += 1
        
        # Update session
        session = self._get_session(hour)
        self.session_performance[session]['pnl'] += pnl
        self.session_performance[session]['trades'] += 1
        if is_win:
            self.session_performance[session]['wins'] += 1
        
        # Learn preferred hours (top 50% by win rate)
        self._update_preferred_hours()
    
    def _get_session(self, hour: int) -> str:
        """Get trading session for hour (UTC)"""
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'london'
        else:
            return 'newyork'
    
    def _update_preferred_hours(self):
        """Update preferred trading hours based on performance"""
        hour_scores = []
        for h, data in self.hourly_performance.items():
            if data['trades'] >= 5:  # Minimum trades to consider
                win_rate = data['wins'] / data['trades'] if data['trades'] > 0 else 0
                hour_scores.append((h, win_rate))
        
        if hour_scores:
            hour_scores.sort(key=lambda x: x[1], reverse=True)
            top_half = [h for h, _ in hour_scores[:len(hour_scores)//2 + 1]]
            self.preferred_hours = top_half if top_half else list(range(24))
    
    def is_good_trading_hour(self, hour: int = None) -> Tuple[bool, str]:
        """Check if current hour is a good trading hour"""
        if hour is None:
            hour = datetime.now().hour
        
        is_preferred = hour in self.preferred_hours
        
        data = self.hourly_performance.get(hour, {'trades': 0})
        if data['trades'] >= 5:
            win_rate = data['wins'] / data['trades']
            reason = f"Hour {hour}: WR={win_rate:.0%} ({data['trades']} trades)"
        else:
            reason = f"Hour {hour}: Not enough data"
            is_preferred = True  # Default to allow
        
        return is_preferred, reason
    
    # ============================================
    # v3.3: Daily Loss Limit Methods
    # ============================================
    
    def check_daily_limit(self, equity: float) -> Tuple[bool, str]:
        """Check if daily loss limit is hit"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.is_daily_limit_hit = False
            self.last_reset_date = today
            logger.info("📆 Daily counters reset")
        
        if self.is_daily_limit_hit:
            return False, f"Daily loss limit hit ({self.daily_loss_limit:.0%})"
        
        daily_loss_pct = abs(min(0, self.daily_pnl)) / equity if equity > 0 else 0
        
        if daily_loss_pct >= self.daily_loss_limit:
            self.is_daily_limit_hit = True
            logger.warning(f"🛑 DAILY LOSS LIMIT HIT: {daily_loss_pct:.1%} >= {self.daily_loss_limit:.0%}")
            return False, f"Daily loss limit: {daily_loss_pct:.1%}"
        
        return True, f"Daily P&L: ${self.daily_pnl:.2f} ({daily_loss_pct:.1%} of limit)"
    
    def record_daily_trade(self, pnl: float):
        """Record trade P&L for daily tracking"""
        self.daily_pnl += pnl
        self.daily_trade_count += 1
    
    # ============================================
    # v3.3: Recovery Mode Methods
    # ============================================
    
    def check_recovery_mode(self, current_equity: float, is_win: bool = None):
        """Check and update recovery mode status"""
        # Update equity peak
        if current_equity > self.equity_peak:
            self.equity_peak = current_equity
        
        # Calculate drawdown
        if self.equity_peak > 0:
            self.current_drawdown = (self.equity_peak - current_equity) / self.equity_peak
        
        # Enter recovery mode?
        if not self.in_recovery_mode and self.current_drawdown >= self.drawdown_threshold:
            self.in_recovery_mode = True
            self.recovery_win_streak = 0
            logger.warning(f"⚠️ RECOVERY MODE ACTIVATED: DD={self.current_drawdown:.1%}")
        
        # Update win streak in recovery
        if self.in_recovery_mode and is_win is not None:
            if is_win:
                self.recovery_win_streak += 1
                if self.recovery_win_streak >= self.recovery_wins_needed:
                    self.in_recovery_mode = False
                    logger.info(f"✅ RECOVERY MODE DEACTIVATED: {self.recovery_win_streak} consecutive wins")
            else:
                self.recovery_win_streak = 0
    
    def get_position_multiplier(self) -> float:
        """Get position size multiplier (reduced in recovery mode)"""
        if self.in_recovery_mode:
            return 0.5  # 50% position size in recovery
        return 1.0
    
    def modify_position_sl(
        self,
        ticket: int,
        new_sl: float,
        new_tp: Optional[float] = None,
    ) -> bool:
        """
        แก้ไข SL/TP ของ position (MasterBrain controls directly)
        
        Args:
            ticket: Position ticket number
            new_sl: New stop loss price
            new_tp: New take profit price (optional)
            
        Returns:
            True if successful
        """
        if not self.mt5_connected or not self.mt5:
            logger.warning("MT5 not connected - cannot modify position")
            return False
        
        try:
            # Get current position
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                logger.warning(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            current_tp = pos.tp if new_tp is None else new_tp
            
            # Build modification request
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": new_sl,
                "tp": current_tp,
            }
            
            result = self.mt5.order_send(request)
            
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                logger.info(f"🧠 MasterBrain: Modified SL {pos.sl:.2f} → {new_sl:.2f}")
                
                # Update active positions tracking
                if ticket in self.active_positions:
                    self.active_positions[ticket]['sl'] = new_sl
                    if new_tp:
                        self.active_positions[ticket]['tp'] = new_tp
                
                return True
            else:
                logger.warning(f"Failed to modify position: {result.retcode}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
    
    def manage_open_positions(self, current_price: float, atr: float):
        """
        จัดการ positions ที่เปิดอยู่ (trailing stop, partial TP)
        
        MasterBrain ตัดสินใจเองว่าจะขยับ SL เมื่อไหร่
        """
        if not self.mt5_connected or not self.mt5:
            return
        
        try:
            positions = self.mt5.positions_get()
            if not positions:
                return
            
            for pos in positions:
                ticket = pos.ticket
                entry = pos.price_open
                current_sl = pos.sl
                is_long = pos.type == 0  # 0 = BUY
                
                # Calculate new trailing SL
                new_sl = self.calculate_trailing_stop(
                    entry_price=entry,
                    current_price=current_price,
                    original_sl=current_sl,
                    atr=atr,
                    is_long=is_long,
                )
                
                if new_sl:
                    # MasterBrain decides to trail
                    profit_pips = (current_price - entry) if is_long else (entry - current_price)
                    logger.info(f"🧠 MasterBrain: Trailing SL - Profit={profit_pips:.1f} pips")
                    
                    self.modify_position_sl(ticket, new_sl)
                    
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def close_position(self, ticket: int) -> bool:
        """
        ปิด position (MasterBrain decides to close)
        """
        if not self.mt5_connected or not self.mt5:
            return False
        
        try:
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            
            # Determine close type
            close_type = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
            
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "price": self.mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else self.mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "MasterBrain close",
            }
            
            result = self.mt5.order_send(request)
            
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                logger.info(f"🧠 MasterBrain: Closed position {ticket}")
                return True
            else:
                logger.warning(f"Failed to close: {result.retcode}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def plan_trade(
        self,
        market_data: Dict[str, float],
        indicators: Dict[str, float],
        tf_data: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        วางแผนการเทรดแบบมนุษย์ (Human-like Trade Planning)
        
        AI คิดเหมือนเทรดเดอร์มืออาชีพ:
        1. วิเคราะห์สถานการณ์ตลาด
        2. หาจุด Entry ที่ดี
        3. กำหนด SL/TP ตามโครงสร้างตลาด
        4. คำนวณ Position Size ตาม Risk
        5. วางแผน Exit Strategy
        
        Returns:
            Trade plan with entry, sl, tp, lot, reasoning
        """
        
        price = market_data.get('price', 0)
        atr = market_data.get('atr', 15)
        trend = market_data.get('trend', 0)
        volatility = market_data.get('volatility', 0.5)
        regime = market_data.get('regime', 'unknown')
        
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        bb_position = indicators.get('bb_position', 0.5)
        
        thinking_steps = []
        
        # ============================================
        # 1. วิเคราะห์ตลาดเหมือนมนุษย์
        # ============================================
        thinking_steps.append("📊 วิเคราะห์ตลาด:")
        
        # Trend analysis
        if trend > 0.5:
            trend_view = "bullish"
            thinking_steps.append("  → Trend: ขาขึ้นชัดเจน ควรหา entry LONG")
        elif trend < -0.5:
            trend_view = "bearish"
            thinking_steps.append("  → Trend: ขาลงชัดเจน ควรหา entry SHORT")
        else:
            trend_view = "neutral"
            thinking_steps.append("  → Trend: ไม่มีทิศทาง ควรรอ")
        
        # Volatility assessment
        if volatility > 0.7:
            vol_view = "high"
            thinking_steps.append("  → Volatility สูง: ขยาย SL, ลด lot")
        elif volatility < 0.3:
            vol_view = "low"
            thinking_steps.append("  → Volatility ต่ำ: SL แคบได้, lot ปกติ")
        else:
            vol_view = "normal"
            thinking_steps.append("  → Volatility ปกติ")
        
        # Multi-TF alignment
        tf_alignment = 0.5
        tf_direction = 'mixed'
        if tf_data:
            tf_alignment, tf_direction, _ = self.mtf_fusion.analyze(tf_data)
            if tf_alignment > 0.7:
                thinking_steps.append(f"  → Multi-TF: {tf_direction} aligned {tf_alignment:.0%}")
            else:
                thinking_steps.append(f"  → Multi-TF: Mixed signals - ลด confidence")
        
        # ============================================
        # 2. ตัดสินใจ Entry เหมือนมนุษย์
        # ============================================
        thinking_steps.append("\n🎯 ตัดสินใจ:")
        
        # Entry decision based on human logic
        should_trade = False
        direction = None
        entry_reason = ""
        
        # Rule 1: Strong trend + aligned TFs
        if trend_view == "bullish" and tf_direction == "bullish" and tf_alignment > 0.7:
            should_trade = True
            direction = "LONG"
            entry_reason = "Trend bullish + TFs aligned"
            thinking_steps.append(f"  → เข้า LONG: {entry_reason}")
        
        # Rule 2: RSI extreme + mean reversion
        elif rsi < 30 and trend_view != "bearish":
            should_trade = True
            direction = "LONG"
            entry_reason = "RSI oversold - mean reversion"
            thinking_steps.append(f"  → เข้า LONG: {entry_reason}")
        
        # Rule 3: Experience says this works
        experience = self._recall_experience(market_data, indicators)
        if "ชนะ" in experience and "70%" in experience:
            should_trade = True
            if direction is None:
                direction = "LONG" if trend > 0 else "WAIT"
            thinking_steps.append(f"  → ประสบการณ์ดี: {experience}")
        
        # Rule 4: Don't trade in danger
        if volatility > 0.8 or rsi > 90 or rsi < 10:
            should_trade = False
            entry_reason = "ตลาดอันตราย - รอ"
            thinking_steps.append(f"  → หยุด: {entry_reason}")
        
        # ============================================
        # 3. คำนวณ SL/TP เหมือนมนุษย์
        # ============================================
        thinking_steps.append("\n📐 คำนวณ Risk:")
        
        if should_trade and direction:
            # SL based on learned params + volatility
            sl_atr = self.learned_params['optimal_sl_atr']
            tp_atr = self.learned_params['optimal_tp_atr']
            
            # Adjust for volatility (human would do this)
            if vol_view == "high":
                sl_atr *= 1.3
                thinking_steps.append(f"  → ขยาย SL เพราะ vol สูง: {sl_atr:.1f}× ATR")
            elif vol_view == "low":
                sl_atr *= 0.8
                thinking_steps.append(f"  → SL แคบเพราะ vol ต่ำ: {sl_atr:.1f}× ATR")
            
            sl_pips = atr * sl_atr
            tp_pips = atr * tp_atr
            
            # Ensure good R:R
            min_rr = 1.5
            if tp_pips / sl_pips < min_rr:
                tp_pips = sl_pips * min_rr
                thinking_steps.append(f"  → ปรับ TP เพื่อ R:R >= {min_rr}")
            
            if direction == "LONG":
                sl = price - sl_pips
                tp = price + tp_pips
            else:
                sl = price + sl_pips
                tp = price - tp_pips
            
            thinking_steps.append(f"  → SL: {sl:.2f} ({sl_pips:.1f} pips)")
            thinking_steps.append(f"  → TP: {tp:.2f} ({tp_pips:.1f} pips)")
            thinking_steps.append(f"  → R:R: {tp_pips/sl_pips:.2f}")
            
            # ============================================
            # 4. คำนวณ Lot Size ตาม Risk
            # ============================================
            equity = 10000  # Default, should be passed
            risk_params = self.calculate_risk_params(
                entry_price=price,
                atr=atr,
                confidence=tf_alignment,
                regime=regime,
                equity=equity,
                is_long=(direction == "LONG"),
            )
            
            lot = risk_params['lot_size']
            risk_pct = risk_params['risk_pct']
            
            thinking_steps.append(f"  → Lot: {lot} (Risk: {risk_pct:.1%})")
            
            # ============================================
            # 5. สร้าง Trade Plan
            # ============================================
            plan = {
                'should_trade': True,
                'direction': direction,
                'entry': price,
                'sl': sl,
                'tp': tp,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'lot': lot,
                'risk_pct': risk_pct,
                'rr': tp_pips / sl_pips,
                'entry_reason': entry_reason,
                'thinking': '\n'.join(thinking_steps),
                'confidence': tf_alignment,
                'exit_strategy': {
                    'trail_at': 1.5,  # Start trailing at 1.5R
                    'move_be_at': 1.0,  # Move SL to BE at 1R
                    'partial_at': 2.0,  # Partial TP at 2R
                },
            }
            
            thinking_steps.append("\n✅ TRADE PLAN READY")
            thinking_steps.append(f"   {direction} @ {price:.2f}")
            thinking_steps.append(f"   SL: {sl:.2f} | TP: {tp:.2f}")
            thinking_steps.append(f"   Lot: {lot} | R:R: {plan['rr']:.2f}")
            
        else:
            plan = {
                'should_trade': False,
                'direction': None,
                'entry': None,
                'sl': None,
                'tp': None,
                'lot': 0,
                'entry_reason': entry_reason or "ไม่มี setup ที่ดี",
                'thinking': '\n'.join(thinking_steps),
                'confidence': 0,
            }
            
            thinking_steps.append("\n⏳ WAIT - ไม่เทรด")
        
        plan['thinking'] = '\n'.join(thinking_steps)
        
        logger.info(f"🧠 Trade Plan:\n{plan['thinking']}")
        
        return plan
    
    def think(
        self,
        market_data: Dict[str, float],
        model_votes: Dict[str, Tuple[str, float]],  # {model: (action, confidence)}
        technical_indicators: Dict[str, float],
        tf_data: Optional[Dict[str, Dict[str, float]]] = None,  # Multi-TF data
    ) -> MasterThought:
        """
        คิดแบบมนุษย์และตัดสินใจ (v3.0 with MultiTF + VectorMemory)
        
        Args:
            market_data: {price, trend, volatility, regime, atr}
            model_votes: {lstm: (action, conf), xgb: (action, conf), ppo: (action, conf)}
            technical_indicators: {rsi, macd, bb_position, etc.}
            tf_data: {'M15': {'trend': 0.5}, 'H1': {...}, 'H4': {...}, 'D1': {...}}
            
        Returns:
            MasterThought with decision
        """
        
        # 1. อ่านตลาดเหมือนมนุษย์
        market_view, market_reasoning = self._read_market_like_human(
            market_data, technical_indicators
        )
        
        # 2. ประเมินความเสี่ยง
        risk_level, risk_reasoning = self._assess_risk(
            market_data, technical_indicators
        )
        
        # 3. กรองอารมณ์ตลาด
        emotion_filter, emotion_reasoning = self._filter_market_emotion(
            technical_indicators
        )
        
        # 4. ดูว่า models บอกอะไร
        model_consensus, model_confidence = self._analyze_model_votes(model_votes)
        
        # 5. เทียบกับประสบการณ์ (v3.0: VectorMemory)
        experience_insight = self._recall_experience_v3(market_data, technical_indicators)
        
        # 6. Multi-TF Fusion (v3.0 NEW)
        tf_alignment = 0.5
        tf_direction = 'mixed'
        tf_insight = ''
        if tf_data:
            tf_alignment, tf_direction, tf_insight = self.mtf_fusion.analyze(tf_data)
        
        # 7. Session Time Filter (v3.0 NEW)
        session_ok, session_reason = self._check_trading_session()
        
        # 8. ตัดสินใจขั้นสุดท้าย
        final_action, master_confidence, override = self._make_master_decision(
            market_view=market_view,
            risk_level=risk_level,
            emotion_filter=emotion_filter,
            model_consensus=model_consensus,
            model_confidence=model_confidence,
            experience_insight=experience_insight,
        )
        
        # 9. Boost/Reduce confidence based on TF alignment (v3.0)
        if tf_data:
            master_confidence = self.mtf_fusion.boost_confidence(
                master_confidence, tf_alignment, tf_direction, final_action
            )
        
        # 10. Session filter override
        if not session_ok and final_action != 'WAIT':
            override = True
            final_action = 'WAIT'
            master_confidence = 0.5
        
        # 11. สร้าง reasoning แบบมนุษย์
        full_reasoning = self._build_human_reasoning_v3(
            market_view=market_view,
            market_reasoning=market_reasoning,
            risk_level=risk_level,
            risk_reasoning=risk_reasoning,
            emotion_filter=emotion_filter,
            emotion_reasoning=emotion_reasoning,
            model_consensus=model_consensus,
            model_confidence=model_confidence,
            experience_insight=experience_insight,
            tf_alignment=tf_alignment,
            tf_direction=tf_direction,
            tf_insight=tf_insight,
            session_ok=session_ok,
            session_reason=session_reason,
            final_action=final_action,
            override=override,
        )
        
        thought = MasterThought(
            market_view=market_view,
            confidence=master_confidence,
            reasoning=full_reasoning,
            suggested_action=final_action,
            override_models=override,
            risk_level=risk_level,
            tf_alignment=tf_alignment,
        )
        
        self.current_view = thought
        self.total_decisions += 1
        if override:
            self.override_count += 1
        
        return thought
    
    def _read_market_like_human(
        self,
        market: Dict[str, float],
        indicators: Dict[str, float],
    ) -> Tuple[str, str]:
        """อ่านตลาดเหมือนมนุษย์"""
        
        trend = market.get('trend', 0)
        volatility = market.get('volatility', 0.5)
        regime = market.get('regime', 'unknown')
        price = market.get('price', 0)
        
        # Human-like interpretation
        if regime == 'volatile' and volatility > 0.7:
            return "dangerous", "ตลาดผันผวนมาก ไม่ควรเทรด"
        
        if trend > 0.6:
            if volatility < 0.4:
                return "bullish", "Uptrend ชัดเจน volatility ต่ำ เหมาะซื้อ"
            else:
                return "bullish", "Uptrend แต่ volatility สูง ระวังหน่อย"
        
        if trend < -0.6:
            if volatility < 0.4:
                return "bearish", "Downtrend ชัดเจน"
            else:
                return "bearish", "Downtrend + volatility สูง"
        
        return "neutral", "ไม่มีทิศทางชัด รอโอกาส"
    
    def _assess_risk(
        self,
        market: Dict[str, float],
        indicators: Dict[str, float],
    ) -> Tuple[str, str]:
        """ประเมินความเสี่ยง"""
        
        volatility = market.get('volatility', 0.5)
        atr = market.get('atr', 0)
        rsi = indicators.get('rsi', 50)
        
        # RSI extremes
        if rsi > 85:
            return "extreme", "RSI > 85 overbought สุดขีด! ห้ามซื้อ"
        if rsi < 15:
            return "extreme", "RSI < 15 oversold สุดขีด! ระวัง"
        
        # Volatility
        if volatility > 0.8:
            return "high", f"Volatility สูงมาก ({volatility:.0%}) ลด lot"
        if volatility > 0.6:
            return "medium", f"Volatility ปานกลาง ระวัง"
        
        return "low", "ความเสี่ยงต่ำ"
    
    def _filter_market_emotion(
        self,
        indicators: Dict[str, float],
    ) -> Tuple[str, str]:
        """กรองอารมณ์ตลาด"""
        
        rsi = indicators.get('rsi', 50)
        bb_position = indicators.get('bb_position', 0.5)  # 0-1, 0.5=middle
        
        # Panic detection
        if rsi < 25 and bb_position < 0.1:
            return "panic", "ตลาด panic selling - อาจถึงจุดกลับตัว"
        
        # Greed detection
        if rsi > 75 and bb_position > 0.9:
            return "greed", "ตลาด greedy - ระวังการปรับฐาน"
        
        # FOMO detection
        if rsi > 65 and bb_position > 0.8:
            return "fomo", "อาจมี FOMO - ไม่ควรไล่ซื้อ"
        
        return "neutral", "อารมณ์ตลาดปกติ"
    
    def _analyze_model_votes(
        self,
        model_votes: Dict[str, Tuple[str, float]],
    ) -> Tuple[str, float]:
        """วิเคราะห์ votes จาก models"""
        
        if not model_votes:
            return "WAIT", 0.0
        
        # Count votes
        action_votes = {}
        total_confidence = 0
        
        for model, (action, conf) in model_votes.items():
            if action not in action_votes:
                action_votes[action] = []
            action_votes[action].append((model, conf))
            total_confidence += conf
        
        # Find consensus
        best_action = max(action_votes.keys(), key=lambda a: len(action_votes[a]))
        avg_confidence = total_confidence / len(model_votes)
        
        # Check if all agree
        if len(action_votes) == 1:
            return best_action, avg_confidence * 1.2  # Boost for consensus
        
        return best_action, avg_confidence
    
    def _recall_experience(
        self,
        market: Dict[str, float],
        indicators: Dict[str, float],
    ) -> str:
        """ดึงประสบการณ์ที่เกี่ยวข้อง (legacy)"""
        
        if len(self.trade_memory) < 5:
            return "ยังไม่มีประสบการณ์เพียงพอ"
        
        # Find similar past situations
        rsi = indicators.get('rsi', 50)
        volatility = market.get('volatility', 0.5)
        
        similar_trades = []
        for trade in self.trade_memory:
            past_rsi = trade.market_state.get('rsi', 50)
            past_vol = trade.market_state.get('volatility', 0.5)
            
            if abs(past_rsi - rsi) < 10 and abs(past_vol - volatility) < 0.2:
                similar_trades.append(trade)
        
        if not similar_trades:
            return "ไม่พบสถานการณ์คล้ายกันในอดีต"
        
        # Analyze results
        wins = len([t for t in similar_trades if t.result == 'win'])
        total = len(similar_trades)
        win_rate = wins / total if total > 0 else 0
        
        if win_rate > 0.6:
            return f"เคยเจอ {total} ครั้ง ชนะ {win_rate:.0%} - สถานการณ์ดี"
        elif win_rate < 0.4:
            return f"เคยเจอ {total} ครั้ง ชนะ {win_rate:.0%} - ระวัง"
        else:
            return f"เคยเจอ {total} ครั้ง ผลลัพธ์ 50-50"
    
    def _recall_experience_v3(
        self,
        market: Dict[str, float],
        indicators: Dict[str, float],
    ) -> str:
        """ดึงประสบการณ์ด้วย VectorMemory (v3.0)"""
        
        # Combine market and indicators for encoding
        combined_state = {**market, **indicators}
        
        # Try vector search first
        if len(self.vector_memory.memories) >= 5:
            features = self._extract_features(combined_state)
            query_embedding = self.vector_memory.encode(features)
            
            similar = self.vector_memory.search_similar(query_embedding, top_k=10)
            
            if similar:
                wins = len([m for m, s in similar if m.result == 'win'])
                total = len(similar)
                avg_similarity = sum(s for _, s in similar) / total
                win_rate = wins / total
                
                if win_rate > 0.6:
                    return f"เคยเจอ {total} ครั้ง (sim:{avg_similarity:.0%}) ชนะ {win_rate:.0%} - สถานการณ์ดี"
                elif win_rate < 0.4:
                    return f"เคยเจอ {total} ครั้ง (sim:{avg_similarity:.0%}) ชนะ {win_rate:.0%} - ระวัง"
                else:
                    return f"เคยเจอ {total} ครั้ง (sim:{avg_similarity:.0%}) ผลลัพธ์ 50-50"
        
        # Fallback to legacy method
        return self._recall_experience(market, indicators)
    
    def _check_trading_session(self) -> Tuple[bool, str]:
        """
        ตรวจสอบ Trading Session (v3.0)
        
        Best times to trade Gold:
        - London Open: 08:00-12:00 UTC (15:00-19:00 ICT)
        - NY Open: 13:00-17:00 UTC (20:00-00:00 ICT)
        - Overlap: 13:00-16:00 UTC (20:00-23:00 ICT)
        
        Avoid:
        - Asian session for Gold (low volatility)
        - Weekend
        """
        from datetime import datetime
        
        now = datetime.now()
        hour = now.hour  # Local time (ICT = UTC+7)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend - no trading
        if weekday >= 5:
            return False, "Weekend - ตลาดปิด"
        
        # Convert to UTC for session check
        utc_hour = (hour - 7) % 24  # ICT to UTC
        
        # London session: 08:00-16:00 UTC
        if 8 <= utc_hour <= 16:
            return True, "London session - สภาพคล่องดี"
        
        # NY session: 13:00-21:00 UTC
        if 13 <= utc_hour <= 21:
            return True, "NY session - สภาพคล่องดี"
        
        # Asian session: 00:00-08:00 UTC
        if 0 <= utc_hour < 8:
            return True, "Asian session - volatility ต่ำ (ระวัง)"
        
        return True, "Market hours"
    
    def _make_master_decision(
        self,
        market_view: str,
        risk_level: str,
        emotion_filter: str,
        model_consensus: str,
        model_confidence: float,
        experience_insight: str,
    ) -> Tuple[str, float, bool]:
        """ตัดสินใจขั้นสุดท้ายเหมือนมนุษย์"""
        
        # Start with model consensus
        action = model_consensus
        confidence = model_confidence
        override = False
        
        # Rule 1: ถ้าตลาดอันตราย → WAIT (override)
        if market_view == "dangerous":
            return "WAIT", 0.9, True  # Override with high confidence
        
        # Rule 2: ถ้า extreme risk → WAIT (override)
        if risk_level == "extreme":
            return "WAIT", 0.85, True
        
        # Rule 3: ถ้า panic/greed → Contrarian thinking
        if emotion_filter == "panic" and model_consensus != "LONG":
            # ตลาด panic อาจเป็นโอกาสซื้อ (contrarian)
            if self._validate_contrarian("LONG"):
                return "LONG", 0.75, True
        
        if emotion_filter == "greed" and model_consensus == "LONG":
            # ตลาด greedy ไม่ควรซื้อ
            return "WAIT", 0.8, True
        
        # Rule 4: ถ้า market view ชัดและ models เห็นด้วย → เพิ่ม confidence
        if market_view == "bullish" and model_consensus == "LONG":
            confidence = min(0.95, confidence * 1.2)
        
        # Rule 5: ถ้า experience insight ดี → เพิ่ม confidence
        if "สถานการณ์ดี" in experience_insight:
            confidence = min(0.95, confidence * 1.1)
        elif "ระวัง" in experience_insight:
            confidence = max(0.3, confidence * 0.8)
        
        return action, confidence, override
    
    def _validate_contrarian(self, action: str) -> bool:
        """Validate contrarian move"""
        # Check if contrarian moves worked in the past
        contrarian_wins = 0
        contrarian_total = 0
        
        for trade in self.trade_memory:
            if trade.lesson and "contrarian" in trade.lesson.lower():
                contrarian_total += 1
                if trade.result == 'win':
                    contrarian_wins += 1
        
        if contrarian_total < 3:
            return True  # Not enough data, allow
        
        return (contrarian_wins / contrarian_total) > 0.5
    
    def _build_human_reasoning(
        self,
        market_view: str,
        market_reasoning: str,
        risk_level: str,
        risk_reasoning: str,
        emotion_filter: str,
        emotion_reasoning: str,
        model_consensus: str,
        model_confidence: float,
        experience_insight: str,
        final_action: str,
        override: bool,
    ) -> str:
        """สร้าง reasoning แบบมนุษย์"""
        
        parts = []
        
        # 1. Market view
        parts.append(f"📊 ตลาด: {market_reasoning}")
        
        # 2. Risk
        if risk_level in ['high', 'extreme']:
            parts.append(f"⚠️ ความเสี่ยง: {risk_reasoning}")
        
        # 3. Emotion
        if emotion_filter != 'neutral':
            parts.append(f"💭 อารมณ์ตลาด: {emotion_reasoning}")
        
        # 4. Models
        parts.append(f"🤖 Models: {model_consensus} ({model_confidence:.0%})")
        
        # 5. Experience
        if "เคยเจอ" in experience_insight:
            parts.append(f"📚 ประสบการณ์: {experience_insight}")
        
        # 6. Final decision
        if override:
            parts.append(f"🧠 MASTER OVERRIDE: {final_action}")
        else:
            parts.append(f"✅ ตัดสินใจ: {final_action}")
        
        return " | ".join(parts)
    
    # ============================================
    # v3.1: S/R-Based Trailing Stop
    # ============================================
    
    def calculate_sr_levels(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        lookback: int = 50,
    ) -> Dict[str, List[float]]:
        """
        หา Support/Resistance levels จาก price action
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            lookback: จำนวน bars ย้อนหลัง
            
        Returns:
            {'supports': [...], 'resistances': [...]}
        """
        
        if len(highs) < lookback:
            return {'supports': [], 'resistances': []}
        
        # Get recent data
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        
        supports = []
        resistances = []
        
        # Find swing highs (resistance) and swing lows (support)
        for i in range(2, len(recent_highs) - 2):
            # Swing high = local maximum
            if (recent_highs[i] > recent_highs[i-1] and 
                recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and 
                recent_highs[i] > recent_highs[i+2]):
                resistances.append(recent_highs[i])
            
            # Swing low = local minimum
            if (recent_lows[i] < recent_lows[i-1] and 
                recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and 
                recent_lows[i] < recent_lows[i+2]):
                supports.append(recent_lows[i])
        
        # Cluster nearby levels (within 0.5% of each other)
        supports = self._cluster_levels(supports, threshold=0.005)
        resistances = self._cluster_levels(resistances, threshold=0.005)
        
        # Sort
        supports.sort(reverse=True)  # High to low
        resistances.sort()  # Low to high
        
        return {
            'supports': supports[-5:],  # Keep top 5
            'resistances': resistances[:5],
        }
    
    def _cluster_levels(
        self,
        levels: List[float],
        threshold: float = 0.005,
    ) -> List[float]:
        """รวม levels ที่ใกล้กันเป็น cluster เดียว"""
        
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - cluster[-1]) / cluster[-1] < threshold:
                cluster.append(level)
            else:
                # Average the cluster
                clustered.append(sum(cluster) / len(cluster))
                cluster = [level]
        
        # Don't forget last cluster
        if cluster:
            clustered.append(sum(cluster) / len(cluster))
        
        return clustered
    
    def calculate_sr_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        original_sl: float,
        supports: List[float],
        resistances: List[float],
        is_long: bool,
        atr: float,
    ) -> Optional[float]:
        """
        คำนวณ Trailing SL ตาม S/R levels
        
        Logic สำหรับ LONG:
        1. ราคาทะลุ resistance แล้วยืนเหนือได้ → ขยับ SL ขึ้นมาที่ resistance นั้น
        2. ราคาไปต่อถึง resistance ถัดไป → ขยับ SL ขึ้นอีก step
        
        Args:
            entry_price: ราคาเข้า
            current_price: ราคาปัจจุบัน
            original_sl: SL ปัจจุบัน
            supports: list of support levels
            resistances: list of resistance levels
            is_long: True=Buy, False=Sell
            atr: ATR for buffer
            
        Returns:
            New SL price or None
        """
        
        buffer = atr * 0.3  # Buffer 0.3 ATR
        
        if is_long:
            # For LONG: Trail SL to broken resistances that become supports
            
            # Find resistances below current price (now become supports)
            broken_resistances = [r for r in resistances if current_price > r + buffer]
            
            if not broken_resistances:
                return None
            
            # Get the highest broken resistance (closest to price)
            new_potential_sl = max(broken_resistances) - buffer
            
            # Only move SL up, never down
            if new_potential_sl > original_sl:
                logger.info(f"🎯 SR Trail: Price broke R={max(broken_resistances):.2f}, moving SL to {new_potential_sl:.2f}")
                return round(new_potential_sl, 2)
        
        else:
            # For SHORT: Trail SL to broken supports that become resistances
            
            # Find supports above current price (now become resistances)
            broken_supports = [s for s in supports if current_price < s - buffer]
            
            if not broken_supports:
                return None
            
            # Get the lowest broken support (closest to price)
            new_potential_sl = min(broken_supports) + buffer
            
            # Only move SL down for shorts
            if new_potential_sl < original_sl:
                logger.info(f"🎯 SR Trail: Price broke S={min(broken_supports):.2f}, moving SL to {new_potential_sl:.2f}")
                return round(new_potential_sl, 2)
        
        return None
    
    def manage_positions_with_sr(
        self,
        current_price: float,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        atr: float,
    ):
        """
        จัดการ positions ด้วย S/R-based trailing
        
        เรียกจาก autonomous_mt5 ทุก iteration
        """
        if not self.mt5_connected or not self.mt5:
            return
        
        try:
            positions = self.mt5.positions_get()
            if not positions:
                return
            
            # Calculate S/R levels
            sr_levels = self.calculate_sr_levels(highs, lows, closes)
            supports = sr_levels['supports']
            resistances = sr_levels['resistances']
            
            for pos in positions:
                ticket = pos.ticket
                entry = pos.price_open
                current_sl = pos.sl
                is_long = pos.type == 0
                
                # Calculate new SL based on S/R
                new_sl = self.calculate_sr_trailing_stop(
                    entry_price=entry,
                    current_price=current_price,
                    original_sl=current_sl,
                    supports=supports,
                    resistances=resistances,
                    is_long=is_long,
                    atr=atr,
                )
                
                if new_sl:
                    # Also check ATR-based trailing (whichever is better)
                    atr_sl = self.calculate_trailing_stop(
                        entry_price=entry,
                        current_price=current_price,
                        original_sl=current_sl,
                        atr=atr,
                        is_long=is_long,
                    )
                    
                    # Use the tighter SL (higher for LONG, lower for SHORT)
                    if is_long:
                        final_sl = max(new_sl, atr_sl or 0)
                    else:
                        final_sl = min(new_sl, atr_sl or float('inf'))
                    
                    if final_sl and final_sl != current_sl:
                        profit_pips = (current_price - entry) if is_long else (entry - current_price)
                        logger.info(f"🧠 MasterBrain SR Trail: Profit={profit_pips:.1f}, New SL={final_sl:.2f}")
                        self.modify_position_sl(ticket, final_sl)
                        
        except Exception as e:
            logger.error(f"Error in SR trailing: {e}")
    
    def _build_human_reasoning_v3(
        self,
        market_view: str,
        market_reasoning: str,
        risk_level: str,
        risk_reasoning: str,
        emotion_filter: str,
        emotion_reasoning: str,
        model_consensus: str,
        model_confidence: float,
        experience_insight: str,
        tf_alignment: float,
        tf_direction: str,
        tf_insight: str,
        session_ok: bool,
        session_reason: str,
        final_action: str,
        override: bool,
    ) -> str:
        """สร้าง reasoning แบบมนุษย์ v3.0 (with MultiTF + Session)"""
        
        parts = []
        
        # 1. Market view
        parts.append(f"📊 ตลาด: {market_reasoning}")
        
        # 2. Multi-TF Alignment (NEW in v3.0)
        if tf_alignment > 0.7:
            parts.append(f"📈 TF: {tf_insight}")
        elif tf_alignment < 0.3:
            parts.append(f"⚠️ TF: Mixed signals")
        
        # 3. Session (NEW in v3.0)
        if not session_ok:
            parts.append(f"🕐 Session: {session_reason}")
        
        # 4. Risk
        if risk_level in ['high', 'extreme']:
            parts.append(f"⚠️ ความเสี่ยง: {risk_reasoning}")
        
        # 5. Emotion
        if emotion_filter != 'neutral':
            parts.append(f"💭 อารมณ์ตลาด: {emotion_reasoning}")
        
        # 6. Models
        parts.append(f"🤖 Models: {model_consensus} ({model_confidence:.0%})")
        
        # 7. Experience
        if "เคยเจอ" in experience_insight:
            parts.append(f"📚 ประสบการณ์: {experience_insight}")
        
        # 8. Final decision
        if override:
            parts.append(f"🧠 MASTER OVERRIDE: {final_action}")
        else:
            parts.append(f"✅ ตัดสินใจ: {final_action}")
        
        return " | ".join(parts)
    
    def record_trade_result(
        self,
        market_state: Dict[str, float],
        action: str,
        result: str,
        pnl: float,
    ):
        """บันทึกผลและเรียนรู้"""
        
        # Determine lesson
        if result == 'win' and self.current_view and self.current_view.override_models:
            lesson = "Master override successful"
        elif result == 'loss' and self.current_view and self.current_view.override_models:
            lesson = "Master override failed - review logic"
        else:
            lesson = f"{action} at {market_state.get('regime', 'unknown')} regime"
        
        memory = TradeMemory(
            timestamp=datetime.now(),
            market_state=market_state,
            action=action,
            result=result,
            pnl=pnl,
            lesson=lesson,
        )
        
        self.trade_memory.append(memory)
        
        # ============================================
        # v3.1: Learn from experience
        # ============================================
        
        # Update streak
        if result == 'win':
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        else:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))
        
        # Learn optimal SL/TP from results
        self._learn_from_trade(market_state, result, pnl)
        
        # Update override win rate
        if self.current_view and self.current_view.override_models:
            override_trades = [
                t for t in self.trade_memory 
                if "override" in t.lesson.lower()
            ]
            if override_trades:
                wins = len([t for t in override_trades if t.result == 'win'])
                self.override_win_rate = wins / len(override_trades)
        
        # Add to training buffer for ML
        features = self._extract_features(market_state)
        action_idx = {'WAIT': 0, 'LONG': 1, 'CLOSE': 2}.get(action, 0)
        label = 1 if result == 'win' else 0  # Win = correct, Loss = incorrect
        
        if label == 1:  # Only learn from wins
            self.training_buffer.append((features, action_idx))
        
        # Train ML if enough data
        if len(self.training_buffer) >= self.min_samples_to_train:
            self._train_network()
        
        streak_info = f"+{self.current_streak}W" if self.current_streak > 0 else f"{self.current_streak}L"
        logger.debug(f"📝 Recorded: {action} → {result}, Streak: {streak_info}, Lesson: {lesson}")
    
    def _learn_from_trade(
        self,
        market_state: Dict[str, float],
        result: str,
        pnl: float,
    ):
        """เรียนรู้ค่า optimal จากผลเทรด"""
        
        regime = market_state.get('regime', 'unknown')
        sl_used = market_state.get('sl_atr', self.learned_params['optimal_sl_atr'])
        tp_used = market_state.get('tp_atr', self.learned_params['optimal_tp_atr'])
        
        # Track performance by SL/TP setting
        key = f"{sl_used:.1f}_{tp_used:.1f}"
        if key not in self.sl_tp_performance:
            self.sl_tp_performance[key] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
        
        if result == 'win':
            self.sl_tp_performance[key]['wins'] += 1
        else:
            self.sl_tp_performance[key]['losses'] += 1
        self.sl_tp_performance[key]['pnl'] += pnl
        
        # Update regime adjustments
        if regime not in self.learned_params['regime_adjustments']:
            self.learned_params['regime_adjustments'][regime] = {
                'wins': 0, 'losses': 0, 'sl_adj': 1.0, 'tp_adj': 1.0
            }
        
        regime_data = self.learned_params['regime_adjustments'][regime]
        if result == 'win':
            regime_data['wins'] += 1
            # ชนะ = TP ดี ไม่ต้องปรับมาก
            regime_data['tp_adj'] = min(1.5, regime_data['tp_adj'] * 1.01)
        else:
            regime_data['losses'] += 1
            # แพ้ = SL อาจแคบไป ปรับกว้างขึ้น
            regime_data['sl_adj'] = min(2.0, regime_data['sl_adj'] * 1.02)
        
        # Find best performing SL/TP combination
        if len(self.sl_tp_performance) >= 5:
            best_key = max(
                self.sl_tp_performance.keys(),
                key=lambda k: self.sl_tp_performance[k]['pnl']
            )
            best_sl, best_tp = map(float, best_key.split('_'))
            
            # Gradually shift towards best
            self.learned_params['optimal_sl_atr'] = (
                self.learned_params['optimal_sl_atr'] * 0.9 + best_sl * 0.1
            )
            self.learned_params['optimal_tp_atr'] = (
                self.learned_params['optimal_tp_atr'] * 0.9 + best_tp * 0.1
            )
        
        # Update streak multiplier
        if self.current_streak > 0:
            # Winning streak = เพิ่ม lot
            self.learned_params['streak_multiplier'] = min(
                1.5, 1.0 + (self.current_streak * 0.05)
            )
        elif self.current_streak < 0:
            # Losing streak = ลด lot
            self.learned_params['streak_multiplier'] = max(
                0.5, 1.0 + (self.current_streak * 0.1)  # ลดเร็วกว่า
            )
        else:
            self.learned_params['streak_multiplier'] = 1.0
    
    def calculate_risk_params(
        self,
        entry_price: float,
        atr: float,
        confidence: float,
        regime: str,
        equity: float,
        is_long: bool = True,
    ) -> Dict[str, float]:
        """
        คำนวณ lot/SL/TP จากประสบการณ์ที่เรียนรู้
        
        Returns:
            Dict with lot_size, sl, tp, sl_pips, tp_pips, risk_pct
        """
        
        # Get learned base values
        base_sl_atr = self.learned_params['optimal_sl_atr']
        base_tp_atr = self.learned_params['optimal_tp_atr']
        base_risk = self.learned_params['optimal_risk_pct']
        streak_mult = self.learned_params['streak_multiplier']
        
        # Adjust by regime (if learned)
        regime_adj = self.learned_params['regime_adjustments'].get(regime, {})
        sl_adj = regime_adj.get('sl_adj', 1.0)
        tp_adj = regime_adj.get('tp_adj', 1.0)
        
        # Adjust by confidence
        conf_adj = 0.5 + (confidence * 0.5)  # 0.5-1.0
        
        # Calculate SL/TP in pips
        sl_pips = atr * base_sl_atr * sl_adj
        tp_pips = atr * base_tp_atr * tp_adj * conf_adj
        
        # Ensure minimum R:R
        min_rr = 1.5
        if tp_pips / sl_pips < min_rr:
            tp_pips = sl_pips * min_rr
        
        # Calculate lot size
        risk_pct = base_risk * streak_mult * conf_adj
        risk_pct = max(0.005, min(0.05, risk_pct))  # 0.5% - 5%
        
        risk_amount = equity * risk_pct
        pip_value_per_lot = 100  # Gold: $100 per pip per lot
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        lot_size = round(max(0.01, min(1.0, lot_size)), 2)
        
        # Calculate actual SL/TP prices
        if is_long:
            sl = entry_price - sl_pips
            tp = entry_price + tp_pips
        else:
            sl = entry_price + sl_pips
            tp = entry_price - tp_pips
        
        return {
            'lot_size': lot_size,
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'sl_pips': round(sl_pips, 2),
            'tp_pips': round(tp_pips, 2),
            'risk_pct': risk_pct,
            'reward_ratio': round(tp_pips / sl_pips, 2),
            'streak_mult': streak_mult,
            'regime_sl_adj': sl_adj,
            'regime_tp_adj': tp_adj,
        }
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        original_sl: float,
        atr: float,
        is_long: bool,
    ) -> Optional[float]:
        """
        คำนวณ trailing stop จากประสบการณ์ที่เรียนรู้
        
        Returns:
            New SL price or None if no update needed
        """
        
        # Calculate profit in ATR terms
        if is_long:
            profit_pips = current_price - entry_price
        else:
            profit_pips = entry_price - current_price
        
        profit_atr = profit_pips / atr if atr > 0 else 0
        sl_distance = abs(entry_price - original_sl)
        profit_rr = profit_pips / sl_distance if sl_distance > 0 else 0
        
        # Get learned trailing params
        trail_start = self.learned_params['trailing_start_profit']
        trail_dist = self.learned_params['trailing_distance']
        
        # Start trailing when profit reaches threshold
        if profit_rr < trail_start:
            return None
        
        # Calculate new SL level (trail by trail_dist ATR behind price)
        if is_long:
            new_sl = current_price - (atr * trail_dist)
            # Only move SL up, never down
            if new_sl <= original_sl:
                return None
            return round(new_sl, 2)
        else:
            new_sl = current_price + (atr * trail_dist)
            # Only move SL down for shorts, never up
            if new_sl >= original_sl:
                return None
            return round(new_sl, 2)
    
    def should_stop_trading(self, current_equity: float) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าควรหยุดเทรดเพราะ drawdown หรือไม่
        
        Returns:
            (should_stop, reason)
        """
        
        # Update peak
        if current_equity > self.equity_peak:
            self.equity_peak = current_equity
        
        # Calculate drawdown
        if self.equity_peak > 0:
            self.current_drawdown = (self.equity_peak - current_equity) / self.equity_peak
        
        # Check against learned threshold
        if self.current_drawdown >= self.learned_drawdown_threshold:
            return True, f"Drawdown {self.current_drawdown:.1%} >= threshold {self.learned_drawdown_threshold:.1%}"
        
        # Check losing streak
        if self.current_streak <= -3:
            return True, f"Losing streak: {abs(self.current_streak)} consecutive losses"
        
        return False, ""
    
    
    def _extract_features(self, market_state: Dict[str, float]) -> np.ndarray:
        """Extract 12 features for ML model"""
        features = np.array([
            market_state.get('price', 0) / 3000,  # Normalized price
            market_state.get('trend', 0),
            market_state.get('volatility', 0.5),
            market_state.get('rsi', 50) / 100,
            market_state.get('atr', 15) / 100,
            1.0 if 'up' in market_state.get('regime', '') else 0.0,
            1.0 if 'down' in market_state.get('regime', '') else 0.0,
            1.0 if 'ranging' in market_state.get('regime', '') else 0.0,
            market_state.get('macd', 0),
            market_state.get('bb_position', 0.5),
            market_state.get('momentum', 0),
            market_state.get('volume_ratio', 1.0),
        ], dtype=np.float32)
        return features
    
    def _get_ml_prediction(
        self,
        market_state: Dict[str, float],
    ) -> Tuple[str, float]:
        """Get prediction from ML network"""
        self.decision_network.eval()
        
        features = self._extract_features(market_state)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.transformer_brain(state_tensor)
        
        action_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action_idx].item()
        
        actions = ['WAIT', 'LONG', 'CLOSE']
        return actions[action_idx], confidence
    
    def _train_network(self):
        """Train Transformer network from experience"""
        if len(self.training_buffer) < self.batch_size:
            return
        
        self.transformer_brain.train()
        
        # Create batch
        indices = np.random.choice(len(self.training_buffer), self.batch_size, replace=False)
        batch = [self.training_buffer[i] for i in indices]
        
        features = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        labels = torch.LongTensor([b[1] for b in batch]).to(self.device)
        
        # Train
        self.optimizer.zero_grad()
        outputs = self.transformer_brain(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=-1)
        self.ml_accuracy = (predictions == labels).float().mean().item()
        
        # Save model periodically
        if len(self.training_buffer) % 100 == 0:
            self._save_model()
        
        logger.debug(f"🧠 Transformer trained: Loss={loss.item():.4f}, Acc={self.ml_accuracy:.1%}")
    
    def _save_model(self):
        """Save Transformer model AND state to SQLite"""
        # 1. Save PyTorch model
        model_path = "ai_agent/models/master_brain_v3.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.transformer_brain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ml_accuracy': self.ml_accuracy,
            'market_states': self.market_states,
        }, model_path)
        
        # 2. Save learned params to SQLite DB
        try:
            import sqlite3
            self.db_path = "trade_memory.db"
            self.db_conn = sqlite3.connect(self.db_path)
            
            # Always ensure table exists
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS master_brain_state (
                    id INTEGER PRIMARY KEY,
                    key TEXT UNIQUE,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db_conn.commit()
            
            state = {
                'version': '3.3',
                'learned_params': self.learned_params,
                'current_streak': self.current_streak,
                'max_win_streak': self.max_win_streak,
                'max_loss_streak': self.max_loss_streak,
                'total_decisions': self.total_decisions,
                'override_win_rate': getattr(self, 'override_win_rate', 0.5),
                'ml_accuracy': self.ml_accuracy,
                'model_accuracy': getattr(self, 'model_accuracy', {}),
                'hourly_performance': getattr(self, 'hourly_performance', {}),
                'session_performance': getattr(self, 'session_performance', {}),
            }
            
            for key, value in state.items():
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO master_brain_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, json.dumps(value)))
            
            self.db_conn.commit()
            self.db_conn.close()
            logger.debug(f"💾 MasterBrain v3.3 saved to trade_memory.db")
        except Exception as e:
            logger.debug(f"Could not save to SQLite: {e}")
        
        logger.debug("💾 MasterBrain v3.3 model saved")
    
    def _load_model(self):
        """Load Transformer model AND state from SQLite"""
        # 1. Load PyTorch model
        model_paths = [
            "ai_agent/models/master_brain_v3.pt",
            "ai_agent/models/master_brain.pt",
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.transformer_brain.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.ml_accuracy = checkpoint.get('ml_accuracy', 0.0)
                    
                    # Restore market states
                    saved_states = checkpoint.get('market_states', {})
                    for state, params in saved_states.items():
                        if state in self.market_states:
                            self.market_states[state] = params
                    
                    break
                except Exception as e:
                    logger.warning(f"Could not load {model_path}: {e}")
        
        # 2. Load learned params from SQLite DB
        try:
            import sqlite3
            self.db_path = "trade_memory.db"
            self.db_conn = sqlite3.connect(self.db_path)
            
            cursor = self.db_conn.execute("SELECT key, value FROM master_brain_state")
            rows = cursor.fetchall()
            
            if rows:
                state = {row[0]: json.loads(row[1]) for row in rows}
                
                if 'learned_params' in state:
                    self.learned_params = state['learned_params']
                if 'model_accuracy' in state and hasattr(self, 'model_accuracy'):
                    self.model_accuracy = state['model_accuracy']
                if 'hourly_performance' in state and hasattr(self, 'hourly_performance'):
                    self.hourly_performance = state['hourly_performance']
                if 'session_performance' in state and hasattr(self, 'session_performance'):
                    self.session_performance = state['session_performance']
                
                if 'current_streak' in state:
                    self.current_streak = state['current_streak']
                self.max_win_streak = state.get('max_win_streak', 0)
                self.max_loss_streak = state.get('max_loss_streak', 0)
                self.total_decisions = state.get('total_decisions', 0)
                self.override_win_rate = state.get('override_win_rate', 0.5)
                
                logger.info(f"📂 MasterBrain v3.3 loaded from DB: {self.total_decisions} decisions")
        except Exception as e:
            logger.debug(f"Could not load from SQLite: {e}")
        
        logger.info(f"📂 MasterBrain v3.3 loaded (Acc: {self.ml_accuracy:.1%})")
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะปัจจุบัน"""
        return {
            "version": "3.1",
            "total_decisions": self.total_decisions,
            "override_count": self.override_count,
            "override_rate": self.override_count / max(1, self.total_decisions),
            "override_win_rate": self.override_win_rate,
            "memory_size": len(self.trade_memory),
            "vector_memory_size": len(self.vector_memory.memories),
            "training_buffer_size": len(self.training_buffer),
            "ml_accuracy": self.ml_accuracy,
            "market_states_count": len(self.market_states),
            "current_streak": self.current_streak,
            "learned_sl_atr": self.learned_params['optimal_sl_atr'],
            "learned_tp_atr": self.learned_params['optimal_tp_atr'],
            "streak_multiplier": self.learned_params['streak_multiplier'],
            "has_transformer": True,
            "has_vector_memory": True,
            "has_adaptive_risk": True,
            "has_mtf_fusion": True,
            "current_view": self.current_view,
        }


def create_master_brain() -> MasterBrain:
    """Factory function"""
    return MasterBrain()


if __name__ == "__main__":
    # Test v3.0
    brain = MasterBrain()
    
    market = {
        'price': 2650.0,
        'trend': 0.7,
        'volatility': 0.3,
        'regime': 'trending_up',
        'atr': 15.0,
    }
    
    indicators = {
        'rsi': 65,
        'macd': 0.5,
        'bb_position': 0.6,
    }
    
    model_votes = {
        'lstm': ('LONG', 0.57),
        'xgb': ('LONG', 0.57),
        'ppo': ('WAIT', 0.19),
    }
    
    # Multi-TF data for v3.0
    tf_data = {
        'M15': {'trend': 0.3, 'momentum': 0.2},
        'H1': {'trend': 0.5, 'momentum': 0.4},
        'H4': {'trend': 0.6, 'momentum': 0.5},
        'D1': {'trend': 0.7, 'momentum': 0.3},
    }
    
    thought = brain.think(market, model_votes, indicators, tf_data)
    
    print("\n" + "="*60)
    print("   MASTER BRAIN v3.0 TEST")
    print("="*60)
    print(f"\n🧠 Reasoning:\n{thought.reasoning}")
    print(f"\n📊 Decision: {thought.suggested_action}")
    print(f"💪 Confidence: {thought.confidence:.1%}")
    print(f"🤖 ML Confidence: {thought.ml_confidence:.1%}")
    print(f"🔄 Override models: {thought.override_models}")
    print(f"⚠️ Risk level: {thought.risk_level}")
    
    status = brain.get_status()
    print(f"\n📊 Status:")
    print(f"   Version: {status['version']}")
    print(f"   Memory: {status['memory_size']} trades")
    print(f"   Streak: {status['current_streak']}")
    print(f"   Learned SL: {status['learned_sl_atr']:.2f}× ATR")
    print(f"   Learned TP: {status['learned_tp_atr']:.2f}× ATR")
    print(f"   Streak Mult: {status['streak_multiplier']:.2f}")
    print(f"   Has Adaptive Risk: {status['has_adaptive_risk']}")
    
    # Test v3.1 Risk Params
    print("\n" + "="*60)
    print("   v3.1 ADAPTIVE RISK TEST")
    print("="*60)
    
    risk_params = brain.calculate_risk_params(
        entry_price=2650.0,
        atr=15.0,
        confidence=0.85,
        regime='trending_up',
        equity=10000,
        is_long=True,
    )
    
    print(f"\n   Lot Size: {risk_params['lot_size']}")
    print(f"   SL: {risk_params['sl']} ({risk_params['sl_pips']:.1f} pips)")
    print(f"   TP: {risk_params['tp']} ({risk_params['tp_pips']:.1f} pips)")
    print(f"   Risk: {risk_params['risk_pct']:.2%}")
    print(f"   R:R: {risk_params['reward_ratio']}")
    print(f"   Streak Mult: {risk_params['streak_mult']:.2f}")
    
    # Test Trailing Stop
    print("\n" + "="*60)
    print("   v3.1 TRAILING STOP TEST")
    print("="*60)
    
    entry = 2650.0
    sl = 2635.0  # 15 pips SL
    
    # Simulate price at 2R profit
    price_at_2r = 2680.0  # +30 pips = 2R
    new_sl = brain.calculate_trailing_stop(
        entry_price=entry,
        current_price=price_at_2r,
        original_sl=sl,
        atr=15.0,
        is_long=True,
    )
    
    print(f"\n   Entry: {entry}")
    print(f"   Original SL: {sl}")
    print(f"   Price at 2R: {price_at_2r}")
    print(f"   New Trailing SL: {new_sl}")
    
    # Test Plan Trade (Human-like)
    print("\n" + "="*60)
    print("   v3.1 HUMAN-LIKE TRADE PLANNING")
    print("="*60)
    
    plan = brain.plan_trade(market, indicators, tf_data)
    print(f"\n{plan['thinking']}")

