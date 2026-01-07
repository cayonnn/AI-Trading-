"""
AI Risk Manager (v2.0 - Adaptive Risk)
======================================
Dynamic lot size, SL, TP calculation based on:
- AI Confidence
- Market Regime
- Volatility
- Win Rate History
- Kelly Criterion
- Win/Loss Streak (NEW in v2.0)
- Adaptive Multipliers (NEW in v2.0)

The AI fully controls trading parameters based on learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from collections import deque


@dataclass
class RiskParameters:
    """Dynamic risk parameters calculated by AI"""
    lot_size: float
    stop_loss: float  # Price level
    take_profit: float  # Price level
    sl_pips: float
    tp_pips: float
    risk_pct: float  # % of equity at risk
    reward_ratio: float
    confidence: float
    reasoning: str
    streak_adjustment: float = 1.0  # Streak-based multiplier


class AIRiskManager:
    """
    AI-controlled risk management system (v2.0 - Adaptive Risk).
    
    New in v2.0:
    - Win/Loss streak tracking
    - Adaptive lot size based on streak
    - Reduce risk after losses, increase after wins
    
    Dynamically calculates:
    - Position size (lot) based on Kelly + Confidence + Streak
    - Stop Loss based on ATR + Regime + Volatility
    - Take Profit based on Confidence + Trend Strength
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.02,  # 2% base risk per trade
        max_risk_pct: float = 0.05,   # 5% max risk per trade
        min_lot: float = 0.01,
        max_lot: float = 1.0,
        min_sl_atr: float = 0.5,
        max_sl_atr: float = 2.0,
        min_rr: float = 1.5,  # Minimum reward:risk ratio
    ):
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.min_sl_atr = min_sl_atr
        self.max_sl_atr = max_sl_atr
        self.min_rr = min_rr
        
        # Learning from history
        self.trade_history: deque = deque(maxlen=100)
        self.win_rate = 0.5
        self.avg_win = 1.0
        self.avg_loss = 1.0
        
        # v2.0: Streak tracking
        self.current_streak = 0  # Positive = wins, Negative = losses
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # v2.0: Adaptive multipliers
        self.streak_config = {
            'win_boost_per_streak': 0.05,   # +5% lot per consecutive win
            'loss_reduce_per_streak': 0.10,  # -10% lot per consecutive loss
            'max_boost': 1.5,               # Max 50% boost
            'max_reduce': 0.5,              # Max 50% reduction
            'reset_after_wins': 3,          # Reset boost after 3 wins
            'reset_after_losses': 2,        # Reset reduce after 2 losses
        }
        
        # Regime-specific parameters learned from experience
        self.regime_parameters = {
            'trending_up': {'sl_mult': 1.2, 'tp_mult': 1.5, 'lot_mult': 1.2},
            'trending_down': {'sl_mult': 1.2, 'tp_mult': 1.5, 'lot_mult': 1.2},
            'ranging': {'sl_mult': 0.8, 'tp_mult': 0.8, 'lot_mult': 0.8},
            'volatile': {'sl_mult': 1.5, 'tp_mult': 1.3, 'lot_mult': 0.5},
            'calm': {'sl_mult': 0.7, 'tp_mult': 1.0, 'lot_mult': 1.0},
            'unknown': {'sl_mult': 1.0, 'tp_mult': 1.0, 'lot_mult': 0.8},
        }
        
        logger.info("AIRiskManager v2.0 initialized - Adaptive Risk | Streak Tracking")
    
    def calculate_parameters(
        self,
        entry_price: float,
        atr: float,
        confidence: float,
        regime: str,
        volatility: float,
        equity: float,
        is_long: bool = True,
        trend_strength: float = 0.5,
    ) -> RiskParameters:
        """
        AI calculates optimal lot size, SL, and TP.
        
        Args:
            entry_price: Current price
            atr: Average True Range
            confidence: AI confidence (0-1)
            regime: Market regime
            volatility: Volatility level (0-1)
            equity: Account equity
            is_long: True for BUY, False for SELL
            trend_strength: Trend strength (0-1)
            
        Returns:
            RiskParameters with all values
        """
        
        # 1. Get regime-specific multipliers
        regime_params = self.regime_parameters.get(
            regime, 
            self.regime_parameters['unknown']
        )
        
        # 2. Calculate dynamic SL multiplier
        sl_mult = self._calculate_sl_multiplier(
            confidence=confidence,
            volatility=volatility,
            regime_mult=regime_params['sl_mult'],
        )
        
        # 3. Calculate dynamic TP multiplier
        tp_mult = self._calculate_tp_multiplier(
            confidence=confidence,
            trend_strength=trend_strength,
            regime_mult=regime_params['tp_mult'],
        )
        
        # 4. Calculate SL/TP prices
        sl_pips = atr * sl_mult
        tp_pips = atr * tp_mult
        
        # Ensure minimum R:R ratio
        if tp_pips / sl_pips < self.min_rr:
            tp_pips = sl_pips * self.min_rr
        
        if is_long:
            stop_loss = entry_price - sl_pips
            take_profit = entry_price + tp_pips
        else:
            stop_loss = entry_price + sl_pips
            take_profit = entry_price - tp_pips
        
        # 5. Calculate position size using Kelly + Confidence
        risk_pct = self._calculate_risk_pct(
            confidence=confidence,
            regime_mult=regime_params['lot_mult'],
            volatility=volatility,
        )
        
        lot_size = self._calculate_lot_size(
            equity=equity,
            risk_pct=risk_pct,
            sl_pips=sl_pips,
            entry_price=entry_price,
        )
        
        # 6. v2.0: Apply streak adjustment
        streak_adj = self._calculate_streak_adjustment()
        lot_size = round(lot_size * streak_adj, 2)
        lot_size = max(self.min_lot, min(self.max_lot, lot_size))
        
        # 7. Build reasoning
        streak_info = ""
        if self.current_streak > 0:
            streak_info = f", Streak=+{self.current_streak}W"
        elif self.current_streak < 0:
            streak_info = f", Streak={self.current_streak}L"
        
        reasoning = (
            f"Regime={regime}, Conf={confidence:.0%}, "
            f"SL={sl_mult:.1f}×ATR, TP={tp_mult:.1f}×ATR, "
            f"Risk={risk_pct:.1%}, R:R={tp_pips/sl_pips:.1f}"
            f"{streak_info}"
        )
        
        logger.debug(f"AI Risk: {reasoning}")
        
        return RiskParameters(
            lot_size=lot_size,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            sl_pips=round(sl_pips, 2),
            tp_pips=round(tp_pips, 2),
            risk_pct=risk_pct,
            reward_ratio=round(tp_pips / sl_pips, 2),
            confidence=confidence,
            reasoning=reasoning,
            streak_adjustment=streak_adj,
        )
    
    def _calculate_streak_adjustment(self) -> float:
        """
        คำนวณ multiplier ตาม win/loss streak
        
        Win streak = เพิ่ม lot
        Loss streak = ลด lot
        """
        if self.current_streak > 0:
            # Winning streak: boost by 5% per win, max 50%
            boost = self.streak_config['win_boost_per_streak'] * self.current_streak
            return min(self.streak_config['max_boost'], 1.0 + boost)
        elif self.current_streak < 0:
            # Losing streak: reduce by 10% per loss, max 50% reduction
            reduction = self.streak_config['loss_reduce_per_streak'] * abs(self.current_streak)
            return max(self.streak_config['max_reduce'], 1.0 - reduction)
        return 1.0
    
    def _calculate_sl_multiplier(
        self,
        confidence: float,
        volatility: float,
        regime_mult: float,
    ) -> float:
        """
        Calculate dynamic SL multiplier.
        
        Higher confidence = tighter SL (more precise entry)
        Higher volatility = wider SL (avoid stop hunting)
        """
        # Base: 1.0 ATR
        base = 1.0
        
        # Confidence factor: 90%+ = 0.8, 60% = 1.2
        conf_factor = 1.3 - (confidence * 0.5)
        
        # Volatility factor: high vol = wider SL
        vol_factor = 0.8 + (volatility * 0.4)
        
        # Combine
        sl_mult = base * conf_factor * vol_factor * regime_mult
        
        # Clamp to limits
        return max(self.min_sl_atr, min(self.max_sl_atr, sl_mult))
    
    def _calculate_tp_multiplier(
        self,
        confidence: float,
        trend_strength: float,
        regime_mult: float,
    ) -> float:
        """
        Calculate dynamic TP multiplier.
        
        Higher confidence = larger TP target
        Stronger trend = larger TP target (let winners run)
        """
        # Base: 2.0 ATR (minimum R:R = 2:1 with 1.0 SL)
        base = 2.0
        
        # Confidence factor: 90%+ = 1.5, 60% = 1.0
        conf_factor = 0.5 + confidence
        
        # Trend factor: strong trend = larger target
        trend_factor = 0.8 + (trend_strength * 0.4)
        
        # Combine
        tp_mult = base * conf_factor * trend_factor * regime_mult
        
        # Clamp to reasonable limits
        return max(1.5, min(5.0, tp_mult))
    
    def _calculate_risk_pct(
        self,
        confidence: float,
        regime_mult: float,
        volatility: float,
    ) -> float:
        """
        Calculate dynamic risk percentage using modified Kelly.
        
        Higher confidence = larger position (more risk)
        Higher volatility = smaller position (less risk)
        """
        # Kelly fraction based on historical performance
        if self.avg_loss > 0:
            kelly = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) / self.avg_win
        else:
            kelly = 0.5
        
        # Clamp Kelly to reasonable range
        kelly = max(0.1, min(0.5, kelly))
        
        # Adjust by confidence (higher confidence = more risk)
        conf_adj = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        
        # Adjust by volatility (higher vol = less risk)
        vol_adj = 1.2 - (volatility * 0.4)  # 0.8 to 1.2
        
        # Calculate final risk
        risk = self.base_risk_pct * kelly * conf_adj * vol_adj * regime_mult
        
        # Clamp to limits
        return max(0.01, min(self.max_risk_pct, risk))
    
    def _calculate_lot_size(
        self,
        equity: float,
        risk_pct: float,
        sl_pips: float,
        entry_price: float,
    ) -> float:
        """
        Calculate lot size based on risk amount.
        
        lot = (equity × risk%) / (sl_pips × pip_value)
        For Gold: 1 lot = 100 oz, pip value ≈ $1 per pip
        """
        # Risk amount in $
        risk_amount = equity * risk_pct
        
        # For Gold (XAUUSD): 1 lot = 100 oz
        # pip value per lot = $1 per 0.01 move = $100 per pip
        pip_value_per_lot = 100  # $100 per pip per lot
        
        # Calculate lot size
        if sl_pips > 0:
            lot = risk_amount / (sl_pips * pip_value_per_lot)
        else:
            lot = self.min_lot
        
        # Round to standard sizes
        lot = round(lot, 2)
        
        # Clamp to limits
        return max(self.min_lot, min(self.max_lot, lot))
    
    def update_from_trade(
        self,
        pnl: float,
        sl_pips: float,
        tp_pips: float,
        regime: str,
        hit_tp: bool,
    ):
        """
        Learn from trade result to improve future decisions.
        """
        self.trade_history.append({
            'pnl': pnl,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'regime': regime,
            'hit_tp': hit_tp,
            'win': pnl > 0,
        })
        
        # Update statistics
        wins = [t for t in self.trade_history if t['win']]
        losses = [t for t in self.trade_history if not t['win']]
        
        if len(self.trade_history) > 0:
            self.win_rate = len(wins) / len(self.trade_history)
        
        if wins:
            self.avg_win = np.mean([t['pnl'] for t in wins])
        if losses:
            self.avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        # Adapt regime parameters based on performance
        self._adapt_regime_parameters(regime, pnl, hit_tp)
    
    def _adapt_regime_parameters(
        self,
        regime: str,
        pnl: float,
        hit_tp: bool,
    ):
        """
        Evolve regime parameters based on results.
        """
        if regime not in self.regime_parameters:
            return
        
        params = self.regime_parameters[regime]
        
        # If hit TP, slightly increase TP multiplier
        if hit_tp and pnl > 0:
            params['tp_mult'] = min(2.0, params['tp_mult'] * 1.02)
            params['lot_mult'] = min(1.5, params['lot_mult'] * 1.01)
        
        # If stopped out, slightly increase SL multiplier
        if not hit_tp and pnl < 0:
            params['sl_mult'] = min(2.0, params['sl_mult'] * 1.02)
            params['lot_mult'] = max(0.5, params['lot_mult'] * 0.99)
        
        logger.debug(f"Adapted {regime} params: {params}")


def create_ai_risk_manager() -> AIRiskManager:
    """Factory function"""
    return AIRiskManager()


if __name__ == "__main__":
    # Test
    rm = create_ai_risk_manager()
    
    params = rm.calculate_parameters(
        entry_price=2650.0,
        atr=15.0,
        confidence=0.85,
        regime='trending_up',
        volatility=0.5,
        equity=10000,
        is_long=True,
        trend_strength=0.7,
    )
    
    print(f"Lot: {params.lot_size}")
    print(f"SL: {params.stop_loss} (-{params.sl_pips} pips)")
    print(f"TP: {params.take_profit} (+{params.tp_pips} pips)")
    print(f"Risk: {params.risk_pct:.1%}")
    print(f"R:R: {params.reward_ratio}")
    print(f"Reasoning: {params.reasoning}")
