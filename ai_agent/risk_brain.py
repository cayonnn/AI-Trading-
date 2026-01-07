"""
Risk Brain Module
==================
Advanced Risk Management ด้วย AI

Features:
1. Dynamic Risk Sizing - ปรับ risk ตาม conditions
2. Correlation Risk - วัด correlation กับ assets อื่น
3. Drawdown Protection - ป้องกัน drawdown
4. Risk-Adjusted Returns - คำนวณ Sharpe, Sortino
5. Position Heat Map - แสดง risk ของ positions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from loguru import logger


@dataclass
class RiskMetrics:
    """Risk Metrics"""
    current_risk_pct: float = 0.0
    max_risk_pct: float = 2.0
    daily_var: float = 0.0  # Value at Risk
    daily_cvar: float = 0.0  # Conditional VaR
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0


@dataclass
class RiskDecision:
    """Risk-based Decision"""
    allow_trade: bool
    max_position_pct: float
    recommended_sl_pct: float
    recommended_tp_pct: float
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    warnings: List[str] = field(default_factory=list)


class RiskBrain:
    """
    Advanced Risk Management AI
    
    ความสามารถ:
    1. ประเมิน risk แบบ real-time
    2. ปรับ position size ตาม risk
    3. ป้องกัน drawdown
    4. คำนวณ risk metrics
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_daily_risk: float = 0.06,
        max_drawdown: float = 0.15,
        var_confidence: float = 0.95,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        
        # Tracking
        self.equity_history: deque = deque(maxlen=252)  # 1 year of daily
        self.trade_returns: deque = deque(maxlen=100)
        self.daily_pnl: deque = deque(maxlen=30)
        
        # State
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.daily_risk_used = 0.0
        self.open_positions_risk = 0.0
        
        # Risk adjustments based on conditions
        self.volatility_multiplier = 1.0
        self.drawdown_multiplier = 1.0
        self.streak_multiplier = 1.0
        
        logger.info("RiskBrain initialized")
    
    def evaluate_risk(
        self,
        equity: float,
        volatility: float,
        confidence: float,
        regime: str,
    ) -> RiskDecision:
        """
        ประเมิน risk และให้คำแนะนำ
        
        Args:
            equity: Current equity
            volatility: Market volatility
            confidence: Trade confidence
            regime: Market regime
            
        Returns:
            RiskDecision with recommendations
        """
        
        warnings = []
        
        # Update equity tracking
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate current drawdown
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # 1. Check drawdown limit
        if current_dd >= self.max_drawdown:
            return RiskDecision(
                allow_trade=False,
                max_position_pct=0,
                recommended_sl_pct=0,
                recommended_tp_pct=0,
                risk_level="extreme",
                warnings=["Maximum drawdown reached - trading halted"],
            )
        
        # 2. Check daily risk limit
        if self.daily_risk_used >= self.max_daily_risk:
            return RiskDecision(
                allow_trade=False,
                max_position_pct=0,
                recommended_sl_pct=0,
                recommended_tp_pct=0,
                risk_level="high",
                warnings=["Daily risk limit reached"],
            )
        
        # 3. Calculate dynamic risk
        base_risk = self.max_risk_per_trade
        
        # Adjust for volatility
        vol_ratio = volatility / 0.015  # 1.5% as baseline
        if vol_ratio > 2:
            self.volatility_multiplier = 0.5
            warnings.append("High volatility - reduced position")
        elif vol_ratio > 1.5:
            self.volatility_multiplier = 0.7
        elif vol_ratio < 0.5:
            self.volatility_multiplier = 1.2
        else:
            self.volatility_multiplier = 1.0
        
        # Adjust for drawdown
        if current_dd > 0.10:
            self.drawdown_multiplier = 0.5
            warnings.append("In drawdown - reduced position")
        elif current_dd > 0.05:
            self.drawdown_multiplier = 0.7
        else:
            self.drawdown_multiplier = 1.0
        
        # Adjust for confidence
        conf_multiplier = confidence if confidence > 0.5 else 0.5
        
        # Adjust for regime
        regime_multiplier = 1.0
        if regime in ["volatile", "unknown"]:
            regime_multiplier = 0.7
            warnings.append(f"Risky regime: {regime}")
        
        # Final risk calculation
        final_risk = (
            base_risk *
            self.volatility_multiplier *
            self.drawdown_multiplier *
            conf_multiplier *
            regime_multiplier
        )
        
        final_risk = min(final_risk, self.max_risk_per_trade)
        
        # Remaining daily risk
        remaining_daily = self.max_daily_risk - self.daily_risk_used
        final_risk = min(final_risk, remaining_daily)
        
        # Determine risk level
        if current_dd > 0.10 or vol_ratio > 2:
            risk_level = "high"
        elif current_dd > 0.05 or vol_ratio > 1.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Recommended SL/TP
        # Minimum 500 pips for XAUUSD (0.5 points = $5.00 per lot)
        min_sl_pct = 0.025  # ~500 pips for Gold at $2000 = $5
        sl_pct = max(min_sl_pct, min(0.03, volatility * 1.5))  # Min 500 pips, Max 600 pips
        tp_pct = sl_pct * 3  # Min 3:1 R:R
        
        return RiskDecision(
            allow_trade=True,
            max_position_pct=final_risk,
            recommended_sl_pct=sl_pct,
            recommended_tp_pct=tp_pct,
            risk_level=risk_level,
            warnings=warnings,
        )
    
    def calculate_var(self, returns: List[float] = None) -> float:
        """Calculate Value at Risk"""
        
        if returns is None:
            returns = list(self.trade_returns)
        
        if len(returns) < 10:
            return 0.02  # Default 2%
        
        # Parametric VaR
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf(1 - self.var_confidence)
        
        var = -(mean + z * std)
        
        return max(var, 0)
    
    def calculate_cvar(self, returns: List[float] = None) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        
        if returns is None:
            returns = list(self.trade_returns)
        
        if len(returns) < 10:
            return 0.03
        
        var = self.calculate_var(returns)
        
        # CVaR is average of returns below VaR
        tail_returns = [r for r in returns if r <= -var]
        
        if tail_returns:
            return -np.mean(tail_returns)
        
        return var * 1.5
    
    def calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        
        returns = list(self.trade_returns)
        
        if len(returns) < 10:
            return 0
        
        mean_return = np.mean(returns) * 252  # Annualized
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0
        
        return (mean_return - risk_free_rate) / std_return
    
    def calculate_sortino(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        
        returns = list(self.trade_returns)
        
        if len(returns) < 10:
            return 0
        
        mean_return = np.mean(returns) * 252
        
        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 10  # Very good
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0
        
        return (mean_return - risk_free_rate) / downside_std
    
    def record_trade(self, pnl_pct: float, position_risk: float):
        """บันทึกผล trade"""
        
        self.trade_returns.append(pnl_pct)
        self.daily_risk_used += position_risk
    
    def record_equity(self, equity: float):
        """บันทึก equity"""
        
        self.equity_history.append({
            "timestamp": datetime.now(),
            "equity": equity,
        })
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_equity = equity
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_risk_used = 0.0
    
    def get_metrics(self) -> RiskMetrics:
        """ดึง Risk Metrics"""
        
        current_dd = (
            (self.peak_equity - self.current_equity) / self.peak_equity
            if self.peak_equity > 0 else 0
        )
        
        # Calculate max drawdown from history
        if self.equity_history:
            equities = [e['equity'] for e in self.equity_history]
            peak = np.maximum.accumulate(equities)
            dd = (peak - equities) / peak
            max_dd = max(dd) if len(dd) > 0 else 0
        else:
            max_dd = 0
        
        return RiskMetrics(
            current_risk_pct=self.daily_risk_used,
            max_risk_pct=self.max_daily_risk,
            daily_var=self.calculate_var(),
            daily_cvar=self.calculate_cvar(),
            sharpe_ratio=self.calculate_sharpe(),
            sortino_ratio=self.calculate_sortino(),
            max_drawdown=max_dd,
            current_drawdown=current_dd,
        )
    
    def get_risk_score(self) -> Tuple[int, str]:
        """
        ดึง Risk Score (0-100)
        
        Returns:
            Tuple of (score, description)
        """
        
        metrics = self.get_metrics()
        
        score = 100
        
        # Penalize for drawdown
        score -= metrics.current_drawdown * 200
        
        # Penalize for high VaR
        score -= metrics.daily_var * 500
        
        # Reward for good Sharpe
        score += min(metrics.sharpe_ratio * 10, 20)
        
        score = max(0, min(100, score))
        
        if score >= 80:
            desc = "Excellent"
        elif score >= 60:
            desc = "Good"
        elif score >= 40:
            desc = "Moderate"
        elif score >= 20:
            desc = "Risky"
        else:
            desc = "Critical"
        
        return int(score), desc


def create_risk_brain() -> RiskBrain:
    """สร้าง RiskBrain"""
    return RiskBrain()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   RISK BRAIN TEST")
    print("="*60)
    
    np.random.seed(42)
    
    rb = create_risk_brain()
    
    # Simulate some trades
    print("\nSimulating trades...")
    for i in range(30):
        pnl = np.random.randn() * 0.01
        rb.record_trade(pnl, 0.02)
        rb.record_equity(10000 + sum(list(rb.trade_returns)) * 10000)
    
    # Evaluate risk
    print("\nEvaluating risk...")
    decision = rb.evaluate_risk(
        equity=10000,
        volatility=0.02,
        confidence=0.75,
        regime="trending_up",
    )
    
    print(f"\nRisk Decision:")
    print(f"  Allow Trade: {decision.allow_trade}")
    print(f"  Max Position: {decision.max_position_pct:.2%}")
    print(f"  Risk Level: {decision.risk_level}")
    print(f"  Recommended SL: {decision.recommended_sl_pct:.2%}")
    print(f"  Recommended TP: {decision.recommended_tp_pct:.2%}")
    if decision.warnings:
        print(f"  Warnings: {decision.warnings}")
    
    # Metrics
    print("\nRisk Metrics:")
    metrics = rb.get_metrics()
    print(f"  VaR (95%): {metrics.daily_var:.2%}")
    print(f"  CVaR: {metrics.daily_cvar:.2%}")
    print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino: {metrics.sortino_ratio:.2f}")
    print(f"  Current DD: {metrics.current_drawdown:.2%}")
    
    score, desc = rb.get_risk_score()
    print(f"\nRisk Score: {score}/100 ({desc})")
