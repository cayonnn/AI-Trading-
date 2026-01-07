"""
Backtest Experience Loader
===========================
Backtest ด้วย historical data แล้วป้อน results เข้า MasterBrain memory

ทำให้ MasterBrain มีประสบการณ์ก่อนเริ่มเทรดจริง
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indicators import calculate_indicators
from ai_agent.master_brain import MasterBrain, create_master_brain


@dataclass
class BacktestTrade:
    """ผลการ backtest trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # "LONG" or "SHORT"
    pnl: float
    pnl_pct: float
    market_state: Dict[str, float]
    indicators: Dict[str, float]
    win: bool


class BacktestExperienceLoader:
    """
    Backtest แล้วป้อนประสบการณ์เข้า MasterBrain
    
    Features:
    1. Simple backtest strategy (EMA crossover)
    2. Record all trades with market state
    3. Load into MasterBrain memory
    """
    
    def __init__(
        self,
        data_path: str = "data/training/GOLD_H1.csv",
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
    ):
        self.data_path = data_path
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        
        self.trades: List[BacktestTrade] = []
        self.master_brain: Optional[MasterBrain] = None
        
    def load_data(self) -> pd.DataFrame:
        """โหลดข้อมูล historical"""
        logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Handle datetime column
        for col in ['datetime', 'timestamp', 'time', 'date']:
            if col in df.columns:
                df['timestamp'] = pd.to_datetime(df[col])
                break
        
        # Calculate indicators
        df = calculate_indicators(df, 'all')
        df = df.dropna()
        
        logger.info(f"Loaded {len(df)} rows with indicators")
        return df
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        max_trades: int = 500,
    ) -> List[BacktestTrade]:
        """
        Run simple backtest to generate experience
        
        Strategy: EMA crossover with RSI filter
        - LONG when EMA20 > EMA50 and RSI < 70
        - Exit when EMA20 < EMA50 or hit SL/TP
        """
        logger.info("Running backtest...")
        
        trades = []
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_state = {}
        entry_indicators = {}
        
        # Calculate EMAs if not present
        if 'ema_20' not in df.columns:
            df['ema_20'] = df['close'].ewm(span=20).mean()
        if 'ema_50' not in df.columns:
            df['ema_50'] = df['close'].ewm(span=50).mean()
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = row['close']
            ema20 = row.get('ema_20', row['close'])
            ema50 = row.get('ema_50', row['close'])
            rsi = row.get('rsi_14', row.get('rsi', 50))
            
            # Calculate trend and volatility
            returns = (price - prev['close']) / prev['close']
            volatility = df['close'].iloc[max(0,i-20):i].pct_change().std() or 0.01
            trend = (ema20 - ema50) / ema50 if ema50 > 0 else 0
            
            # Market state for memory
            market_state = {
                'price': price,
                'trend': trend,
                'volatility': min(1.0, volatility * 100),
                'rsi': rsi,
                'regime': 'trending_up' if trend > 0.01 else 'trending_down' if trend < -0.01 else 'ranging',
            }
            
            indicators = {
                'rsi': rsi,
                'ema_cross': 1 if ema20 > ema50 else -1,
                'momentum': returns * 100,
            }
            
            # Entry logic
            if position == 0:
                # LONG signal
                if ema20 > ema50 and prev.get('ema_20', 0) <= prev.get('ema_50', 0) and rsi < 70:
                    position = 1
                    entry_price = price
                    entry_time = row.get('timestamp', datetime.now())
                    entry_state = market_state.copy()
                    entry_indicators = indicators.copy()
            
            # Exit logic
            elif position == 1:
                pnl_pips = price - entry_price
                
                # Check SL/TP
                hit_sl = pnl_pips <= -self.sl_pips
                hit_tp = pnl_pips >= self.tp_pips
                # Signal exit
                signal_exit = ema20 < ema50
                
                if hit_sl or hit_tp or signal_exit:
                    pnl_pct = (price - entry_price) / entry_price
                    
                    trade = BacktestTrade(
                        entry_time=entry_time,
                        exit_time=row.get('timestamp', datetime.now()),
                        entry_price=entry_price,
                        exit_price=price,
                        direction="LONG",
                        pnl=pnl_pips,
                        pnl_pct=pnl_pct,
                        market_state=entry_state,
                        indicators=entry_indicators,
                        win=pnl_pips > 0,
                    )
                    trades.append(trade)
                    
                    position = 0
                    entry_price = 0
                    
                    if len(trades) >= max_trades:
                        break
        
        self.trades = trades
        
        wins = len([t for t in trades if t.win])
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0
        total_pnl = sum(t.pnl for t in trades)
        
        logger.success(f"Backtest complete: {len(trades)} trades")
        logger.info(f"  Win Rate: {win_rate:.1%} ({wins}W / {losses}L)")
        logger.info(f"  Total P&L: {total_pnl:.2f} pips")
        
        return trades
    
    def load_into_master_brain(
        self,
        master_brain: Optional[MasterBrain] = None,
    ) -> MasterBrain:
        """ป้อนประสบการณ์เข้า MasterBrain"""
        
        if master_brain is None:
            master_brain = create_master_brain()
        
        self.master_brain = master_brain
        
        logger.info(f"Loading {len(self.trades)} trades into MasterBrain memory...")
        
        for trade in self.trades:
            master_brain.record_trade_result(
                market_state=trade.market_state,
                action=trade.direction,
                result="win" if trade.win else "loss",
                pnl=trade.pnl,
            )
        
        status = master_brain.get_status()
        logger.success(f"MasterBrain memory loaded:")
        logger.info(f"  Memory size: {status['memory_size']} trades")
        
        return master_brain
    
    def run_full_pipeline(self) -> MasterBrain:
        """Run full pipeline: load data → backtest → load memory"""
        
        df = self.load_data()
        self.run_backtest(df)
        return self.load_into_master_brain()


def preload_master_brain_experience(
    master_brain: MasterBrain,
    data_path: str = "data/training/GOLD_H1.csv",
    max_trades: int = 100,
) -> MasterBrain:
    """
    Convenience function to preload experience into existing MasterBrain
    
    Usage in autonomous_ai.py:
        self.master_brain = create_master_brain()
        preload_master_brain_experience(self.master_brain)
    """
    loader = BacktestExperienceLoader(data_path=data_path)
    
    df = loader.load_data()
    loader.run_backtest(df, max_trades=max_trades)
    
    return loader.load_into_master_brain(master_brain)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   BACKTEST EXPERIENCE LOADER")
    print("="*60)
    
    loader = BacktestExperienceLoader()
    master = loader.run_full_pipeline()
    
    print("\n" + "="*60)
    print("   MASTER BRAIN STATUS")
    print("="*60)
    
    status = master.get_status()
    print(f"\n  Total decisions: {status['total_decisions']}")
    print(f"  Override count: {status['override_count']}")
    print(f"  Memory size: {status['memory_size']} trades")
    
    # Test a new decision
    print("\n" + "="*60)
    print("   TESTING MASTER WITH EXPERIENCE")
    print("="*60)
    
    test_market = {
        'price': 2650.0,
        'trend': 0.1,
        'volatility': 0.3,
        'regime': 'trending_up',
        'atr': 15.0,
    }
    
    test_indicators = {
        'rsi': 55,
        'ema_cross': 1,
        'momentum': 0.1,
    }
    
    test_votes = {
        'lstm': ('LONG', 0.6),
        'xgb': ('LONG', 0.65),
        'ppo': ('LONG', 0.55),
    }
    
    thought = master.think(test_market, test_votes, test_indicators)
    
    print(f"\n  Decision: {thought.suggested_action}")
    print(f"  Confidence: {thought.confidence:.1%}")
    print(f"  Override: {thought.override_models}")
    print(f"\n  Reasoning:\n  {thought.reasoning}")
