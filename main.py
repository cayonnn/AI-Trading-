"""
AI Trading System - Full Integration
=====================================
ระบบเทรด AI อัตโนมัติเต็มรูปแบบ

Features:
- PPO Agent ที่เรียนรู้ด้วยตัวเอง
- MT5 Integration (Paper/Live)
- Continuous Learning จากผลเทรด
- Auto-retraining เมื่อ Performance ลด

Usage:
    python main.py train          # เทรน PPO Agent
    python main.py paper          # Paper Trading
    python main.py backtest       # Backtest
    python main.py status         # ดูสถานะ
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta
from loguru import logger
import json

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports
from ai_agent.ppo_agent import PPOAgent, TradingEnvironment, DEVICE, train_ppo_agent
from ai_agent.trade_memory import TradeMemory, TradeRecord
from ai_agent.strategy_library import StrategyLibrary

import pandas as pd
import numpy as np

# MT5 (optional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
        level=log_level,
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
        level="DEBUG",
    )


def load_data(source: str = "csv") -> pd.DataFrame:
    """Load market data"""
    if source == "mt5" and MT5_AVAILABLE:
        if not mt5.initialize():
            logger.error("MT5 init failed")
            return None
        
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 5000)
        if rates is None:
            logger.error("Failed to fetch MT5 data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'tick_volume': 'volume'})
        mt5.shutdown()
        return df
    else:
        path = "data/training/GOLD_H1.csv"
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            return None
        
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df


def train_agent(n_episodes: int = 100, data_source: str = "csv"):
    """Train PPO Agent"""
    print("\n" + "="*60)
    print("   PPO AI TRADING AGENT - TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {n_episodes}")
    print("="*60 + "\n")
    
    # Load data
    df = load_data(data_source)
    if df is None:
        return
    
    logger.info(f"Data loaded: {len(df)} bars")
    
    # Train
    agent, history = train_ppo_agent(
        data_path="data/training/GOLD_H1.csv",
        n_episodes=n_episodes,
    )
    
    # Summary
    print("\n" + "="*60)
    print("   TRAINING COMPLETE")
    print("="*60)
    
    if history:
        final = history[-1]
        print(f"Final Win Rate: {final.get('win_rate', 0):.1%}")
        print(f"Final P&L: ${final.get('total_pnl', 0):.2f}")
        print(f"Total Episodes: {len(history)}")
    
    print("="*60)
    print(f"Model saved to: ai_agent/models/")


def run_paper_trading(interval: int = 60):
    """Run Paper Trading with PPO Agent"""
    print("\n" + "="*60)
    print("   PPO AI - PAPER TRADING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Interval: {interval}s")
    print("="*60 + "\n")
    
    # Load agent
    state_dim = 8 + 3
    agent = PPOAgent(state_dim=state_dim)
    loaded = agent.load("best")
    
    if not loaded:
        logger.warning("No trained model found - training new agent...")
        train_agent(n_episodes=50)
        agent.load("best")
    
    # Memory
    memory = TradeMemory()
    
    # Trading state
    capital = 10000.0
    position = None
    trades = []
    
    logger.info("Starting paper trading... (Ctrl+C to stop)")
    
    try:
        while True:
            # Load fresh data
            df = load_data("csv")  # Use MT5 for live
            if df is None:
                time.sleep(interval)
                continue
            
            # Create environment
            env = TradingEnvironment(df.tail(500).reset_index(drop=True))
            
            # Get current state
            state = env._get_state()
            current_price = df['close'].iloc[-1]
            
            # Get action from PPO
            action, _, _ = agent.select_action(state)
            
            action_names = {0: "WAIT", 1: "LONG", 2: "CLOSE"}
            
            if action == 1 and position is None:
                # Open Long
                position = {
                    "entry_price": current_price,
                    "entry_time": datetime.now(),
                    "size": 0.1,
                }
                logger.info(f"🔵 OPEN LONG @ ${current_price:.2f}")
                
            elif action == 2 and position is not None:
                # Close position
                exit_price = current_price
                pnl = (exit_price - position["entry_price"]) * position["size"] * 100
                
                icon = "✅" if pnl > 0 else "❌"
                logger.info(f"{icon} CLOSE @ ${exit_price:.2f} | P&L: ${pnl:.2f}")
                
                capital += pnl
                trades.append({
                    "entry": position["entry_price"],
                    "exit": exit_price,
                    "pnl": pnl,
                    "time": datetime.now(),
                })
                position = None
            else:
                logger.debug(f"Action: {action_names[action]} | Price: ${current_price:.2f}")
            
            # Stats
            if trades:
                win_rate = sum(1 for t in trades if t["pnl"] > 0) / len(trades)
                total_pnl = sum(t["pnl"] for t in trades)
                logger.info(f"Stats: {len(trades)} trades | Win Rate: {win_rate:.0%} | P&L: ${total_pnl:.2f}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\nStopping paper trading...")
        
        # Final report
        print("\n" + "="*60)
        print("   PAPER TRADING REPORT")
        print("="*60)
        print(f"Total Trades: {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            print(f"Win Rate: {wins/len(trades):.1%}")
            print(f"Total P&L: ${sum(t['pnl'] for t in trades):.2f}")
            print(f"Final Capital: ${capital:.2f}")
        print("="*60)


def run_backtest():
    """Run backtest"""
    print("\n" + "="*60)
    print("   PPO AI - BACKTEST")
    print("="*60)
    
    # Load data
    df = load_data("csv")
    if df is None:
        return
    
    # Load agent
    state_dim = 8 + 3
    agent = PPOAgent(state_dim=state_dim)
    loaded = agent.load("best")
    
    if not loaded:
        logger.warning("No trained model - training first...")
        train_agent(n_episodes=30)
        agent.load("best")
    
    # Backtest
    env = TradingEnvironment(df)
    state = env.reset()
    
    total_reward = 0
    
    while True:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
        state = next_state
    
    # Results
    perf = env.get_performance()
    
    print("="*60)
    print("   BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {perf.get('n_trades', 0)}")
    print(f"Win Rate: {perf.get('win_rate', 0):.1%}")
    print(f"Total P&L: ${perf.get('total_pnl', 0):.2f}")
    print(f"Return: {perf.get('return_pct', 0):.2%}")
    print(f"Final Capital: ${perf.get('final_capital', 10000):.2f}")
    print("="*60)


def show_status():
    """Show system status"""
    print("\n" + "="*60)
    print("   AI TRADING SYSTEM STATUS")
    print("="*60)
    
    print(f"\nDevice: {DEVICE}")
    print(f"MT5 Available: {MT5_AVAILABLE}")
    
    # Check models
    models_dir = "ai_agent/models"
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        print(f"\nModels saved: {len(models)}")
        for m in models:
            path = os.path.join(models_dir, m)
            size = os.path.getsize(path) / 1024
            print(f"  - {m} ({size:.1f} KB)")
    
    # Check memory
    memory = TradeMemory()
    stats = memory.get_performance_stats()
    
    print(f"\nTrade Memory:")
    print(f"  Total Trades: {stats.get('total_trades', 0)}")
    print(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
    print(f"  Total P&L: ${stats.get('total_pnl', 0):.2f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("command", choices=["train", "paper", "backtest", "status"],
                       help="Command to run")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of training episodes")
    parser.add_argument("--interval", type=int, default=60,
                       help="Paper trading interval in seconds")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    setup_logging("DEBUG" if args.debug else "INFO")
    
    if args.command == "train":
        train_agent(n_episodes=args.episodes)
    elif args.command == "paper":
        run_paper_trading(interval=args.interval)
    elif args.command == "backtest":
        run_backtest()
    elif args.command == "status":
        show_status()


if __name__ == "__main__":
    main()
