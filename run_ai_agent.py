"""
Run Sniper AI Agent
====================
เรียกใช้ AI Agent สำหรับ Paper/Live Trading

Usage:
    python run_ai_agent.py --mode paper
    python run_ai_agent.py --mode live
"""

import sys
import os
import argparse
import time
from datetime import datetime
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agent.master_agent import MasterTradingAgent, AgentConfig, AgentMode
from ai_agent.learning_engine import DEVICE

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available - using simulated data")

import pandas as pd
import numpy as np


def setup_logging():
    """ตั้งค่า Logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/ai_agent_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
    )


def fetch_mt5_data(symbol: str = "XAUUSD", timeframe: str = "H1", bars: int = 300) -> pd.DataFrame:
    """ดึงข้อมูลจาก MT5"""
    if not MT5_AVAILABLE:
        return None
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return None
    
    # Map timeframe
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    # Fetch data
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to fetch {symbol} data")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'tick_volume': 'volume'})
    
    return df


def fetch_csv_data(path: str = "data/training/GOLD_H1.csv") -> pd.DataFrame:
    """ดึงข้อมูลจาก CSV"""
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    df.columns = [c.lower() for c in df.columns]
    
    return df


def simulate_trade_result(trade: dict, current_price: float) -> dict:
    """จำลองผลการเทรด"""
    entry = trade['entry_price']
    sl = trade['stop_loss']
    tp = trade['take_profit']
    
    # Random outcome for simulation
    if np.random.random() > 0.5:  # 50% chance TP
        exit_price = tp
        pnl = (tp - entry) * trade['position_size'] * 100
        exit_reason = "tp"
    else:  # 50% chance SL
        exit_price = sl
        pnl = (sl - entry) * trade['position_size'] * 100
        exit_reason = "sl"
    
    return {
        "trade_id": trade['trade_id'],
        "exit_price": exit_price,
        "pnl": pnl,
        "exit_reason": exit_reason,
        "duration_bars": np.random.randint(1, 48),
    }


def run_paper_trading(agent: MasterTradingAgent, data: pd.DataFrame, interval: int = 5):
    """
    รัน Paper Trading
    
    Args:
        agent: MasterTradingAgent instance
        data: Market data
        interval: Seconds between checks
    """
    logger.info("="*60)
    logger.info("🤖 SNIPER AI AGENT - PAPER TRADING")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data: {len(data)} bars")
    logger.info(f"Check interval: {interval}s")
    logger.info("="*60)
    
    try:
        while True:
            # Get latest data window
            window_size = 300
            if len(data) >= window_size:
                market_data = data.tail(window_size).copy()
            else:
                market_data = data.copy()
            
            # Run agent
            trade = agent.run_once(market_data)
            
            if trade:
                # Simulate trade result (for paper trading)
                time.sleep(1)  # Wait a bit
                result = simulate_trade_result(trade, market_data['close'].iloc[-1])
                agent.learn(result)
            
            # Print status every minute
            status = agent.get_status()
            logger.info(f"Status: {status['state']} | Trades: {status['session_trades']} | "
                       f"Win Rate: {status['session_win_rate']:.0%} | P&L: ${status['session_pnl']:.2f}")
            
            # Wait
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\n⛔ Stopping agent...")
        agent.save_state()
        
        # Final report
        print("\n" + "="*60)
        print("📊 FINAL REPORT")
        print("="*60)
        status = agent.get_status()
        for k, v in status.items():
            print(f"  {k}: {v}")
        print("="*60)


def run_backtest(agent: MasterTradingAgent, data: pd.DataFrame):
    """
    รัน Backtest
    """
    logger.info("="*60)
    logger.info("🤖 SNIPER AI AGENT - BACKTEST")
    logger.info("="*60)
    
    window_size = 300
    
    # For backtest, increase daily limit or disable it
    original_max_daily = agent.config.max_daily_trades
    agent.config.max_daily_trades = 999  # Unlimited for backtest
    
    current_day = None
    trade_entry_bar = None
    
    for i in range(window_size, len(data)):
        # Get window
        market_data = data.iloc[i-window_size:i].copy()
        current_price = market_data['close'].iloc[-1]
        
        # Check for new day (reset daily stats)
        if hasattr(market_data.index, 'date'):
            bar_date = market_data.index[-1].date()
        else:
            bar_date = i // 24  # Assume H1 data, new "day" every 24 bars
        
        if current_day is not None and bar_date != current_day:
            agent.reset_daily()
        current_day = bar_date
        
        # Check existing position
        if agent.current_position:
            # Check SL/TP
            pos = agent.current_position
            high = market_data['high'].iloc[-1]
            low = market_data['low'].iloc[-1]
            
            exit_price = None
            exit_reason = None
            
            if low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "sl"
            elif high >= pos['take_profit']:
                exit_price = pos['take_profit']
                exit_reason = "tp"
            
            if exit_price:
                duration = i - trade_entry_bar if trade_entry_bar else 1
                pnl = (exit_price - pos['entry_price']) * pos['position_size'] * 100
                agent.learn({
                    "trade_id": pos['trade_id'],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "duration_bars": duration,
                })
                trade_entry_bar = None
        else:
            # Try to trade
            trade = agent.run_once(market_data)
            if trade:
                trade_entry_bar = i
        
        # Progress
        if i % 500 == 0:
            progress = (i - window_size) / (len(data) - window_size) * 100
            stats = agent.memory.get_performance_stats()
            win_rate = stats.get('win_rate', 0)
            total_pnl = stats.get('total_pnl', 0)
            logger.info(f"Progress: {progress:.1f}% | Trades: {agent.session_trades} | "
                       f"Win Rate: {win_rate:.1%} | P&L: ${total_pnl:.2f}")
    
    # Restore config
    agent.config.max_daily_trades = original_max_daily
    
    # Final report
    logger.info("="*60)
    logger.info("📊 BACKTEST COMPLETE")
    logger.info("="*60)
    status = agent.get_status()
    for k, v in status.items():
        logger.info(f"  {k}: {v}")
    
    # Memory stats
    memory_stats = agent.memory.get_performance_stats()
    logger.info("\nTrade Memory Stats:")
    for k, v in memory_stats.items():
        logger.info(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Run Sniper AI Agent")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default="paper")
    parser.add_argument("--symbol", default="GOLD")
    parser.add_argument("--confidence", type=float, default=0.80)
    parser.add_argument("--rr-ratio", type=float, default=3.0)
    parser.add_argument("--interval", type=int, default=5)
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Config
    config = AgentConfig(
        symbol=args.symbol,
        min_confidence=args.confidence,
        min_rr_ratio=args.rr_ratio,
    )
    
    # Mode
    if args.mode == "live":
        mode = AgentMode.LIVE
    elif args.mode == "backtest":
        mode = AgentMode.BACKTEST
    else:
        mode = AgentMode.PAPER
    
    # Create agent
    agent = MasterTradingAgent(config=config, mode=mode)
    
    # Load state if exists
    agent.load_state()
    
    # Get data
    if MT5_AVAILABLE and args.mode in ["paper", "live"]:
        data = fetch_mt5_data(symbol="XAUUSD")
    else:
        data = fetch_csv_data()
    
    if data is None or len(data) == 0:
        logger.error("No data available!")
        return
    
    # Run
    if args.mode == "backtest":
        run_backtest(agent, data)
    else:
        run_paper_trading(agent, data, interval=args.interval)


if __name__ == "__main__":
    main()
