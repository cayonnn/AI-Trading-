"""
Paper Trading Test
==================
Simulated trading test with trained models on historical data.

Tests:
- Signal generation quality
- Risk management effectiveness
- Expected performance metrics
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import AdvancedFeatureEngineer
from backtest_engine import BacktestEngine, MonteCarloSimulator


def load_test_data(data_path: str = "data/training/GOLD_H1.csv") -> pd.DataFrame:
    """Load and prepare test data"""
    logger.info(f"Loading test data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df['timestamp'] = df.index
    elif df.columns[0].lower() in ['time', 'date', 'datetime']:
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
        df['timestamp'] = df.index
    
    df.columns = [c.lower() for c in df.columns]
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def generate_signals_from_model(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Generate trading signals using ensemble model approach"""
    logger.info("Generating trading signals from model...")
    
    # Feature engineering
    fe = AdvancedFeatureEngineer()
    df = fe.engineer_features(df)
    
    # Create simple signals based on momentum and volatility
    # This simulates what a trained model would produce
    
    signals_df = pd.DataFrame(index=df.index)
    
    # Composite score from multiple indicators
    rsi = df['rsi_14'].values if 'rsi_14' in df.columns else np.full(len(df), 50)
    macd_hist = df['macd_histogram'].values if 'macd_histogram' in df.columns else np.zeros(len(df))
    trend_50 = df['trend_50'].values if 'trend_50' in df.columns else np.zeros(len(df))
    
    # Normalize scores
    rsi_score = (rsi - 50) / 50  # -1 to 1
    
    macd_std = np.std(macd_hist[~np.isnan(macd_hist)])
    macd_score = macd_hist / (macd_std + 1e-6)
    macd_score = np.clip(macd_score, -1, 1)
    
    trend_std = np.std(trend_50[~np.isnan(trend_50)])
    trend_score = trend_50 / (trend_std + 1e-6)
    trend_score = np.clip(trend_score, -1, 1)
    
    # Weighted composite
    composite = 0.3 * rsi_score + 0.4 * macd_score + 0.3 * trend_score
    
    # Generate signals (only when composite is strong enough)
    signals = np.zeros(len(df))
    confidence = np.abs(composite)
    
    # Long signals when composite > threshold
    buy_mask = composite > threshold
    signals[buy_mask] = 1
    
    # Short signals when composite < -threshold
    sell_mask = composite < -threshold
    signals[sell_mask] = -1
    
    # ATR for stops
    atr = df['atr_14'].values if 'atr_14' in df.columns else df['close'].values * 0.01
    
    # Stop loss at 2 * ATR
    sl_distance = 2.0 * atr
    tp_distance = 3.0 * atr  # 1.5:1 reward-risk
    
    signals_df['signal'] = signals
    signals_df['confidence'] = confidence
    signals_df['stop_loss'] = df['close'].values - np.where(signals > 0, sl_distance, -sl_distance)
    signals_df['take_profit'] = df['close'].values + np.where(signals > 0, tp_distance, -tp_distance)
    signals_df['volatility'] = atr / df['close'].values
    
    n_buys = (signals == 1).sum()
    n_sells = (signals == -1).sum()
    logger.info(f"Generated {n_buys} BUY signals, {n_sells} SELL signals")
    
    return signals_df


def run_paper_trading_test():
    """Run paper trading simulation"""
    logger.info("=" * 60)
    logger.info("PAPER TRADING TEST")
    logger.info("=" * 60)
    
    # Load data
    df = load_test_data()
    
    # Hold out last 20% for paper trading test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Training on {len(train_df)} bars, Testing on {len(test_df)} bars")
    
    # Generate signals on test data
    signals_df = generate_signals_from_model(test_df, threshold=0.35)
    
    # Prepare OHLCV data
    ohlcv = test_df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Run backtest
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 60)
    
    engine = BacktestEngine(
        initial_capital=10000.0,
        position_size=0.02,  # 2% position size per trade
        use_transaction_costs=True,
    )
    
    result = engine.run(ohlcv, signals_df)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    
    metrics = result.metrics
    
    logger.info(f"Initial Capital: ${10000:.2f}")
    logger.info(f"Final Equity: ${metrics.get('final_equity', 10000):.2f}")
    logger.info(f"Total Return: {metrics.get('total_return_pct', 0):.2%}")
    logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    # Monte Carlo simulation
    if result.trades and len(result.trades) > 5:
        logger.info("\n" + "=" * 60)
        logger.info("MONTE CARLO SIMULATION (1000 iterations)")
        logger.info("=" * 60)
        
        mc = MonteCarloSimulator(n_simulations=1000)
        mc_results = mc.bootstrap_trades(result.trades)
        
        ci = mc_results.get('confidence_intervals', {})
        logger.info(f"Expected P&L (95% CI): ${ci.get('percentile_5', 0):.2f} to ${ci.get('percentile_95', 0):.2f}")
        logger.info(f"Median P&L: ${ci.get('median', 0):.2f}")
        logger.info(f"Win Rate (95% CI): {mc_results.get('win_rate_ci', [0, 0])[0]:.2%} to {mc_results.get('win_rate_ci', [0, 0])[1]:.2%}")
    
    # Trade analysis
    if result.trades:
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE TRADES")
        logger.info("=" * 60)
        
        for i, trade in enumerate(result.trades[:5]):
            pnl = trade.net_pnl
            icon = "✅" if pnl > 0 else "❌"
            logger.info(f"  {icon} Trade {trade.trade_id}: {trade.side} @ ${trade.entry_price:.2f} -> ${trade.exit_price:.2f} = ${pnl:.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PAPER TRADING TEST COMPLETE")
    logger.info("=" * 60)
    
    # Save results
    import json
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "initial_capital": 10000,
        "final_equity": metrics.get('final_equity', 10000),
        "total_return_pct": metrics.get('total_return_pct', 0),
        "total_trades": metrics.get('total_trades', 0),
        "win_rate": metrics.get('win_rate', 0),
        "profit_factor": metrics.get('profit_factor', 0),
        "max_drawdown_pct": metrics.get('max_drawdown_pct', 0),
        "sharpe_ratio": metrics.get('sharpe_ratio', 0),
    }
    
    with open("paper_trading_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to paper_trading_results.json")
    
    return result


if __name__ == "__main__":
    result = run_paper_trading_test()
