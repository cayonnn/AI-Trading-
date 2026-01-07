
import os
import pandas as pd
from mt5_data_provider import MT5DataProvider
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

def fetch_data():
    """Fetch training data from MT5"""
    provider = MT5DataProvider()
    
    if not provider.connect():
        logger.error("Failed to connect to MT5")
        return

    # Try symbols: generic Forex XAUUSD or CFD GOLD
    # Based on previous logs, user has "GOLD"
    symbols_to_try = ["GOLD", "XAUUSD"]
    symbol = None
    
    for s in symbols_to_try:
        info = provider.get_symbol_info(s)
        if info:
            symbol = s
            logger.info(f"Found symbol: {symbol}")
            break
    
    if not symbol:
        logger.error("Could not find GOLD or XAUUSD symbol in Market Watch")
        return

    # Settings
    timeframes = ["M15", "H1", "H4", "D1"]
    bars = 10000 # Fetch 10k bars for training
    output_dir = "data/training"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving data to {output_dir}")
    
    for tf in timeframes:
        logger.info(f"Fetching {symbol} {tf} ({bars} bars)...")
        df = provider.get_ohlcv(symbol, tf, bars=bars)
        
        if df is not None and not df.empty:
            filename = f"{output_dir}/{symbol}_{tf}.csv"
            df.to_csv(filename, index=False)
            logger.success(f"Saved {len(df)} rows to {filename}")
            
            # Show stats
            logger.info(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            logger.info(f"  Price: {df['low'].min():.2f} - {df['high'].max():.2f}")
        else:
            logger.warning(f"Failed to fetch data for {tf}")

    provider.disconnect()
    logger.info("Data fetching complete")

if __name__ == "__main__":
    fetch_data()
