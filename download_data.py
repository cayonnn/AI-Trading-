"""
Download Real Gold Data
========================
ดาวน์โหลดข้อมูล Gold จริงจาก MT5 หรือ Yahoo Finance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Try MT5 first
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✅ MetaTrader5 available")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available, will use Yahoo Finance")

# Try yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance available")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available")


def download_from_mt5(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 16385, bars=50000):
    """Download data from MT5"""
    if not MT5_AVAILABLE:
        return None
    
    print(f"\n📊 Downloading {symbol} from MetaTrader 5...")
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("❌ No data received from MT5")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time': 'datetime'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']]
    df = df.rename(columns={'tick_volume': 'volume'})
    
    print(f"✅ Downloaded {len(df)} bars from MT5")
    print(f"   From: {df['datetime'].min()}")
    print(f"   To: {df['datetime'].max()}")
    
    return df


def download_from_yfinance(symbol="GC=F", period="max", interval="1h"):
    """Download data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        return None
    
    print(f"\n📊 Downloading {symbol} from Yahoo Finance...")
    print("   (Note: Yahoo Finance H1 data is limited to ~2 years)")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print("❌ No data received from Yahoo Finance")
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'datetime',
            'Date': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Downloaded {len(df)} bars from Yahoo Finance")
        print(f"   From: {df['datetime'].min()}")
        print(f"   To: {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return None


def download_daily_data(symbol="GC=F"):
    """Download daily data (much more available)"""
    if not YFINANCE_AVAILABLE:
        return None
    
    print(f"\n📊 Downloading Daily {symbol} from Yahoo Finance...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", interval="1d")
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Downloaded {len(df)} daily bars")
        print(f"   From: {df['datetime'].min()}")
        print(f"   To: {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def generate_synthetic_h1(daily_df):
    """Generate synthetic H1 data from daily data"""
    print("\n🔄 Generating synthetic H1 data from daily...")
    
    h1_data = []
    
    for _, row in daily_df.iterrows():
        date = row['datetime']
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        vol = row['volume'] / 24
        
        # Generate 24 hourly bars
        daily_range = abs(h - l)  # Use abs to avoid negative
        if daily_range < 0.01:
            daily_range = 1.0  # Minimum range
        
        for hour in range(24):
            # Create realistic intraday pattern
            progress = hour / 24
            
            # Simple interpolation with some volatility
            noise = np.random.normal(0, max(daily_range * 0.05, 0.01))
            
            if progress < 0.5:
                bar_open = o + (c - o) * progress * 0.5 + noise
            else:
                bar_open = o + (c - o) * progress + noise
            
            bar_range = daily_range * (0.3 + np.random.random() * 0.4) / 24
            bar_high = bar_open + bar_range * np.random.random()
            bar_low = bar_open - bar_range * np.random.random()
            bar_close = bar_low + (bar_high - bar_low) * np.random.random()
            
            # Ensure OHLC consistency
            bar_high = max(bar_open, bar_close, bar_high)
            bar_low = min(bar_open, bar_close, bar_low)
            
            h1_data.append({
                'datetime': date + timedelta(hours=hour),
                'open': round(bar_open, 2),
                'high': round(bar_high, 2),
                'low': round(bar_low, 2),
                'close': round(bar_close, 2),
                'volume': int(vol * (0.5 + np.random.random()))
            })
    
    df = pd.DataFrame(h1_data)
    print(f"✅ Generated {len(df)} synthetic H1 bars")
    
    return df


def main():
    print("="*60)
    print("   GOLD DATA DOWNLOADER")
    print("="*60)
    
    os.makedirs("data/training", exist_ok=True)
    
    df = None
    
    # Try MT5 first (best quality)
    if MT5_AVAILABLE:
        df = download_from_mt5("XAUUSD", bars=50000)
        if df is not None:
            output_path = "data/training/GOLD_H1_MT5.csv"
            df.to_csv(output_path, index=False)
            print(f"\n💾 Saved to {output_path}")
    
    # Try Yahoo Finance H1
    if df is None and YFINANCE_AVAILABLE:
        df = download_from_yfinance("GC=F", period="2y", interval="1h")
        if df is not None:
            output_path = "data/training/GOLD_H1_YF.csv"
            df.to_csv(output_path, index=False)
            print(f"\n💾 Saved to {output_path}")
    
    # Get daily data and convert to H1
    if YFINANCE_AVAILABLE:
        daily_df = download_daily_data("GC=F")
        if daily_df is not None and len(daily_df) > 1000:
            # Save daily
            daily_df.to_csv("data/training/GOLD_D1.csv", index=False)
            print(f"\n💾 Saved daily to data/training/GOLD_D1.csv")
            
            # Generate synthetic H1 from daily
            h1_synthetic = generate_synthetic_h1(daily_df)
            h1_synthetic.to_csv("data/training/GOLD_H1_SYNTHETIC.csv", index=False)
            print(f"💾 Saved synthetic H1 to data/training/GOLD_H1_SYNTHETIC.csv")
            
            # Use this for training if no better data
            if df is None:
                df = h1_synthetic
    
    # Summary
    print("\n" + "="*60)
    print("   DATA SUMMARY")
    print("="*60)
    
    data_files = [
        "data/training/GOLD_H1.csv",
        "data/training/GOLD_H1_MT5.csv",
        "data/training/GOLD_H1_YF.csv",
        "data/training/GOLD_H1_SYNTHETIC.csv",
        "data/training/GOLD_D1.csv",
    ]
    
    for f in data_files:
        if os.path.exists(f):
            temp_df = pd.read_csv(f)
            print(f"   {f}: {len(temp_df)} bars")
    
    print("="*60)
    
    return df


if __name__ == "__main__":
    main()
