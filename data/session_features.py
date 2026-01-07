"""
Session Features Module
========================
เพิ่ม Features ที่เกี่ยวกับ Trading Session และ Correlation

Features:
1. Session Time Features - London, NY, Tokyo, Sydney
2. DXY Correlation - Dollar Index inverse correlation
3. Day of Week Features - วันจันทร์-ศุกร์มีผลต่างกัน
4. Time of Day Features - ชั่วโมงที่เทรดได้ดี
5. Session Overlap Features - London/NY overlap
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


def add_session_features(df: pd.DataFrame, timezone_offset: int = 7) -> pd.DataFrame:
    """
    เพิ่ม Session-based features
    
    Args:
        df: DataFrame with datetime/timestamp column
        timezone_offset: GMT offset (default: GMT+7 for Thailand)
        
    Returns:
        DataFrame with session features
    """
    
    df = df.copy()
    
    # Get datetime column
    if 'datetime' in df.columns:
        dt_col = pd.to_datetime(df['datetime'], utc=True)
    elif 'timestamp' in df.columns:
        dt_col = pd.to_datetime(df['timestamp'], utc=True)
    elif 'time' in df.columns:
        dt_col = pd.to_datetime(df['time'], utc=True)
    else:
        logger.warning("No datetime column found")
        return df
    
    # Convert to local timezone hour
    utc_hour = dt_col.dt.hour
    
    # === Session Features ===
    
    # Sydney: 22:00 - 07:00 UTC
    df['session_sydney'] = ((utc_hour >= 22) | (utc_hour < 7)).astype(int)
    
    # Tokyo: 00:00 - 09:00 UTC
    df['session_tokyo'] = ((utc_hour >= 0) & (utc_hour < 9)).astype(int)
    
    # London: 08:00 - 17:00 UTC
    df['session_london'] = ((utc_hour >= 8) & (utc_hour < 17)).astype(int)
    
    # New York: 13:00 - 22:00 UTC
    df['session_newyork'] = ((utc_hour >= 13) & (utc_hour < 22)).astype(int)
    
    # London/NY Overlap: 13:00 - 17:00 UTC (Best for Gold!)
    df['session_overlap'] = ((utc_hour >= 13) & (utc_hour < 17)).astype(int)
    
    # Active session count
    df['session_active_count'] = (
        df['session_sydney'] + df['session_tokyo'] + 
        df['session_london'] + df['session_newyork']
    )
    
    # === Time Features ===
    
    # Hour of day (cyclical)
    hour = dt_col.dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (cyclical)
    day = dt_col.dt.dayofweek  # 0 = Monday
    df['day_sin'] = np.sin(2 * np.pi * day / 7)
    df['day_cos'] = np.cos(2 * np.pi * day / 7)
    
    # Binary day features
    df['is_monday'] = (day == 0).astype(int)
    df['is_friday'] = (day == 4).astype(int)
    
    # Week of month
    df['week_of_month'] = dt_col.dt.day // 7
    
    # Month end (NFP usually first Friday)
    df['is_month_start'] = (dt_col.dt.day <= 5).astype(int)
    
    # === Session Quality for Gold ===
    
    # Best time for Gold (London + NY)
    df['gold_session_quality'] = (
        df['session_london'] * 0.4 +
        df['session_newyork'] * 0.3 +
        df['session_overlap'] * 0.3
    )
    
    logger.info(f"Added {13} session features")
    
    return df


def add_dxy_correlation_features(
    df: pd.DataFrame,
    dxy_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    เพิ่ม DXY correlation features
    
    Args:
        df: Gold price DataFrame
        dxy_data: DXY price DataFrame (optional)
        
    Returns:
        DataFrame with DXY correlation features
    """
    
    df = df.copy()
    
    if dxy_data is not None and len(dxy_data) > 0:
        # Merge DXY data
        try:
            # Calculate DXY returns
            dxy_data = dxy_data.copy()
            dxy_data['dxy_returns'] = dxy_data['close'].pct_change()
            dxy_data['dxy_ma20'] = dxy_data['close'].rolling(20).mean()
            dxy_data['dxy_strength'] = dxy_data['close'] / dxy_data['dxy_ma20'] - 1
            
            # Merge (assuming same datetime index)
            # In production, would need proper datetime alignment
            df['dxy_returns'] = dxy_data['dxy_returns'].values[:len(df)]
            df['dxy_strength'] = dxy_data['dxy_strength'].values[:len(df)]
            
            # Gold/DXY inverse correlation
            df['gold_dxy_corr'] = df['close'].pct_change().rolling(20).corr(df['dxy_returns'])
            
            logger.info("Added DXY correlation features")
        except Exception as e:
            logger.warning(f"Could not add DXY features: {e}")
            _add_synthetic_dxy_features(df)
    else:
        # Add synthetic DXY-like features
        _add_synthetic_dxy_features(df)
    
    return df


def _add_synthetic_dxy_features(df: pd.DataFrame) -> None:
    """Add synthetic DXY-like features when real DXY not available"""
    
    # Gold typically has inverse correlation with USD
    # Use momentum indicators as proxy
    
    returns = df['close'].pct_change()
    
    # Synthetic DXY strength (inverted gold momentum)
    df['dxy_proxy'] = -returns.rolling(20).mean() * 100
    
    # Dollar regime proxy
    # When gold is weak, dollar is likely strong
    gold_ma = df['close'].rolling(50).mean()
    df['dollar_regime'] = (df['close'] < gold_ma).astype(int)
    
    # Correlation with inverse (proxy for DXY correlation)
    inverse_returns = -returns
    df['inverse_corr_20'] = returns.rolling(20).corr(inverse_returns.shift(1))
    
    logger.info("Added synthetic DXY proxy features")


def add_volatility_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่ม Volatility features ตาม session
    """
    
    df = df.copy()
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr_14 = tr.rolling(14).mean()
    
    # Normalized ATR
    df['atr_pct'] = atr_14 / close * 100
    
    # Volatility relative to session (if session features exist)
    if 'session_london' in df.columns:
        # Calculate session-specific volatility
        for session in ['sydney', 'tokyo', 'london', 'newyork']:
            col = f'session_{session}'
            if col in df.columns:
                session_mask = df[col] == 1
                if session_mask.sum() > 20:
                    session_vol = tr[session_mask].rolling(20).mean()
                    overall_vol = tr.rolling(20).mean()
                    df[f'{session}_vol_ratio'] = session_vol / overall_vol
    
    # Volatility expansion/contraction
    df['vol_expansion'] = (atr_14 > atr_14.rolling(20).mean()).astype(int)
    
    logger.info("Added volatility session features")
    
    return df


def create_all_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง features ทั้งหมด
    """
    
    df = add_session_features(df)
    df = add_dxy_correlation_features(df)
    df = add_volatility_session_features(df)
    
    return df


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   SESSION FEATURES TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='H')
    prices = 2000 + np.cumsum(np.random.randn(n) * 3)
    
    df = pd.DataFrame({
        "datetime": dates,
        "open": prices - np.random.rand(n) * 2,
        "high": prices + np.random.rand(n) * 4,
        "low": prices - np.random.rand(n) * 4,
        "close": prices,
        "volume": np.random.randint(1000, 5000, n),
    })
    
    # Add features
    df = create_all_session_features(df)
    
    print(f"\nNew columns: {[c for c in df.columns if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]}")
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"\nSample session features:")
    print(df[['datetime', 'session_london', 'session_newyork', 'session_overlap', 'gold_session_quality']].head(10))
