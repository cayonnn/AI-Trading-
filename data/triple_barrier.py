"""
Triple Barrier Labeling
========================
Implementation based on Lopez de Prado's "Advances in Financial Machine Learning" (2018).

Triple barrier method assigns labels based on which barrier is hit first:
- Upper barrier (profit-taking): +1
- Lower barrier (stop-loss): -1
- Vertical barrier (time limit): 0 or based on final price

This provides more realistic labels than simple next-bar direction prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger


def get_daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Calculate daily volatility using exponential moving average of returns.
    
    Args:
        close: Close prices
        span: EMA span for volatility
        
    Returns:
        Volatility series
    """
    returns = close.pct_change()
    volatility = returns.ewm(span=span).std()
    return volatility


def get_vertical_barrier(df: pd.DataFrame, t0: pd.DatetimeIndex, num_bars: int) -> pd.Series:
    """
    Get vertical barrier timestamps (max holding period).
    
    Args:
        df: DataFrame with price data
        t0: Series of entry timestamps
        num_bars: Maximum number of bars to hold position
        
    Returns:
        Series of exit timestamps
    """
    t1 = df.index.searchsorted(t0 + pd.Timedelta(hours=num_bars))
    t1 = t1[t1 < len(df.index)]
    t1 = pd.Series(df.index[t1], index=t0[:len(t1)])
    return t1


def get_horizontal_barriers(
    close: pd.Series,
    volatility: pd.Series,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate horizontal barriers based on volatility.
    
    Args:
        close: Close prices
        volatility: Volatility series
        pt_sl: Tuple of (profit-taking multiplier, stop-loss multiplier)
        min_ret: Minimum return threshold
        
    Returns:
        Tuple of (upper barrier, lower barrier)
    """
    # Use ATR-like volatility scaling
    pt_multiple, sl_multiple = pt_sl
    
    # Upper barrier (profit-taking)
    upper = close * (1 + pt_multiple * volatility + min_ret)
    
    # Lower barrier (stop-loss)
    lower = close * (1 - sl_multiple * volatility - min_ret)
    
    return upper, lower


def apply_triple_barrier(
    df: pd.DataFrame,
    pt_sl: Tuple[float, float] = (1.5, 1.0),
    max_holding_bars: int = 24,
    volatility_span: int = 20,
    min_ret: float = 0.001
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to price data.
    
    Args:
        df: DataFrame with 'close', 'high', 'low' columns
        pt_sl: Tuple of (profit-taking multiplier, stop-loss multiplier)
        max_holding_bars: Maximum bars to hold position (vertical barrier)
        volatility_span: Span for volatility calculation
        min_ret: Minimum return threshold to trigger barrier
        
    Returns:
        DataFrame with added 'triple_barrier_label' column
    """
    logger.info(f"Applying triple barrier labeling: pt_sl={pt_sl}, max_bars={max_holding_bars}")
    
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Calculate volatility
    volatility = get_daily_volatility(close, volatility_span)
    
    # Initialize labels
    labels = []
    touch_times = []
    returns_list = []
    
    for i in range(len(df) - max_holding_bars):
        entry_price = close.iloc[i]
        entry_vol = volatility.iloc[i]
        
        if pd.isna(entry_vol) or entry_vol == 0:
            labels.append(0)
            touch_times.append(0)
            returns_list.append(0)
            continue
        
        # Calculate barriers
        upper_barrier = entry_price * (1 + pt_sl[0] * entry_vol + min_ret)
        lower_barrier = entry_price * (1 - pt_sl[1] * entry_vol - min_ret)
        
        # Look forward for barrier touches
        label = 0
        touch_bar = max_holding_bars
        final_return = 0
        
        for j in range(1, max_holding_bars + 1):
            future_idx = i + j
            if future_idx >= len(df):
                break
                
            future_high = high.iloc[future_idx]
            future_low = low.iloc[future_idx]
            future_close = close.iloc[future_idx]
            
            # Check upper barrier (profit-taking)
            if future_high >= upper_barrier:
                label = 1
                touch_bar = j
                final_return = (upper_barrier - entry_price) / entry_price
                break
            
            # Check lower barrier (stop-loss)
            if future_low <= lower_barrier:
                label = -1
                touch_bar = j
                final_return = (lower_barrier - entry_price) / entry_price
                break
        
        # If no barrier touched, use vertical barrier (time exit)
        if label == 0 and i + max_holding_bars < len(df):
            exit_price = close.iloc[i + max_holding_bars]
            final_return = (exit_price - entry_price) / entry_price
            
            # Assign label based on return direction at exit
            if final_return > min_ret:
                label = 1
            elif final_return < -min_ret:
                label = -1
            else:
                label = 0
            touch_bar = max_holding_bars
        
        labels.append(label)
        touch_times.append(touch_bar)
        returns_list.append(final_return)
    
    # Pad the end with NaN
    labels.extend([np.nan] * max_holding_bars)
    touch_times.extend([np.nan] * max_holding_bars)
    returns_list.extend([np.nan] * max_holding_bars)
    
    df['triple_barrier_label'] = labels
    df['barrier_touch_time'] = touch_times
    df['expected_return'] = returns_list
    
    # Log statistics
    label_counts = pd.Series(labels).dropna().value_counts()
    logger.info(f"Triple barrier labels: {label_counts.to_dict()}")
    
    return df


def create_meta_labels(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    pt_sl: Tuple[float, float] = (1.5, 1.0),
    max_holding_bars: int = 24
) -> pd.DataFrame:
    """
    Create meta-labels for position sizing based on primary model signals.
    
    Meta-labeling: Secondary model learns when primary model's signals are correct.
    - 1: Primary signal was correct (profitable)
    - 0: Primary signal was wrong (loss)
    
    Args:
        df: DataFrame with price data
        primary_signal: Primary model's directional predictions (1=long, -1=short)
        pt_sl: Profit-taking/stop-loss multipliers
        max_holding_bars: Maximum holding period
        
    Returns:
        DataFrame with 'meta_label' column
    """
    logger.info("Creating meta-labels for position sizing...")
    
    df = df.copy()
    
    # First apply triple barrier labeling
    df = apply_triple_barrier(df, pt_sl, max_holding_bars)
    
    # Create meta-labels
    # Signal correct if: (long AND price went up) OR (short AND price went down)
    meta_labels = []
    
    for i, (signal, barrier_label) in enumerate(zip(primary_signal, df['triple_barrier_label'])):
        if pd.isna(barrier_label) or pd.isna(signal) or signal == 0:
            meta_labels.append(np.nan)
        elif (signal > 0 and barrier_label > 0) or (signal < 0 and barrier_label < 0):
            meta_labels.append(1)  # Correct signal
        else:
            meta_labels.append(0)  # Wrong signal
    
    df['meta_label'] = meta_labels
    
    # Log meta-label distribution
    meta_counts = pd.Series(meta_labels).dropna().value_counts()
    logger.info(f"Meta-labels: {meta_counts.to_dict()}")
    
    return df


def convert_to_binary_target(triple_barrier_labels: pd.Series) -> pd.Series:
    """
    Convert triple barrier labels (-1, 0, 1) to binary (0, 1) for classification.
    
    Strategy:
    - Label 1 (hit upper barrier) -> 1 (up)
    - Label -1 (hit lower barrier) -> 0 (down)
    - Label 0 (vertical barrier) -> Based on return direction or filter out
    
    Args:
        triple_barrier_labels: Series with (-1, 0, 1) labels
        
    Returns:
        Binary series (0, 1)
    """
    binary = triple_barrier_labels.copy()
    
    # Map: 1 -> 1, -1 -> 0, 0 -> 0 (neutral treated as down for now)
    binary = binary.map({1: 1, -1: 0, 0: 0})
    
    return binary


if __name__ == "__main__":
    # Test with sample data
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Load data
    df = pd.read_csv('data/training/GOLD_H1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"Original data shape: {df.shape}")
    
    # Apply triple barrier
    df_labeled = apply_triple_barrier(df, pt_sl=(1.5, 1.0), max_holding_bars=24)
    
    print(f"\nLabel distribution:")
    print(df_labeled['triple_barrier_label'].value_counts())
    
    print(f"\nSample labels:")
    print(df_labeled[['close', 'triple_barrier_label', 'barrier_touch_time', 'expected_return']].head(10))
