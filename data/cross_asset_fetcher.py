"""
Cross-Asset Data Fetcher
=========================
Fetch additional data sources for cross-asset features:
- USD Index (DXY)
- VIX (Volatility Index)
- Silver (for gold-silver ratio)
- S&P 500

These correlations help predict gold price movements.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime, timedelta


class CrossAssetFetcher:
    """Fetch cross-asset data from Yahoo Finance."""
    
    # Ticker symbols for various assets
    TICKERS = {
        'dxy': 'DX-Y.NYB',      # US Dollar Index
        'vix': '^VIX',          # CBOE Volatility Index
        'silver': 'SI=F',       # Silver Futures
        'sp500': '^GSPC',       # S&P 500 Index
        'gold': 'GC=F',         # Gold Futures (for reference)
        'oil': 'CL=F',          # Crude Oil Futures
        'treasury_10y': '^TNX', # 10-Year Treasury Yield
        'eurusd': 'EURUSD=X',   # EUR/USD Exchange Rate
    }
    
    # Mapping from interval string to yfinance format
    INTERVAL_MAP = {
        'H1': '1h',
        '1h': '1h',
        'H4': '1h',  # Will resample
        '4h': '1h',
        'D1': '1d',
        '1d': '1d',
    }
    
    def __init__(self):
        """Initialize the cross-asset fetcher."""
        self.data_cache = {}
    
    def fetch_asset(
        self,
        asset_key: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch single asset data.
        
        Args:
            asset_key: Key from TICKERS dict
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        if asset_key not in self.TICKERS:
            raise ValueError(f"Unknown asset: {asset_key}. Available: {list(self.TICKERS.keys())}")
        
        ticker = self.TICKERS[asset_key]
        yf_interval = self.INTERVAL_MAP.get(interval, interval)
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {asset_key} ({ticker}) from {start_date} to {end_date}")
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No data returned for {asset_key}")
                return pd.DataFrame()
            
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add asset identifier
            df['asset'] = asset_key
            
            logger.info(f"Fetched {len(df)} rows for {asset_key}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {asset_key}: {e}")
            return pd.DataFrame()
    
    def fetch_all_assets(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1h',
        assets: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.
        
        Args:
            start_date: Start date
            end_date: End date
            interval: Time interval
            assets: List of asset keys to fetch (default: all)
            
        Returns:
            Dict of asset_key -> DataFrame
        """
        if assets is None:
            assets = ['dxy', 'vix', 'silver', 'sp500']
        
        data = {}
        for asset_key in assets:
            df = self.fetch_asset(asset_key, start_date, end_date, interval)
            if not df.empty:
                data[asset_key] = df
        
        return data
    
    def merge_with_gold(
        self,
        gold_df: pd.DataFrame,
        cross_assets: Dict[str, pd.DataFrame],
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Merge cross-asset data with gold price data.
        
        Args:
            gold_df: Gold price DataFrame with timestamp index
            cross_assets: Dict of asset DataFrames
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
            
        Returns:
            Merged DataFrame with cross-asset features
        """
        logger.info(f"Merging {len(cross_assets)} cross-assets with gold data")
        
        df = gold_df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        
        for asset_key, asset_df in cross_assets.items():
            # Ensure asset index is datetime
            if not isinstance(asset_df.index, pd.DatetimeIndex):
                asset_df.index = pd.to_datetime(asset_df.index)
            
            # Select relevant columns and rename
            price_col = 'close' if 'close' in asset_df.columns else asset_df.columns[0]
            asset_close = asset_df[[price_col]].copy()
            asset_close.columns = [f'{asset_key}_close']
            
            # Merge using nearest timestamp
            df = df.join(asset_close, how='left')
        
        # Fill missing values
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'interpolate':
            df = df.interpolate(method='linear')
        
        df = df.bfill()  # Fill any remaining NaN at the start
        
        logger.info(f"Merged data shape: {df.shape}")
        
        return df
    
    def create_cross_asset_features(
        self,
        df: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create features from cross-asset data.
        
        Features created:
        - Price ratios (gold/silver, gold/dxy, etc.)
        - Correlation rolling windows
        - Return differences (momentum relative to other assets)
        
        Args:
            df: DataFrame with gold and cross-asset prices
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating cross-asset features...")
        
        df = df.copy()
        gold_close = df['close']
        
        # Find available cross-asset columns
        cross_cols = [col for col in df.columns if col.endswith('_close') and col != 'close']
        
        for col in cross_cols:
            asset_name = col.replace('_close', '')
            asset_price = df[col]
            
            # Price ratio
            df[f'{asset_name}_ratio'] = gold_close / (asset_price + 1e-8)
            
            # Return correlation
            gold_returns = gold_close.pct_change()
            asset_returns = asset_price.pct_change()
            
            for period in lookback_periods:
                # Rolling correlation
                df[f'{asset_name}_corr_{period}'] = gold_returns.rolling(period).corr(asset_returns)
                
                # Relative momentum (gold returns - asset returns)
                df[f'{asset_name}_rel_mom_{period}'] = (
                    gold_returns.rolling(period).sum() - 
                    asset_returns.rolling(period).sum()
                )
            
            # Z-score of ratio
            ratio = df[f'{asset_name}_ratio']
            df[f'{asset_name}_ratio_zscore'] = (ratio - ratio.rolling(20).mean()) / (ratio.rolling(20).std() + 1e-8)
        
        # Special features
        if 'dxy_close' in df.columns:
            # Gold typically moves inverse to USD
            df['gold_dxy_inverse'] = -df['dxy_ratio_zscore'] if 'dxy_ratio_zscore' in df.columns else 0
        
        if 'vix_close' in df.columns:
            # High VIX often bullish for gold (safe haven)
            vix = df['vix_close']
            df['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
            df['vix_regime'] = pd.cut(df['vix_zscore'], bins=[-5, -1, 1, 5], labels=[0, 1, 2])
        
        if 'silver_close' in df.columns:
            # Gold/Silver ratio - high ratio often means gold overvalued relative to silver
            df['gold_silver_ratio'] = gold_close / (df['silver_close'] + 1e-8)
        
        # Count new features
        new_cols = [col for col in df.columns if col not in cross_cols and col != 'close' and col != 'asset']
        logger.info(f"Created {len(new_cols)} cross-asset features")
        
        return df


def fetch_and_merge_cross_assets(
    gold_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch cross-assets and merge with gold data.
    
    Args:
        gold_df: Gold price DataFrame
        start_date: Override start date
        end_date: Override end date
        
    Returns:
        DataFrame with cross-asset features
    """
    fetcher = CrossAssetFetcher()
    
    # Determine date range from gold data
    if start_date is None:
        if isinstance(gold_df.index, pd.DatetimeIndex):
            start_date = gold_df.index.min().strftime('%Y-%m-%d')
        elif 'timestamp' in gold_df.columns:
            start_date = pd.to_datetime(gold_df['timestamp']).min().strftime('%Y-%m-%d')
        else:
            start_date = '2024-01-01'
    
    if end_date is None:
        if isinstance(gold_df.index, pd.DatetimeIndex):
            end_date = gold_df.index.max().strftime('%Y-%m-%d')
        elif 'timestamp' in gold_df.columns:
            end_date = pd.to_datetime(gold_df['timestamp']).max().strftime('%Y-%m-%d')
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch cross-asset data
    cross_assets = fetcher.fetch_all_assets(start_date, end_date, interval='1h')
    
    # Merge with gold
    df = fetcher.merge_with_gold(gold_df, cross_assets)
    
    # Create features
    df = fetcher.create_cross_asset_features(df)
    
    return df


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Test fetching
    fetcher = CrossAssetFetcher()
    
    # Fetch individual assets
    print("=== Testing Cross-Asset Fetcher ===\n")
    
    dxy = fetcher.fetch_asset('dxy', '2024-10-01', '2024-12-31')
    print(f"DXY data: {len(dxy)} rows")
    print(dxy.head())
    
    vix = fetcher.fetch_asset('vix', '2024-10-01', '2024-12-31')
    print(f"\nVIX data: {len(vix)} rows")
    print(vix.head())
    
    # Test with gold data
    print("\n=== Testing with Gold Data ===\n")
    gold_df = pd.read_csv('data/training/GOLD_H1.csv')
    gold_df['timestamp'] = pd.to_datetime(gold_df['timestamp'])
    gold_df = gold_df.set_index('timestamp')
    
    # Fetch and merge
    merged_df = fetch_and_merge_cross_assets(gold_df)
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"New columns: {[col for col in merged_df.columns if col not in gold_df.columns]}")
