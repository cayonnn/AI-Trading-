"""
Advanced Feature Engineering Module
====================================
Production-grade feature engineering for ML trading models.

Features:
- Lag Features (Returns, Volatility)
- Rolling Statistics (Mean, Std, Skew, Kurtosis)
- Technical Patterns
- Time-based Features
- Cross-asset Correlation Features
- Feature Selection & Validation

Author: AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Lag periods
    lag_periods: List[int] = None
    # Rolling windows
    rolling_windows: List[int] = None
    # Include time features
    include_time_features: bool = True
    # Include pattern features
    include_patterns: bool = True
    # Target column
    target_column: str = 'target'
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10, 20]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]


class AdvancedFeatureEngineer:
    """
    Production-grade feature engineering pipeline.
    
    Creates 100+ features from OHLCV data for ML models.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self.scaler = RobustScaler()
        self._fitted = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting advanced feature engineering...")
        df = df.copy()
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        # 1. Price-based features
        df = self._add_price_features(df)
        logger.info(f"  [1/7] Price features added: {len(df.columns)} columns")
        
        # 2. Lag features
        df = self._add_lag_features(df)
        logger.info(f"  [2/7] Lag features added: {len(df.columns)} columns")
        
        # 3. Rolling statistics
        df = self._add_rolling_features(df)
        logger.info(f"  [3/7] Rolling features added: {len(df.columns)} columns")
        
        # 4. Momentum features
        df = self._add_momentum_features(df)
        logger.info(f"  [4/7] Momentum features added: {len(df.columns)} columns")
        
        # 5. Volatility features
        df = self._add_volatility_features(df)
        logger.info(f"  [5/7] Volatility features added: {len(df.columns)} columns")
        
        # 6. Time features
        if self.config.include_time_features:
            df = self._add_time_features(df)
            logger.info(f"  [6/7] Time features added: {len(df.columns)} columns")
        
        # 7. Pattern features
        if self.config.include_patterns:
            df = self._add_pattern_features(df)
            logger.info(f"  [7/7] Pattern features added: {len(df.columns)} columns")
        
        # Clean up
        df = self._clean_features(df)
        
        self.feature_names = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 'time', 'target', 'date'] and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int', 'float']]
        
        logger.success(f"Feature engineering complete: {len(self.feature_names)} features created")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Candle features
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['close']
        
        # Body to range ratio
        df['body_range_ratio'] = np.abs(df['body']) / (df['range'] + 1e-10)
        
        # Is bullish/bearish
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (np.abs(df['body_pct']) < 0.001).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        features_to_lag = ['returns', 'log_returns', 'range_pct', 'body_pct', 'volume']
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in self.config.lag_periods:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Cumulative returns
        for period in self.config.lag_periods:
            df[f'cum_returns_{period}'] = df['returns'].rolling(window=period).sum()
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        for window in self.config.rolling_windows:
            # Mean
            df[f'close_ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Standard deviation
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            
            # Min/Max
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Position in range
            df[f'price_position_{window}'] = (
                (df['close'] - df[f'close_min_{window}']) / 
                (df[f'close_max_{window}'] - df[f'close_min_{window}'] + 1e-10)
            )
            
            # Skewness and Kurtosis
            df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            
            # Z-score
            df[f'close_zscore_{window}'] = (
                (df['close'] - df[f'close_ma_{window}']) / 
                (df[f'close_std_{window}'] + 1e-10)
            )
            
            # Volume ratio
            df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)
            
            # Trend strength
            df[f'trend_{window}'] = df['close'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=False
            )
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # RSI for multiple periods
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI divergence
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(14)
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_change'] = df['macd_histogram'].diff()
        
        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10) * 100
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # ATR
        for period in [7, 14, 21]:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.ewm(span=period, adjust=False).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close'] * 100
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_pct_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
        
        # Keltner Channels
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        df['keltner_upper'] = ema20 + (2 * df['atr_14'])
        df['keltner_lower'] = ema20 - (2 * df['atr_14'])
        df['keltner_width'] = (df['keltner_upper'] - df['keltner_lower']) / ema20
        
        # Squeeze indicator (BB inside Keltner)
        df['squeeze'] = ((df['bb_lower_20'] > df['keltner_lower']) & 
                         (df['bb_upper_20'] < df['keltner_upper'])).astype(int)
        
        # Historical volatility
        for period in [10, 20, 50]:
            df[f'hist_vol_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Volatility ratio
        df['vol_ratio'] = df['hist_vol_10'] / (df['hist_vol_50'] + 1e-10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            ts = df['timestamp']
            # Handle timezone-aware datetime
            if hasattr(ts.dtype, 'tz') and ts.dtype.tz is not None:
                ts = ts.dt.tz_localize(None)
            elif not pd.api.types.is_datetime64_any_dtype(ts):
                ts = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
        elif df.index.dtype == 'datetime64[ns]':
            ts = pd.to_datetime(df.index)
        else:
            return df
        
        # Hour of day (cyclical encoding)
        df['hour'] = ts.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (cyclical encoding)
        df['day_of_week'] = ts.dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (cyclical encoding)
        df['month'] = ts.dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Is Asian/London/NY session
        df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_newyork'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Is weekend
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        # Doji
        df['doji'] = (np.abs(df['body_pct']) < 0.001).astype(int)
        
        # Hammer (small body, long lower shadow)
        df['hammer'] = (
            (df['lower_shadow'] > 2 * np.abs(df['body'])) & 
            (df['upper_shadow'] < np.abs(df['body']))
        ).astype(int)
        
        # Shooting star (small body, long upper shadow)
        df['shooting_star'] = (
            (df['upper_shadow'] > 2 * np.abs(df['body'])) & 
            (df['lower_shadow'] < np.abs(df['body']))
        ).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['body'].shift(1) < 0) &  # Previous bearish
            (df['body'] > 0) &  # Current bullish
            (df['open'] < df['close'].shift(1)) &  # Opens below prev close
            (df['close'] > df['open'].shift(1))  # Closes above prev open
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['body'].shift(1) > 0) &  # Previous bullish
            (df['body'] < 0) &  # Current bearish
            (df['open'] > df['close'].shift(1)) &  # Opens above prev close
            (df['close'] < df['open'].shift(1))  # Closes below prev open
        ).astype(int)
        
        # Consecutive candles
        df['consecutive_bullish'] = df['is_bullish'].rolling(window=3).sum()
        df['consecutive_bearish'] = (1 - df['is_bullish']).rolling(window=3).sum()
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_high_streak'] = df['higher_high'].rolling(window=5).sum()
        df['lower_low_streak'] = df['lower_low'].rolling(window=5).sum()
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaNs
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop remaining NaNs (usually at the beginning)
        df = df.dropna()
        
        return df
    
    def select_features(
        self, 
        df: pd.DataFrame, 
        target: pd.Series,
        n_features: int = 50,
        method: str = 'mutual_info'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top N features using statistical methods.
        
        Args:
            df: Feature DataFrame
            target: Target variable
            n_features: Number of features to select
            method: 'mutual_info' or 'correlation'
            
        Returns:
            Tuple of (selected features DataFrame, feature names)
        """
        logger.info(f"Selecting top {n_features} features using {method}...")
        
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
        X = df[feature_cols]
        
        if method == 'mutual_info':
            # Mutual Information
            mi_scores = mutual_info_classif(X, target, random_state=42)
            feature_scores = pd.Series(mi_scores, index=feature_cols)
        else:
            # Correlation
            correlations = X.corrwith(target).abs()
            feature_scores = correlations
        
        # Select top features
        top_features = feature_scores.nlargest(n_features).index.tolist()
        
        logger.info(f"Top 10 features: {top_features[:10]}")
        
        return df[top_features], top_features
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


def create_target(
    df: pd.DataFrame, 
    horizon: int = 5,  # Look ahead 5 bars (5 hours for H1)
    target_type: str = 'binary',
    threshold: float = 0.001,  # 0.1% minimum move for LONG signal
    use_max_return: bool = True  # Use max potential return within horizon
) -> pd.Series:
    """
    Create professional target variable for ML.
    
    PRODUCTION-GRADE LABELING:
    - Uses 5-bar horizon instead of 1 bar (captures real trends)
    - Requires 0.1% minimum move (filters noise)
    - Option to use max return within horizon (better for trading)
    
    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon (bars ahead) - default 5
        target_type: 'binary' (up/down) or 'regression' (returns)
        threshold: Minimum return for positive class - default 0.1%
        use_max_return: If True, use best return within horizon window
        
    Returns:
        Target series (1=LONG opportunity, 0=WAIT/DOWN)
    """
    if use_max_return and target_type == 'binary':
        # Calculate max return within the next 'horizon' bars
        # This better captures if price goes up at any point within horizon
        future_highs = df['high'].shift(-1).rolling(window=horizon, min_periods=1).max()
        max_returns = (future_highs.shift(1-horizon) / df['close']) - 1
        
        # Also calculate close-to-close return for confirmation
        future_close = df['close'].shift(-horizon)
        close_returns = (future_close / df['close']) - 1
        
        # Label = 1 if BOTH conditions met:
        # 1. Max potential return >= threshold (opportunity existed)
        # 2. Close return is positive (trend was bullish)
        target = ((max_returns >= threshold) & (close_returns > 0)).astype(int)
    else:
        # Simple close-to-close return
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        if target_type == 'binary':
            target = (future_returns > threshold).astype(int)
        else:
            target = future_returns
    
    return target


if __name__ == "__main__":
    from loguru import logger
    import sys
    
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Test with sample data
    logger.info("="*60)
    logger.info("  ADVANCED FEATURE ENGINEERING - TEST")
    logger.info("="*60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': 2000 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, n)
    })
    df['high'] = df['open'] + np.abs(np.random.randn(n) * 2)
    df['low'] = df['open'] - np.abs(np.random.randn(n) * 2)
    df['close'] = df['open'] + np.random.randn(n) * 1
    
    # Engineer features
    fe = AdvancedFeatureEngineer()
    df_features = fe.engineer_features(df)
    
    logger.info(f"\nOriginal shape: {df.shape}")
    logger.info(f"Features shape: {df_features.shape}")
    logger.info(f"Total features: {len(fe.feature_names)}")
    logger.info(f"\nSample features:\n{fe.feature_names[:20]}")
