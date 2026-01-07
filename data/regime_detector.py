"""
Market Regime Detection
========================
Implementation based on Two Sigma research using Gaussian Mixture Models (GMM).

Detects market regimes (Bull, Bear, Sideways, Crisis) and adds regime as a feature.
This allows the model to adapt strategies based on market conditions.

Reference: Two Sigma - "Machine Learning for Market Regime Modeling"
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List, Dict
from loguru import logger


class MarketRegimeDetector:
    """
    Detect market regimes using Gaussian Mixture Models (GMM).
    
    Regimes are typically:
    - 0: Low volatility / Sideways
    - 1: Bullish trend / Low volatility
    - 2: Bearish trend / High volatility
    - 3: Crisis / Very high volatility
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_volatility: int = 20,
        lookback_momentum: int = 10,
        random_state: int = 42
    ):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_volatility: Lookback period for volatility calculation
            lookback_momentum: Lookback period for momentum calculation
            random_state: Random state for reproducibility
        """
        self.n_regimes = n_regimes
        self.lookback_volatility = lookback_volatility
        self.lookback_momentum = lookback_momentum
        self.random_state = random_state
        
        self.gmm = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.regime_labels = {}
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Features:
        - Volatility (rolling std of returns)
        - Momentum (rolling return)
        - Volume relative change
        - High-Low range
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        returns = df['close'].pct_change()
        
        # Volatility - rolling std of returns
        features['volatility'] = returns.rolling(self.lookback_volatility).std() * np.sqrt(252)
        
        # Momentum - cumulative return over lookback period
        features['momentum'] = returns.rolling(self.lookback_momentum).sum()
        
        # Trend strength - absolute momentum
        features['trend_strength'] = features['momentum'].abs()
        
        # Range - (high - low) / close
        features['range'] = (df['high'] - df['low']) / df['close']
        features['avg_range'] = features['range'].rolling(self.lookback_volatility).mean()
        
        # Volume change (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol_mean = df['volume'].rolling(self.lookback_volatility).mean()
            features['volume_ratio'] = df['volume'] / (vol_mean + 1e-8)
        
        # Volatility regime (zscore)
        vol_mean = features['volatility'].rolling(60).mean()
        vol_std = features['volatility'].rolling(60).std()
        features['vol_zscore'] = (features['volatility'] - vol_mean) / (vol_std + 1e-8)
        
        # Momentum regime (zscore)
        mom_mean = features['momentum'].rolling(60).mean()
        mom_std = features['momentum'].rolling(60).std()
        features['mom_zscore'] = (features['momentum'] - mom_mean) / (mom_std + 1e-8)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def fit(self, df: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit GMM model to detect market regimes.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            self
        """
        logger.info(f"Fitting market regime detector with {self.n_regimes} regimes...")
        
        # Calculate features
        features = self._calculate_features(df)
        
        # Remove NaN rows
        features_clean = features.dropna()
        
        if len(features_clean) < 100:
            logger.warning(f"Not enough data for regime detection: {len(features_clean)} rows")
            return self
        
        # Scale features
        X = self.scaler.fit_transform(features_clean.values)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=self.random_state,
            n_init=10,
            max_iter=200
        )
        self.gmm.fit(X)
        
        # Label regimes based on characteristics
        self._label_regimes(features_clean, X)
        
        logger.info(f"Regime detection fitted. BIC: {self.gmm.bic(X):.2f}")
        
        return self
    
    def _label_regimes(self, features: pd.DataFrame, X: np.ndarray):
        """
        Interpret and label regimes based on their characteristics.
        """
        # Get regime assignments
        regimes = self.gmm.predict(X)
        
        # Calculate mean characteristics for each regime
        regime_chars = {}
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_chars[i] = {
                'volatility': features.iloc[mask]['volatility'].mean(),
                'momentum': features.iloc[mask]['momentum'].mean(),
                'count': mask.sum()
            }
        
        # Sort regimes by volatility (ascending)
        sorted_by_vol = sorted(regime_chars.items(), key=lambda x: x[1]['volatility'])
        
        # Assign labels
        if self.n_regimes == 3:
            self.regime_labels = {
                sorted_by_vol[0][0]: ('low_vol', 'Low Volatility'),
                sorted_by_vol[1][0]: ('normal', 'Normal'),
                sorted_by_vol[2][0]: ('high_vol', 'High Volatility')
            }
        elif self.n_regimes == 4:
            # Also consider momentum for 4 regimes
            self.regime_labels = {
                sorted_by_vol[0][0]: ('calm', 'Calm'),
                sorted_by_vol[1][0]: ('normal', 'Normal'),
                sorted_by_vol[2][0]: ('volatile', 'Volatile'),
                sorted_by_vol[3][0]: ('crisis', 'Crisis')
            }
        
        # Log regime characteristics
        for regime_id, chars in regime_chars.items():
            label = self.regime_labels.get(regime_id, ('unknown', 'Unknown'))[1]
            logger.info(
                f"Regime {regime_id} ({label}): "
                f"vol={chars['volatility']:.4f}, mom={chars['momentum']:.4f}, "
                f"count={chars['count']}"
            )
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict market regime for new data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with regime predictions
        """
        if self.gmm is None:
            logger.warning("Model not fitted. Call fit() first.")
            return pd.Series(index=df.index, data=0)
        
        # Calculate features
        features = self._calculate_features(df)
        
        # Handle NaN
        valid_mask = ~features.isna().any(axis=1)
        
        # Predict
        regimes = pd.Series(index=df.index, data=np.nan)
        
        if valid_mask.sum() > 0:
            X = self.scaler.transform(features[valid_mask].values)
            regimes[valid_mask] = self.gmm.predict(X)
        
        return regimes.fillna(0).astype(int)
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability of each regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime probabilities
        """
        if self.gmm is None:
            logger.warning("Model not fitted. Call fit() first.")
            return pd.DataFrame()
        
        # Calculate features
        features = self._calculate_features(df)
        valid_mask = ~features.isna().any(axis=1)
        
        # Initialize probabilities
        proba_cols = [f'regime_{i}_prob' for i in range(self.n_regimes)]
        proba_df = pd.DataFrame(index=df.index, columns=proba_cols, data=0.0)
        
        if valid_mask.sum() > 0:
            X = self.scaler.transform(features[valid_mask].values)
            proba = self.gmm.predict_proba(X)
            proba_df.loc[valid_mask, proba_cols] = proba
        
        return proba_df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime and regime probability features to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added regime features
        """
        logger.info("Adding regime features...")
        
        df = df.copy()
        
        # Fit if not already fitted
        if self.gmm is None:
            self.fit(df)
        
        # Add regime prediction
        df['market_regime'] = self.predict(df)
        
        # Add regime probabilities
        proba_df = self.predict_proba(df)
        for col in proba_df.columns:
            df[col] = proba_df[col]
        
        # Add regime-based features
        df['is_high_vol_regime'] = (df['market_regime'] == self.n_regimes - 1).astype(int)
        df['is_low_vol_regime'] = (df['market_regime'] == 0).astype(int)
        
        # Regime transition (when regime changes)
        df['regime_change'] = (df['market_regime'] != df['market_regime'].shift(1)).astype(int)
        
        logger.info(f"Added regime features. Regime distribution: {df['market_regime'].value_counts().to_dict()}")
        
        return df


def detect_and_add_regimes(
    df: pd.DataFrame,
    n_regimes: int = 3,
    fit_on_train: bool = True,
    train_ratio: float = 0.7
) -> pd.DataFrame:
    """
    Convenience function to detect regimes and add as features.
    
    Args:
        df: DataFrame with OHLCV data
        n_regimes: Number of regimes
        fit_on_train: If True, fit only on training portion to prevent leakage
        train_ratio: Ratio of data to use for fitting
        
    Returns:
        DataFrame with regime features
    """
    detector = MarketRegimeDetector(n_regimes=n_regimes)
    
    if fit_on_train:
        # Fit only on training data
        train_size = int(len(df) * train_ratio)
        detector.fit(df.iloc[:train_size])
    else:
        detector.fit(df)
    
    return detector.add_regime_features(df)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Load data
    df = pd.read_csv('data/training/GOLD_H1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print(f"Data shape: {df.shape}")
    
    # Detect regimes
    detector = MarketRegimeDetector(n_regimes=3)
    detector.fit(df)
    
    # Add features
    df_with_regimes = detector.add_regime_features(df)
    
    print(f"\nRegime distribution:")
    print(df_with_regimes['market_regime'].value_counts())
    
    print(f"\nNew columns: {[col for col in df_with_regimes.columns if col not in df.columns]}")
