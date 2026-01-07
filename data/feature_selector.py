"""
Feature Selector Module
=======================
Select best features for trading models.

Features:
- Correlation-based selection
- Feature importance ranking
- Recursive feature elimination
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from loguru import logger


class FeatureSelector:
    """
    Select optimal features for trading models.
    
    Usage:
        selector = FeatureSelector()
        selected_features = selector.select(df, target_col='close', n_features=30)
    """
    
    # Features known to be useful for gold trading
    PRIORITY_FEATURES = [
        'rsi_14', 'rsi_7', 'rsi_21',
        'macd', 'macd_signal', 'macd_histogram',
        'ema_9', 'ema_21', 'ema_50',
        'atr', 'atr_pct',
        'bb_pct', 'bb_width',
        'adx', 'plus_di', 'minus_di',
        'stoch_k', 'stoch_d',
        'momentum', 'roc',
        'returns', 'log_returns'
    ]
    
    # Features to exclude (potential data leakage or noise)
    EXCLUDE_FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'dividends', 'stock splits', 'capital gains'
    ]
    
    def __init__(
        self,
        method: str = 'combined',
        n_features: int = 30,
        correlation_threshold: float = 0.95
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('importance', 'correlation', 'combined')
            n_features: Number of features to select
            correlation_threshold: Remove features with correlation above this
        """
        self.method = method
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.selected_features: List[str] = []
        self.feature_scores: dict = {}
    
    def select(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        n_features: Optional[int] = None
    ) -> List[str]:
        """
        Select best features from DataFrame.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            n_features: Override number of features
            
        Returns:
            List of selected feature names
        """
        if n_features is None:
            n_features = self.n_features
        
        logger.info(f"Selecting {n_features} features from {len(df.columns)} columns")
        
        # Get feature columns (exclude target and excluded)
        feature_cols = [
            col for col in df.columns 
            if col not in self.EXCLUDE_FEATURES and col != target_col
        ]
        
        # Create target (next return direction)
        df = df.copy()
        df['_target'] = (df[target_col].shift(-1) > df[target_col]).astype(int)
        df = df.dropna()
        
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df['_target']
        
        # Step 1: Remove highly correlated features
        feature_cols = self._remove_correlated(X, feature_cols)
        X = X[feature_cols]
        
        logger.info(f"After correlation filter: {len(feature_cols)} features")
        
        # Step 2: Score features
        if self.method == 'importance':
            scores = self._score_by_importance(X, y)
        elif self.method == 'correlation':
            scores = self._score_by_mutual_info(X, y)
        else:  # combined
            imp_scores = self._score_by_importance(X, y)
            mi_scores = self._score_by_mutual_info(X, y)
            scores = {k: (imp_scores.get(k, 0) + mi_scores.get(k, 0)) / 2 
                     for k in feature_cols}
        
        self.feature_scores = scores
        
        # Step 3: Select top features (prioritize known good features)
        selected = self._select_top_features(feature_cols, scores, n_features)
        
        self.selected_features = selected
        
        logger.info(f"Selected {len(selected)} features")
        logger.info(f"Top 10: {selected[:10]}")
        
        return selected
    
    def _remove_correlated(
        self,
        X: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = X[feature_cols].corr().abs()
        
        # Upper triangle mask
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [
            column for column in upper.columns 
            if any(upper[column] > self.correlation_threshold)
        ]
        
        # Keep priority features even if correlated
        to_drop = [f for f in to_drop if f not in self.PRIORITY_FEATURES]
        
        return [f for f in feature_cols if f not in to_drop]
    
    def _score_by_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict:
        """Score features by Random Forest importance."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Normalize
        max_imp = max(importance.values()) if importance else 1
        return {k: v / max_imp for k, v in importance.items()}
    
    def _score_by_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict:
        """Score features by mutual information."""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        importance = dict(zip(X.columns, mi_scores))
        
        # Normalize
        max_imp = max(importance.values()) if importance else 1
        return {k: v / max_imp for k, v in importance.items()}
    
    def _select_top_features(
        self,
        feature_cols: List[str],
        scores: dict,
        n_features: int
    ) -> List[str]:
        """Select top features, prioritizing known good ones."""
        selected = []
        
        # First, add priority features that exist
        for feat in self.PRIORITY_FEATURES:
            if feat in feature_cols and feat not in selected:
                selected.append(feat)
                if len(selected) >= n_features:
                    return selected
        
        # Then add by score
        remaining = [f for f in feature_cols if f not in selected]
        sorted_features = sorted(remaining, key=lambda x: scores.get(x, 0), reverse=True)
        
        for feat in sorted_features:
            if len(selected) >= n_features:
                break
            selected.append(feat)
        
        return selected
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature scores as DataFrame."""
        return pd.DataFrame([
            {'feature': k, 'score': v}
            for k, v in sorted(
                self.feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])


def select_features(
    df: pd.DataFrame,
    target_col: str = 'close',
    n_features: int = 30
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to select features.
    
    Args:
        df: DataFrame with all features
        target_col: Target column
        n_features: Number of features to select
        
    Returns:
        Tuple of (filtered DataFrame, selected feature names)
    """
    selector = FeatureSelector(n_features=n_features)
    selected = selector.select(df, target_col, n_features)
    
    # Keep target column and selected features
    cols_to_keep = ['open', 'high', 'low', 'close', 'volume'] + selected
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    return df[cols_to_keep], selected


if __name__ == "__main__":
    # Test feature selector
    print("=== Testing Feature Selector ===")
    
    # Create sample data
    n_samples = 1000
    df = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 2000,
        'high': np.random.randn(n_samples).cumsum() + 2005,
        'low': np.random.randn(n_samples).cumsum() + 1995,
        'close': np.random.randn(n_samples).cumsum() + 2000,
        'volume': np.random.randint(1000, 10000, n_samples),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd': np.random.randn(n_samples) * 5,
        'atr': np.abs(np.random.randn(n_samples)) * 10,
        'ema_21': np.random.randn(n_samples).cumsum() + 1998,
        'noise_1': np.random.randn(n_samples),
        'noise_2': np.random.randn(n_samples),
        'returns': np.random.randn(n_samples) * 0.01
    })
    
    selector = FeatureSelector(n_features=5)
    selected = selector.select(df, 'close', 5)
    
    print(f"\nSelected features: {selected}")
    print(f"\nFeature scores:\n{selector.get_feature_scores()}")
