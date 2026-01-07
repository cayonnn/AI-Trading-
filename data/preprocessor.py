"""
Data Preprocessor Module
========================
Production-grade data preprocessing for ML/DL models.

Features:
- Sequence creation for LSTM/CNN
- Feature scaling (MinMax, Standard, Robust)
- Train/validation/test splitting
- Walk-forward validation support
- Feature selection
"""

from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger


class ScalerType(Enum):
    """Scaler types for feature normalization."""
    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"


@dataclass
class DataSplit:
    """Container for train/validation/test splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: object
    
    def summary(self) -> str:
        """Return summary of data splits."""
        return (
            f"Train: {self.X_train.shape}, "
            f"Val: {self.X_val.shape}, "
            f"Test: {self.X_test.shape}"
        )


class DataPreprocessor:
    """
    Preprocess data for machine learning models.
    
    Usage:
        preprocessor = DataPreprocessor(
            sequence_length=60,
            prediction_horizon=1,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        splits = preprocessor.prepare_data(df, target_col='close')
    """
    
    # Default feature columns to exclude from training
    EXCLUDE_COLUMNS = ['open', 'high', 'low', 'volume', 'dividends', 'stock_splits']
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: ScalerType = ScalerType.ROBUST,
        include_target_in_features: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Number of time steps for sequences
            prediction_horizon: Steps ahead to predict
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            scaler_type: Type of feature scaler
            include_target_in_features: Include target column in input features
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler_type = scaler_type
        self.include_target_in_features = include_target_in_features
        
        self.scaler = None
        self.target_scaler = None
        self.feature_names = None
        
        logger.info(
            f"DataPreprocessor initialized: seq_len={sequence_length}, "
            f"horizon={prediction_horizon}, train={train_ratio}, val={val_ratio}"
        )
    
    def _create_scaler(self) -> object:
        """Create scaler based on type."""
        if self.scaler_type == ScalerType.MINMAX:
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        elif self.scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        else:
            return MinMaxScaler()
    
    def _select_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            exclude_cols: Columns to exclude
            
        Returns:
            Tuple of (feature DataFrame, feature names)
        """
        exclude = set(exclude_cols or self.EXCLUDE_COLUMNS)
        
        # Keep target if needed
        if self.include_target_in_features:
            exclude.discard(target_col)
        
        # Select numeric columns only
        feature_cols = [
            col for col in df.columns
            if col not in exclude and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
        
        return df[feature_cols], feature_cols
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling NaN and infinite values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill
        df = df.ffill().bfill()
        
        # Drop any remaining NaN rows
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        return df
    
    def _create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models.
        
        Args:
            data: Feature array
            target: Target array
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y)
    
    def _create_classification_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Create classification target (1 = up, 0 = down).
        
        Args:
            df: Input DataFrame
            target_col: Column to use for creating target
            threshold: Minimum change to be classified as up/down
            
        Returns:
            Binary target array
        """
        # Future returns
        future_returns = df[target_col].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Create target: 1 if price goes up, 0 if down
        target = (future_returns > threshold).astype(int)
        
        return target.values
    
    def _create_regression_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'close'
    ) -> np.ndarray:
        """
        Create regression target (future price).
        
        Args:
            df: Input DataFrame
            target_col: Column to use as target
            
        Returns:
            Target array
        """
        # Shift target to get future value
        target = df[target_col].shift(-self.prediction_horizon)
        
        return target.values
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        task: str = 'regression',
        exclude_cols: Optional[List[str]] = None
    ) -> DataSplit:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame with features
            target_col: Target column name
            task: 'regression' or 'classification'
            exclude_cols: Columns to exclude from features
            
        Returns:
            DataSplit with train/val/test splits
        """
        logger.info(f"Preparing data for {task}...")
        
        # Select features
        feature_df, feature_names = self._select_features(df, target_col, exclude_cols)
        self.feature_names = feature_names
        
        # Clean data
        feature_df = self._clean_data(feature_df)
        
        # Create target
        if task == 'classification':
            target = self._create_classification_target(df.loc[feature_df.index], target_col)
        else:
            target = self._create_regression_target(df.loc[feature_df.index], target_col)
        
        # Remove NaN from target
        valid_mask = ~np.isnan(target)
        feature_df = feature_df[valid_mask]
        target = target[valid_mask]
        
        # Scale features
        self.scaler = self._create_scaler()
        scaled_features = self.scaler.fit_transform(feature_df.values)
        
        # Scale target for regression
        if task == 'regression':
            self.target_scaler = self._create_scaler()
            target = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(scaled_features, target)
        
        logger.info(f"Created {len(X)} sequences of shape {X[0].shape}")
        
        # Split data
        n_samples = len(X)
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        split = DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            scaler=self.scaler
        )
        
        logger.info(f"Data split: {split.summary()}")
        
        return split
    
    def prepare_for_xgboost(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        task: str = 'classification',
        exclude_cols: Optional[List[str]] = None
    ) -> DataSplit:
        """
        Prepare data for XGBoost (no sequences).
        
        Args:
            df: Input DataFrame with features
            target_col: Target column name
            task: 'regression' or 'classification'
            exclude_cols: Columns to exclude
            
        Returns:
            DataSplit with train/val/test splits
        """
        logger.info(f"Preparing data for XGBoost ({task})...")
        
        # Select features
        feature_df, feature_names = self._select_features(df, target_col, exclude_cols)
        self.feature_names = feature_names
        
        # Clean data
        feature_df = self._clean_data(feature_df)
        
        # Create target
        if task == 'classification':
            target = self._create_classification_target(df.loc[feature_df.index], target_col)
        else:
            target = self._create_regression_target(df.loc[feature_df.index], target_col)
        
        # Remove NaN from target
        valid_mask = ~np.isnan(target)
        feature_df = feature_df[valid_mask]
        target = target[valid_mask]
        
        X = feature_df.values
        y = target
        
        # Split data
        n_samples = len(X)
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        split = DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            scaler=None
        )
        
        logger.info(f"XGBoost data split: {split.summary()}")
        
        return split
    
    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        train_period: int = 180,
        test_period: int = 30,
        step: int = 30
    ) -> List[DataSplit]:
        """
        Create walk-forward validation splits.
        
        Args:
            df: Input DataFrame
            target_col: Target column
            train_period: Days for training
            test_period: Days for testing
            step: Step size in days
            
        Returns:
            List of DataSplit objects
        """
        splits = []
        
        # Assume daily data or convert to daily periods
        n_samples = len(df)
        
        start = 0
        while start + train_period + test_period <= n_samples:
            train_end = start + train_period
            test_end = train_end + test_period
            
            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]
            
            # Prepare train data
            feature_df, feature_names = self._select_features(train_df, target_col)
            feature_df = self._clean_data(feature_df)
            
            target_train = self._create_classification_target(train_df.loc[feature_df.index], target_col)
            
            # Prepare test data
            test_feature_df = test_df[feature_names]
            test_feature_df = self._clean_data(test_feature_df)
            target_test = self._create_classification_target(test_df.loc[test_feature_df.index], target_col)
            
            # Remove NaN
            valid_train = ~np.isnan(target_train)
            valid_test = ~np.isnan(target_test)
            
            X_train = feature_df[valid_train].values
            y_train = target_train[valid_train]
            X_test = test_feature_df[valid_test].values
            y_test = target_test[valid_test]
            
            split = DataSplit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test[:len(X_test)//2],  # First half as val
                y_val=y_test[:len(y_test)//2],
                X_test=X_test[len(X_test)//2:],  # Second half as test
                y_test=y_test[len(y_test)//2:],
                feature_names=feature_names,
                scaler=None
            )
            
            splits.append(split)
            start += step
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        return splits
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Original scale predictions
        """
        if self.target_scaler is None:
            return predictions
        
        return self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()


def prepare_training_data(
    df: pd.DataFrame,
    sequence_length: int = 60,
    task: str = 'classification'
) -> DataSplit:
    """
    Convenience function to prepare training data.
    
    Args:
        df: DataFrame with OHLCV and indicators
        sequence_length: Sequence length for LSTM
        task: 'regression' or 'classification'
        
    Returns:
        DataSplit with train/val/test data
    """
    preprocessor = DataPreprocessor(
        sequence_length=sequence_length,
        prediction_horizon=1,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    return preprocessor.prepare_data(df, target_col='close', task=task)


if __name__ == "__main__":
    # Test with sample data
    from data.data_fetcher import DataFetcher
    from data.indicators import calculate_indicators
    
    print("=== Testing Data Preprocessor ===")
    
    # Fetch and prepare data
    fetcher = DataFetcher()
    df = fetcher.fetch('GC=F', '1h', lookback_days=90)
    df = calculate_indicators(df, 'minimal')
    
    print(f"Original shape: {df.shape}")
    
    # Test LSTM preprocessing
    preprocessor = DataPreprocessor(sequence_length=60)
    splits = preprocessor.prepare_data(df, task='classification')
    
    print(f"\nLSTM Data:")
    print(f"  X_train shape: {splits.X_train.shape}")
    print(f"  y_train shape: {splits.y_train.shape}")
    print(f"  Features: {len(splits.feature_names)}")
    
    # Test XGBoost preprocessing
    xgb_splits = preprocessor.prepare_for_xgboost(df, task='classification')
    
    print(f"\nXGBoost Data:")
    print(f"  X_train shape: {xgb_splits.X_train.shape}")
    print(f"  y_train shape: {xgb_splits.y_train.shape}")
