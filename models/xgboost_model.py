"""
XGBoost Model Module
====================
Production-grade XGBoost for trend classification and regression.

Features:
- Walk-forward validation support
- Feature importance analysis
- Hyperparameter optimization
- Cross-validation with time series splits
- Probability calibration
"""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger


class XGBoostModel:
    """
    XGBoost model wrapper for trading signal classification/regression.
    
    Features:
    - Time-series aware training
    - Feature importance tracking
    - Model persistence
    - Comprehensive metrics
    
    Usage:
        model = XGBoostModel(task='classification')
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        importance = model.get_feature_importance()
    """
    
    DEFAULT_PARAMS = {
        'classification': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1
        },
        'regression': {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    def __init__(
        self,
        task: str = 'classification',
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize XGBoost model.
        
        Args:
            task: 'classification' or 'regression'
            params: Custom XGBoost parameters
            feature_names: List of feature names
        """
        self.task = task
        self.feature_names = feature_names
        
        # Merge default params with custom params
        default_params = self.DEFAULT_PARAMS[task].copy()
        if params:
            default_params.update(params)
        
        self.params = default_params
        
        # Create model
        if task == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        
        # Training info
        self.best_iteration = None
        self.feature_importance_ = None
        self.training_history = {}
        
        logger.info(f"XGBoostModel created for {task}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'XGBoostModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Rounds without improvement to stop
            verbose: Print training progress
            
        Returns:
            self
        """
        logger.info(f"Training XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Fit model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store best iteration
        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Log training results
        if X_val is not None:
            val_pred = self.predict(X_val)
            metrics = self.evaluate(y_val, val_pred)
            logger.info(f"Validation metrics: {metrics}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Features
            
        Returns:
            Probability of positive class
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        return self.model.predict_proba(X)[:, 1]
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            self.feature_importance_ = dict(zip(self.feature_names, importance))
        else:
            self.feature_importance_ = dict(zip(
                [f'feature_{i}' for i in range(len(importance))],
                importance
            ))
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            self._calculate_feature_importance()
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance_.items()
        ])
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of metrics
        """
        if self.task == 'classification':
            # Check if multiclass
            n_classes = len(np.unique(y_true))
            avg = 'macro' if n_classes > 2 else 'binary'
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0, average=avg),
                'recall': recall_score(y_true, y_pred, zero_division=0, average=avg),
                'f1': f1_score(y_true, y_pred, zero_division=0, average=avg)
            }
            
            if y_proba is not None:
                try:
                    if n_classes > 2:
                        metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_proba)
                except ValueError:
                    metrics['auc'] = 0.5
        else:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of splits
            
        Returns:
            Dictionary of metrics for each fold
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = {
            'accuracy' if self.task == 'classification' else 'rmse': [],
            'precision' if self.task == 'classification' else 'mae': [],
            'recall' if self.task == 'classification' else 'r2': [],
            'f1' if self.task == 'classification' else 'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create new model for this fold
            if self.task == 'classification':
                fold_model = xgb.XGBClassifier(**self.params)
            else:
                fold_model = xgb.XGBRegressor(**self.params)
            
            # Train
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            metrics = self.evaluate(y_val, y_pred)
            
            for key in cv_metrics:
                if key in metrics:
                    cv_metrics[key].append(metrics[key])
            
            logger.info(f"Fold {fold + 1}: {metrics}")
        
        # Calculate mean and std
        cv_summary = {}
        for key, values in cv_metrics.items():
            if values:
                cv_summary[f'{key}_mean'] = np.mean(values)
                cv_summary[f'{key}_std'] = np.std(values)
        
        logger.info(f"CV Summary: {cv_summary}")
        
        return cv_metrics
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using booster for compatibility
        self.model.get_booster().save_model(str(filepath.with_suffix('.json')))
        
        # Save metadata (convert numpy types to Python types)
        feature_imp = None
        if self.feature_importance_:
            feature_imp = {k: float(v) for k, v in self.feature_importance_.items()}
        
        metadata = {
            'task': self.task,
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in self.params.items()},
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'feature_importance': feature_imp
        }
        
        with open(filepath.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'XGBoostModel':
        """Load model from file."""
        filepath = Path(filepath)
        
        # Load model
        self.model.load_model(str(filepath.with_suffix('.json')))
        
        # Load metadata
        meta_path = filepath.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            self.task = metadata.get('task', self.task)
            self.feature_names = metadata.get('feature_names')
            self.best_iteration = metadata.get('best_iteration')
            self.feature_importance_ = metadata.get('feature_importance')
        
        logger.info(f"Model loaded from {filepath}")
        return self


class XGBoostEnsemble:
    """
    Ensemble of XGBoost models with walk-forward training.
    
    Usage:
        ensemble = XGBoostEnsemble(n_models=5)
        ensemble.fit_walk_forward(X, y, train_size=180, test_size=30)
        predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        n_models: int = 5,
        task: str = 'classification',
        params: Optional[Dict] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            n_models: Number of models in ensemble
            task: 'classification' or 'regression'
            params: XGBoost parameters
        """
        self.n_models = n_models
        self.task = task
        self.params = params
        self.models: List[XGBoostModel] = []
        self.weights: List[float] = []
    
    def fit_walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: int,
        test_size: int,
        step: Optional[int] = None
    ) -> 'XGBoostEnsemble':
        """
        Train models using walk-forward validation.
        
        Args:
            X: Features
            y: Targets
            train_size: Training window size
            test_size: Test window size
            step: Step size between windows
            
        Returns:
            self
        """
        if step is None:
            step = test_size
        
        n_samples = len(X)
        start = 0
        
        while start + train_size + test_size <= n_samples and len(self.models) < self.n_models:
            train_end = start + train_size
            test_end = train_end + test_size
            
            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            
            # Train model
            model = XGBoostModel(task=self.task, params=self.params)
            model.fit(X_train, y_train, X_test, y_test, verbose=False)
            
            # Evaluate and weight by performance
            y_pred = model.predict(X_test)
            metrics = model.evaluate(y_test, y_pred)
            
            weight = metrics.get('accuracy', metrics.get('r2', 0.5))
            
            self.models.append(model)
            self.weights.append(weight)
            
            logger.info(f"Model {len(self.models)}: weight={weight:.4f}")
            
            start += step
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction."""
        if not self.models:
            raise ValueError("No models trained")
        
        if self.task == 'classification':
            # Weighted voting
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                predictions += weight * model.predict_proba(X)
            return (predictions >= 0.5).astype(int)
        else:
            # Weighted average
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                predictions += weight * model.predict(X)
            return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction."""
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")
        
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict_proba(X)
        
        return predictions


if __name__ == "__main__":
    # Test XGBoost model
    print("=== Testing XGBoost Model ===")
    
    # Create dummy data
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Train model
    model = XGBoostModel(task='classification')
    model.fit(X_train, y_train, X_val, y_val, verbose=False)
    
    # Evaluate
    predictions = model.predict(X_test)
    metrics = model.evaluate(y_test, predictions)
    
    print(f"\nTest metrics: {metrics}")
    print(f"\nTop 10 features:\n{model.get_feature_importance(top_n=10)}")
    
    # Test cross-validation
    print("\n=== Cross-validation ===")
    model.cross_validate(X, y, n_splits=3)
