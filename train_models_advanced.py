"""
Advanced Model Training Pipeline v3.0
=====================================
Production-grade training with:
- Walk-forward validation
- SHAP feature selection
- Optuna hyperparameter optimization
- Class imbalance handling
- Comprehensive metrics
- Model ensembling support

Author: AI Trading System
Version: 3.0.0
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import AdvancedFeatureEngineer, create_target
from data.preprocessor import DataPreprocessor
from models.lstm_model import LSTMPredictor
from models.xgboost_model import XGBoostModel

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Run: pip install optuna")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class AdvancedModelTrainer:
    """Production-grade model training pipeline v3.0"""
    
    def __init__(
        self,
        data_path: str = "data/training/GOLD_H1.csv",
        checkpoint_dir: str = "models/checkpoints",
        use_optuna: bool = True,
        n_trials: int = 50,
        n_features: int = 50,  # Number of features to select
    ):
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.n_trials = n_trials
        self.n_features = n_features
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = RobustScaler()
        self.selected_features = []
        self.best_params = {}
        self.metrics = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load data and engineer features"""
        logger.info("=" * 60)
        logger.info("  LOADING AND PREPARING DATA")
        logger.info("=" * 60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Handle different datetime column names
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        else:
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
            
        df = df.sort_values('timestamp')
        logger.info(f"Loaded {len(df)} rows from {self.data_path}")
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df)
        
        # Create target
        df['target'] = create_target(df, horizon=1, target_type='binary')
        df = df.dropna()
        
        logger.info(f"Features engineered: {len(self.feature_engineer.feature_names)}")
        logger.info(f"Final dataset: {len(df)} rows")
        
        # Log class distribution
        class_dist = df['target'].value_counts()
        logger.info(f"Class distribution: 0={class_dist.get(0, 0)}, 1={class_dist.get(1, 0)}")
        
        return df
    
    def split_data_walk_forward(
        self, 
        df: pd.DataFrame, 
        n_splits: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Walk-forward split for time series"""
        splits = []
        n = len(df)
        
        # Each fold uses expanding training window
        for i in range(n_splits):
            train_end = int(n * (0.4 + i * 0.1))
            val_end = int(n * (0.5 + i * 0.1))
            
            if val_end >= n:
                break
                
            train = df.iloc[:train_end]
            val = df.iloc[train_end:val_end]
            test = df.iloc[val_end:int(val_end + n * 0.1)]
            
            if len(test) > 0:
                splits.append((train, val, test))
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def select_features_shap(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        feature_names: List[str],
        top_n: int = 50
    ) -> List[str]:
        """Select top features using SHAP values"""
        logger.info(f"\nSelecting top {top_n} features using SHAP...")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using all features")
            return feature_names[:top_n]
        
        # Train a quick XGBoost for feature selection
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train[:1000])  # Sample for speed
        
        # Get feature importance
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        importance = np.abs(shap_values).mean(axis=0)
        
        # Select top features
        indices = np.argsort(importance)[::-1][:top_n]
        selected = [feature_names[i] for i in indices]
        
        logger.info(f"Selected {len(selected)} features")
        logger.info(f"Top 10: {selected[:10]}")
        
        return selected
    
    def optimize_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict:
        """Optimize XGBoost hyperparameters with Optuna"""
        logger.info("\nOptimizing XGBoost with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            }
            
            model = XGBoostModel(task='classification', params=params)
            model.fit(X_train, y_train, X_val, y_val, verbose=False)
            
            preds = model.predict(X_val)
            
            # Use F1 score for optimization (better for imbalanced data)
            f1 = f1_score(y_val, preds)
            
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Best F1 Score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc'] = 0.5
        
        return metrics
    
    def train_xgboost(
        self, 
        train: pd.DataFrame, 
        val: pd.DataFrame, 
        test: pd.DataFrame,
        optimize: bool = True
    ) -> Tuple[XGBoostModel, Dict]:
        """Train XGBoost model"""
        logger.info("\n" + "=" * 60)
        logger.info("  TRAINING XGBOOST MODEL")
        logger.info("=" * 60)
        
        feature_cols = self.feature_engineer.feature_names
        
        X_train = train[feature_cols].values
        y_train = train['target'].values
        X_val = val[feature_cols].values
        y_val = val['target'].values
        X_test = test[feature_cols].values
        y_test = test['target'].values
        
        # Feature selection with SHAP
        if len(feature_cols) > self.n_features:
            self.selected_features = self.select_features_shap(
                X_train, y_train, feature_cols, self.n_features
            )
            # Filter to selected features
            sel_indices = [feature_cols.index(f) for f in self.selected_features]
            X_train = X_train[:, sel_indices]
            X_val = X_val[:, sel_indices]
            X_test = X_test[:, sel_indices]
            feature_cols = self.selected_features
        else:
            self.selected_features = feature_cols
        
        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Optimize hyperparameters
        if optimize and self.use_optuna:
            best_params = self.optimize_xgboost(X_train, y_train, X_val, y_val)
            best_params['early_stopping_rounds'] = 50
            best_params['scale_pos_weight'] = scale_pos_weight
        else:
            best_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 50,
                'scale_pos_weight': scale_pos_weight
            }
        
        self.best_params['xgboost'] = best_params
        
        # Train final model
        logger.info("\nTraining final XGBoost model...")
        model = XGBoostModel(
            task='classification',
            params=best_params,
            feature_names=feature_cols
        )
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        # Evaluate on all sets
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_preds)
        val_metrics = self.calculate_metrics(y_val, val_preds)
        test_metrics = self.calculate_metrics(y_test, test_preds)
        
        logger.success(f"Train: Acc={train_metrics['accuracy']:.2%}, F1={train_metrics['f1']:.2%}")
        logger.success(f"Val:   Acc={val_metrics['accuracy']:.2%}, F1={val_metrics['f1']:.2%}")
        logger.success(f"Test:  Acc={test_metrics['accuracy']:.2%}, F1={test_metrics['f1']:.2%}")
        
        # Feature importance
        importance = model.get_feature_importance(top_n=15)
        logger.info(f"\nTop 15 Features:\n{importance}")
        
        # Save model and selected features
        model_path = f"{self.checkpoint_dir}/xgboost_best.json"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save selected features
        features_path = f"{self.checkpoint_dir}/selected_features.json"
        with open(features_path, 'w') as f:
            json.dump(self.selected_features, f, indent=2)
        logger.info(f"Selected features saved to {features_path}")
        
        return model, test_metrics
    
    def train_lstm(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[LSTMPredictor, Dict]:
        """Train LSTM model with Multi-Head Attention (v2.0)"""
        logger.info("\n" + "=" * 60)
        logger.info("  TRAINING LSTM MODEL v2.0")
        logger.info("=" * 60)
        
        # Use selected features if available
        feature_cols = self.selected_features if self.selected_features else self.feature_engineer.feature_names
        seq_len = 120  # UPGRADED: 60 → 120
        
        # Scale features
        train_features = train[feature_cols].values
        val_features = val[feature_cols].values
        test_features = test[feature_cols].values
        
        self.scaler.fit(train_features)
        train_scaled = self.scaler.transform(train_features)
        val_scaled = self.scaler.transform(val_features)
        test_scaled = self.scaler.transform(test_features)
        
        # Save scaler
        scaler_path = f"{self.checkpoint_dir}/scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Create sequences with float32 to reduce memory
        def create_sequences(features, targets, seq_length):
            n_samples = len(features) - seq_length
            n_features = features.shape[1]
            # Pre-allocate with float32 to save memory (3.24 GiB → 1.62 GiB)
            X = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
            y = np.zeros(n_samples, dtype=np.float32)
            for i in range(n_samples):
                X[i] = features[i:i + seq_length].astype(np.float32)
                y[i] = targets[i + seq_length]
            return X, y
        
        X_train, y_train = create_sequences(train_scaled, train['target'].values, seq_len)
        X_val, y_val = create_sequences(val_scaled, val['target'].values, seq_len)
        X_test, y_test = create_sequences(test_scaled, test['target'].values, seq_len)
        
        logger.info(f"LSTM shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Class weight: {class_weight:.2f}")
        
        # Train LSTM v2.0 with Multi-Head Attention
        model = LSTMPredictor(
            input_size=X_train.shape[2],
            hidden_size=256,    # UPGRADED: 128 → 256
            num_layers=3,       # UPGRADED: 2 → 3
            dropout=0.3,
            task='classification',
            learning_rate=0.0003,  # Slightly lower for larger model
            class_weight=class_weight,
            bidirectional=True,
            use_attention=True
        )
        
        batch_size = 64  # UPGRADED: fixed 64
        
        history = model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=200,         # UPGRADED: 100 → 200
            batch_size=batch_size,
            patience=30,        # UPGRADED: 20 → 30
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Evaluate
        def predict_batched(model, X, batch_size=256):
            preds = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                preds.extend(model.predict(batch))
            return np.array(preds)
        
        train_preds = predict_batched(model, X_train)
        val_preds = predict_batched(model, X_val)
        test_preds = predict_batched(model, X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_preds)
        val_metrics = self.calculate_metrics(y_val, val_preds)
        test_metrics = self.calculate_metrics(y_test, test_preds)
        
        logger.success(f"Train: Acc={train_metrics['accuracy']:.2%}, F1={train_metrics['f1']:.2%}")
        logger.success(f"Val:   Acc={val_metrics['accuracy']:.2%}, F1={val_metrics['f1']:.2%}")
        logger.success(f"Test:  Acc={test_metrics['accuracy']:.2%}, F1={test_metrics['f1']:.2%}")
        
        # Save model
        model_path = f"{self.checkpoint_dir}/lstm_best.pt"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, test_metrics
    
    def run_training(
        self, 
        train_lstm: bool = True, 
        train_xgb: bool = True, 
        optimize: bool = True
    ):
        """Run full training pipeline"""
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("  ADVANCED MODEL TRAINING PIPELINE v3.0")
        logger.info("=" * 60)
        logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Optuna optimization: {self.use_optuna and optimize}")
        logger.info(f"Feature selection: Top {self.n_features} features")
        logger.info("=" * 60)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Simple split for main training
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        
        logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        
        results = {}
        
        # Train XGBoost first (for feature selection)
        if train_xgb:
            xgb_model, xgb_metrics = self.train_xgboost(train, val, test, optimize=optimize)
            results['xgboost'] = xgb_metrics
            self.metrics['xgboost'] = xgb_metrics
        
        # Train LSTM
        if train_lstm:
            # Load selected features if not already loaded (for --lstm-only mode)
            if not self.selected_features:
                features_path = f"{self.checkpoint_dir}/selected_features.json"
                if os.path.exists(features_path):
                    import json
                    with open(features_path, 'r') as f:
                        data = json.load(f)
                    self.selected_features = data.get('selected_features', data) if isinstance(data, dict) else data
                    logger.info(f"Loaded {len(self.selected_features)} selected features from {features_path}")
                    
            lstm_model, lstm_metrics = self.train_lstm(train, val, test)
            results['lstm'] = lstm_metrics
            self.metrics['lstm'] = lstm_metrics
        
        # Save results
        results_path = f"{self.checkpoint_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'metrics': results,
                'best_params': self.best_params,
                'n_features': len(self.selected_features),
                'selected_features': self.selected_features[:20]  # Top 20
            }, f, indent=2)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("\n" + "=" * 60)
        logger.info("  TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f} minutes")
        logger.info(f"Features used: {len(self.selected_features)}")
        
        if 'xgboost' in results:
            m = results['xgboost']
            logger.info(f"XGBoost - Accuracy: {m['accuracy']:.2%}, F1: {m['f1']:.2%}")
        if 'lstm' in results:
            m = results['lstm']
            logger.info(f"LSTM    - Accuracy: {m['accuracy']:.2%}, F1: {m['f1']:.2%}")
        
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Model Training v3.0")
    parser.add_argument('--data', type=str, default='data/training/GOLD_H1.csv')
    parser.add_argument('--no-optuna', action='store_true', help='Disable Optuna')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--features', type=int, default=50, help='Number of features to select')
    parser.add_argument('--xgb-only', action='store_true', help='Train XGBoost only')
    parser.add_argument('--lstm-only', action='store_true', help='Train LSTM only')
    
    args = parser.parse_args()
    
    trainer = AdvancedModelTrainer(
        data_path=args.data,
        use_optuna=not args.no_optuna,
        n_trials=args.trials,
        n_features=args.features
    )
    
    train_lstm = not args.xgb_only
    train_xgb = not args.lstm_only
    
    trainer.run_training(
        train_lstm=train_lstm,
        train_xgb=train_xgb,
        optimize=not args.no_optuna
    )


if __name__ == "__main__":
    main()
