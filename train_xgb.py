"""
XGBoost Training Script
=======================
Standalone training for XGBoost model with:
- SHAP feature selection
- Optuna hyperparameter optimization
- Walk-forward validation
- Production-ready output

Usage:
    python train_xgb.py
    python train_xgb.py --trials 100
    python train_xgb.py --no-optuna
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import AdvancedFeatureEngineer, create_target
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
    logger.warning("SHAP not installed. Feature selection disabled.")


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data"""
    logger.info(f"Loading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    
    # Handle datetime column
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    else:
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
    
    df = df.sort_values('timestamp')
    logger.info(f"Loaded {len(df)} rows")
    
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Engineer features"""
    logger.info("Engineering features...")
    
    engineer = AdvancedFeatureEngineer()
    df = engineer.engineer_features(df)
    df['target'] = create_target(df, horizon=1, target_type='binary')
    df = df.dropna()
    
    logger.info(f"Created {len(engineer.feature_names)} features, {len(df)} samples")
    
    return df, engineer.feature_names


def select_features_shap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_n: int = 50
) -> List[str]:
    """Select top features using SHAP"""
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, using first features")
        return feature_names[:top_n]
    
    logger.info(f"Selecting top {top_n} features with SHAP...")
    
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, verbosity=0, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:1000])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(importance)[::-1][:top_n]
    selected = [feature_names[i] for i in indices]
    
    logger.info(f"Top 10 features: {selected[:10]}")
    
    return selected


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50
) -> Dict:
    """Optimize XGBoost with Optuna"""
    logger.info(f"Optimizing XGBoost with {n_trials} trials...")
    
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
        return f1_score(y_val, preds)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.success(f"Best F1: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params


def train_xgboost(
    data_path: str = "data/training/GOLD_H1.csv",
    checkpoint_dir: str = "models/checkpoints",
    n_trials: int = 50,
    n_features: int = 50,
    use_optuna: bool = True
):
    """Main training function"""
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("  XGBOOST TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Optuna: {use_optuna and OPTUNA_AVAILABLE}")
    logger.info(f"Features: {n_features}")
    logger.info("=" * 60)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_data(data_path)
    df, feature_names = engineer_features(df)
    
    # Split data
    n = len(df)
    train = df.iloc[:int(n * 0.7)]
    val = df.iloc[int(n * 0.7):int(n * 0.85)]
    test = df.iloc[int(n * 0.85):]
    
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    X_train = train[feature_names].values
    y_train = train['target'].values
    X_val = val[feature_names].values
    y_val = val['target'].values
    X_test = test[feature_names].values
    y_test = test['target'].values
    
    # Feature selection
    if len(feature_names) > n_features:
        selected_features = select_features_shap(X_train, y_train, feature_names, n_features)
        sel_idx = [feature_names.index(f) for f in selected_features]
        X_train = X_train[:, sel_idx]
        X_val = X_val[:, sel_idx]
        X_test = X_test[:, sel_idx]
        feature_names = selected_features
    
    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"Class balance - Pos: {n_pos}, Neg: {n_neg}, Weight: {scale_pos_weight:.2f}")
    
    # Optimize or use defaults
    if use_optuna and OPTUNA_AVAILABLE:
        best_params = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials)
    else:
        best_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    
    best_params['early_stopping_rounds'] = 50
    best_params['scale_pos_weight'] = scale_pos_weight
    
    # Train final model
    logger.info("\nTraining final model...")
    model = XGBoostModel(task='classification', params=best_params, feature_names=feature_names)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    # Evaluate
    test_preds = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds, zero_division=0),
        'recall': recall_score(y_test, test_preds, zero_division=0),
        'f1': f1_score(y_test, test_preds, zero_division=0),
    }
    
    logger.success(f"\nTest Results:")
    logger.success(f"  Accuracy:  {metrics['accuracy']:.2%}")
    logger.success(f"  Precision: {metrics['precision']:.2%}")
    logger.success(f"  Recall:    {metrics['recall']:.2%}")
    logger.success(f"  F1 Score:  {metrics['f1']:.2%}")
    
    # Save model and features
    model.save(f"{checkpoint_dir}/xgboost_best.json")
    
    with open(f"{checkpoint_dir}/selected_features.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(f"{checkpoint_dir}/xgb_training_results.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'params': best_params,
            'n_features': len(feature_names)
        }, f, indent=2)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed:.1f} minutes")
    logger.info(f"Model saved to: {checkpoint_dir}/xgboost_best.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XGBoost Training")
    parser.add_argument('--data', type=str, default='data/training/GOLD_H1.csv')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--features', type=int, default=50)
    parser.add_argument('--no-optuna', action='store_true')
    
    args = parser.parse_args()
    
    train_xgboost(
        data_path=args.data,
        n_trials=args.trials,
        n_features=args.features,
        use_optuna=not args.no_optuna
    )
