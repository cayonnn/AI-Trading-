"""
Train Models with Advanced Features
====================================
Train LSTM and XGBoost with enhanced features:
1. Cross-Asset Features (DXY/VIX proxies)
2. Session Features
3. SHAP Feature Selection
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.indicators import calculate_indicators
from data.cross_asset_features import add_cross_asset_features
from data.session_features import create_all_session_features
from data.shap_feature_selection import SHAPFeatureSelector
from models.lstm_model import LSTMPredictor
from models.xgboost_model import XGBoostModel


def train_with_advanced_features():
    """Train models with advanced features"""
    
    # Configure logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   ADVANCED MODEL TRAINING")
    print("="*60)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    logger.info("Loading training data...")
    
    data_path = "data/training/GOLD_H1.csv"
    df = pd.read_csv(data_path)
    
    # Handle datetime
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows")
    
    # ==========================================
    # 2. Add All Features
    # ==========================================
    logger.info("Adding features...")
    
    # Technical indicators
    logger.info("  Adding technical indicators...")
    df = calculate_indicators(df, 'all')
    
    # Cross-asset features (DXY/VIX proxies)
    logger.info("  Adding cross-asset features...")
    df = add_cross_asset_features(df)
    
    # Session features
    logger.info("  Adding session features...")
    df = create_all_session_features(df)
    
    # Handle NaN - fill instead of drop to preserve data
    initial_len = len(df)
    nan_cols = df.columns[df.isna().any()].tolist()
    logger.info(f"  Columns with NaN: {len(nan_cols)}")
    
    # Fill NaN with 0 or forward fill for price data
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].ffill()
        else:
            df[col] = df[col].fillna(0)
    
    # Drop only rows with critical NaN (OHLC)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    logger.info(f"  Rows after cleanup: {len(df):,} (dropped {initial_len - len(df):,})")
    
    # ==========================================
    # 3. Create Target
    # ==========================================
    logger.info("Creating target variable...")
    
    # Binary classification: 1 = price up, 0 = price down
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # ==========================================
    # 4. Prepare Features
    # ==========================================
    logger.info("Preparing features...")
    
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'datetime', 'time', 'date', 'target', 'tick_volume', 'spread']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Total features before selection: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ==========================================
    # 5. SHAP Feature Selection
    # ==========================================
    logger.info("Running SHAP feature selection...")
    
    selector = SHAPFeatureSelector(n_top_features=50)
    X_df = pd.DataFrame(X, columns=feature_cols)
    selected_features = selector.fit(X_df, y)
    
    # Save feature selection
    selector.save("models/checkpoints/selected_features.json")
    
    # Use selected features
    X_selected = X_df[selected_features].values.astype(np.float32)
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # ==========================================
    # 6. Train/Test Split
    # ==========================================
    logger.info("Splitting data...")
    
    n_samples = len(X_selected)
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    
    X_train = X_selected[:train_size]
    y_train = y[:train_size]
    X_val = X_selected[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X_selected[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # ==========================================
    # 7. Train LSTM
    # ==========================================
    logger.info("")
    logger.info("="*50)
    logger.info("TRAINING LSTM MODEL")
    logger.info("="*50)
    
    # Create sequences for LSTM
    seq_len = 60
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    
    logger.info(f"LSTM training shape: {X_train_seq.shape}")
    
    # Create and train LSTM
    lstm = LSTMPredictor(
        input_size=len(selected_features),
        hidden_size=128,
        num_layers=2,
        task='classification',
        dropout=0.3,
        bidirectional=True,
        use_attention=True,
    )
    
    lstm.fit(
        X_train_seq, y_train_seq,
        X_val=X_val_seq, y_val=y_val_seq,
        epochs=50,
        batch_size=64,
        patience=20,  # ให้ train นานขึ้น
    )
    
    # Evaluate (use batched prediction to avoid OOM)
    logger.info("Evaluating LSTM...")
    all_preds = []
    batch_size = 1000
    for i in range(0, len(X_test_seq), batch_size):
        batch = X_test_seq[i:i+batch_size]
        preds = lstm.predict(batch)
        all_preds.extend(preds)
    lstm_preds = np.array(all_preds)
    lstm_acc = (lstm_preds == y_test_seq).mean()
    logger.success(f"LSTM Test Accuracy: {lstm_acc:.2%}")
    
    # Save
    lstm.save("models/checkpoints/lstm_advanced.pt")
    
    # ==========================================
    # 8. Train XGBoost
    # ==========================================
    logger.info("")
    logger.info("="*50)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("="*50)
    
    xgb = XGBoostModel(
        task='classification',
        params={
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    )
    
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # Evaluate
    xgb_preds = xgb.predict(X_test)
    xgb_acc = (xgb_preds == y_test).mean()
    logger.success(f"XGBoost Test Accuracy: {xgb_acc:.2%}")
    
    # Feature importance
    importance = xgb.get_feature_importance()
    if importance is not None:
        logger.info("Top 10 Features:")
        # Handle DataFrame or array
        if isinstance(importance, pd.DataFrame):
            print(importance.head(10).to_string(index=False))
        else:
            if hasattr(importance, 'ndim') and importance.ndim > 1:
                importance = importance.flatten()
            imp_df = pd.DataFrame({
                'feature': selected_features[:len(importance)],
                'importance': list(importance)
            }).sort_values('importance', ascending=False)
            print(imp_df.head(10).to_string(index=False))
    
    # Save
    xgb.save("models/checkpoints/xgboost_advanced.json")
    
    # ==========================================
    # 9. Summary
    # ==========================================
    print("\n" + "="*60)
    print("   TRAINING COMPLETE")
    print("="*60)
    print(f"   LSTM Accuracy:    {lstm_acc:.2%}")
    print(f"   XGBoost Accuracy: {xgb_acc:.2%}")
    print(f"   Features Used:    {len(selected_features)}")
    print(f"   Models saved to:  models/checkpoints/")
    print("="*60)
    
    # Save results
    import json
    results = {
        'timestamp': datetime.now().isoformat(),
        'lstm_accuracy': float(lstm_acc),
        'xgboost_accuracy': float(xgb_acc),
        'features_count': len(selected_features),
        'selected_features': selected_features,
        'training_samples': len(X_train),
    }
    
    with open("models/checkpoints/advanced_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    train_with_advanced_features()
