"""
Walk-Forward Validation
========================
ทดสอบ AI แบบ out-of-sample ด้วย walk-forward

Features:
1. Rolling Window Training
2. Out-of-Sample Testing
3. Performance Metrics per Window
4. Overall Statistics
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.indicators import calculate_indicators
from models.lstm_model import LSTMPredictor
from models.xgboost_model import XGBoostModel
from data.preprocessor import DataPreprocessor


@dataclass
class WalkForwardResult:
    """ผลลัพธ์ของแต่ละ window"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: List[int] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)


@dataclass
class WalkForwardSummary:
    """สรุปผลทั้งหมด"""
    total_windows: int
    avg_accuracy: float
    std_accuracy: float
    avg_f1: float
    best_window: int
    worst_window: int
    total_predictions: int
    overall_accuracy: float


class WalkForwardValidator:
    """
    Walk-Forward Validation System
    
    ใช้สำหรับทดสอบ model แบบ realistic โดย:
    1. Train บน window เก่า
    2. Test บน window ใหม่ (ที่ model ไม่เคยเห็น)
    3. เลื่อน window ไปเรื่อยๆ
    """
    
    def __init__(
        self,
        train_days: int = 180,   # 6 months training
        test_days: int = 30,     # 1 month testing
        step_days: int = 30,     # Step forward 1 month
        model_type: str = 'xgboost',  # 'lstm' or 'xgboost'
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.model_type = model_type
        
        self.results: List[WalkForwardResult] = []
        
        logger.info(f"WalkForwardValidator: train={train_days}d, test={test_days}d, step={step_days}d")
    
    def validate(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
    ) -> WalkForwardSummary:
        """
        รัน Walk-Forward Validation
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            WalkForwardSummary
        """
        
        logger.info(f"Starting Walk-Forward Validation on {len(data)} samples")
        
        # Convert to hourly samples (assuming H1 data)
        train_samples = self.train_days * 24
        test_samples = self.test_days * 24
        step_samples = self.step_days * 24
        
        n_samples = len(data)
        window_id = 0
        start_idx = 0
        
        all_predictions = []
        all_actuals = []
        
        while start_idx + train_samples + test_samples <= n_samples:
            train_end = start_idx + train_samples
            test_end = train_end + test_samples
            
            # Split data
            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()
            
            # Train and evaluate
            result = self._train_and_evaluate(
                window_id, train_data, test_data, target_col
            )
            
            self.results.append(result)
            all_predictions.extend(result.predictions)
            all_actuals.extend(result.actuals)
            
            logger.info(
                f"Window {window_id}: Acc={result.accuracy:.1%}, "
                f"F1={result.f1:.1%}, Samples={result.test_samples}"
            )
            
            # Move window forward
            start_idx += step_samples
            window_id += 1
        
        # Calculate summary
        accuracies = [r.accuracy for r in self.results]
        f1s = [r.f1 for r in self.results]
        
        # Overall accuracy
        if all_actuals:
            overall_acc = sum(
                1 for p, a in zip(all_predictions, all_actuals) if p == a
            ) / len(all_actuals)
        else:
            overall_acc = 0
        
        summary = WalkForwardSummary(
            total_windows=len(self.results),
            avg_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            avg_f1=np.mean(f1s),
            best_window=int(np.argmax(accuracies)),
            worst_window=int(np.argmin(accuracies)),
            total_predictions=len(all_predictions),
            overall_accuracy=overall_acc,
        )
        
        return summary
    
    def _train_and_evaluate(
        self,
        window_id: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
    ) -> WalkForwardResult:
        """Train and evaluate on one window"""
        
        # Get feature columns
        exclude_cols = ['timestamp', 'datetime', 'time', 'date', target_col]
        feature_cols = [c for c in train_data.columns if c not in exclude_cols]
        
        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data[target_col].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # Train model
        if self.model_type == 'xgboost':
            model = XGBoostModel(
                task='classification',
                params={
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                }
            )
            model.fit(X_train, y_train, verbose=False)
            predictions = model.predict(X_test)
            
        else:  # LSTM
            # For LSTM, need sequences
            seq_len = 60
            if len(X_train) > seq_len:
                X_train_seq = self._create_sequences(X_train, seq_len)
                y_train_seq = y_train[seq_len:]
                X_test_seq = self._create_sequences(X_test, seq_len)
                y_test = y_test[seq_len:]
                
                model = LSTMPredictor(
                    input_size=X_train.shape[1],
                    hidden_size=64,
                    num_layers=1,
                    task='classification',
                )
                model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, patience=3)
                predictions = model.predict(X_test_seq)
            else:
                predictions = np.zeros(len(y_test))
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        return WalkForwardResult(
            window_id=window_id,
            train_start=str(train_data.index[0]) if hasattr(train_data.index[0], '__str__') else "0",
            train_end=str(train_data.index[-1]) if hasattr(train_data.index[-1], '__str__') else str(len(train_data)),
            test_start=str(test_data.index[0]) if hasattr(test_data.index[0], '__str__') else "0",
            test_end=str(test_data.index[-1]) if hasattr(test_data.index[-1], '__str__') else str(len(test_data)),
            train_samples=len(train_data),
            test_samples=len(test_data),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            predictions=predictions.tolist(),
            actuals=y_test.tolist(),
        )
    
    def _create_sequences(self, data: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sequences for LSTM"""
        sequences = []
        for i in range(seq_len, len(data)):
            sequences.append(data[i-seq_len:i])
        return np.array(sequences)
    
    def print_report(self, summary: WalkForwardSummary):
        """พิมพ์รายงาน"""
        
        print("\n" + "="*60)
        print("   WALK-FORWARD VALIDATION REPORT")
        print("="*60)
        print(f"\nWindows: {summary.total_windows}")
        print(f"Total Predictions: {summary.total_predictions}")
        print(f"\n--- Performance ---")
        print(f"Average Accuracy: {summary.avg_accuracy:.2%} (±{summary.std_accuracy:.2%})")
        print(f"Average F1 Score: {summary.avg_f1:.2%}")
        print(f"Overall Accuracy: {summary.overall_accuracy:.2%}")
        print(f"\nBest Window: #{summary.best_window} ({self.results[summary.best_window].accuracy:.2%})")
        print(f"Worst Window: #{summary.worst_window} ({self.results[summary.worst_window].accuracy:.2%})")
        
        print("\n--- Per-Window Results ---")
        for r in self.results:
            print(f"  Window {r.window_id}: Acc={r.accuracy:.1%}, F1={r.f1:.1%}, N={r.test_samples}")


def run_walk_forward_validation():
    """รัน Walk-Forward Validation"""
    
    # Configure logger
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   WALK-FORWARD VALIDATION")
    print("="*60)
    
    # Load data
    data_path = "data/training/GOLD_H1.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Handle datetime
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")
    
    # Calculate indicators
    logger.info("Calculating indicators...")
    df = calculate_indicators(df, 'all')
    df = df.dropna()
    logger.info(f"Data shape after indicators: {df.shape}")
    
    # Create target (1 = price up, 0 = price down/flat)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    # Run Walk-Forward
    validator = WalkForwardValidator(
        train_days=180,  # 6 months
        test_days=30,    # 1 month
        step_days=30,    # Move 1 month forward
        model_type='xgboost',
    )
    
    summary = validator.validate(df, target_col='target')
    
    # Print report
    validator.print_report(summary)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'window': r.window_id,
            'accuracy': r.accuracy,
            'precision': r.precision,
            'recall': r.recall,
            'f1': r.f1,
            'samples': r.test_samples,
        }
        for r in validator.results
    ])
    
    results_df.to_csv('models/checkpoints/walk_forward_results.csv', index=False)
    logger.info("Results saved to models/checkpoints/walk_forward_results.csv")
    
    return summary


if __name__ == "__main__":
    run_walk_forward_validation()
