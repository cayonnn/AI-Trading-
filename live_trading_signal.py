"""
Live Trading Signal Generator
==============================
Generate real trading signals using the trained Hedge Fund XGBoost model.

Usage:
    python live_trading_signal.py --data data/training/GOLD_H1.csv
    python live_trading_signal.py --latest  # Use latest data point

Author: AI Trading System
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_engineering import AdvancedFeatureEngineer
from data.regime_detector import MarketRegimeDetector
from models.xgboost_model import XGBoostModel


class LiveTradingSignal:
    """Generate live trading signals from trained model."""
    
    def __init__(
        self,
        model_path: str = "models/checkpoints/xgboost_hedge_fund.json",
        results_path: str = "models/checkpoints/hedge_fund_results.json"
    ):
        self.model_path = model_path
        self.results_path = results_path
        
        self.model = None
        self.feature_names = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector(n_regimes=3)
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and feature names."""
        logger.info("Loading trained model...")
        
        # Load feature names from results first
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                results = json.load(f)
                self.feature_names = results.get('selected_features', [])
        
        if not self.feature_names:
            logger.warning("No feature names found, will use all available features")
        else:
            logger.info(f"Loaded {len(self.feature_names)} features")
        
        # Create model instance and load weights
        self.model = XGBoostModel(
            task='classification',
            feature_names=self.feature_names
        )
        self.model.load(self.model_path)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with features."""
        # Set index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
        
        # Engineer features
        df = df.reset_index()
        df = self.feature_engineer.engineer_features(df)
        df = df.set_index('timestamp')
        
        # Add regime features
        try:
            self.regime_detector.fit(df)
            df = self.regime_detector.add_regime_features(df)
        except Exception as e:
            logger.warning(f"Could not add regime features: {e}")
        
        return df
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        n_latest: int = 1
    ) -> Dict:
        """
        Generate trading signal for the latest data points.
        
        Args:
            df: DataFrame with OHLCV data
            n_latest: Number of latest signals to generate
            
        Returns:
            Dict with signal information
        """
        logger.info("="*60)
        logger.info("  GENERATING TRADING SIGNAL")
        logger.info("="*60)
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Get available features
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in df.columns]
        else:
            # Use all numeric features except target columns
            exclude = ['target', 'triple_barrier_label', 'expected_return', 'barrier_touch_time']
            available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col not in exclude]
        
        logger.info(f"Using {len(available_features)} features")
        
        # Drop NaN rows
        df = df.dropna(subset=available_features)
        
        if len(df) == 0:
            logger.error("No valid data after cleaning")
            return {'error': 'No valid data'}
        
        # Get latest rows
        latest_df = df.tail(n_latest)
        X = latest_df[available_features].values
        
        # Predict probabilities
        proba = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        signals = []
        
        for i, (idx, row) in enumerate(latest_df.iterrows()):
            # Handle different proba shapes
            if proba.ndim == 1:
                prob_up = float(proba[i])
            elif proba.shape[1] >= 2:
                prob_up = float(proba[i, 1])
            else:
                prob_up = float(proba[i, 0])
            
            prob_down = 1 - prob_up
            pred = predictions[i]
            
            # Determine signal strength
            confidence = max(prob_up, prob_down)
            
            if pred == 1:
                signal = "BUY" if confidence > 0.55 else "WEAK_BUY"
                direction = "📈 LONG"
            else:
                signal = "SELL" if confidence > 0.55 else "WEAK_SELL"
                direction = "📉 SHORT"
            
            signal_info = {
                'timestamp': str(idx),
                'close': float(row['close']),
                'signal': signal,
                'direction': direction,
                'probability_up': float(prob_up),
                'probability_down': float(prob_down),
                'confidence': float(confidence),
                'prediction': int(pred),
                'regime': int(row.get('market_regime', 0))
            }
            signals.append(signal_info)
        
        return signals
    
    def print_signal(self, signals: list):
        """Print signals in a formatted way."""
        print("\n" + "="*60)
        print("  [AI] TRADING SIGNAL - GOLD H1")
        print("="*60)
        
        for sig in signals:
            print(f"\n[TIME] {sig['timestamp']}")
            print(f"[PRICE] ${sig['close']:.2f}")
            
            direction = "[LONG >>>]" if sig['prediction'] == 1 else "[SHORT <<<]"
            print(f"\n{direction}")
            print(f"[SIGNAL] {sig['signal']}")
            print(f"[CONFIDENCE] {sig['confidence']:.1%}")
            print(f"[P(Up)] {sig['probability_up']:.1%}")
            print(f"[P(Down)] {sig['probability_down']:.1%}")
            print(f"[REGIME] {sig['regime']}")
        
        print("\n" + "="*60)
        print("[WARNING] RISK NOTICE:")
        print("   - Model accuracy: ~51% (Walk-Forward)")
        print("   - Use proper risk management")
        print("   - Never risk more than 1-2% per trade")
        print("   - This is NOT financial advice")
        print("="*60)
    
    def get_trade_recommendation(self, signals: list) -> Dict:
        """
        Get specific trade recommendation.
        
        Returns:
            Dict with entry, stop-loss, and take-profit levels
        """
        if not signals:
            return {}
        
        latest = signals[-1]
        close = latest['close']
        confidence = latest['confidence']
        
        # Calculate levels based on ATR (assume ~$10-15 for GOLD H1)
        atr = close * 0.005  # Approximate 0.5% ATR
        
        if latest['prediction'] == 1:  # BUY
            entry = close
            stop_loss = close - (atr * 1.5)  # 1.5 ATR stop
            take_profit = close + (atr * 2.0)  # 2.0 ATR target (1:1.33 R:R)
            
            if confidence > 0.55:
                take_profit = close + (atr * 2.5)  # Higher target for high confidence
            
            trade = {
                'action': 'BUY',
                'entry': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_pips': round(entry - stop_loss, 2),
                'reward_pips': round(take_profit - entry, 2),
                'risk_reward_ratio': round((take_profit - entry) / (entry - stop_loss), 2)
            }
        else:  # SELL
            entry = close
            stop_loss = close + (atr * 1.5)
            take_profit = close - (atr * 2.0)
            
            if confidence > 0.55:
                take_profit = close - (atr * 2.5)
            
            trade = {
                'action': 'SELL',
                'entry': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_pips': round(stop_loss - entry, 2),
                'reward_pips': round(entry - take_profit, 2),
                'risk_reward_ratio': round((entry - take_profit) / (stop_loss - entry), 2)
            }
        
        return trade
    
    def print_trade_recommendation(self, trade: Dict):
        """Print trade recommendation."""
        if not trade:
            return
        
        action = trade['action']
        marker = "[+]" if action == "BUY" else "[-]"
        
        print(f"\n{marker} TRADE RECOMMENDATION - {action}")
        print("-" * 40)
        print(f"   Entry:       ${trade['entry']:.2f}")
        print(f"   Stop Loss:   ${trade['stop_loss']:.2f} (Risk: ${trade['risk_pips']:.2f})")
        print(f"   Take Profit: ${trade['take_profit']:.2f} (Reward: ${trade['reward_pips']:.2f})")
        print(f"   R:R Ratio:   1:{trade['risk_reward_ratio']:.2f}")
        print("-" * 40)
        
        # Position sizing example
        account_balance = 10000  # Example
        risk_percent = 0.01  # 1% risk
        risk_amount = account_balance * risk_percent
        position_size = risk_amount / trade['risk_pips'] if trade['risk_pips'] > 0 else 0
        
        print(f"\n[POSITION] Sizing (Example: ${account_balance:,} account, {risk_percent:.0%} risk)")
        print(f"   Risk Amount: ${risk_amount:.2f}")
        print(f"   Position Size: {position_size:.4f} lots")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Signal Generator")
    parser.add_argument('--data', type=str, default='data/training/GOLD_H1.csv',
                       help='Path to price data')
    parser.add_argument('--n', type=int, default=1,
                       help='Number of latest signals to show')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Generate signal
    generator = LiveTradingSignal()
    signals = generator.generate_signal(df, n_latest=args.n)
    
    if args.json:
        print(json.dumps(signals, indent=2))
    else:
        generator.print_signal(signals)
        
        # Get trade recommendation
        trade = generator.get_trade_recommendation(signals)
        generator.print_trade_recommendation(trade)


if __name__ == "__main__":
    main()
