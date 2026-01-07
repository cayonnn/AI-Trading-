"""
Cross-Asset Correlation Features
=================================
เพิ่ม features จาก DXY (Dollar Index) และ VIX ที่มีผลต่อ Gold

Gold correlations:
- DXY: Strong negative correlation (-0.7 to -0.9)
- VIX: Positive correlation during crisis (0.3 to 0.6)
- Real Yields: Negative correlation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger


class CrossAssetFeatures:
    """
    Generate features from cross-asset correlations
    
    เนื่องจากไม่มี real-time DXY/VIX data
    จะใช้ proxy features จาก Gold price behavior
    """
    
    def __init__(self):
        # Correlation windows
        self.windows = [5, 10, 20, 50]
        
        logger.info("CrossAssetFeatures initialized")
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """เพิ่ม cross-asset features"""
        
        df = df.copy()
        
        # 1. DXY Proxy Features
        df = self._add_dxy_proxy(df)
        
        # 2. VIX Proxy Features  
        df = self._add_vix_proxy(df)
        
        # 3. Risk-On/Risk-Off Features
        df = self._add_risk_sentiment(df)
        
        # 4. Correlation Regime Features
        df = self._add_correlation_regime(df)
        
        logger.info(f"Added cross-asset features")
        
        return df
    
    def _add_dxy_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DXY Proxy - Dollar strength from Gold inverse
        
        เมื่อ Gold ลง = Dollar strength
        เมื่อ Gold ขึ้น = Dollar weakness
        """
        
        returns = df['close'].pct_change()
        
        # DXY proxy = inverted gold returns (smoothed)
        df['dxy_proxy_5'] = -returns.rolling(5).mean() * 100
        df['dxy_proxy_20'] = -returns.rolling(20).mean() * 100
        
        # DXY trend strength
        df['dxy_trend'] = df['dxy_proxy_20'].rolling(10).apply(
            lambda x: 1 if x.mean() > 0 else (-1 if x.mean() < 0 else 0)
        )
        
        # DXY momentum
        df['dxy_momentum'] = df['dxy_proxy_5'] - df['dxy_proxy_20']
        
        # Dollar regime (strong/weak)
        ma50 = df['close'].rolling(50).mean()
        df['dollar_regime'] = np.where(df['close'] < ma50, 1, 0)  # Strong dollar = Gold below MA
        
        return df
    
    def _add_vix_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VIX Proxy - Volatility from Gold price action
        
        High volatility = High VIX (fear)
        Low volatility = Low VIX (complacency)
        """
        
        # True Range for volatility
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        # Normalized ATR as VIX proxy (scaled to typical VIX range)
        atr_14 = tr.rolling(14).mean()
        df['vix_proxy'] = (atr_14 / close * 100) * 10  # Scale to ~15-30 range
        
        # VIX percentile (relative fear level)
        df['vix_percentile'] = df['vix_proxy'].rolling(100).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.001)
        )
        
        # VIX spike detection
        vix_ma = df['vix_proxy'].rolling(20).mean()
        df['vix_spike'] = (df['vix_proxy'] > vix_ma * 1.5).astype(int)
        
        # VIX trend (increasing fear or decreasing)
        df['vix_trend'] = df['vix_proxy'].diff(5).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        return df
    
    def _add_risk_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Risk-On / Risk-Off sentiment
        
        Risk-Off = Gold ขึ้น + Volatility สูง
        Risk-On = Gold ลง + Volatility ต่ำ
        """
        
        returns = df['close'].pct_change()
        
        # Rolling metrics
        ret_5 = returns.rolling(5).sum()
        vol_5 = returns.rolling(5).std()
        
        # Risk sentiment score (-1 to 1)
        # Risk-Off: Gold up + high vol
        # Risk-On: Gold down + low vol
        
        ret_norm = ret_5 / (ret_5.rolling(50).std() + 0.0001)
        vol_norm = vol_5 / (vol_5.rolling(50).mean() + 0.0001)
        
        df['risk_sentiment'] = (ret_norm + vol_norm) / 2
        
        # Binary risk flag
        df['risk_off'] = (df['risk_sentiment'] > 0.5).astype(int)
        df['risk_on'] = (df['risk_sentiment'] < -0.5).astype(int)
        
        # Safe haven demand (Gold as safe haven)
        df['safe_haven_demand'] = np.where(
            (df['vix_proxy'] > df['vix_proxy'].rolling(20).mean()) & 
            (returns.rolling(5).sum() > 0),
            1, 0
        )
        
        return df
    
    def _add_correlation_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlation regime detection
        
        ตรวจจับว่า Gold อยู่ใน regime ไหน:
        1. Dollar-driven
        2. Risk-driven
        3. Inflation-driven
        """
        
        returns = df['close'].pct_change()
        
        # Calculate rolling characteristics
        vol = returns.rolling(20).std()
        trend = df['close'].rolling(20).mean() - df['close'].rolling(50).mean()
        momentum = returns.rolling(10).sum()
        
        # Regime scoring
        # Dollar regime: Low vol, clear trend
        dollar_score = (1 / (vol + 0.001)) * np.abs(trend) / (df['close'] + 0.001)
        
        # Risk regime: High vol, momentum-driven
        risk_score = vol * np.abs(momentum)
        
        # Normalize scores
        df['dollar_regime_score'] = dollar_score / (dollar_score.rolling(50).max() + 0.001)
        df['risk_regime_score'] = risk_score / (risk_score.rolling(50).max() + 0.001)
        
        # Dominant regime
        df['regime_type'] = np.where(
            df['dollar_regime_score'] > df['risk_regime_score'],
            1,  # Dollar-driven
            0   # Risk-driven
        )
        
        return df
    
    def get_feature_names(self) -> list:
        """ดึงรายชื่อ features ที่เพิ่ม"""
        
        return [
            # DXY Proxy
            'dxy_proxy_5', 'dxy_proxy_20', 'dxy_trend', 'dxy_momentum', 'dollar_regime',
            # VIX Proxy
            'vix_proxy', 'vix_percentile', 'vix_spike', 'vix_trend',
            # Risk Sentiment
            'risk_sentiment', 'risk_off', 'risk_on', 'safe_haven_demand',
            # Correlation Regime
            'dollar_regime_score', 'risk_regime_score', 'regime_type',
        ]


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """สร้าง cross-asset features"""
    caf = CrossAssetFeatures()
    return caf.add_features(df)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   CROSS-ASSET FEATURES TEST")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    prices = 2000 + np.cumsum(np.random.randn(n) * 3)
    
    df = pd.DataFrame({
        "datetime": pd.date_range('2024-01-01', periods=n, freq='H'),
        "open": prices - np.random.rand(n) * 2,
        "high": prices + np.random.rand(n) * 5,
        "low": prices - np.random.rand(n) * 5,
        "close": prices,
        "volume": np.random.randint(1000, 5000, n),
    })
    
    # Add features
    df = add_cross_asset_features(df)
    
    # Show new columns
    caf = CrossAssetFeatures()
    new_cols = caf.get_feature_names()
    
    print(f"\nAdded {len(new_cols)} cross-asset features:")
    for col in new_cols:
        if col in df.columns:
            print(f"  ✅ {col}")
    
    print(f"\nSample values:")
    print(df[['close', 'dxy_proxy_20', 'vix_proxy', 'risk_sentiment']].tail(5))
