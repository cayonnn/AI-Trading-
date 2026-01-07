"""
Multi-Timeframe Analysis Module
================================
วิเคราะห์หลาย Timeframes พร้อมกัน

Features:
1. MTF Trend Analysis - ดู trend จากหลาย TF
2. MTF Confluence - หา confluence ระหว่าง TFs
3. Higher TF Bias - ดู big picture จาก TF ใหญ่
4. Entry TF Timing - จับจังหวะจาก TF เล็ก
5. TF Sync Detection - ตรวจจับว่า TFs sync กันหรือไม่
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from loguru import logger


class Timeframe(Enum):
    """Timeframes"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


@dataclass
class TimeframeAnalysis:
    """การวิเคราะห์แต่ละ Timeframe"""
    timeframe: str
    trend: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0-1
    momentum: float  # -1 to 1
    rsi: float
    above_ma20: bool
    above_ma50: bool


@dataclass
class MTFSignal:
    """สัญญาณจาก MTF Analysis"""
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float
    aligned_timeframes: int
    total_timeframes: int
    higher_tf_bias: str
    entry_tf_signal: str
    confluence_score: float


class MultiTimeframe:
    """
    Multi-Timeframe Analysis Engine
    
    ความสามารถ:
    1. วิเคราะห์หลาย TFs พร้อมกัน
    2. หา confluence
    3. ระบุ Higher TF bias
    4. จับจังหวะ entry จาก Lower TF
    """
    
    def __init__(
        self,
        timeframes: List[str] = None,
    ):
        if timeframes is None:
            timeframes = ["M15", "H1", "H4", "D1"]
        
        self.timeframes = timeframes
        self.tf_weights = {
            "M1": 0.05,
            "M5": 0.08,
            "M15": 0.12,
            "M30": 0.15,
            "H1": 0.20,
            "H4": 0.20,
            "D1": 0.15,
            "W1": 0.05,
        }
        
        # Store analyses
        self.tf_analyses: Dict[str, TimeframeAnalysis] = {}
        
        logger.info(f"MultiTimeframe initialized with TFs: {timeframes}")
    
    def analyze(
        self,
        data_dict: Dict[str, pd.DataFrame] = None,
        base_data: pd.DataFrame = None,
    ) -> MTFSignal:
        """
        วิเคราะห์หลาย Timeframes
        
        Args:
            data_dict: Dict ของ data แต่ละ TF
            base_data: Base data (ถ้าไม่มี data_dict จะ resample)
            
        Returns:
            MTFSignal
        """
        
        if data_dict is None and base_data is not None:
            # Resample from base data
            data_dict = self._resample_data(base_data)
        
        if not data_dict:
            return self._empty_signal()
        
        # Analyze each timeframe
        analyses = {}
        for tf, data in data_dict.items():
            if len(data) >= 50:
                analyses[tf] = self._analyze_single_tf(tf, data)
        
        self.tf_analyses = analyses
        
        # Calculate confluence
        return self._calculate_mtf_signal(analyses)
    
    def _resample_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample data to different timeframes"""
        
        result = {}
        
        # Base is assumed to be M1 or similar
        # For simplicity, just use the same data for demonstration
        # In production, use proper resampling
        
        for tf in self.timeframes:
            result[tf] = data.copy()
        
        return result
    
    def _analyze_single_tf(self, tf: str, data: pd.DataFrame) -> TimeframeAnalysis:
        """วิเคราะห์ Timeframe เดียว"""
        
        close = data['close']
        
        # Moving averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
        
        current = close.iloc[-1]
        
        # Trend
        trend_pct = (current - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
        
        if trend_pct > 0.01:
            trend = "up"
        elif trend_pct < -0.01:
            trend = "down"
        else:
            trend = "sideways"
        
        trend_strength = min(abs(trend_pct) * 10, 1.0)
        
        # Momentum
        momentum = np.clip(trend_pct * 20, -1, 1)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        return TimeframeAnalysis(
            timeframe=tf,
            trend=trend,
            trend_strength=trend_strength,
            momentum=momentum,
            rsi=rsi,
            above_ma20=current > ma20,
            above_ma50=current > ma50,
        )
    
    def _calculate_mtf_signal(
        self,
        analyses: Dict[str, TimeframeAnalysis],
    ) -> MTFSignal:
        """คำนวณสัญญาณ MTF"""
        
        if not analyses:
            return self._empty_signal()
        
        # Count aligned timeframes
        bullish_count = 0
        bearish_count = 0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for tf, analysis in analyses.items():
            weight = self.tf_weights.get(tf, 0.1)
            total_weight += weight
            
            if analysis.trend == "up":
                bullish_count += 1
                weighted_score += weight
            elif analysis.trend == "down":
                bearish_count += 1
                weighted_score -= weight
        
        total_tfs = len(analyses)
        
        # Determine direction
        if bullish_count > total_tfs * 0.6:
            direction = 1
        elif bearish_count > total_tfs * 0.6:
            direction = -1
        else:
            direction = 0
        
        # Confidence based on alignment
        aligned = max(bullish_count, bearish_count)
        confidence = aligned / total_tfs if total_tfs > 0 else 0
        
        # Higher TF bias
        higher_tfs = ["H4", "D1", "W1"]
        higher_bullish = sum(1 for tf in higher_tfs 
                           if tf in analyses and analyses[tf].trend == "up")
        higher_bearish = sum(1 for tf in higher_tfs 
                           if tf in analyses and analyses[tf].trend == "down")
        
        if higher_bullish > higher_bearish:
            higher_tf_bias = "bullish"
        elif higher_bearish > higher_bullish:
            higher_tf_bias = "bearish"
        else:
            higher_tf_bias = "neutral"
        
        # Entry TF signal (lower TFs)
        entry_tfs = ["M15", "M30", "H1"]
        entry_bullish = sum(1 for tf in entry_tfs
                          if tf in analyses and analyses[tf].trend == "up")
        entry_bearish = sum(1 for tf in entry_tfs
                          if tf in analyses and analyses[tf].trend == "down")
        
        if entry_bullish > entry_bearish:
            entry_tf_signal = "bullish"
        elif entry_bearish > entry_bullish:
            entry_tf_signal = "bearish"
        else:
            entry_tf_signal = "neutral"
        
        # Confluence score
        # Perfect confluence = same direction across all TFs
        confluence_score = aligned / total_tfs if total_tfs > 0 else 0
        
        # Boost if higher TF and entry TF agree
        if higher_tf_bias == entry_tf_signal and higher_tf_bias != "neutral":
            confluence_score = min(confluence_score * 1.2, 1.0)
        
        return MTFSignal(
            direction=direction,
            confidence=confidence,
            aligned_timeframes=aligned,
            total_timeframes=total_tfs,
            higher_tf_bias=higher_tf_bias,
            entry_tf_signal=entry_tf_signal,
            confluence_score=confluence_score,
        )
    
    def should_trade_mtf(self, signal: MTFSignal) -> Tuple[bool, str]:
        """ตัดสินใจว่าควรเทรดหรือไม่ตาม MTF"""
        
        # Need minimum alignment
        if signal.confluence_score < 0.5:
            return False, f"Low confluence ({signal.confluence_score:.1%})"
        
        # Need clear direction
        if signal.direction == 0:
            return False, "No clear direction"
        
        # Higher TF should agree
        if signal.direction > 0 and signal.higher_tf_bias == "bearish":
            return False, "Higher TF bias is bearish"
        
        if signal.direction < 0 and signal.higher_tf_bias == "bullish":
            return False, "Higher TF bias is bullish"
        
        # Entry TF should align
        if signal.direction > 0 and signal.entry_tf_signal != "bullish":
            return False, "Entry TF not bullish"
        
        if signal.direction < 0 and signal.entry_tf_signal != "bearish":
            return False, "Entry TF not bearish"
        
        return True, f"MTF aligned ({signal.aligned_timeframes}/{signal.total_timeframes})"
    
    def get_tf_summary(self) -> Dict[str, str]:
        """ดึงสรุปแต่ละ TF"""
        
        return {
            tf: f"{a.trend} ({a.trend_strength:.0%})"
            for tf, a in self.tf_analyses.items()
        }
    
    def _empty_signal(self) -> MTFSignal:
        return MTFSignal(
            direction=0,
            confidence=0,
            aligned_timeframes=0,
            total_timeframes=0,
            higher_tf_bias="neutral",
            entry_tf_signal="neutral",
            confluence_score=0,
        )


def create_multi_timeframe() -> MultiTimeframe:
    """สร้าง MultiTimeframe"""
    return MultiTimeframe()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   MULTI-TIMEFRAME ANALYSIS TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data for different TFs
    def create_tf_data(trend: float, n: int = 100):
        prices = 2000 + np.cumsum(np.random.randn(n) * 3 + trend)
        return pd.DataFrame({
            "open": prices - np.random.rand(n) * 2,
            "high": prices + np.random.rand(n) * 4,
            "low": prices - np.random.rand(n) * 4,
            "close": prices,
        })
    
    data_dict = {
        "M15": create_tf_data(0.1),
        "H1": create_tf_data(0.15),
        "H4": create_tf_data(0.2),
        "D1": create_tf_data(0.25),
    }
    
    # Analyze
    mtf = create_multi_timeframe()
    signal = mtf.analyze(data_dict=data_dict)
    
    print(f"\nDirection: {signal.direction}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Confluence: {signal.confluence_score:.1%}")
    print(f"Higher TF Bias: {signal.higher_tf_bias}")
    print(f"Entry TF Signal: {signal.entry_tf_signal}")
    print(f"Aligned: {signal.aligned_timeframes}/{signal.total_timeframes}")
    
    should, reason = mtf.should_trade_mtf(signal)
    print(f"\nShould Trade: {should} ({reason})")
    
    print(f"\nTF Summary:")
    for tf, summary in mtf.get_tf_summary().items():
        print(f"  {tf}: {summary}")
