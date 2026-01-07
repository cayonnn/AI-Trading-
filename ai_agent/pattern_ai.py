"""
Pattern AI Module
==================
Deep Learning Pattern Recognition สำหรับ Trading

Features:
1. CNN-based Pattern Recognition - จดจำรูปแบบราคา
2. Candlestick Pattern Detection - ตรวจจับ candle patterns
3. Chart Pattern Recognition - หา Head & Shoulders, Triangles, etc.
4. Price Action Signals - วิเคราะห์ price action
5. Predictive Patterns - ทำนายจาก patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from loguru import logger


class PatternType(Enum):
    """ประเภท Pattern"""
    # Candlestick
    DOJI = "doji"
    HAMMER = "hammer"
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    PINBAR_BULL = "pinbar_bull"
    PINBAR_BEAR = "pinbar_bear"
    
    # Chart Patterns
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inv_head_shoulders"
    TRIANGLE_UP = "triangle_up"
    TRIANGLE_DOWN = "triangle_down"
    FLAG_BULL = "flag_bull"
    FLAG_BEAR = "flag_bear"
    
    # Price Action
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    PULLBACK = "pullback"
    REVERSAL = "reversal"


@dataclass
class DetectedPattern:
    """Pattern ที่ตรวจพบ"""
    pattern_type: PatternType
    confidence: float
    direction: int  # 1 = bullish, -1 = bearish
    price_level: float
    target_price: float = 0.0
    stop_price: float = 0.0
    detected_at: int = 0  # bar index


class PatternCNN(nn.Module):
    """CNN สำหรับ Pattern Recognition"""
    
    def __init__(self, input_channels: int = 5, num_patterns: int = 20):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(10)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 10, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_pattern = nn.Linear(64, num_patterns)
        self.fc_direction = nn.Linear(64, 3)  # bullish, neutral, bearish
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Pool and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Outputs
        pattern_logits = self.fc_pattern(x)
        direction_logits = self.fc_direction(x)
        
        return pattern_logits, direction_logits


class PatternAI:
    """
    Advanced Pattern Recognition AI
    
    ความสามารถ:
    1. ตรวจจับ Candlestick Patterns
    2. หา Chart Patterns
    3. วิเคราะห์ Price Action
    4. ทำนาย direction
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pattern CNN (for future use with training)
        self.pattern_cnn = PatternCNN().to(self.device)
        self.pattern_cnn.eval()
        
        # Pattern history for learning
        self.pattern_history: List[Dict] = []
        
        # Performance tracking
        self.pattern_performance: Dict[str, Dict] = {}
        
        logger.info(f"PatternAI initialized on {self.device}")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        วิเคราะห์ patterns ทั้งหมด
        """
        
        if len(data) < self.lookback:
            return self._empty_result()
        
        # 1. Candlestick patterns
        candle_patterns = self._detect_candlestick_patterns(data)
        
        # 2. Chart patterns
        chart_patterns = self._detect_chart_patterns(data)
        
        # 3. Price action signals
        price_action = self._analyze_price_action(data)
        
        # 4. Combine signals
        all_patterns = candle_patterns + chart_patterns
        
        # 5. Calculate overall signal
        bullish_score = sum(p.confidence for p in all_patterns if p.direction > 0)
        bearish_score = sum(p.confidence for p in all_patterns if p.direction < 0)
        
        if bullish_score > bearish_score:
            direction = 1
            net_score = bullish_score - bearish_score
        elif bearish_score > bullish_score:
            direction = -1
            net_score = bearish_score - bullish_score
        else:
            direction = 0
            net_score = 0
        
        # 6. Best pattern
        best_pattern = max(all_patterns, key=lambda x: x.confidence) if all_patterns else None
        
        return {
            "patterns_found": len(all_patterns),
            "candlestick_patterns": candle_patterns,
            "chart_patterns": chart_patterns,
            "price_action": price_action,
            "direction": direction,
            "confidence": min(net_score, 1.0),
            "best_pattern": best_pattern,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
        }
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """ตรวจจับ Candlestick Patterns"""
        
        patterns = []
        
        o = data['open'].values
        h = data['high'].values
        l = data['low'].values
        c = data['close'].values
        
        # Check last few candles
        for i in range(-5, 0):
            if abs(i) > len(data):
                continue
            
            body = c[i] - o[i]
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            candle_range = h[i] - l[i]
            
            if candle_range == 0:
                continue
            
            body_pct = abs(body) / candle_range
            upper_pct = upper_wick / candle_range
            lower_pct = lower_wick / candle_range
            
            # Doji
            if body_pct < 0.1:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOJI,
                    confidence=0.6,
                    direction=0,
                    price_level=c[i],
                    detected_at=len(data) + i,
                ))
            
            # Hammer (bullish)
            if lower_pct > 0.6 and body_pct < 0.3 and upper_pct < 0.1:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.HAMMER,
                    confidence=0.7,
                    direction=1,
                    price_level=c[i],
                    detected_at=len(data) + i,
                ))
            
            # Bullish Pinbar
            if lower_pct > 0.5 and body > 0:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.PINBAR_BULL,
                    confidence=0.65,
                    direction=1,
                    price_level=c[i],
                    detected_at=len(data) + i,
                ))
            
            # Bearish Pinbar
            if upper_pct > 0.5 and body < 0:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.PINBAR_BEAR,
                    confidence=0.65,
                    direction=-1,
                    price_level=c[i],
                    detected_at=len(data) + i,
                ))
            
            # Engulfing patterns (need previous candle)
            if i < -1:
                prev_body = c[i-1] - o[i-1]
                
                # Bullish Engulfing
                if prev_body < 0 and body > 0 and abs(body) > abs(prev_body) * 1.5:
                    if o[i] <= c[i-1] and c[i] >= o[i-1]:
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.ENGULFING_BULL,
                            confidence=0.75,
                            direction=1,
                            price_level=c[i],
                            detected_at=len(data) + i,
                        ))
                
                # Bearish Engulfing
                if prev_body > 0 and body < 0 and abs(body) > abs(prev_body) * 1.5:
                    if o[i] >= c[i-1] and c[i] <= o[i-1]:
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.ENGULFING_BEAR,
                            confidence=0.75,
                            direction=-1,
                            price_level=c[i],
                            detected_at=len(data) + i,
                        ))
        
        return patterns
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """ตรวจจับ Chart Patterns"""
        
        patterns = []
        
        h = data['high'].values
        l = data['low'].values
        c = data['close'].values
        
        # Need enough data
        if len(data) < 30:
            return patterns
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(h) - 5):
            if h[i] == max(h[i-5:i+6]):
                swing_highs.append((i, h[i]))
            if l[i] == min(l[i-5:i+6]):
                swing_lows.append((i, l[i]))
        
        # Double Top
        if len(swing_highs) >= 2:
            last_two = swing_highs[-2:]
            if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
                # Similar heights = double top
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_TOP,
                    confidence=0.7,
                    direction=-1,
                    price_level=c[-1],
                    target_price=min(l[-20:]),
                    stop_price=max(h[-20:]) * 1.01,
                    detected_at=len(data) - 1,
                ))
        
        # Double Bottom
        if len(swing_lows) >= 2:
            last_two = swing_lows[-2:]
            if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
                # Similar lows = double bottom
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    confidence=0.7,
                    direction=1,
                    price_level=c[-1],
                    target_price=max(h[-20:]),
                    stop_price=min(l[-20:]) * 0.99,
                    detected_at=len(data) - 1,
                ))
        
        # Breakout detection
        recent_high = max(h[-20:-1])
        recent_low = min(l[-20:-1])
        
        if c[-1] > recent_high:
            patterns.append(DetectedPattern(
                pattern_type=PatternType.BREAKOUT,
                confidence=0.8,
                direction=1,
                price_level=c[-1],
                detected_at=len(data) - 1,
            ))
        
        if c[-1] < recent_low:
            patterns.append(DetectedPattern(
                pattern_type=PatternType.BREAKDOWN,
                confidence=0.8,
                direction=-1,
                price_level=c[-1],
                detected_at=len(data) - 1,
            ))
        
        return patterns
    
    def _analyze_price_action(self, data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ Price Action"""
        
        c = data['close'].values
        h = data['high'].values
        l = data['low'].values
        
        # Recent momentum
        momentum_5 = (c[-1] - c[-5]) / c[-5] if len(c) >= 5 else 0
        momentum_10 = (c[-1] - c[-10]) / c[-10] if len(c) >= 10 else 0
        momentum_20 = (c[-1] - c[-20]) / c[-20] if len(c) >= 20 else 0
        
        # Trend strength
        higher_highs = sum(1 for i in range(-20, -1) if h[i] > h[i-1])
        higher_lows = sum(1 for i in range(-20, -1) if l[i] > l[i-1])
        lower_highs = sum(1 for i in range(-20, -1) if h[i] < h[i-1])
        lower_lows = sum(1 for i in range(-20, -1) if l[i] < l[i-1])
        
        uptrend_score = (higher_highs + higher_lows) / 38
        downtrend_score = (lower_highs + lower_lows) / 38
        
        # Price position
        range_high = max(h[-20:])
        range_low = min(l[-20:])
        range_position = (c[-1] - range_low) / (range_high - range_low + 1e-10)
        
        return {
            "momentum_5": momentum_5,
            "momentum_10": momentum_10,
            "momentum_20": momentum_20,
            "uptrend_score": uptrend_score,
            "downtrend_score": downtrend_score,
            "range_position": range_position,
            "at_range_high": range_position > 0.9,
            "at_range_low": range_position < 0.1,
        }
    
    def record_pattern_result(
        self,
        pattern_type: PatternType,
        was_correct: bool,
    ):
        """บันทึกผลของ pattern prediction"""
        
        key = pattern_type.value
        
        if key not in self.pattern_performance:
            self.pattern_performance[key] = {"correct": 0, "total": 0}
        
        self.pattern_performance[key]["total"] += 1
        if was_correct:
            self.pattern_performance[key]["correct"] += 1
    
    def get_pattern_accuracy(self, pattern_type: PatternType) -> float:
        """ดึงความแม่นยำของ pattern"""
        
        key = pattern_type.value
        
        if key not in self.pattern_performance:
            return 0.5
        
        stats = self.pattern_performance[key]
        return stats["correct"] / stats["total"] if stats["total"] > 0 else 0.5
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "patterns_found": 0,
            "candlestick_patterns": [],
            "chart_patterns": [],
            "price_action": {},
            "direction": 0,
            "confidence": 0,
            "best_pattern": None,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "patterns_tracked": len(self.pattern_performance),
            "pattern_accuracy": {
                k: v["correct"] / v["total"] if v["total"] > 0 else 0
                for k, v in self.pattern_performance.items()
            },
        }


def create_pattern_ai() -> PatternAI:
    """สร้าง PatternAI"""
    return PatternAI()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   PATTERN AI TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data
    n = 100
    prices = 2000 + np.cumsum(np.random.randn(n) * 3)
    
    data = pd.DataFrame({
        "open": prices - np.random.rand(n) * 3,
        "high": prices + np.random.rand(n) * 5,
        "low": prices - np.random.rand(n) * 5,
        "close": prices,
        "volume": np.random.randint(1000, 5000, n),
    })
    
    # Analyze
    pai = create_pattern_ai()
    result = pai.analyze(data)
    
    print(f"\nPatterns Found: {result['patterns_found']}")
    print(f"Direction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if result['best_pattern']:
        bp = result['best_pattern']
        print(f"\nBest Pattern: {bp.pattern_type.value}")
        print(f"  Confidence: {bp.confidence:.1%}")
        print(f"  Direction: {'Bullish' if bp.direction > 0 else 'Bearish' if bp.direction < 0 else 'Neutral'}")
    
    print(f"\nPrice Action:")
    pa = result['price_action']
    print(f"  Momentum 5: {pa.get('momentum_5', 0):.2%}")
    print(f"  Range Position: {pa.get('range_position', 0):.1%}")
