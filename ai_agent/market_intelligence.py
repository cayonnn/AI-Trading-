"""
Market Intelligence Module
===========================
วิเคราะห์ตลาดอย่างลึกซึ้งด้วย Multi-Factor Analysis

Features:
1. Multi-Timeframe Analysis - วิเคราะห์หลาย timeframes
2. Market Structure - ตรวจจับ HH/HL/LH/LL
3. Smart Money Concepts - Order Blocks, FVG, Liquidity
4. Confluence Detection - รวม signals จากหลายแหล่ง
5. Trend Strength Analysis - วัดความแข็งแกร่งของ trend
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from loguru import logger


class MarketBias(Enum):
    """Market Bias"""
    STRONG_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    STRONG_BEARISH = -2


@dataclass
class PriceLevel:
    """ระดับราคาสำคัญ"""
    price: float
    level_type: str  # 'support', 'resistance', 'order_block', 'fvg'
    strength: float  # 0-1
    touched: int = 0
    created_at: int = 0  # bar index


@dataclass
class MarketStructure:
    """โครงสร้างตลาด"""
    trend: str  # 'uptrend', 'downtrend', 'ranging'
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    last_break: str = ""  # 'higher_high', 'lower_low', 'choch'
    structure_shift: bool = False


@dataclass
class ConfluenceSignal:
    """สัญญาณจากหลายแหล่ง"""
    direction: int  # 1 = bullish, -1 = bearish
    strength: float  # 0-1
    factors: Dict[str, float] = field(default_factory=dict)
    description: str = ""


class MarketIntelligence:
    """
    Advanced Market Intelligence
    
    ความสามารถ:
    1. วิเคราะห์ Market Structure
    2. หา Key Levels (Support/Resistance)
    3. ตรวจจับ Smart Money Patterns
    4. Multi-Timeframe Confluence
    5. Trend Strength Scoring
    """
    
    def __init__(self):
        # Key levels
        self.support_levels: List[PriceLevel] = []
        self.resistance_levels: List[PriceLevel] = []
        self.order_blocks: List[PriceLevel] = []
        
        # State
        self.current_bias = MarketBias.NEUTRAL
        self.last_structure: Optional[MarketStructure] = None
        
        logger.info("MarketIntelligence initialized")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        วิเคราะห์ตลาดแบบครบถ้วน
        
        Returns:
            Dict containing all analysis results
        """
        
        if len(data) < 50:
            return self._empty_analysis()
        
        # 1. Market Structure
        structure = self._analyze_structure(data)
        
        # 2. Key Levels
        levels = self._find_key_levels(data)
        
        # 3. Trend Analysis
        trend_info = self._analyze_trend(data)
        
        # 4. Momentum
        momentum = self._analyze_momentum(data)
        
        # 5. Volume Analysis
        volume_info = self._analyze_volume(data)
        
        # 6. Smart Money Concepts
        smc = self._analyze_smart_money(data)
        
        # 7. Multi-factor Confluence
        confluence = self._calculate_confluence(
            structure, trend_info, momentum, volume_info, smc
        )
        
        # 8. Overall Bias
        bias = self._determine_bias(confluence)
        self.current_bias = bias
        
        return {
            "structure": structure,
            "levels": levels,
            "trend": trend_info,
            "momentum": momentum,
            "volume": volume_info,
            "smc": smc,
            "confluence": confluence,
            "bias": bias.name,
            "bias_score": bias.value,
            "trade_score": confluence.strength,
            "recommendation": self._get_recommendation(confluence, bias),
        }
    
    def _analyze_structure(self, data: pd.DataFrame) -> MarketStructure:
        """วิเคราะห์ Market Structure"""
        
        close = data['close'].values
        high = data['high'].values if 'high' in data else close + 1
        low = data['low'].values if 'low' in data else close - 1
        
        # Find swing points (simple method)
        swing_highs = []
        swing_lows = []
        lookback = 5
        
        for i in range(lookback, len(high) - lookback):
            # Swing High
            if high[i] == max(high[i-lookback:i+lookback+1]):
                swing_highs.append(high[i])
            
            # Swing Low
            if low[i] == min(low[i-lookback:i+lookback+1]):
                swing_lows.append(low[i])
        
        # Determine trend from swing points
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
            hl = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
            lh = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False
            ll = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
            
            if hh and hl:
                trend = "uptrend"
                last_break = "higher_high"
            elif ll and lh:
                trend = "downtrend"
                last_break = "lower_low"
            else:
                trend = "ranging"
                last_break = "choch" if (hh and ll) or (lh and hl) else ""
        else:
            trend = "unknown"
            last_break = ""
        
        structure = MarketStructure(
            trend=trend,
            swing_highs=swing_highs[-5:] if swing_highs else [],
            swing_lows=swing_lows[-5:] if swing_lows else [],
            last_break=last_break,
            structure_shift=last_break == "choch",
        )
        
        self.last_structure = structure
        return structure
    
    def _find_key_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """หา Support/Resistance levels"""
        
        close = data['close'].values
        high = data['high'].values if 'high' in data else close
        low = data['low'].values if 'low' in data else close
        
        current_price = close[-1]
        
        # Simple S/R based on swing points
        supports = []
        resistances = []
        
        # Recent swing lows as support
        for i in range(5, len(low) - 1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                if low[i] < current_price:
                    supports.append(low[i])
        
        # Recent swing highs as resistance
        for i in range(5, len(high) - 1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                if high[i] > current_price:
                    resistances.append(high[i])
        
        # Keep closest levels
        supports = sorted(supports, reverse=True)[:3]
        resistances = sorted(resistances)[:3]
        
        return {
            "supports": supports,
            "resistances": resistances,
            "nearest_support": supports[0] if supports else current_price * 0.99,
            "nearest_resistance": resistances[0] if resistances else current_price * 1.01,
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ Trend"""
        
        close = data['close']
        
        # Moving Averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
        
        current = close.iloc[-1]
        
        # Trend direction
        above_ma20 = current > ma20
        above_ma50 = current > ma50
        ma20_above_ma50 = ma20 > ma50
        
        # Trend strength (ADX-like calculation)
        returns = close.pct_change()
        up_moves = returns.where(returns > 0, 0).rolling(14).sum()
        down_moves = abs(returns.where(returns < 0, 0).rolling(14).sum())
        
        di_plus = up_moves.iloc[-1] if not pd.isna(up_moves.iloc[-1]) else 0
        di_minus = down_moves.iloc[-1] if not pd.isna(down_moves.iloc[-1]) else 0
        
        if di_plus + di_minus > 0:
            trend_strength = abs(di_plus - di_minus) / (di_plus + di_minus)
        else:
            trend_strength = 0
        
        # Determine direction
        if above_ma20 and above_ma50 and ma20_above_ma50:
            direction = "strong_up"
            score = 0.8 + trend_strength * 0.2
        elif above_ma20 and above_ma50:
            direction = "up"
            score = 0.6 + trend_strength * 0.2
        elif not above_ma20 and not above_ma50 and not ma20_above_ma50:
            direction = "strong_down"
            score = -0.8 - trend_strength * 0.2
        elif not above_ma20 and not above_ma50:
            direction = "down"
            score = -0.6 - trend_strength * 0.2
        else:
            direction = "mixed"
            score = 0
        
        return {
            "direction": direction,
            "strength": trend_strength,
            "score": np.clip(score, -1, 1),
            "ma20": ma20,
            "ma50": ma50,
            "above_ma20": above_ma20,
            "above_ma50": above_ma50,
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ Momentum"""
        
        close = data['close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        macd_value = macd.iloc[-1]
        signal_value = signal.iloc[-1]
        hist_value = histogram.iloc[-1]
        
        # Momentum score
        rsi_score = (rsi - 50) / 50  # -1 to 1
        macd_score = np.sign(hist_value) * min(abs(hist_value) / 2, 1)
        
        momentum_score = (rsi_score + macd_score) / 2
        
        return {
            "rsi": rsi,
            "rsi_zone": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
            "macd": macd_value,
            "macd_signal": signal_value,
            "macd_histogram": hist_value,
            "macd_cross": "bullish" if macd_value > signal_value else "bearish",
            "score": np.clip(momentum_score, -1, 1),
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """วิเคราะห์ Volume"""
        
        if 'volume' not in data.columns and 'tick_volume' not in data.columns:
            return {"available": False, "score": 0}
        
        vol_col = 'volume' if 'volume' in data.columns else 'tick_volume'
        volume = data[vol_col]
        
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Check if volume confirms price move
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        volume_confirms = (price_change > 0 and volume_ratio > 1.2) or \
                         (price_change < 0 and volume_ratio > 1.2)
        
        return {
            "available": True,
            "current": current_volume,
            "average": avg_volume,
            "ratio": volume_ratio,
            "high_volume": volume_ratio > 1.5,
            "confirms_move": volume_confirms,
            "score": min(volume_ratio - 1, 1) if volume_confirms else 0,
        }
    
    def _analyze_smart_money(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Smart Money Concepts Analysis"""
        
        close = data['close'].values
        high = data['high'].values if 'high' in data else close
        low = data['low'].values if 'low' in data else close
        
        current = close[-1]
        
        # Order Block Detection (simplified)
        order_blocks = []
        for i in range(3, len(close) - 1):
            # Bullish OB: Strong down move followed by up move
            if close[i-1] < close[i-2] and close[i] > close[i-1]:
                if (close[i-2] - close[i-1]) / close[i-2] > 0.005:
                    order_blocks.append({
                        "type": "bullish",
                        "price": low[i-1],
                        "valid": low[i-1] < current,
                    })
            
            # Bearish OB: Strong up move followed by down move
            if close[i-1] > close[i-2] and close[i] < close[i-1]:
                if (close[i-1] - close[i-2]) / close[i-2] > 0.005:
                    order_blocks.append({
                        "type": "bearish",
                        "price": high[i-1],
                        "valid": high[i-1] > current,
                    })
        
        # Fair Value Gap Detection (simplified)
        fvgs = []
        for i in range(2, len(close)):
            # Bullish FVG: Gap up
            if low[i] > high[i-2]:
                fvgs.append({
                    "type": "bullish",
                    "top": low[i],
                    "bottom": high[i-2],
                    "filled": current < low[i],
                })
            
            # Bearish FVG: Gap down
            if high[i] < low[i-2]:
                fvgs.append({
                    "type": "bearish",
                    "top": low[i-2],
                    "bottom": high[i],
                    "filled": current > high[i],
                })
        
        # Valid OBs and FVGs near price
        valid_bullish_ob = [ob for ob in order_blocks 
                           if ob['type'] == 'bullish' and ob['valid']]
        valid_bearish_ob = [ob for ob in order_blocks
                           if ob['type'] == 'bearish' and ob['valid']]
        
        # Score based on SMC
        ob_score = len(valid_bullish_ob) * 0.1 - len(valid_bearish_ob) * 0.1
        
        return {
            "order_blocks": order_blocks[-5:],
            "fvgs": fvgs[-5:],
            "bullish_obs": len(valid_bullish_ob),
            "bearish_obs": len(valid_bearish_ob),
            "score": np.clip(ob_score, -1, 1),
        }
    
    def _calculate_confluence(
        self,
        structure: MarketStructure,
        trend: Dict,
        momentum: Dict,
        volume: Dict,
        smc: Dict,
    ) -> ConfluenceSignal:
        """คำนวณ Confluence จากทุก factors"""
        
        factors = {}
        
        # Structure weight: 25%
        if structure.trend == "uptrend":
            factors["structure"] = 0.8
        elif structure.trend == "downtrend":
            factors["structure"] = -0.8
        else:
            factors["structure"] = 0
        
        # Trend weight: 30%
        factors["trend"] = trend.get("score", 0)
        
        # Momentum weight: 25%
        factors["momentum"] = momentum.get("score", 0)
        
        # Volume weight: 10%
        factors["volume"] = volume.get("score", 0) if volume.get("available") else 0
        
        # SMC weight: 10%
        factors["smc"] = smc.get("score", 0)
        
        # Weighted sum
        weights = {
            "structure": 0.25,
            "trend": 0.30,
            "momentum": 0.25,
            "volume": 0.10,
            "smc": 0.10,
        }
        
        total_score = sum(
            factors.get(k, 0) * weights.get(k, 0)
            for k in weights
        )
        
        direction = 1 if total_score > 0 else -1 if total_score < 0 else 0
        strength = abs(total_score)
        
        return ConfluenceSignal(
            direction=direction,
            strength=strength,
            factors=factors,
            description=self._describe_confluence(factors),
        )
    
    def _describe_confluence(self, factors: Dict) -> str:
        """อธิบาย confluence"""
        
        bullish = [k for k, v in factors.items() if v > 0.3]
        bearish = [k for k, v in factors.items() if v < -0.3]
        
        if len(bullish) >= 3:
            return f"Strong bullish: {', '.join(bullish)}"
        elif len(bearish) >= 3:
            return f"Strong bearish: {', '.join(bearish)}"
        elif len(bullish) > len(bearish):
            return f"Bullish bias: {', '.join(bullish)}"
        elif len(bearish) > len(bullish):
            return f"Bearish bias: {', '.join(bearish)}"
        else:
            return "No clear bias"
    
    def _determine_bias(self, confluence: ConfluenceSignal) -> MarketBias:
        """กำหนด Market Bias"""
        
        if confluence.strength > 0.6:
            if confluence.direction > 0:
                return MarketBias.STRONG_BULLISH
            else:
                return MarketBias.STRONG_BEARISH
        elif confluence.strength > 0.3:
            if confluence.direction > 0:
                return MarketBias.BULLISH
            else:
                return MarketBias.BEARISH
        else:
            return MarketBias.NEUTRAL
    
    def _get_recommendation(
        self,
        confluence: ConfluenceSignal,
        bias: MarketBias,
    ) -> Dict[str, Any]:
        """สร้างคำแนะนำการเทรด"""
        
        if bias == MarketBias.STRONG_BULLISH:
            return {
                "action": "LONG",
                "confidence": 0.85,
                "reason": confluence.description,
            }
        elif bias == MarketBias.BULLISH:
            return {
                "action": "LONG",
                "confidence": 0.70,
                "reason": confluence.description,
            }
        elif bias == MarketBias.STRONG_BEARISH:
            return {
                "action": "SHORT",
                "confidence": 0.85,
                "reason": confluence.description,
            }
        elif bias == MarketBias.BEARISH:
            return {
                "action": "SHORT",
                "confidence": 0.70,
                "reason": confluence.description,
            }
        else:
            return {
                "action": "WAIT",
                "confidence": 0.50,
                "reason": "No clear signal",
            }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Empty analysis for insufficient data"""
        return {
            "structure": MarketStructure("unknown"),
            "levels": {"supports": [], "resistances": []},
            "trend": {"direction": "unknown", "score": 0},
            "momentum": {"rsi": 50, "score": 0},
            "volume": {"available": False},
            "smc": {"score": 0},
            "confluence": ConfluenceSignal(0, 0),
            "bias": "NEUTRAL",
            "bias_score": 0,
            "trade_score": 0,
            "recommendation": {"action": "WAIT", "confidence": 0},
        }


def create_market_intelligence() -> MarketIntelligence:
    """สร้าง MarketIntelligence"""
    return MarketIntelligence()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   MARKET INTELLIGENCE TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data
    n = 200
    trend = np.cumsum(np.random.randn(n) * 0.5) + 2000
    noise = np.random.randn(n) * 2
    prices = trend + noise
    
    data = pd.DataFrame({
        "open": prices - np.random.rand(n) * 2,
        "high": prices + np.random.rand(n) * 5,
        "low": prices - np.random.rand(n) * 5,
        "close": prices,
        "volume": np.random.randint(1000, 5000, n),
    })
    
    # Analyze
    mi = create_market_intelligence()
    analysis = mi.analyze(data)
    
    print(f"\nMarket Bias: {analysis['bias']}")
    print(f"Trade Score: {analysis['trade_score']:.2f}")
    print(f"\nTrend: {analysis['trend']['direction']} (score: {analysis['trend']['score']:.2f})")
    print(f"RSI: {analysis['momentum']['rsi']:.1f} ({analysis['momentum']['rsi_zone']})")
    print(f"MACD: {analysis['momentum']['macd_cross']}")
    print(f"\nStructure: {analysis['structure'].trend}")
    print(f"Confluence: {analysis['confluence'].description}")
    print(f"\nRecommendation: {analysis['recommendation']}")
