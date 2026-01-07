"""
Strategy Library - Sniper Trading Focus
=========================================
คลังกลยุทธ์สำหรับ Sniper Trading

หลักการ Sniper:
- LONG เท่านั้น
- เข้าน้อย เลือกเฉพาะสัญญาณชัวร์
- SL สั้น (ความเสี่ยงต่ำ)
- TP ยาว (กำไรสูง)
- R:R อย่างน้อย 1:3
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from enum import Enum


class TradeDecision(Enum):
    """ผลการตัดสินใจ"""
    LONG = "LONG"
    WAIT = "WAIT"  # ไม่เทรด รอสัญญาณที่ดีกว่า


@dataclass
class SniperSignal:
    """สัญญาณ Sniper Trading"""
    decision: TradeDecision
    confidence: float  # 0.0 - 1.0
    strategy_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    r_ratio: float  # Risk-Reward Ratio
    reason: str
    
    # ข้อมูลเพิ่มเติม
    market_regime: str = ""
    expected_duration: int = 0  # bars
    priority: int = 0  # สำหรับจัดลำดับสัญญาณ


class BaseStrategy:
    """Base class สำหรับกลยุทธ์ทั้งหมด"""
    
    def __init__(
        self,
        min_confidence: float = 0.75,  # ขั้นต่ำ 75% confidence
        min_rr_ratio: float = 3.0,     # R:R อย่างน้อย 1:3
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 5.0,
    ):
        self.min_confidence = min_confidence
        self.min_rr_ratio = min_rr_ratio
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.name = "base_strategy"
        
    def analyze(self, data: pd.DataFrame) -> Optional[SniperSignal]:
        """วิเคราะห์และสร้างสัญญาณ - Override ใน Subclass"""
        raise NotImplementedError
    
    def calculate_sl_tp(
        self,
        entry: float,
        atr: float,
        direction: str = "LONG"
    ) -> Tuple[float, float, float]:
        """
        คำนวณ SL/TP แบบ Sniper
        
        Returns:
            (stop_loss, take_profit, r_ratio)
        """
        sl_distance = atr * self.sl_atr_multiplier
        tp_distance = atr * self.tp_atr_multiplier
        
        if direction == "LONG":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
        
        r_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        return sl, tp, r_ratio


class TrendSniperStrategy(BaseStrategy):
    """
    Trend Sniper - เข้า LONG เมื่อ Trend แข็งแกร่ง
    
    เงื่อนไข:
    - Price อยู่เหนือ MA(50) และ MA(200)
    - MACD เป็นบวกและเพิ่มขึ้น
    - RSI ไม่ Overbought (< 70)
    - มี Pullback ให้เข้า
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "trend_sniper"
        self.ma_fast_period = 50
        self.ma_slow_period = 200
        
    def analyze(self, data: pd.DataFrame) -> Optional[SniperSignal]:
        """วิเคราะห์สัญญาณ Trend Sniper"""
        if len(data) < self.ma_slow_period + 10:
            return None
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        close = current['close']
        
        # คำนวณ Indicators
        ma50 = data['close'].rolling(self.ma_fast_period).mean().iloc[-1]
        ma200 = data['close'].rolling(self.ma_slow_period).mean().iloc[-1]
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
        current_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2]
        
        # ATR
        high = data['high']
        low = data['low']
        prev_close = data['close'].shift(1)
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # === เงื่อนไข Sniper ===
        confidence = 0.0
        reasons = []
        
        # 1. Price เหนือ MAs (แนวโน้มขึ้น)
        if close > ma50 and close > ma200:
            confidence += 0.25
            reasons.append("Price > MA50 & MA200")
        else:
            return None  # ไม่ผ่านเงื่อนไขหลัก
        
        # 2. MA50 เหนือ MA200 (Golden Cross)
        if ma50 > ma200:
            confidence += 0.15
            reasons.append("MA50 > MA200 (Golden)")
        
        # 3. MACD เป็นบวกและเพิ่มขึ้น
        if current_macd_hist > 0 and current_macd_hist > prev_macd_hist:
            confidence += 0.20
            reasons.append("MACD Bullish & Rising")
        elif current_macd_hist > 0:
            confidence += 0.10
            reasons.append("MACD Bullish")
        
        # 4. RSI ไม่ Overbought
        if 40 <= current_rsi <= 65:
            confidence += 0.20
            reasons.append(f"RSI Healthy ({current_rsi:.0f})")
        elif current_rsi < 40:
            confidence += 0.15
            reasons.append(f"RSI Oversold ({current_rsi:.0f})")
        elif current_rsi > 70:
            confidence -= 0.15
            reasons.append(f"RSI Overbought ({current_rsi:.0f})")
        
        # 5. Pullback ให้เข้า
        pullback = (close - ma50) / ma50
        if -0.02 <= pullback <= 0.01:
            confidence += 0.20
            reasons.append("Good Pullback to MA50")
        
        # === ตัดสินใจ ===
        if confidence < self.min_confidence:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=0,
                take_profit=0,
                r_ratio=0,
                reason=f"Confidence ต่ำ ({confidence:.0%})",
            )
        
        # คำนวณ SL/TP
        sl, tp, rr = self.calculate_sl_tp(close, atr, "LONG")
        
        if rr < self.min_rr_ratio:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                r_ratio=rr,
                reason=f"R:R ต่ำเกินไป ({rr:.1f}:1)",
            )
        
        return SniperSignal(
            decision=TradeDecision.LONG,
            confidence=confidence,
            strategy_name=self.name,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            r_ratio=rr,
            reason=" | ".join(reasons),
            market_regime="trending",
            expected_duration=48,  # 48 bars
            priority=int(confidence * 100),
        )


class BreakoutSniperStrategy(BaseStrategy):
    """
    Breakout Sniper - เข้า LONG เมื่อทะลุ Resistance
    
    เงื่อนไข:
    - ทะลุ Resistance ชัดเจน
    - Volume สูงกว่าปกติ
    - RSI แข็งแกร่งแต่ไม่ Extreme
    - Candle Bullish ใหญ่
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "breakout_sniper"
        self.lookback_period = 20
        
    def analyze(self, data: pd.DataFrame) -> Optional[SniperSignal]:
        """วิเคราะห์สัญญาณ Breakout"""
        if len(data) < self.lookback_period + 10:
            return None
        
        current = data.iloc[-1]
        close = current['close']
        open_price = current['open']
        high = current['high']
        volume = current['volume'] if 'volume' in current else 1
        
        # หา Resistance (High สูงสุดใน N แท่งก่อนหน้า)
        lookback = data.iloc[-self.lookback_period-1:-1]
        resistance = lookback['high'].max()
        support = lookback['low'].min()
        
        # ATR
        high_col = data['high']
        low_col = data['low']
        prev_close = data['close'].shift(1)
        tr = pd.concat([
            high_col - low_col,
            abs(high_col - prev_close),
            abs(low_col - prev_close)
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Volume MA
        volume_ma = data['volume'].rolling(20).mean().iloc[-1] if 'volume' in data.columns else volume
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # === เงื่อนไข Breakout Sniper ===
        confidence = 0.0
        reasons = []
        
        # 1. ทะลุ Resistance
        breakout_pct = (close - resistance) / resistance
        if breakout_pct > 0.001:  # ทะลุอย่างน้อย 0.1%
            confidence += 0.30
            reasons.append(f"Breakout Resistance ({breakout_pct:.2%})")
        else:
            return None  # ไม่มี Breakout
        
        # 2. Candle Bullish ใหญ่
        body = close - open_price
        body_pct = body / open_price
        if body_pct > 0.003:  # Body มากกว่า 0.3%
            confidence += 0.20
            reasons.append(f"Strong Bullish Candle ({body_pct:.2%})")
        
        # 3. Volume สูง
        if 'volume' in data.columns and volume > volume_ma * 1.5:
            confidence += 0.20
            reasons.append(f"High Volume ({volume/volume_ma:.1f}x)")
        else:
            confidence += 0.05  # ไม่มีข้อมูล Volume
        
        # 4. RSI แข็งแกร่ง
        if 55 <= current_rsi <= 75:
            confidence += 0.15
            reasons.append(f"Strong RSI ({current_rsi:.0f})")
        elif current_rsi > 75:
            confidence -= 0.10
            reasons.append(f"RSI Too High ({current_rsi:.0f})")
        
        # 5. Range was tight (Consolidation before breakout)
        range_pct = (resistance - support) / support
        if range_pct < 0.03:  # Range < 3%
            confidence += 0.15
            reasons.append("Tight Consolidation")
        
        # === ตัดสินใจ ===
        if confidence < self.min_confidence:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=0,
                take_profit=0,
                r_ratio=0,
                reason=f"Confidence ต่ำ ({confidence:.0%})",
            )
        
        # คำนวณ SL/TP - SL ต่ำกว่า Resistance เดิม
        sl = resistance - (atr * 0.5)  # SL ใต้ Resistance
        tp_distance = atr * self.tp_atr_multiplier
        tp = close + tp_distance
        sl_distance = close - sl
        rr = tp_distance / sl_distance if sl_distance > 0 else 0
        
        if rr < self.min_rr_ratio:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                r_ratio=rr,
                reason=f"R:R ต่ำเกินไป ({rr:.1f}:1)",
            )
        
        return SniperSignal(
            decision=TradeDecision.LONG,
            confidence=confidence,
            strategy_name=self.name,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            r_ratio=rr,
            reason=" | ".join(reasons),
            market_regime="breakout",
            expected_duration=24,
            priority=int(confidence * 100),
        )


class MomentumSniperStrategy(BaseStrategy):
    """
    Momentum Sniper - เข้า LONG เมื่อ Momentum แข็งแกร่งมาก
    
    เงื่อนไข:
    - Momentum สูงผิดปกติ
    - MACD Cross ขึ้น
    - Volume Spike
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "momentum_sniper"
        
    def analyze(self, data: pd.DataFrame) -> Optional[SniperSignal]:
        """วิเคราะห์สัญญาณ Momentum"""
        if len(data) < 50:
            return None
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        close = current['close']
        
        # Momentum
        momentum_10 = close - data['close'].iloc[-11]
        momentum_20 = close - data['close'].iloc[-21]
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        
        macd_crossed_up = macd.iloc[-1] > macd_signal.iloc[-1] and \
                          macd.iloc[-2] <= macd_signal.iloc[-2]
        
        # ATR
        high = data['high']
        low = data['low']
        prev_close = data['close'].shift(1)
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Momentum Z-score
        momentum_mean = (data['close'] - data['close'].shift(10)).rolling(50).mean().iloc[-1]
        momentum_std = (data['close'] - data['close'].shift(10)).rolling(50).std().iloc[-1]
        momentum_zscore = (momentum_10 - momentum_mean) / (momentum_std + 1e-10)
        
        # === เงื่อนไข Momentum Sniper ===
        confidence = 0.0
        reasons = []
        
        # 1. Momentum Z-score สูง
        if momentum_zscore > 2.0:
            confidence += 0.35
            reasons.append(f"Extreme Momentum (Z={momentum_zscore:.1f})")
        elif momentum_zscore > 1.5:
            confidence += 0.25
            reasons.append(f"Strong Momentum (Z={momentum_zscore:.1f})")
        elif momentum_zscore > 1.0:
            confidence += 0.15
            reasons.append(f"Momentum (Z={momentum_zscore:.1f})")
        else:
            return None  # ไม่มี Momentum
        
        # 2. MACD Cross Up
        if macd_crossed_up:
            confidence += 0.25
            reasons.append("MACD Cross Up")
        
        # 3. Both Momentums positive
        if momentum_10 > 0 and momentum_20 > 0:
            confidence += 0.20
            reasons.append("Dual Momentum Positive")
        
        # 4. Momentum accelerating
        if momentum_10 > momentum_20 / 2:
            confidence += 0.15
            reasons.append("Momentum Accelerating")
        
        # === ตัดสินใจ ===
        if confidence < self.min_confidence:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=0,
                take_profit=0,
                r_ratio=0,
                reason=f"Confidence ต่ำ ({confidence:.0%})",
            )
        
        sl, tp, rr = self.calculate_sl_tp(close, atr, "LONG")
        
        if rr < self.min_rr_ratio:
            return SniperSignal(
                decision=TradeDecision.WAIT,
                confidence=confidence,
                strategy_name=self.name,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                r_ratio=rr,
                reason=f"R:R ต่ำเกินไป ({rr:.1f}:1)",
            )
        
        return SniperSignal(
            decision=TradeDecision.LONG,
            confidence=confidence,
            strategy_name=self.name,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            r_ratio=rr,
            reason=" | ".join(reasons),
            market_regime="momentum",
            expected_duration=12,
            priority=int(confidence * 100),
        )


class StrategyLibrary:
    """
    คลังกลยุทธ์ทั้งหมด
    
    หน้าที่:
    - เก็บกลยุทธ์ทั้งหมด
    - เรียกใช้ทุกกลยุทธ์และรวบรวมสัญญาณ
    - เลือกสัญญาณที่ดีที่สุด
    """
    
    def __init__(
        self,
        min_confidence: float = 0.75,
        min_rr_ratio: float = 3.0,
    ):
        self.min_confidence = min_confidence
        self.min_rr_ratio = min_rr_ratio
        
        # สร้างกลยุทธ์ทั้งหมด
        self.strategies = {
            "trend_sniper": TrendSniperStrategy(
                min_confidence=min_confidence,
                min_rr_ratio=min_rr_ratio,
            ),
            "breakout_sniper": BreakoutSniperStrategy(
                min_confidence=min_confidence,
                min_rr_ratio=min_rr_ratio,
            ),
            "momentum_sniper": MomentumSniperStrategy(
                min_confidence=min_confidence,
                min_rr_ratio=min_rr_ratio,
            ),
        }
        
        # เก็บประสิทธิภาพแต่ละกลยุทธ์
        self.strategy_weights = {name: 1.0 for name in self.strategies}
        
        logger.info(f"StrategyLibrary initialized with {len(self.strategies)} strategies")
    
    def analyze_all(self, data: pd.DataFrame) -> List[SniperSignal]:
        """เรียกใช้ทุกกลยุทธ์และรวบรวมสัญญาณ"""
        signals = []
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(data)
                if signal:
                    # ปรับ confidence ตาม weight
                    signal.confidence *= self.strategy_weights[name]
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
        
        return signals
    
    def get_best_signal(self, data: pd.DataFrame) -> Optional[SniperSignal]:
        """เลือกสัญญาณที่ดีที่สุดจากทุกกลยุทธ์"""
        signals = self.analyze_all(data)
        
        # กรองเฉพาะ LONG
        long_signals = [s for s in signals if s.decision == TradeDecision.LONG]
        
        if not long_signals:
            # ไม่มี LONG signal, ส่งคืน WAIT ถ้ามี
            wait_signals = [s for s in signals if s.decision == TradeDecision.WAIT]
            if wait_signals:
                return max(wait_signals, key=lambda x: x.confidence)
            return None
        
        # เลือกสัญญาณที่ดีที่สุด
        # เกณฑ์: Confidence สูง, R:R สูง
        def score(signal):
            return signal.confidence * 0.6 + (signal.r_ratio / 10) * 0.4
        
        best = max(long_signals, key=score)
        return best
    
    def update_weights(self, strategy_name: str, won: bool):
        """อัพเดต weight ของกลยุทธ์จากผลเทรด"""
        if strategy_name not in self.strategy_weights:
            return
        
        if won:
            self.strategy_weights[strategy_name] *= 1.05  # เพิ่ม 5%
        else:
            self.strategy_weights[strategy_name] *= 0.95  # ลด 5%
        
        # จำกัด weight
        self.strategy_weights[strategy_name] = max(0.5, min(1.5, self.strategy_weights[strategy_name]))
        
        logger.debug(f"Updated {strategy_name} weight to {self.strategy_weights[strategy_name]:.2f}")
    
    def get_status(self) -> Dict:
        """ดึงสถานะของ library"""
        return {
            "n_strategies": len(self.strategies),
            "strategy_names": list(self.strategies.keys()),
            "weights": self.strategy_weights,
            "min_confidence": self.min_confidence,
            "min_rr_ratio": self.min_rr_ratio,
        }


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Test with sample data
    np.random.seed(42)
    n = 300
    
    # Create trending data
    base_price = 2300
    trend = np.linspace(0, 100, n)
    noise = np.random.randn(n) * 5
    
    prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(n) * 2,
        'high': prices + np.abs(np.random.randn(n) * 5),
        'low': prices - np.abs(np.random.randn(n) * 5),
        'close': prices,
        'volume': np.random.randint(10000, 100000, n),
    })
    
    # Test Strategy Library
    library = StrategyLibrary(min_confidence=0.75, min_rr_ratio=3.0)
    
    signal = library.get_best_signal(data)
    
    if signal:
        print(f"\n{'='*60}")
        print(f"SIGNAL: {signal.decision.value}")
        print(f"Strategy: {signal.strategy_name}")
        print(f"Confidence: {signal.confidence:.0%}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"SL: ${signal.stop_loss:.2f}")
        print(f"TP: ${signal.take_profit:.2f}")
        print(f"R:R = 1:{signal.r_ratio:.1f}")
        print(f"Reason: {signal.reason}")
        print(f"{'='*60}")
    else:
        print("\nNo signal generated - Market conditions not suitable")
    
    print("\n✅ Strategy Library Test Complete")
