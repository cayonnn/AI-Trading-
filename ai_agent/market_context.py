"""
Market Context Module
======================
เข้าใจ Context ของตลาดอย่างลึกซึ้ง

Features:
1. Session Analysis - วิเคราะห์ตาม Trading Sessions
2. News/Event Awareness - รู้ว่ามี events สำคัญ
3. Correlation Analysis - ดูความสัมพันธ์กับ assets อื่น
4. Liquidity Analysis - วิเคราะห์ liquidity
5. Market Phase Detection - ตรวจจับ accumulation/distribution
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from loguru import logger


class TradingSession(Enum):
    """Trading Sessions"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OFF_HOURS = "off_hours"


class MarketPhase(Enum):
    """Market Phases"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    RANGING = "ranging"


@dataclass
class ContextAnalysis:
    """ผลการวิเคราะห์ Context"""
    session: TradingSession
    session_quality: float  # 0-1
    market_phase: MarketPhase
    liquidity_score: float  # 0-1
    volatility_state: str  # 'low', 'normal', 'high', 'extreme'
    trade_allowed: bool
    context_score: float  # 0-100
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MarketContext:
    """
    Deep Market Context Awareness
    
    ความสามารถ:
    1. รู้ว่าตอนนี้อยู่ session ไหน
    2. ประเมิน liquidity
    3. ตรวจจับ market phase
    4. วิเคราะห์ volatility state
    """
    
    def __init__(self, timezone_offset: int = 7):  # Default: GMT+7
        self.timezone_offset = timezone_offset
        
        # Session performance history
        self.session_performance: Dict[str, Dict] = {
            session.value: {"wins": 0, "total": 0}
            for session in TradingSession
        }
        
        # Volatility tracking
        self.volatility_history: List[float] = []
        
        # Volume tracking
        self.volume_history: List[float] = []
        
        logger.info("MarketContext initialized")
    
    def analyze(
        self,
        data: pd.DataFrame,
        current_time: datetime = None,
    ) -> ContextAnalysis:
        """
        วิเคราะห์ Context ทั้งหมด
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        warnings = []
        recommendations = []
        
        # 1. Determine current session
        session = self._get_current_session(current_time)
        session_quality = self._get_session_quality(session)
        
        # 2. Analyze market phase
        market_phase = self._detect_market_phase(data)
        
        # 3. Calculate liquidity score
        liquidity_score = self._analyze_liquidity(data)
        
        # 4. Analyze volatility state
        volatility_state = self._analyze_volatility_state(data)
        
        # 5. Generate warnings
        if session_quality < 0.5:
            warnings.append(f"Low quality session: {session.value}")
        
        if liquidity_score < 0.4:
            warnings.append("Low liquidity detected")
        
        if volatility_state == "extreme":
            warnings.append("Extreme volatility - reduce position size")
        
        # 6. Generate recommendations
        if market_phase == MarketPhase.MARKUP:
            recommendations.append("Favor long positions")
        elif market_phase == MarketPhase.MARKDOWN:
            recommendations.append("Favor short positions or wait")
        
        if session in [TradingSession.LONDON, TradingSession.OVERLAP_LONDON_NY]:
            recommendations.append("Best trading time for Gold")
        
        # 7. Determine if trading is allowed
        trade_allowed = (
            session_quality >= 0.3 and
            liquidity_score >= 0.3 and
            volatility_state != "extreme"
        )
        
        # 8. Calculate context score
        context_score = self._calculate_context_score(
            session_quality, liquidity_score, volatility_state, market_phase
        )
        
        return ContextAnalysis(
            session=session,
            session_quality=session_quality,
            market_phase=market_phase,
            liquidity_score=liquidity_score,
            volatility_state=volatility_state,
            trade_allowed=trade_allowed,
            context_score=context_score,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _get_current_session(self, current_time: datetime) -> TradingSession:
        """ระบุ session ปัจจุบัน"""
        
        # Convert to UTC
        utc_hour = (current_time.hour - self.timezone_offset) % 24
        
        # Trading sessions (UTC)
        # Sydney: 22:00 - 07:00
        # Tokyo: 00:00 - 09:00
        # London: 08:00 - 17:00
        # New York: 13:00 - 22:00
        
        if 13 <= utc_hour < 17:
            return TradingSession.OVERLAP_LONDON_NY
        elif 8 <= utc_hour < 17:
            return TradingSession.LONDON
        elif 13 <= utc_hour < 22:
            return TradingSession.NEW_YORK
        elif 0 <= utc_hour < 9:
            return TradingSession.TOKYO
        elif 22 <= utc_hour or utc_hour < 7:
            return TradingSession.SYDNEY
        else:
            return TradingSession.OFF_HOURS
    
    def _get_session_quality(self, session: TradingSession) -> float:
        """ประเมินคุณภาพของ session สำหรับ Gold"""
        
        session_qualities = {
            TradingSession.OVERLAP_LONDON_NY: 1.0,  # Best for Gold
            TradingSession.LONDON: 0.9,
            TradingSession.NEW_YORK: 0.8,
            TradingSession.TOKYO: 0.5,
            TradingSession.SYDNEY: 0.4,
            TradingSession.OFF_HOURS: 0.2,
        }
        
        base_quality = session_qualities.get(session, 0.5)
        
        # Adjust based on historical performance
        stats = self.session_performance.get(session.value, {"wins": 0, "total": 0})
        if stats["total"] >= 10:
            win_rate = stats["wins"] / stats["total"]
            base_quality = base_quality * 0.5 + win_rate * 0.5
        
        return base_quality
    
    def _detect_market_phase(self, data: pd.DataFrame) -> MarketPhase:
        """ตรวจจับ Market Phase"""
        
        if len(data) < 50:
            return MarketPhase.RANGING
        
        close = data['close'].values
        volume = data.get('volume', data.get('tick_volume', pd.Series([1000]*len(data)))).values
        
        # Calculate trends
        short_trend = (close[-1] - close[-20]) / close[-20]
        long_trend = (close[-1] - close[-50]) / close[-50]
        
        # Volume analysis
        recent_vol = np.mean(volume[-10:])
        past_vol = np.mean(volume[-50:-10])
        vol_change = (recent_vol - past_vol) / (past_vol + 1)
        
        # Detect phase
        if short_trend > 0.01 and long_trend > 0:
            if vol_change > 0.2:
                return MarketPhase.MARKUP
            else:
                return MarketPhase.DISTRIBUTION
        elif short_trend < -0.01 and long_trend < 0:
            if vol_change > 0.2:
                return MarketPhase.MARKDOWN
            else:
                return MarketPhase.ACCUMULATION
        else:
            return MarketPhase.RANGING
    
    def _analyze_liquidity(self, data: pd.DataFrame) -> float:
        """วิเคราะห์ Liquidity"""
        
        if 'volume' not in data.columns and 'tick_volume' not in data.columns:
            return 0.6  # Default
        
        vol_col = 'volume' if 'volume' in data.columns else 'tick_volume'
        volume = data[vol_col].values
        
        if len(volume) < 20:
            return 0.5
        
        # Compare recent volume to average
        recent_vol = np.mean(volume[-5:])
        avg_vol = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
        
        ratio = recent_vol / (avg_vol + 1)
        
        # Normalize to 0-1
        liquidity_score = min(ratio, 2) / 2
        
        return liquidity_score
    
    def _analyze_volatility_state(self, data: pd.DataFrame) -> str:
        """วิเคราะห์สถานะ Volatility"""
        
        if len(data) < 20:
            return "normal"
        
        close = data['close']
        returns = close.pct_change().dropna()
        
        current_vol = returns.iloc[-20:].std() * np.sqrt(252) * 100  # Annualized %
        
        # Track volatility
        self.volatility_history.append(current_vol)
        if len(self.volatility_history) > 100:
            self.volatility_history = self.volatility_history[-100:]
        
        # Determine state based on percentiles
        if len(self.volatility_history) >= 20:
            percentile = np.percentile(self.volatility_history, [25, 50, 75, 95])
            
            if current_vol < percentile[0]:
                return "low"
            elif current_vol < percentile[2]:
                return "normal"
            elif current_vol < percentile[3]:
                return "high"
            else:
                return "extreme"
        
        # Default thresholds (for Gold)
        if current_vol < 10:
            return "low"
        elif current_vol < 20:
            return "normal"
        elif current_vol < 35:
            return "high"
        else:
            return "extreme"
    
    def _calculate_context_score(
        self,
        session_quality: float,
        liquidity_score: float,
        volatility_state: str,
        market_phase: MarketPhase,
    ) -> float:
        """คำนวณ Context Score (0-100)"""
        
        score = 50.0
        
        # Session quality (30 points)
        score += session_quality * 30
        
        # Liquidity (20 points)
        score += liquidity_score * 20
        
        # Volatility (20 points)
        vol_scores = {"low": 10, "normal": 20, "high": 15, "extreme": 5}
        score += vol_scores.get(volatility_state, 10)
        
        # Market phase (30 points)
        phase_scores = {
            MarketPhase.MARKUP: 25,
            MarketPhase.MARKDOWN: 20,
            MarketPhase.ACCUMULATION: 15,
            MarketPhase.DISTRIBUTION: 10,
            MarketPhase.RANGING: 10,
        }
        score += phase_scores.get(market_phase, 10)
        
        return min(100, max(0, score))
    
    def record_trade(
        self,
        session: TradingSession,
        was_win: bool,
    ):
        """บันทึกผล trade ตาม session"""
        
        key = session.value
        if key in self.session_performance:
            self.session_performance[key]["total"] += 1
            if was_win:
                self.session_performance[key]["wins"] += 1
    
    def get_best_trading_time(self) -> Tuple[str, float]:
        """หาเวลาที่ดีที่สุดสำหรับเทรด"""
        
        best_session = max(
            self.session_performance.items(),
            key=lambda x: x[1]["wins"] / x[1]["total"] if x[1]["total"] > 0 else 0
        )
        
        win_rate = (
            best_session[1]["wins"] / best_session[1]["total"]
            if best_session[1]["total"] > 0 else 0
        )
        
        return best_session[0], win_rate


def create_market_context() -> MarketContext:
    """สร้าง MarketContext"""
    return MarketContext()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   MARKET CONTEXT TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data
    n = 100
    prices = 2000 + np.cumsum(np.random.randn(n) * 3)
    
    data = pd.DataFrame({
        "open": prices - np.random.rand(n) * 2,
        "high": prices + np.random.rand(n) * 4,
        "low": prices - np.random.rand(n) * 4,
        "close": prices,
        "volume": np.random.randint(1000, 5000, n),
    })
    
    # Analyze
    mc = create_market_context()
    result = mc.analyze(data)
    
    print(f"\nContext Analysis:")
    print(f"  Session: {result.session.value}")
    print(f"  Session Quality: {result.session_quality:.1%}")
    print(f"  Market Phase: {result.market_phase.value}")
    print(f"  Liquidity: {result.liquidity_score:.1%}")
    print(f"  Volatility: {result.volatility_state}")
    print(f"  Context Score: {result.context_score:.0f}/100")
    print(f"  Trade Allowed: {result.trade_allowed}")
    
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    if result.recommendations:
        print(f"  Recommendations: {result.recommendations}")
