"""
Sentiment Integration v1.0
===========================
News and Market Sentiment Analysis for Trading

Features:
1. Economic Calendar Integration
2. Sentiment Analysis from multiple sources
3. Risk event filtering
4. Sentiment-based confidence adjustment

Note: Uses free APIs where available
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from loguru import logger


@dataclass
class SentimentSignal:
    """Market sentiment signal"""
    source: str
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    impact: str  # 'high', 'medium', 'low'
    event: str
    timestamp: datetime
    

@dataclass
class SentimentSummary:
    """Aggregated sentiment summary"""
    overall: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # -1.0 to 1.0
    high_impact_events: List[str]
    should_avoid: bool
    avoid_reason: str
    signals: List[SentimentSignal]


class SentimentIntegration:
    """
    รวบรวม sentiment จากหลายแหล่ง
    
    Sources:
    1. DXY (Dollar Index) - via OANDA/Alpha Vantage
    2. Economic Calendar - via ForexFactory/Investing.com
    3. VIX (Fear Index) - proxy from Gold volatility
    4. Market Structure - from price action
    """
    
    def __init__(
        self,
        cache_duration_minutes: int = 30,
    ):
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache: Dict[str, Any] = {}
        self.cache_time: Dict[str, datetime] = {}
        
        # Sentiment history
        self.sentiment_history: deque = deque(maxlen=100)
        
        # High-impact events to avoid
        self.high_impact_events = [
            "FOMC", "NFP", "CPI", "GDP", "Interest Rate",
            "Fed Chair", "ECB", "BOE", "BOJ", "RBA",
            "Unemployment", "Retail Sales", "PMI",
        ]
        
        # Event impact on Gold
        self.gold_impact = {
            "FOMC": {"direction": "volatile", "strength": 0.9},
            "NFP": {"direction": "volatile", "strength": 0.8},
            "CPI": {"direction": "volatile", "strength": 0.8},
            "Interest Rate": {"direction": "inverse", "strength": 0.7},
            "USD strength": {"direction": "bearish", "strength": 0.6},
            "USD weakness": {"direction": "bullish", "strength": 0.6},
            "Risk-off": {"direction": "bullish", "strength": 0.7},
            "Risk-on": {"direction": "bearish", "strength": 0.5},
        }
        
        # Current sentiment state
        self.current_dxy_trend: str = "neutral"
        self.current_vix_level: str = "normal"
        self.upcoming_events: List[Dict] = []
        
        logger.info("SentimentIntegration initialized")
    
    def analyze(
        self,
        market_data: Optional[Dict] = None,
    ) -> SentimentSummary:
        """
        Analyze overall market sentiment
        
        Args:
            market_data: Optional current market data
            
        Returns:
            SentimentSummary with overall sentiment and signals
        """
        signals: List[SentimentSignal] = []
        
        # 1. Analyze DXY (Dollar) effect on Gold
        dxy_signal = self._analyze_dxy_effect(market_data)
        if dxy_signal:
            signals.append(dxy_signal)
        
        # 2. Analyze VIX/Fear level
        vix_signal = self._analyze_vix_proxy(market_data)
        if vix_signal:
            signals.append(vix_signal)
        
        # 3. Check upcoming high-impact events
        event_signal = self._check_upcoming_events()
        if event_signal:
            signals.append(event_signal)
        
        # 4. Analyze market structure sentiment
        structure_signal = self._analyze_market_structure(market_data)
        if structure_signal:
            signals.append(structure_signal)
        
        # 5. Time-based sentiment (sessions)
        session_signal = self._get_session_sentiment()
        if session_signal:
            signals.append(session_signal)
        
        # Aggregate signals
        return self._aggregate_signals(signals)
    
    def _analyze_dxy_effect(self, market_data: Optional[Dict]) -> Optional[SentimentSignal]:
        """
        Analyze DXY effect on Gold (inverse correlation)
        
        Gold typically moves inversely to USD
        """
        if market_data is None:
            return None
        
        # Use DXY proxy if available (from cross_asset_features)
        dxy_change = market_data.get("dxy_change_pct", 0)
        
        if abs(dxy_change) < 0.1:
            return None  # No significant change
        
        if dxy_change > 0.3:
            # Strong USD = Bearish for Gold
            sentiment = "bearish"
            strength = min(abs(dxy_change) / 1.0, 0.8)
        elif dxy_change < -0.3:
            # Weak USD = Bullish for Gold
            sentiment = "bullish"
            strength = min(abs(dxy_change) / 1.0, 0.8)
        else:
            sentiment = "neutral"
            strength = 0.3
        
        return SentimentSignal(
            source="DXY",
            sentiment=sentiment,
            strength=strength,
            impact="medium",
            event=f"USD {'strengthening' if dxy_change > 0 else 'weakening'}",
            timestamp=datetime.now(),
        )
    
    def _analyze_vix_proxy(self, market_data: Optional[Dict]) -> Optional[SentimentSignal]:
        """
        Analyze VIX/Fear proxy for Gold
        
        High VIX = Risk-off = Bullish for Gold (safe haven)
        """
        if market_data is None:
            return None
        
        volatility = market_data.get("volatility", 0.5)
        
        if volatility > 0.7:
            # High volatility = Fear = Gold bullish
            return SentimentSignal(
                source="VIX_proxy",
                sentiment="bullish",
                strength=0.6,
                impact="medium",
                event="High market volatility (Risk-off)",
                timestamp=datetime.now(),
            )
        elif volatility < 0.3:
            # Low volatility = Complacency = Gold bearish
            return SentimentSignal(
                source="VIX_proxy",
                sentiment="bearish",
                strength=0.4,
                impact="low",
                event="Low volatility (Risk-on)",
                timestamp=datetime.now(),
            )
        
        return None
    
    def _check_upcoming_events(self) -> Optional[SentimentSignal]:
        """
        Check for upcoming high-impact economic events
        
        This uses static schedule - could be enhanced with API
        """
        now = datetime.now()
        hour = now.hour
        day = now.weekday()  # 0=Monday
        
        # Known high-impact times (UTC)
        high_impact_times = [
            # NFP - First Friday 13:30 UTC
            {"day": 4, "hour": 13, "event": "NFP", "impact": "high"},
            # FOMC - Wednesdays 19:00 UTC (8 times/year)
            {"day": 2, "hour": 19, "event": "FOMC", "impact": "high"},
            # CPI - Mid-month 13:30 UTC
            {"day": None, "hour": 13, "event": "CPI", "impact": "high"},
        ]
        
        # Check if we're near a high-impact time
        for event in high_impact_times:
            if event["day"] is not None and day != event["day"]:
                continue
            
            # Within 1 hour of event
            if abs(hour - event["hour"]) <= 1:
                return SentimentSignal(
                    source="Calendar",
                    sentiment="volatile",
                    strength=0.9,
                    impact="high",
                    event=f"Possible {event['event']} announcement",
                    timestamp=datetime.now(),
                )
        
        return None
    
    def _analyze_market_structure(self, market_data: Optional[Dict]) -> Optional[SentimentSignal]:
        """
        Analyze market structure for sentiment
        """
        if market_data is None:
            return None
        
        trend = market_data.get("trend", 0)
        momentum = market_data.get("momentum", 0)
        regime = market_data.get("regime", "unknown")
        
        # Combine trend and momentum
        if trend > 0.5 and momentum > 0.3:
            return SentimentSignal(
                source="Structure",
                sentiment="bullish",
                strength=min((trend + momentum) / 2, 0.8),
                impact="medium",
                event="Strong upward momentum",
                timestamp=datetime.now(),
            )
        elif trend < -0.5 and momentum < -0.3:
            return SentimentSignal(
                source="Structure",
                sentiment="bearish",
                strength=min(abs((trend + momentum) / 2), 0.8),
                impact="medium",
                event="Strong downward momentum",
                timestamp=datetime.now(),
            )
        
        return None
    
    def _get_session_sentiment(self) -> Optional[SentimentSignal]:
        """
        Get sentiment based on trading session
        """
        hour = datetime.now().hour  # UTC
        
        # Convert to trading session
        if 0 <= hour < 8:
            session = "asian"
            sentiment = "neutral"
            strength = 0.3
            event = "Asian session - low liquidity for Gold"
        elif 8 <= hour < 13:
            session = "london"
            sentiment = "neutral"
            strength = 0.5
            event = "London session - moderate activity"
        elif 13 <= hour < 17:
            session = "overlap"
            sentiment = "neutral"
            strength = 0.7
            event = "London/NY overlap - high liquidity"
        elif 17 <= hour < 22:
            session = "newyork"
            sentiment = "neutral"
            strength = 0.6
            event = "New York session - good liquidity"
        else:
            session = "off"
            sentiment = "neutral"
            strength = 0.2
            event = "Off-hours - avoid trading"
        
        return SentimentSignal(
            source="Session",
            sentiment=sentiment,
            strength=strength,
            impact="low",
            event=event,
            timestamp=datetime.now(),
        )
    
    def _aggregate_signals(self, signals: List[SentimentSignal]) -> SentimentSummary:
        """
        Aggregate multiple signals into one summary
        """
        if not signals:
            return SentimentSummary(
                overall="neutral",
                confidence=0.0,
                high_impact_events=[],
                should_avoid=False,
                avoid_reason="",
                signals=[],
            )
        
        # Calculate overall sentiment
        bullish_score = 0.0
        bearish_score = 0.0
        high_impact = []
        should_avoid = False
        avoid_reason = ""
        
        for signal in signals:
            weight = signal.strength
            
            if signal.impact == "high":
                weight *= 1.5
                high_impact.append(signal.event)
                # High-impact volatile events = avoid trading
                if signal.sentiment == "volatile":
                    should_avoid = True
                    avoid_reason = f"High-impact event: {signal.event}"
            
            if signal.sentiment == "bullish":
                bullish_score += weight
            elif signal.sentiment == "bearish":
                bearish_score += weight
        
        # Overall sentiment
        net_score = bullish_score - bearish_score
        
        if net_score > 0.3:
            overall = "bullish"
        elif net_score < -0.3:
            overall = "bearish"
        else:
            overall = "neutral"
        
        # Confidence is magnitude of net score, capped at 1.0
        confidence = min(abs(net_score), 1.0)
        if net_score < 0:
            confidence = -confidence  # Negative for bearish
        
        return SentimentSummary(
            overall=overall,
            confidence=confidence,
            high_impact_events=high_impact,
            should_avoid=should_avoid,
            avoid_reason=avoid_reason,
            signals=signals,
        )
    
    def adjust_confidence(
        self,
        base_confidence: float,
        action: str,  # 'LONG' or 'SHORT'
        sentiment: SentimentSummary,
    ) -> Tuple[float, str]:
        """
        Adjust trading confidence based on sentiment
        
        Returns:
            (adjusted_confidence, reason)
        """
        if sentiment.should_avoid:
            return 0.0, sentiment.avoid_reason
        
        adjustment = 0.0
        reasons = []
        
        # Sentiment alignment bonus/penalty
        if action == "LONG" and sentiment.overall == "bullish":
            adjustment += sentiment.confidence * 0.1
            reasons.append("Sentiment aligned")
        elif action == "LONG" and sentiment.overall == "bearish":
            adjustment -= abs(sentiment.confidence) * 0.15
            reasons.append("Sentiment against")
        elif action == "SHORT" and sentiment.overall == "bearish":
            adjustment += abs(sentiment.confidence) * 0.1
            reasons.append("Sentiment aligned")
        elif action == "SHORT" and sentiment.overall == "bullish":
            adjustment -= sentiment.confidence * 0.15
            reasons.append("Sentiment against")
        
        # High-impact events reduce confidence
        if sentiment.high_impact_events:
            adjustment -= 0.10
            reasons.append(f"High-impact: {len(sentiment.high_impact_events)}")
        
        adjusted = max(0.0, min(1.0, base_confidence + adjustment))
        
        return adjusted, " | ".join(reasons) if reasons else "No adjustment"
    
    def get_status(self) -> Dict:
        """Get current sentiment status"""
        summary = self.analyze(None)
        return {
            "overall": summary.overall,
            "confidence": summary.confidence,
            "should_avoid": summary.should_avoid,
            "avoid_reason": summary.avoid_reason,
            "high_impact_events": summary.high_impact_events,
            "signal_count": len(summary.signals),
        }


def create_sentiment_integration() -> SentimentIntegration:
    """Factory function"""
    return SentimentIntegration()


# Singleton
_sentiment_instance: Optional[SentimentIntegration] = None


def get_sentiment() -> SentimentIntegration:
    """Get singleton instance"""
    global _sentiment_instance
    if _sentiment_instance is None:
        _sentiment_instance = SentimentIntegration()
    return _sentiment_instance


# Test
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("=" * 60)
    print("   SENTIMENT INTEGRATION TEST")
    print("=" * 60)
    
    sentiment = create_sentiment_integration()
    
    # Test with sample market data
    market_data = {
        "trend": 0.6,
        "momentum": 0.4,
        "volatility": 0.5,
        "regime": "trending_up",
        "dxy_change_pct": -0.5,  # USD weakening
    }
    
    summary = sentiment.analyze(market_data)
    
    print(f"\nOverall Sentiment: {summary.overall}")
    print(f"Confidence: {summary.confidence:.2f}")
    print(f"Should Avoid: {summary.should_avoid}")
    if summary.avoid_reason:
        print(f"Avoid Reason: {summary.avoid_reason}")
    
    print(f"\nSignals ({len(summary.signals)}):")
    for signal in summary.signals:
        print(f"  [{signal.source}] {signal.sentiment} ({signal.strength:.0%}) - {signal.event}")
    
    # Test confidence adjustment
    print("\n\nConfidence Adjustment Test:")
    for action in ["LONG", "SHORT"]:
        adj, reason = sentiment.adjust_confidence(0.70, action, summary)
        print(f"  {action}: 70% -> {adj:.0%} ({reason})")
