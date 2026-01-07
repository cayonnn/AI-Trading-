"""
AI Advanced Features
=====================
ฟีเจอร์ขั้นสูงสำหรับ AI Trading

Components:
1. Multi-Timeframe Analysis - วิเคราะห์หลาย timeframe
2. News/Event Filter - หลีกเลี่ยง high-impact news
3. Correlation Analysis - ดูความสัมพันธ์กับตลาดอื่น
4. Session Filter - เทรดเฉพาะช่วงเวลาที่ดี
5. Advanced Signal Generator - รวมทุกอย่างเข้าด้วยกัน
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class TimeFrame(Enum):
    """Timeframes"""
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class TradingSession(Enum):
    """Trading sessions"""
    ASIAN = "asian"          # 00:00 - 08:00 GMT
    EUROPEAN = "european"    # 08:00 - 16:00 GMT
    AMERICAN = "american"    # 13:00 - 21:00 GMT
    OVERLAP_EU_US = "overlap"  # 13:00 - 16:00 GMT (best for gold)


class NewsImpact(Enum):
    """News impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"


@dataclass
class MarketContext:
    """บริบทตลาดแบบครบถ้วน"""
    # Multi-timeframe trends
    trend_m15: int  # 1=up, 0=neutral, -1=down
    trend_h1: int
    trend_h4: int
    trend_d1: int
    trend_alignment: float  # 0-1, 1=all aligned
    
    # Volatility
    volatility_h1: float
    volatility_percentile: float  # 0-100
    atr: float
    
    # Session
    current_session: TradingSession
    session_quality: float  # 0-1
    
    # Correlations
    dxy_correlation: float
    oil_correlation: float
    spx_correlation: float
    
    # News
    has_high_impact_news: bool
    hours_to_news: int
    
    # Overall
    trade_allowed: bool
    confidence_multiplier: float
    recommended_action: str


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Analysis
    =========================
    วิเคราะห์หลาย timeframe เพื่อหาจุด confluence
    """
    
    MT5_TIMEFRAMES = {
        TimeFrame.M15: "TIMEFRAME_M15",
        TimeFrame.H1: "TIMEFRAME_H1",
        TimeFrame.H4: "TIMEFRAME_H4",
        TimeFrame.D1: "TIMEFRAME_D1",
    }
    
    def __init__(self, symbol: str = "GOLD"):
        self.symbol = symbol
        self.data_cache = {}
    
    def get_data(self, timeframe: TimeFrame, bars: int = 200) -> Optional[pd.DataFrame]:
        """ดึงข้อมูลจาก MT5"""
        if not MT5_AVAILABLE:
            return None
        
        if not mt5.initialize():
            return None
        
        tf_map = {
            TimeFrame.M15: mt5.TIMEFRAME_M15,
            TimeFrame.H1: mt5.TIMEFRAME_H1,
            TimeFrame.H4: mt5.TIMEFRAME_H4,
            TimeFrame.D1: mt5.TIMEFRAME_D1,
        }
        
        rates = mt5.copy_rates_from_pos(self.symbol, tf_map[timeframe], 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'})
        
        self.data_cache[timeframe] = df
        return df
    
    def analyze_trend(self, df: pd.DataFrame) -> int:
        """วิเคราะห์ trend: 1=up, 0=neutral, -1=down"""
        if df is None or len(df) < 50:
            return 0
        
        close = df['close']
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        
        # Price above both MAs = uptrend
        if current > ma20 > ma50:
            return 1
        # Price below both MAs = downtrend
        elif current < ma20 < ma50:
            return -1
        else:
            return 0
    
    def get_multi_timeframe_signal(self) -> Dict:
        """วิเคราะห์ทุก timeframe"""
        trends = {}
        
        for tf in [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]:
            df = self.get_data(tf)
            if df is not None:
                trends[tf.value] = self.analyze_trend(df)
            else:
                trends[tf.value] = 0
        
        # Calculate alignment
        trend_values = list(trends.values())
        if all(t == 1 for t in trend_values):
            alignment = 1.0  # All bullish
        elif all(t == -1 for t in trend_values):
            alignment = 1.0  # All bearish
        elif all(t >= 0 for t in trend_values) and 1 in trend_values:
            alignment = 0.7  # Mostly bullish
        elif all(t <= 0 for t in trend_values) and -1 in trend_values:
            alignment = 0.7  # Mostly bearish
        else:
            alignment = 0.0  # Mixed signals
        
        # Dominant trend
        avg_trend = sum(trend_values) / len(trend_values)
        if avg_trend > 0.25:
            dominant = "BULLISH"
        elif avg_trend < -0.25:
            dominant = "BEARISH"
        else:
            dominant = "NEUTRAL"
        
        return {
            'trends': trends,
            'alignment': alignment,
            'dominant': dominant,
            'score': abs(avg_trend),  # 0-1
        }


class NewsEventFilter:
    """
    News & Event Filter
    ====================
    หลีกเลี่ยงการเทรดช่วง high-impact news
    """
    
    # Major economic events that affect gold
    HIGH_IMPACT_EVENTS = [
        "NFP",           # Non-Farm Payrolls
        "FOMC",          # Federal Reserve
        "CPI",           # Consumer Price Index
        "PPI",           # Producer Price Index
        "GDP",           # Gross Domestic Product
        "Interest Rate", # Rate decisions
        "Unemployment",  # Jobless claims
        "Retail Sales",  # Retail sales
        "PMI",           # Purchasing Manager Index
    ]
    
    # Avoid trading X hours before/after high impact news
    BUFFER_HOURS_BEFORE = 2
    BUFFER_HOURS_AFTER = 1
    
    def __init__(self, calendar_file: str = "ai_agent/data/economic_calendar.json"):
        self.calendar_file = calendar_file
        self.events = []
        self._load_calendar()
    
    def _load_calendar(self):
        """Load economic calendar from file"""
        if os.path.exists(self.calendar_file):
            try:
                with open(self.calendar_file, 'r') as f:
                    self.events = json.load(f)
            except:
                self.events = []
    
    def add_event(self, event_time: datetime, event_name: str, impact: str = "high"):
        """Add an event to calendar"""
        self.events.append({
            'time': event_time.isoformat(),
            'name': event_name,
            'impact': impact,
        })
        self._save_calendar()
    
    def _save_calendar(self):
        """Save calendar to file"""
        os.makedirs(os.path.dirname(self.calendar_file), exist_ok=True)
        with open(self.calendar_file, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def check_news_filter(self, check_time: datetime = None) -> Tuple[bool, Dict]:
        """
        ตรวจสอบว่ามี high-impact news ใกล้ๆ หรือไม่
        
        Returns: (can_trade, info)
        """
        if check_time is None:
            check_time = datetime.now()
        
        # Weekly high-impact times (GMT)
        # These are typical times for major US economic releases
        
        # NFP - First Friday of month, 13:30 GMT
        # FOMC - Variable, usually mid-month Wed
        # CPI - Usually mid-month
        
        result = {
            'has_high_impact': False,
            'next_event': None,
            'hours_to_event': 999,
            'can_trade': True,
            'reason': "No high-impact news nearby",
        }
        
        # Check day of week
        day = check_time.weekday()
        hour = check_time.hour
        day_of_month = check_time.day
        
        # First Friday of month - likely NFP
        if day == 4 and day_of_month <= 7:  # Friday
            if 11 <= hour <= 16:  # Around NFP time (13:30 GMT)
                result['has_high_impact'] = True
                result['can_trade'] = False
                result['reason'] = "NFP release window - avoid trading"
        
        # Wednesday mid-month - possible FOMC
        if day == 2 and 12 <= day_of_month <= 20:
            if 18 <= hour <= 21:  # FOMC usually 18:00-20:00 GMT
                result['has_high_impact'] = True
                result['can_trade'] = False
                result['reason'] = "Possible FOMC window - avoid trading"
        
        # Check stored events
        for event in self.events:
            try:
                event_time = datetime.fromisoformat(event['time'])
                diff = (event_time - check_time).total_seconds() / 3600
                
                if event.get('impact', 'medium') == 'high':
                    if -self.BUFFER_HOURS_AFTER <= diff <= self.BUFFER_HOURS_BEFORE:
                        result['has_high_impact'] = True
                        result['can_trade'] = False
                        result['next_event'] = event['name']
                        result['hours_to_event'] = diff
                        result['reason'] = f"High-impact event: {event['name']}"
                        break
                    
                    if 0 < diff < result['hours_to_event']:
                        result['hours_to_event'] = diff
                        result['next_event'] = event['name']
            except:
                continue
        
        return result['can_trade'], result


class SessionAnalyzer:
    """
    Trading Session Analyzer
    =========================
    วิเคราะห์ช่วงเวลาที่ดีที่สุดสำหรับเทรด Gold
    """
    
    # Gold trading sessions (GMT)
    SESSIONS = {
        TradingSession.ASIAN: (0, 8),       # 00:00-08:00
        TradingSession.EUROPEAN: (8, 16),   # 08:00-16:00
        TradingSession.AMERICAN: (13, 21),  # 13:00-21:00
        TradingSession.OVERLAP_EU_US: (13, 16),  # Best time
    }
    
    # Session quality for Gold (0-1)
    SESSION_QUALITY = {
        TradingSession.ASIAN: 0.6,
        TradingSession.EUROPEAN: 0.8,
        TradingSession.AMERICAN: 0.9,
        TradingSession.OVERLAP_EU_US: 1.0,  # Best
    }
    
    def get_current_session(self, current_time: datetime = None) -> Tuple[TradingSession, float]:
        """Get current trading session and quality"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        hour = current_time.hour
        
        # Check for best overlap first
        if 13 <= hour < 16:
            return TradingSession.OVERLAP_EU_US, self.SESSION_QUALITY[TradingSession.OVERLAP_EU_US]
        
        # Check other sessions
        if 0 <= hour < 8:
            return TradingSession.ASIAN, self.SESSION_QUALITY[TradingSession.ASIAN]
        elif 8 <= hour < 16:
            return TradingSession.EUROPEAN, self.SESSION_QUALITY[TradingSession.EUROPEAN]
        elif 13 <= hour < 21:
            return TradingSession.AMERICAN, self.SESSION_QUALITY[TradingSession.AMERICAN]
        else:
            return TradingSession.ASIAN, 0.4  # Late night = low quality


class CorrelationAnalyzer:
    """
    Correlation Analysis
    =====================
    วิเคราะห์ความสัมพันธ์กับตลาดอื่น
    
    Gold correlations:
    - USD (DXY): Negative correlation (-0.8 typical)
    - S&P 500: Mixed, positive in crisis
    - Oil: Positive correlation (commodities)
    - Bonds: Positive correlation (safe haven)
    """
    
    # Typical historical correlations
    TYPICAL_CORRELATIONS = {
        'DXY': -0.8,     # US Dollar Index
        'SPX': 0.2,      # S&P 500
        'OIL': 0.5,      # Crude Oil
        'BONDS': 0.6,    # US Bonds
    }
    
    def __init__(self, symbol: str = "GOLD"):
        self.symbol = symbol
        self.correlation_data = {}
    
    def analyze_dxy_correlation(self, gold_df: pd.DataFrame, dxy_df: pd.DataFrame = None) -> float:
        """วิเคราะห์ correlation กับ DXY"""
        if dxy_df is None:
            # Return typical value if no DXY data
            return self.TYPICAL_CORRELATIONS['DXY']
        
        if len(gold_df) < 20 or len(dxy_df) < 20:
            return self.TYPICAL_CORRELATIONS['DXY']
        
        # Calculate returns correlation
        gold_returns = gold_df['close'].pct_change().dropna()
        dxy_returns = dxy_df['close'].pct_change().dropna()
        
        # Align data
        min_len = min(len(gold_returns), len(dxy_returns))
        correlation = np.corrcoef(
            gold_returns.tail(min_len),
            dxy_returns.tail(min_len)
        )[0, 1]
        
        return correlation if not np.isnan(correlation) else self.TYPICAL_CORRELATIONS['DXY']
    
    def get_correlation_signal(self, gold_df: pd.DataFrame) -> Dict:
        """Get trading signal based on correlations"""
        
        # Simplified: use typical correlations
        # In production, would fetch real DXY, SPX, OIL data
        
        return {
            'dxy_correlation': self.TYPICAL_CORRELATIONS['DXY'],
            'spx_correlation': self.TYPICAL_CORRELATIONS['SPX'],
            'oil_correlation': self.TYPICAL_CORRELATIONS['OIL'],
            'signal': 'neutral',  # Would be calculated from actual data
            'confidence': 0.5,
        }


class AdvancedSignalGenerator:
    """
    Advanced Signal Generator
    ==========================
    รวมทุก component เข้าด้วยกัน
    """
    
    def __init__(self, symbol: str = "GOLD"):
        self.symbol = symbol
        
        # Components
        self.mtf_analyzer = MultiTimeframeAnalyzer(symbol)
        self.news_filter = NewsEventFilter()
        self.session_analyzer = SessionAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(symbol)
        
        logger.info("AdvancedSignalGenerator initialized")
    
    def get_full_context(self, h1_df: pd.DataFrame = None) -> MarketContext:
        """Get complete market context"""
        
        current_time = datetime.utcnow()
        
        # 1. Multi-timeframe analysis
        mtf_result = self.mtf_analyzer.get_multi_timeframe_signal()
        
        # 2. News filter
        can_trade_news, news_info = self.news_filter.check_news_filter(current_time)
        
        # 3. Session analysis
        current_session, session_quality = self.session_analyzer.get_current_session(current_time)
        
        # 4. Correlation analysis
        corr_result = self.correlation_analyzer.get_correlation_signal(h1_df)
        
        # 5. Volatility analysis
        if h1_df is not None and len(h1_df) > 20:
            returns = h1_df['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * 100
            
            # Historical percentile
            hist_vol = returns.rolling(20).std() * 100
            vol_percentile = (hist_vol < volatility).mean() * 100
            
            # ATR
            high_low = h1_df['high'] - h1_df['low']
            atr = high_low.rolling(14).mean().iloc[-1]
        else:
            volatility = 1.0
            vol_percentile = 50
            atr = 10.0
        
        # 6. Calculate confidence multiplier
        confidence_factors = [
            mtf_result['alignment'],          # 0-1
            session_quality,                   # 0-1
            1.0 if can_trade_news else 0.0,   # 0-1
        ]
        confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
        
        # 7. Determine if trading is allowed
        trade_allowed = (
            can_trade_news and
            session_quality >= 0.5 and
            mtf_result['alignment'] >= 0.5
        )
        
        # 8. Recommended action
        if not trade_allowed:
            if not can_trade_news:
                recommended = "WAIT - High impact news nearby"
            elif session_quality < 0.5:
                recommended = "WAIT - Poor session quality"
            else:
                recommended = "WAIT - No trend alignment"
        else:
            if mtf_result['dominant'] == "BULLISH":
                recommended = "LOOK_FOR_BUY"
            elif mtf_result['dominant'] == "BEARISH":
                recommended = "LOOK_FOR_SELL"
            else:
                recommended = "WAIT - No clear trend"
        
        return MarketContext(
            trend_m15=mtf_result['trends'].get('M15', 0),
            trend_h1=mtf_result['trends'].get('H1', 0),
            trend_h4=mtf_result['trends'].get('H4', 0),
            trend_d1=mtf_result['trends'].get('D1', 0),
            trend_alignment=mtf_result['alignment'],
            volatility_h1=volatility,
            volatility_percentile=vol_percentile,
            atr=atr,
            current_session=current_session,
            session_quality=session_quality,
            dxy_correlation=corr_result['dxy_correlation'],
            oil_correlation=corr_result['oil_correlation'],
            spx_correlation=corr_result['spx_correlation'],
            has_high_impact_news=news_info.get('has_high_impact', False),
            hours_to_news=int(news_info.get('hours_to_event', 999)),
            trade_allowed=trade_allowed,
            confidence_multiplier=confidence_multiplier,
            recommended_action=recommended,
        )
    
    def should_trade(
        self,
        ai_action: int,
        ai_confidence: float,
        h1_df: pd.DataFrame = None,
    ) -> Tuple[bool, Dict]:
        """
        ตัดสินใจว่าควรเทรดหรือไม่
        
        Returns: (should_trade, details)
        """
        context = self.get_full_context(h1_df)
        
        details = {
            'context': context,
            'ai_action': ai_action,
            'ai_confidence': ai_confidence,
            'final_confidence': ai_confidence * context.confidence_multiplier,
            'should_trade': False,
            'reason': "",
        }
        
        # Check basic conditions
        if not context.trade_allowed:
            details['reason'] = context.recommended_action
            return False, details
        
        # Check AI confidence
        final_confidence = ai_confidence * context.confidence_multiplier
        min_confidence = 0.6
        
        if final_confidence < min_confidence:
            details['reason'] = f"Confidence too low ({final_confidence:.1%} < {min_confidence:.1%})"
            return False, details
        
        # Check trend alignment for entry
        if ai_action == 1:  # BUY
            if context.trend_d1 == -1:  # D1 downtrend
                details['reason'] = "Avoid BUY - Daily trend is DOWN"
                return False, details
            
            if context.trend_h4 == -1 and context.trend_h1 == -1:
                details['reason'] = "Avoid BUY - H4 and H1 trends are DOWN"
                return False, details
        
        # All checks passed
        details['should_trade'] = True
        details['reason'] = f"All conditions met (confidence: {final_confidence:.1%})"
        
        return True, details
    
    def get_enhanced_parameters(
        self,
        base_sl: float,
        base_tp: float,
        h1_df: pd.DataFrame = None,
    ) -> Dict:
        """
        ปรับ SL/TP ตาม advanced analysis
        """
        context = self.get_full_context(h1_df)
        
        # Adjust based on volatility
        if context.volatility_percentile > 80:
            # High volatility = wider SL
            sl_multiplier = 1.5
            tp_multiplier = 1.3
        elif context.volatility_percentile < 20:
            # Low volatility = tighter SL
            sl_multiplier = 0.8
            tp_multiplier = 0.9
        else:
            sl_multiplier = 1.0
            tp_multiplier = 1.0
        
        # Adjust based on session
        if context.session_quality >= 0.9:
            # Best session = can afford tighter SL
            sl_multiplier *= 0.9
        
        # Adjust based on trend alignment
        if context.trend_alignment >= 0.8:
            # Strong alignment = extend TP
            tp_multiplier *= 1.2
        
        return {
            'sl_pips': base_sl * sl_multiplier,
            'tp_pips': base_tp * tp_multiplier,
            'sl_multiplier': sl_multiplier,
            'tp_multiplier': tp_multiplier,
            'context': context,
        }


# Test
if __name__ == "__main__":
    print("="*60)
    print("   ADVANCED SIGNAL GENERATOR - TEST")
    print("="*60)
    
    generator = AdvancedSignalGenerator("GOLD")
    
    # Create dummy data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='H')
    df = pd.DataFrame({
        'datetime': dates,
        'open': 2000 + np.random.randn(200).cumsum(),
        'high': 2005 + np.random.randn(200).cumsum(),
        'low': 1995 + np.random.randn(200).cumsum(),
        'close': 2000 + np.random.randn(200).cumsum(),
        'volume': np.random.randint(100, 1000, 200),
    })
    
    # Get context
    context = generator.get_full_context(df)
    
    print(f"\n📊 Market Context:")
    print(f"   Trend M15/H1/H4/D1: {context.trend_m15}/{context.trend_h1}/{context.trend_h4}/{context.trend_d1}")
    print(f"   Trend Alignment: {context.trend_alignment:.1%}")
    print(f"   Session: {context.current_session.value} (Quality: {context.session_quality:.1%})")
    print(f"   Volatility: {context.volatility_h1:.2f}% (Percentile: {context.volatility_percentile:.0f}%)")
    print(f"   News Nearby: {context.has_high_impact_news}")
    print(f"   Trade Allowed: {context.trade_allowed}")
    print(f"   Confidence Multiplier: {context.confidence_multiplier:.1%}")
    print(f"   Recommended: {context.recommended_action}")
    
    # Test should_trade
    should, details = generator.should_trade(ai_action=1, ai_confidence=0.7, h1_df=df)
    print(f"\n🎯 Should Trade: {should}")
    print(f"   Reason: {details['reason']}")
    
    print("\n" + "="*60)
    print("   TEST COMPLETE!")
    print("="*60)
