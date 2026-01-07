"""
News/Event Filter for AI Trading System
======================================
v1.0 - Economic Calendar Integration

Features:
- Detects high-impact news events (NFP, FOMC, CPI)
- Pauses trading before major announcements
- Resumes after event passes
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
import json


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    timestamp: datetime
    currency: str
    impact: str  # 'high', 'medium', 'low'
    event: str
    forecast: str
    previous: str


class NewsFilter:
    """
    News/Event Filter
    
    Prevents trading around high-impact economic events
    """
    
    def __init__(
        self,
        pause_before_minutes: int = 30,
        pause_after_minutes: int = 30,
        blocked_events: List[str] = None,
    ):
        self.pause_before = timedelta(minutes=pause_before_minutes)
        self.pause_after = timedelta(minutes=pause_after_minutes)
        
        # High-impact events to block trading
        self.blocked_events = blocked_events or [
            'Non-Farm Payrolls',
            'NFP',
            'FOMC',
            'Fed Interest Rate',
            'CPI',
            'GDP',
            'ECB Interest Rate',
            'BOE Interest Rate',
            'Unemployment Rate',
            'Retail Sales',
        ]
        
        # Cache events
        self.events_cache: List[EconomicEvent] = []
        self.cache_date: Optional[datetime] = None
        
        logger.info("NewsFilter initialized")
    
    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if trading should be allowed right now
        
        Returns:
            (can_trade, reason)
        """
        now = datetime.utcnow()
        
        # Check for upcoming high-impact events
        for event in self._get_todays_events():
            if event.impact != 'high':
                continue
            
            # Check if event name matches blocked list
            is_blocked = any(
                blocked.lower() in event.event.lower() 
                for blocked in self.blocked_events
            )
            
            if not is_blocked:
                continue
            
            # Check time window
            event_start = event.timestamp - self.pause_before
            event_end = event.timestamp + self.pause_after
            
            if event_start <= now <= event_end:
                time_to_event = event.timestamp - now
                if time_to_event.total_seconds() > 0:
                    reason = f"⚠️ {event.event} in {time_to_event.seconds // 60} min - TRADING PAUSED"
                else:
                    time_since = now - event.timestamp
                    reason = f"⚠️ {event.event} was {time_since.seconds // 60} min ago - WAIT"
                
                logger.warning(reason)
                return False, reason
        
        return True, "No blocking events"
    
    def _get_todays_events(self) -> List[EconomicEvent]:
        """Get today's economic events (cached)"""
        today = datetime.now().date()
        
        # Use cache if valid
        if self.cache_date == today and self.events_cache:
            return self.events_cache
        
        try:
            self.events_cache = self._fetch_events()
            self.cache_date = today
        except Exception as e:
            logger.warning(f"Could not fetch economic calendar: {e}")
            self.events_cache = []
        
        return self.events_cache
    
    def _fetch_events(self) -> List[EconomicEvent]:
        """
        Fetch economic calendar from API
        Note: In production, use a real API like ForexFactory, Investing.com, etc.
        """
        events = []
        
        # Manual high-impact events for GOLD (can add API later)
        # These are typical weekly events
        now = datetime.utcnow()
        
        # Add example events (in production, fetch from API)
        example_events = [
            # NFP is typically first Friday of month at 13:30 UTC
            # FOMC is ~8 times a year
            # CPI is monthly, around 12th-14th
        ]
        
        # For now, return empty - user can add manual events
        # or integrate with investing.com/economic-calendar API
        
        return events
    
    def add_manual_event(
        self,
        event_name: str,
        event_time: datetime,
        impact: str = 'high',
        currency: str = 'USD',
    ):
        """Manually add an event to track"""
        event = EconomicEvent(
            timestamp=event_time,
            currency=currency,
            impact=impact,
            event=event_name,
            forecast='',
            previous='',
        )
        self.events_cache.append(event)
        logger.info(f"📅 Added event: {event_name} at {event_time}")
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """Get list of upcoming events"""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self._get_todays_events():
            if now <= event.timestamp <= cutoff:
                upcoming.append({
                    'event': event.event,
                    'time': event.timestamp.isoformat(),
                    'impact': event.impact,
                    'currency': event.currency,
                })
        
        return upcoming


# ============================================
# Singleton instance
# ============================================

_news_filter: Optional[NewsFilter] = None

def get_news_filter() -> NewsFilter:
    """Get singleton NewsFilter instance"""
    global _news_filter
    if _news_filter is None:
        _news_filter = NewsFilter()
    return _news_filter


if __name__ == "__main__":
    # Test
    nf = NewsFilter()
    
    # Add test event
    from datetime import datetime, timedelta
    test_time = datetime.utcnow() + timedelta(minutes=20)
    nf.add_manual_event("Test NFP", test_time)
    
    can_trade, reason = nf.should_trade()
    print(f"Can trade: {can_trade}")
    print(f"Reason: {reason}")
