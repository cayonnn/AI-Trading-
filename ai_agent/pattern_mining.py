"""
Pattern Mining System for MasterBrain Pre-training
===================================================
ขุด patterns จาก historical data เพื่อ pre-train MasterBrain

Features:
1. Candlestick Pattern Recognition
2. Regime + Hour + Volatility Analysis
3. Indicator Confluence Detection
4. Win Rate Calculation for Each Pattern
5. Pre-seed MasterBrain with Good/Bad Patterns
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger

# Import indicators if available
try:
    from data.indicators import TechnicalIndicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False


@dataclass
class TradingPattern:
    """Pattern ที่ขุดได้"""
    pattern_id: str
    pattern_type: str  # "candlestick", "regime", "indicator", "confluence"
    name: str
    description: str
    
    # Statistics
    occurrences: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    
    # Conditions
    conditions: Dict[str, Any] = None
    
    # Classification
    is_good: bool = False  # Win rate > 55%
    is_bad: bool = False   # Win rate < 40%
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict) -> 'TradingPattern':
        return TradingPattern(**d)


class PatternMiner:
    """
    ขุด Patterns จาก Historical Data
    
    วิเคราะห์:
    1. Candlestick patterns (Doji, Hammer, Engulfing, etc.)
    2. Market regime patterns
    3. Time-based patterns (hour, day of week, session)
    4. Indicator confluence patterns
    5. Price action patterns
    """
    
    def __init__(
        self,
        data_path: str = "data/training/GOLD_H1.csv",
        lookback_bars: int = 5,
        forward_bars: int = 10,
        min_occurrences: int = 30,
        good_win_rate: float = 0.55,
        bad_win_rate: float = 0.40,
    ):
        self.data_path = data_path
        self.lookback_bars = lookback_bars
        self.forward_bars = forward_bars
        self.min_occurrences = min_occurrences
        self.good_win_rate = good_win_rate
        self.bad_win_rate = bad_win_rate
        
        # Pattern storage
        self.patterns: Dict[str, TradingPattern] = {}
        self.regime_patterns: Dict[str, Dict] = {}
        self.hour_patterns: Dict[int, Dict] = {}
        self.day_patterns: Dict[int, Dict] = {}
        self.volatility_patterns: Dict[str, Dict] = {}
        self.confluence_patterns: Dict[str, Dict] = {}
        
        # Load data
        self.df = None
        self._load_data()
        
        logger.info(f"PatternMiner initialized with {len(self.df)} bars")
    
    def _load_data(self):
        """โหลดและเตรียม data"""
        self.df = pd.read_csv(self.data_path)
        self.df.columns = [c.lower() for c in self.df.columns]
        
        # Parse datetime
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'], utc=True)
            self.df.set_index('datetime', inplace=True)
        
        # Extract time features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        
        # Calculate returns
        self.df['returns'] = self.df['close'].pct_change()
        
        # Calculate basic indicators
        self._calculate_indicators()
        
        logger.info(f"Loaded {len(self.df)} bars from {self.data_path}")

    
    def _calculate_indicators(self):
        """คำนวณ indicators พื้นฐาน"""
        df = self.df
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Volatility (normalized)
        df['volatility'] = df['atr'] / df['close']
        df['volatility_level'] = pd.cut(
            df['volatility'],
            bins=[0, 0.005, 0.015, 0.025, 1.0],
            labels=['very_low', 'low', 'medium', 'high']
        )
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple trend
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Regime
        df['regime'] = 'ranging'
        df.loc[(df['trend'] == 1) & (df['volatility'] < 0.015), 'regime'] = 'trending_up_low_vol'
        df.loc[(df['trend'] == 1) & (df['volatility'] >= 0.015), 'regime'] = 'trending_up_high_vol'
        df.loc[(df['trend'] == -1) & (df['volatility'] < 0.015), 'regime'] = 'trending_down_low_vol'
        df.loc[(df['trend'] == -1) & (df['volatility'] >= 0.015), 'regime'] = 'trending_down_high_vol'
        
        # Candlestick properties
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = (df['close'] > df['open']).astype(bool)
        
        # Session
        df['session'] = 'asia'
        df.loc[df['hour'].between(8, 15), 'session'] = 'london'
        df.loc[df['hour'].between(13, 21), 'session'] = 'newyork'
        
        # Future returns for labeling
        df['future_return'] = df['close'].shift(-self.forward_bars) / df['close'] - 1
        df['future_max'] = df['high'].rolling(self.forward_bars).max().shift(-self.forward_bars)
        df['future_min'] = df['low'].rolling(self.forward_bars).min().shift(-self.forward_bars)
        
        # Label: 1 = good long, 0 = wait, -1 = good short
        threshold = 0.002  # 0.2% profit
        df['label'] = 0
        df.loc[df['future_return'] > threshold, 'label'] = 1
        df.loc[df['future_return'] < -threshold, 'label'] = -1
    
    def mine_all_patterns(self) -> Dict[str, List[TradingPattern]]:
        """ขุด patterns ทั้งหมด"""
        logger.info("="*60)
        logger.info("   PATTERN MINING")
        logger.info("="*60)
        
        # 1. Mine candlestick patterns
        candlestick_patterns = self._mine_candlestick_patterns()
        logger.info(f"Candlestick patterns: {len(candlestick_patterns)}")
        
        # 2. Mine regime patterns
        regime_patterns = self._mine_regime_patterns()
        logger.info(f"Regime patterns: {len(regime_patterns)}")
        
        # 3. Mine hour patterns
        hour_patterns = self._mine_hour_patterns()
        logger.info(f"Hour patterns: {len(hour_patterns)}")
        
        # 4. Mine day patterns
        day_patterns = self._mine_day_patterns()
        logger.info(f"Day patterns: {len(day_patterns)}")
        
        # 5. Mine session patterns
        session_patterns = self._mine_session_patterns()
        logger.info(f"Session patterns: {len(session_patterns)}")
        
        # 6. Mine RSI patterns
        rsi_patterns = self._mine_rsi_patterns()
        logger.info(f"RSI patterns: {len(rsi_patterns)}")
        
        # 7. Mine confluence patterns
        confluence_patterns = self._mine_confluence_patterns()
        logger.info(f"Confluence patterns: {len(confluence_patterns)}")
        
        # Combine all
        all_patterns = {
            'candlestick': candlestick_patterns,
            'regime': regime_patterns,
            'hour': hour_patterns,
            'day': day_patterns,
            'session': session_patterns,
            'rsi': rsi_patterns,
            'confluence': confluence_patterns,
        }
        
        # Summary
        total_good = sum(1 for p_list in all_patterns.values() for p in p_list if p.is_good)
        total_bad = sum(1 for p_list in all_patterns.values() for p in p_list if p.is_bad)
        
        logger.info("="*60)
        logger.info(f"   MINING COMPLETE")
        logger.info(f"   Good patterns (WR > {self.good_win_rate:.0%}): {total_good}")
        logger.info(f"   Bad patterns (WR < {self.bad_win_rate:.0%}): {total_bad}")
        logger.info("="*60)
        
        return all_patterns
    
    def _mine_candlestick_patterns(self) -> List[TradingPattern]:
        """ขุด candlestick patterns"""
        patterns = []
        df = self.df.copy()
        # Only keep rows with valid labels
        df = df[df['label'].notna()]
        
        # Ensure is_bullish is boolean
        df['is_bullish'] = df['is_bullish'].fillna(False).astype(bool)
        
        # Doji
        doji_mask = df['body'] < df['atr'] * 0.1
        patterns.append(self._create_pattern_from_mask(
            'candlestick_doji', 'candlestick', 'Doji',
            'แท่งเทียนที่ open = close (ตลาดลังเล)',
            doji_mask, df
        ))
        
        # Hammer (bullish)
        is_bearish = ~df['is_bullish']
        hammer_mask = (
            (df['lower_shadow'] > df['body'] * 2) &
            (df['upper_shadow'] < df['body'] * 0.5) &
            is_bearish
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_hammer', 'candlestick', 'Hammer',
            'แท่งค้อน - สัญญาณกลับตัวขึ้น',
            hammer_mask, df
        ))
        
        # Shooting Star (bearish)
        star_mask = (
            (df['upper_shadow'] > df['body'] * 2) &
            (df['lower_shadow'] < df['body'] * 0.5) &
            df['is_bullish']
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_shooting_star', 'candlestick', 'Shooting Star',
            'ดาวตก - สัญญาณกลับตัวลง',
            star_mask, df
        ))
        
        # Bullish Engulfing
        prev_bearish = (~df['is_bullish']).shift(1).fillna(False)
        bull_engulf_mask = (
            df['is_bullish'] &
            prev_bearish &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_bull_engulfing', 'candlestick', 'Bullish Engulfing',
            'กลืนขึ้น - สัญญาณขาขึ้น',
            bull_engulf_mask, df
        ))
        
        # Bearish Engulfing
        prev_bullish = df['is_bullish'].shift(1).fillna(False)
        bear_engulf_mask = (
            (~df['is_bullish']) &
            prev_bullish &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_bear_engulfing', 'candlestick', 'Bearish Engulfing',
            'กลืนลง - สัญญาณขาลง',
            bear_engulf_mask, df
        ))
        
        # Big bullish candle
        big_bull_mask = (
            df['is_bullish'] &
            (df['body'] > df['atr'] * 1.5)
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_big_bullish', 'candlestick', 'Big Bullish',
            'แท่งเขียวใหญ่ - momentum ขาขึ้น',
            big_bull_mask, df
        ))
        
        # Big bearish candle
        big_bear_mask = (
            (~df['is_bullish']) &
            (df['body'] > df['atr'] * 1.5)
        )
        patterns.append(self._create_pattern_from_mask(
            'candlestick_big_bearish', 'candlestick', 'Big Bearish',
            'แท่งแดงใหญ่ - momentum ขาลง',
            big_bear_mask, df
        ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_regime_patterns(self) -> List[TradingPattern]:
        """ขุด regime patterns"""
        patterns = []
        df = self.df[self.df['label'].notna() & self.df['regime'].notna()]
        
        for regime in df['regime'].unique():
            mask = df['regime'] == regime
            patterns.append(self._create_pattern_from_mask(
                f'regime_{regime}', 'regime', f'Regime: {regime}',
                f'Market regime: {regime}',
                mask, df
            ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_hour_patterns(self) -> List[TradingPattern]:
        """ขุด hour patterns"""
        patterns = []
        df = self.df[self.df['label'].notna()]
        
        for hour in range(24):
            mask = df['hour'] == hour
            patterns.append(self._create_pattern_from_mask(
                f'hour_{hour}', 'hour', f'Hour {hour}:00',
                f'เทรดที่ชั่วโมง {hour}:00 UTC',
                mask, df
            ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_day_patterns(self) -> List[TradingPattern]:
        """ขุด day of week patterns"""
        patterns = []
        df = self.df[self.df['label'].notna()]
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day_num in range(7):
            mask = df['day_of_week'] == day_num
            if mask.sum() >= self.min_occurrences:
                patterns.append(self._create_pattern_from_mask(
                    f'day_{days[day_num].lower()}', 'day', f'{days[day_num]}',
                    f'เทรดวัน{days[day_num]}',
                    mask, df
                ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_session_patterns(self) -> List[TradingPattern]:
        """ขุด session patterns"""
        patterns = []
        df = self.df[self.df['label'].notna()]
        
        for session in ['asia', 'london', 'newyork']:
            mask = df['session'] == session
            patterns.append(self._create_pattern_from_mask(
                f'session_{session}', 'session', f'Session: {session.title()}',
                f'เทรดช่วง {session} session',
                mask, df
            ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_rsi_patterns(self) -> List[TradingPattern]:
        """ขุด RSI patterns"""
        patterns = []
        df = self.df[self.df['label'].notna() & self.df['rsi'].notna()]
        
        # RSI Overbought
        overbought_mask = df['rsi'] > 70
        patterns.append(self._create_pattern_from_mask(
            'rsi_overbought', 'indicator', 'RSI Overbought',
            'RSI > 70 - Overbought zone',
            overbought_mask, df
        ))
        
        # RSI Oversold
        oversold_mask = df['rsi'] < 30
        patterns.append(self._create_pattern_from_mask(
            'rsi_oversold', 'indicator', 'RSI Oversold',
            'RSI < 30 - Oversold zone',
            oversold_mask, df
        ))
        
        # RSI Neutral
        neutral_mask = df['rsi'].between(40, 60)
        patterns.append(self._create_pattern_from_mask(
            'rsi_neutral', 'indicator', 'RSI Neutral',
            'RSI 40-60 - Neutral zone',
            neutral_mask, df
        ))
        
        # RSI Extreme Overbought
        extreme_ob_mask = df['rsi'] > 80
        patterns.append(self._create_pattern_from_mask(
            'rsi_extreme_overbought', 'indicator', 'RSI Extreme Overbought',
            'RSI > 80 - Extreme overbought',
            extreme_ob_mask, df
        ))
        
        # RSI Extreme Oversold
        extreme_os_mask = df['rsi'] < 20
        patterns.append(self._create_pattern_from_mask(
            'rsi_extreme_oversold', 'indicator', 'RSI Extreme Oversold',
            'RSI < 20 - Extreme oversold',
            extreme_os_mask, df
        ))
        
        return [p for p in patterns if p is not None]
    
    def _mine_confluence_patterns(self) -> List[TradingPattern]:
        """ขุด confluence patterns (หลาย conditions รวมกัน)"""
        patterns = []
        df = self.df[self.df['label'].notna() & self.df['rsi'].notna()]
        
        # Trending up + London session
        mask = (df['regime'].str.contains('trending_up')) & (df['session'] == 'london')
        patterns.append(self._create_pattern_from_mask(
            'confluence_trend_up_london', 'confluence', 'Trend Up + London',
            'Uptrend ใน London session',
            mask, df
        ))
        
        # Trending down + London session
        mask = (df['regime'].str.contains('trending_down')) & (df['session'] == 'london')
        patterns.append(self._create_pattern_from_mask(
            'confluence_trend_down_london', 'confluence', 'Trend Down + London',
            'Downtrend ใน London session',
            mask, df
        ))
        
        # Ranging + Asia session
        mask = (df['regime'] == 'ranging') & (df['session'] == 'asia')
        patterns.append(self._create_pattern_from_mask(
            'confluence_ranging_asia', 'confluence', 'Ranging + Asia',
            'Sideways ใน Asia session',
            mask, df
        ))
        
        # RSI oversold + Uptrend
        mask = (df['rsi'] < 30) & (df['trend'] == 1)
        patterns.append(self._create_pattern_from_mask(
            'confluence_rsi_os_uptrend', 'confluence', 'RSI Oversold + Uptrend',
            'RSI < 30 ใน Uptrend - mean reversion opportunity',
            mask, df
        ))
        
        # RSI overbought + Downtrend
        mask = (df['rsi'] > 70) & (df['trend'] == -1)
        patterns.append(self._create_pattern_from_mask(
            'confluence_rsi_ob_downtrend', 'confluence', 'RSI Overbought + Downtrend',
            'RSI > 70 ใน Downtrend - shorting opportunity',
            mask, df
        ))
        
        # Big candle + High volatility
        mask = (df['body'] > df['atr'] * 1.5) & (df['volatility'] > 0.015)
        patterns.append(self._create_pattern_from_mask(
            'confluence_big_candle_high_vol', 'confluence', 'Big Candle + High Volatility',
            'แท่งใหญ่ในช่วง volatility สูง',
            mask, df
        ))
        
        # New York overlap (London + NY)
        mask = df['hour'].between(13, 16)
        patterns.append(self._create_pattern_from_mask(
            'confluence_ny_overlap', 'confluence', 'NY Overlap',
            'ช่วง overlap London-NY (13:00-16:00 UTC)',
            mask, df
        ))
        
        return [p for p in patterns if p is not None]
    
    def _create_pattern_from_mask(
        self,
        pattern_id: str,
        pattern_type: str,
        name: str,
        description: str,
        mask: pd.Series,
        df: pd.DataFrame,
    ) -> Optional[TradingPattern]:
        """สร้าง pattern จาก boolean mask"""
        
        subset = df[mask]
        
        if len(subset) < self.min_occurrences:
            return None
        
        # Calculate statistics
        wins = (subset['label'] == 1).sum()
        losses = (subset['label'] == -1).sum()
        total = wins + losses
        
        if total == 0:
            return None
        
        win_rate = wins / total
        avg_return = subset['future_return'].mean()
        total_return = subset['future_return'].sum()
        
        pattern = TradingPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            name=name,
            description=description,
            occurrences=len(subset),
            wins=int(wins),
            losses=int(losses),
            total_pnl=float(total_return * 10000),  # Pips
            avg_pnl=float(avg_return * 10000),      # Pips
            win_rate=float(win_rate),
            conditions={},
            is_good=bool(win_rate >= self.good_win_rate),
            is_bad=bool(win_rate <= self.bad_win_rate),
        )
        
        return pattern
    
    def save_patterns(self, filepath: str = "ai_agent/data/mined_patterns.json"):
        """บันทึก patterns"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        all_patterns = self.mine_all_patterns()
        
        # Convert to dict
        data = {}
        for category, pattern_list in all_patterns.items():
            data[category] = [p.to_dict() for p in pattern_list]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved patterns to {filepath}")
        
        return all_patterns
    
    def load_patterns(self, filepath: str = "ai_agent/data/mined_patterns.json") -> Dict:
        """โหลด patterns"""
        if not os.path.exists(filepath):
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_good_patterns(self) -> List[TradingPattern]:
        """ดึง patterns ที่ดี"""
        all_patterns = self.mine_all_patterns()
        good = []
        for pattern_list in all_patterns.values():
            good.extend([p for p in pattern_list if p.is_good])
        return sorted(good, key=lambda p: -p.win_rate)
    
    def get_bad_patterns(self) -> List[TradingPattern]:
        """ดึง patterns ที่ไม่ดี"""
        all_patterns = self.mine_all_patterns()
        bad = []
        for pattern_list in all_patterns.values():
            bad.extend([p for p in pattern_list if p.is_bad])
        return sorted(bad, key=lambda p: p.win_rate)
    
    def print_summary(self):
        """พิมพ์สรุป patterns"""
        all_patterns = self.mine_all_patterns()
        
        print("\n" + "="*70)
        print("   GOOD PATTERNS (Win Rate > 55%)")
        print("="*70)
        
        good = self.get_good_patterns()
        for p in good[:15]:
            print(f"   ✅ {p.name:<30} WR: {p.win_rate:.1%} ({p.occurrences} trades)")
        
        print("\n" + "="*70)
        print("   BAD PATTERNS (Win Rate < 40%) - AVOID!")
        print("="*70)
        
        bad = self.get_bad_patterns()
        for p in bad[:15]:
            print(f"   ❌ {p.name:<30} WR: {p.win_rate:.1%} ({p.occurrences} trades)")
    
    def pre_train_master_brain(self, master_brain) -> Dict:
        """
        Pre-train MasterBrain ด้วย patterns ที่ขุดได้
        
        Args:
            master_brain: MasterBrain instance
            
        Returns:
            Pre-training statistics
        """
        all_patterns = self.mine_all_patterns()
        
        stats = {
            'good_patterns_loaded': 0,
            'bad_situations_loaded': 0,
            'regime_confidence_set': 0,
            'hour_confidence_set': 0,
        }
        
        # 1. Pre-seed bad situations
        for pattern_list in all_patterns.values():
            for p in pattern_list:
                if p.is_bad and p.pattern_type in ['regime', 'confluence']:
                    # Create situation key
                    situation_key = p.pattern_id.replace('regime_', '').replace('confluence_', '')
                    master_brain.bad_situations[situation_key] = {
                        'wins': p.wins,
                        'losses': p.losses,
                        'total_pnl': p.total_pnl,
                    }
                    stats['bad_situations_loaded'] += 1
        
        # 2. Pre-seed regime confidence
        regime_patterns = all_patterns.get('regime', [])
        for p in regime_patterns:
            regime_name = p.pattern_id.replace('regime_', '')
            master_brain.regime_confidence[regime_name] = p.win_rate
            stats['regime_confidence_set'] += 1
        
        # 3. Pre-seed hour confidence
        hour_patterns = all_patterns.get('hour', [])
        for p in hour_patterns:
            hour = int(p.pattern_id.replace('hour_', ''))
            master_brain.hour_confidence[hour] = p.win_rate
            stats['hour_confidence_set'] += 1
        
        # 4. Pre-seed winning patterns for reference
        for pattern_list in all_patterns.values():
            for p in pattern_list:
                if p.is_good:
                    master_brain.winning_patterns.append({
                        'id': p.pattern_id,
                        'name': p.name,
                        'win_rate': p.win_rate,
                        'occurrences': p.occurrences,
                    })
                    stats['good_patterns_loaded'] += 1
        
        logger.info(f"🧠 Pre-trained MasterBrain with {stats['good_patterns_loaded']} good patterns, "
                   f"{stats['bad_situations_loaded']} bad situations")
        
        return stats


def mine_and_pretrain():
    """ขุด patterns และ pre-train MasterBrain"""
    
    print("="*70)
    print("   PATTERN MINING & PRE-TRAINING")
    print("="*70)
    
    # Create miner
    miner = PatternMiner(
        data_path="data/training/GOLD_H1.csv",
        lookback_bars=5,
        forward_bars=10,
        min_occurrences=50,
        good_win_rate=0.55,
        bad_win_rate=0.40,
    )
    
    # Mine patterns
    miner.print_summary()
    
    # Save patterns
    miner.save_patterns()
    
    # Pre-train MasterBrain
    print("\n" + "="*70)
    print("   PRE-TRAINING MASTERBRAIN")
    print("="*70)
    
    try:
        from ai_agent.master_brain import MasterBrain
        
        brain = MasterBrain()
        stats = miner.pre_train_master_brain(brain)
        
        print(f"\n   ✅ Good patterns loaded: {stats['good_patterns_loaded']}")
        print(f"   ⛔ Bad situations loaded: {stats['bad_situations_loaded']}")
        print(f"   📊 Regime confidence set: {stats['regime_confidence_set']}")
        print(f"   ⏰ Hour confidence set: {stats['hour_confidence_set']}")
        
        # Save
        brain._save_model()
        print("\n   💾 MasterBrain saved with pre-trained patterns!")
        
    except Exception as e:
        print(f"\n   ⚠️ Could not pre-train MasterBrain: {e}")
    
    print("\n" + "="*70)
    print("   COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    mine_and_pretrain()
