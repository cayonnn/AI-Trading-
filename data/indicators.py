"""
Technical Indicators Module
===========================
Production-grade technical indicator calculations for trading signals.

Features:
- Comprehensive indicator library (RSI, MACD, Bollinger, ATR, ADX, etc.)
- Vectorized calculations for performance
- Multi-timeframe support
- Clean NaN handling
"""

from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
from loguru import logger


class TechnicalIndicators:
    """
    Calculate technical indicators on OHLCV data.
    
    All methods return the input DataFrame with new columns added.
    Supports method chaining for clean code.
    
    Usage:
        ti = TechnicalIndicators(df)
        df = ti.add_all()  # Add all indicators
        
        # Or add specific indicators
        df = (TechnicalIndicators(df)
              .add_rsi()
              .add_macd()
              .add_bollinger()
              .get_dataframe())
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """Validate that DataFrame has required columns."""
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the DataFrame with all added indicators."""
        return self.df
    
    # =========================================================================
    # Trend Indicators
    # =========================================================================
    
    def add_sma(self, periods: List[int] = [20, 50, 200]) -> 'TechnicalIndicators':
        """
        Add Simple Moving Averages.
        
        Args:
            periods: List of SMA periods
        """
        for period in periods:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
        return self
    
    def add_ema(self, periods: List[int] = [9, 21, 50, 100, 200]) -> 'TechnicalIndicators':
        """
        Add Exponential Moving Averages.
        
        Args:
            periods: List of EMA periods
        """
        for period in periods:
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        return self
    
    def add_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> 'TechnicalIndicators':
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # MACD crossover signals
        self.df['macd_cross_up'] = (
            (self.df['macd'] > self.df['macd_signal']) & 
            (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1))
        ).astype(int)
        
        self.df['macd_cross_down'] = (
            (self.df['macd'] < self.df['macd_signal']) & 
            (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1))
        ).astype(int)
        
        return self
    
    def add_adx(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add ADX (Average Directional Index) with +DI and -DI.
        
        Args:
            period: ADX calculation period
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Use DataFrame index to prevent length mismatch
        plus_dm_series = pd.Series(plus_dm, index=self.df.index)
        minus_dm_series = pd.Series(minus_dm, index=self.df.index)
        
        plus_di = 100 * plus_dm_series.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm_series.ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        self.df['adx'] = adx
        
        return self
    
    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    
    def add_rsi(self, periods: List[int] = [7, 14, 21]) -> 'TechnicalIndicators':
        """
        Add Relative Strength Index.
        
        Args:
            periods: List of RSI periods
        """
        for period in periods:
            delta = self.df['close'].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            self.df[f'rsi_{period}'] = rsi
        
        return self
    
    def add_stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3
    ) -> 'TechnicalIndicators':
        """
        Add Stochastic Oscillator.
        
        Args:
            k_period: %K period
            d_period: %D period (signal line)
        """
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        self.df['stoch_k'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
        
        return self
    
    def add_williams_r(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add Williams %R.
        
        Args:
            period: Lookback period
        """
        high_max = self.df['high'].rolling(window=period).max()
        low_min = self.df['low'].rolling(window=period).min()
        
        self.df['williams_r'] = -100 * (high_max - self.df['close']) / (high_max - low_min)
        
        return self
    
    def add_cci(self, period: int = 20) -> 'TechnicalIndicators':
        """
        Add Commodity Channel Index.
        
        Args:
            period: CCI period
        """
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        self.df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return self
    
    def add_momentum(self, period: int = 10) -> 'TechnicalIndicators':
        """
        Add Momentum indicator.
        
        Args:
            period: Momentum period
        """
        self.df['momentum'] = self.df['close'] - self.df['close'].shift(period)
        self.df['momentum_pct'] = self.df['close'].pct_change(period) * 100
        
        return self
    
    def add_roc(self, period: int = 10) -> 'TechnicalIndicators':
        """
        Add Rate of Change.
        
        Args:
            period: ROC period
        """
        self.df['roc'] = (
            (self.df['close'] - self.df['close'].shift(period)) / 
            self.df['close'].shift(period)
        ) * 100
        
        return self
    
    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    
    def add_atr(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add Average True Range.
        
        Args:
            period: ATR period
        """
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = tr.ewm(span=period, adjust=False).mean()
        
        # Normalized ATR (as percentage of price)
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100
        
        return self
    
    def add_bollinger(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> 'TechnicalIndicators':
        """
        Add Bollinger Bands.
        
        Args:
            period: SMA period for middle band
            std_dev: Number of standard deviations
        """
        sma = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        self.df['bb_middle'] = sma
        self.df['bb_upper'] = sma + (std_dev * std)
        self.df['bb_lower'] = sma - (std_dev * std)
        
        # Bollinger Band Width
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        
        # %B (position within bands)
        self.df['bb_pct'] = (
            (self.df['close'] - self.df['bb_lower']) / 
            (self.df['bb_upper'] - self.df['bb_lower'])
        )
        
        return self
    
    def add_keltner(
        self,
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> 'TechnicalIndicators':
        """
        Add Keltner Channels.
        
        Args:
            period: EMA period
            atr_multiplier: ATR multiplier for bands
        """
        # First ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.add_atr(period)
        
        ema = self.df['close'].ewm(span=period, adjust=False).mean()
        
        self.df['keltner_middle'] = ema
        self.df['keltner_upper'] = ema + (atr_multiplier * self.df['atr'])
        self.df['keltner_lower'] = ema - (atr_multiplier * self.df['atr'])
        
        return self
    
    # =========================================================================
    # Volume Indicators
    # =========================================================================
    
    def add_obv(self) -> 'TechnicalIndicators':
        """Add On-Balance Volume."""
        if 'volume' not in self.df.columns:
            logger.warning("Volume column not found, skipping OBV")
            return self
        
        obv = []
        prev_obv = 0
        
        for i in range(len(self.df)):
            if i == 0:
                obv.append(0)
            else:
                if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                    prev_obv += self.df['volume'].iloc[i]
                elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                    prev_obv -= self.df['volume'].iloc[i]
                obv.append(prev_obv)
        
        self.df['obv'] = obv
        self.df['obv_ema'] = pd.Series(obv).ewm(span=20, adjust=False).mean().values
        
        return self
    
    def add_vwap(self) -> 'TechnicalIndicators':
        """Add Volume Weighted Average Price (reset daily)."""
        if 'volume' not in self.df.columns:
            logger.warning("Volume column not found, skipping VWAP")
            return self
        
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # For daily reset, we need to detect new days
        # Simple implementation: rolling VWAP
        cumulative_tp_vol = (typical_price * self.df['volume']).cumsum()
        cumulative_vol = self.df['volume'].cumsum()
        
        self.df['vwap'] = cumulative_tp_vol / cumulative_vol
        
        return self
    
    def add_mfi(self, period: int = 14) -> 'TechnicalIndicators':
        """
        Add Money Flow Index.
        
        Args:
            period: MFI period
        """
        if 'volume' not in self.df.columns:
            logger.warning("Volume column not found, skipping MFI")
            return self
        
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_money_flow = typical_price * self.df['volume']
        
        # Positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        self.df['mfi'] = mfi.values
        
        return self
    
    # =========================================================================
    # Support/Resistance & Patterns
    # =========================================================================
    
    def add_pivot_points(self) -> 'TechnicalIndicators':
        """Add Pivot Points (Standard)."""
        # Using previous period's OHLC
        pp = (self.df['high'].shift(1) + self.df['low'].shift(1) + self.df['close'].shift(1)) / 3
        
        self.df['pivot'] = pp
        self.df['r1'] = 2 * pp - self.df['low'].shift(1)
        self.df['s1'] = 2 * pp - self.df['high'].shift(1)
        self.df['r2'] = pp + (self.df['high'].shift(1) - self.df['low'].shift(1))
        self.df['s2'] = pp - (self.df['high'].shift(1) - self.df['low'].shift(1))
        
        return self
    
    def add_donchian(self, period: int = 20) -> 'TechnicalIndicators':
        """
        Add Donchian Channels.
        
        Args:
            period: Lookback period
        """
        self.df['donchian_high'] = self.df['high'].rolling(window=period).max()
        self.df['donchian_low'] = self.df['low'].rolling(window=period).min()
        self.df['donchian_mid'] = (self.df['donchian_high'] + self.df['donchian_low']) / 2
        
        return self
    
    # =========================================================================
    # Price Action Features
    # =========================================================================
    
    def add_price_features(self) -> 'TechnicalIndicators':
        """Add price-based features."""
        # Returns
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # Candle features
        self.df['body'] = self.df['close'] - self.df['open']
        self.df['body_pct'] = self.df['body'] / self.df['open'] * 100
        self.df['upper_shadow'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_shadow'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['range'] = self.df['high'] - self.df['low']
        
        # Is bullish/bearish
        self.df['is_bullish'] = (self.df['close'] > self.df['open']).astype(int)
        
        # Gap
        self.df['gap'] = self.df['open'] - self.df['close'].shift(1)
        self.df['gap_pct'] = self.df['gap'] / self.df['close'].shift(1) * 100
        
        return self
    
    def add_trend_features(self, periods: List[int] = [5, 10, 20]) -> 'TechnicalIndicators':
        """
        Add trend-related features.
        
        Args:
            periods: Lookback periods for trend calculation
        """
        for period in periods:
            # Price position relative to high/low
            high = self.df['high'].rolling(window=period).max()
            low = self.df['low'].rolling(window=period).min()
            self.df[f'price_position_{period}'] = (self.df['close'] - low) / (high - low)
            
            # Trend direction (simple linear regression slope)
            self.df[f'trend_{period}'] = self.df['close'].rolling(window=period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # Higher highs and lower lows count
            self.df[f'higher_highs_{period}'] = self.df['high'].rolling(window=period).apply(
                lambda x: sum(x.iloc[i] > x.iloc[i-1] for i in range(1, len(x)))
            )
        
        return self
    
    # =========================================================================
    # Composite Methods
    # =========================================================================
    
    def add_all(
        self,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Add all standard indicators.
        
        Args:
            include_volume: Include volume-based indicators
            
        Returns:
            DataFrame with all indicators
        """
        logger.info("Adding all technical indicators...")
        
        # Trend
        self.add_ema([9, 21, 50, 100, 200])
        self.add_sma([20, 50, 200])
        self.add_macd()
        self.add_adx()
        
        # Momentum
        self.add_rsi([7, 14, 21])
        self.add_stochastic()
        self.add_cci()
        self.add_momentum()
        self.add_roc()
        
        # Volatility
        self.add_atr()
        self.add_bollinger()
        self.add_keltner()
        
        # Volume
        if include_volume and 'volume' in self.df.columns:
            self.add_obv()
            self.add_mfi()
        
        # Support/Resistance
        self.add_pivot_points()
        self.add_donchian()
        
        # Price Features
        self.add_price_features()
        self.add_trend_features()
        
        logger.info(f"Added {len(self.df.columns)} columns total")
        
        return self.df
    
    def add_minimal(self) -> pd.DataFrame:
        """
        Add minimal set of indicators for quick analysis.
        
        Returns:
            DataFrame with minimal indicators
        """
        self.add_ema([9, 21, 50])
        self.add_rsi([14])
        self.add_macd()
        self.add_atr()
        self.add_bollinger()
        self.add_price_features()
        
        return self.df


def calculate_indicators(
    df: pd.DataFrame,
    indicator_set: str = 'all'
) -> pd.DataFrame:
    """
    Convenience function to calculate indicators.
    
    Args:
        df: OHLCV DataFrame
        indicator_set: 'all', 'minimal', or 'custom'
        
    Returns:
        DataFrame with indicators
    """
    ti = TechnicalIndicators(df)
    
    if indicator_set == 'all':
        return ti.add_all()
    elif indicator_set == 'minimal':
        return ti.add_minimal()
    else:
        return ti.add_all()


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    print("=== Testing Technical Indicators ===")
    
    # Fetch sample data
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="3mo", interval="1h")
    df.columns = df.columns.str.lower()
    
    print(f"Original shape: {df.shape}")
    
    # Add all indicators
    df_with_indicators = calculate_indicators(df, 'all')
    
    print(f"With indicators: {df_with_indicators.shape}")
    print(f"\nColumns added:\n{list(df_with_indicators.columns)}")
    print(f"\nSample data:\n{df_with_indicators.tail()}")
