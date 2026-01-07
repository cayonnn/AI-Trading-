"""
MT5 Data Provider - Production Grade
====================================
Professional data fetching from MetaTrader 5
"""

import MetaTrader5 as mt5
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from loguru import logger
import sys
import time

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


class MT5DataProvider:
    """Production-grade MT5 data provider"""
    
    def __init__(self, login=None, password=None, server=None, path=None):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect = 3
        logger.info("MT5 Data Provider initialized")
    
    def connect(self) -> bool:
        """Connect to MT5"""
        try:
            logger.info("Connecting to MT5...")
            if self.path:
                init_result = mt5.initialize(path=self.path)
            else:
                init_result = mt5.initialize()
            
            if not init_result:
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            
            logger.info("MT5 initialized")
            
            if self.login and self.password and self.server:
                logger.info(f"Logging in to {self.login}@{self.server}...")
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"Login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                logger.info("Login successful")
                account = mt5.account_info()
                if account:
                    logger.info(f"Account: {account.name}")
                    logger.info(f"Balance: ${account.balance:,.2f}")
            
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("MT5 connection established")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
    
    def ensure_connected(self) -> bool:
        """Ensure connected, reconnect if needed"""
        if not mt5.terminal_info():
            logger.warning("Terminal disconnected")
            self.connected = False
        
        if not self.connected:
            if self.reconnect_attempts < self.max_reconnect:
                self.reconnect_attempts += 1
                logger.info(f"Reconnecting... ({self.reconnect_attempts}/{self.max_reconnect})")
                time.sleep(2)
                return self.connect()
            else:
                logger.error("Max reconnection attempts reached")
                return False
        return True
    
    def get_ohlcv(self, symbol: str, timeframe: str = "H1", bars: int = 1000, start_pos: int = 0) -> Optional[pd.DataFrame]:
        """Get OHLCV data"""
        if not self.ensure_connected():
            return None
        
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1
        }
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, bars)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get {symbol} data: {mt5.last_error()}")
                return None
            
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'tick_volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            logger.info(f"Fetched {len(df)} bars of {symbol} {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return None
    
    def get_multi_timeframe(self, symbol: str, timeframes: List[str] = None, bars: int = 1000) -> Dict[str, pd.DataFrame]:
        """Get multi-timeframe data"""
        if timeframes is None:
            timeframes = ["W1", "D1", "H4", "H1", "M30", "M15"]
        logger.info(f"Fetching MTF data for {symbol}...")
        mtf_data = {}
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf, bars)
            if df is not None:
                mtf_data[tf] = df
        logger.info(f"Fetched {len(mtf_data)}/{len(timeframes)} timeframes")
        return mtf_data
    
    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask price"""
        if not self.ensure_connected():
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return (tick.bid, tick.ask)
        return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.ensure_connected():
            return None
        info = mt5.symbol_info(symbol)
        if not info:
            return None
        return {
            'symbol': info.name, 'description': info.description,
            'point': info.point, 'digits': info.digits,
            'spread': info.spread, 'volume_min': info.volume_min,
            'volume_max': info.volume_max, 'volume_step': info.volume_step,
            'contract_size': info.trade_contract_size,
            'bid': info.bid, 'ask': info.ask
        }

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.ensure_connected():
            return None

        account = mt5.account_info()
        if not account:
            return None

        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'profit': account.profit,
            'leverage': account.leverage,
            'currency': account.currency,
            'name': account.name,
            'company': account.company
        }


if __name__ == "__main__":
    logger.info("")
    logger.info("="*60)
    logger.info("  MT5 DATA PROVIDER TEST")
    logger.info("="*60)
    logger.info("")
    
    provider = MT5DataProvider()
    if provider.connect():
        info = provider.get_symbol_info("XAUUSD")
        if info:
            logger.info(f"\nSymbol: {info['symbol']}")
            logger.info(f"Bid: {info['bid']:.2f}")
            logger.info(f"Ask: {info['ask']:.2f}")
        
        df = provider.get_ohlcv("XAUUSD", "H1", bars=100)
        if df is not None:
            logger.info(f"\nH1 Data: {len(df)} bars")
            logger.info(f"Latest close: {df['close'].iloc[-1]:.2f}")
        
        mtf = provider.get_multi_timeframe("XAUUSD", ["D1", "H4", "H1"], bars=100)
        logger.info(f"\nMTF Data: {len(mtf)} timeframes")
        
        provider.disconnect()
    else:
        logger.warning("Could not connect to MT5")
    
    logger.info("")
    logger.info("="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)
