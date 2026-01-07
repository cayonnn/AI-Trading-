"""
Trade Memory System
===================
จำทุกการเทรดและเรียนรู้จากประสบการณ์

Features:
- บันทึกทุกการเทรด
- วิเคราะห์รูปแบบที่ชนะ/แพ้
- จำสภาพตลาดขณะเทรด
- ดึงประสบการณ์ที่คล้ายกัน
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import os


@dataclass
class TradeRecord:
    """บันทึกการเทรดหนึ่งรายการ"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # LONG only for sniper
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_bars: int
    exit_reason: str  # sl, tp, signal
    strategy_used: str
    confidence: float
    
    # Market context at entry
    market_regime: str  # trending, ranging, volatile
    atr: float
    rsi: float
    macd_histogram: float
    trend_strength: float
    volatility: float
    
    # Performance
    was_winner: bool = False
    r_multiple: float = 0.0  # Actual R (profit/risk)
    
    def __post_init__(self):
        self.was_winner = self.pnl > 0
        risk = abs(self.entry_price - self.stop_loss)
        if risk > 0:
            self.r_multiple = self.pnl / (risk * self.quantity * 100)


@dataclass
class PatternAnalysis:
    """การวิเคราะห์รูปแบบที่พบ"""
    pattern_id: str
    description: str
    win_rate: float
    avg_r_multiple: float
    sample_count: int
    conditions: Dict
    recommended_action: str


class TradeMemory:
    """
    ระบบความจำการเทรด - AI จำและเรียนรู้จากทุกการเทรด
    
    ความสามารถ:
    - บันทึกทุกการเทรดพร้อม Context
    - วิเคราะห์รูปแบบที่ชนะ/แพ้
    - ค้นหาสถานการณ์ที่คล้ายกัน
    - แนะนำ Action จากประสบการณ์
    """
    
    def __init__(self, db_path: str = "ai_agent/trade_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self.min_samples_for_pattern = 10
        
        logger.info("TradeMemory initialized")
    
    def _init_database(self):
        """สร้างฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ตาราง Trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity REAL,
                pnl REAL,
                pnl_pct REAL,
                duration_bars INTEGER,
                exit_reason TEXT,
                strategy_used TEXT,
                confidence REAL,
                market_regime TEXT,
                atr REAL,
                rsi REAL,
                macd_histogram REAL,
                trend_strength REAL,
                volatility REAL,
                was_winner INTEGER,
                r_multiple REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ตาราง Patterns ที่ค้นพบ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                description TEXT,
                win_rate REAL,
                avg_r_multiple REAL,
                sample_count INTEGER,
                conditions TEXT,
                recommended_action TEXT,
                last_updated TEXT
            )
        ''')
        
        # ตาราง Strategy Performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT PRIMARY KEY,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                profit_factor REAL,
                avg_r_multiple REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def remember(self, trade: TradeRecord):
        """บันทึกการเทรด"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            trade.trade_id,
            trade.timestamp.isoformat() if isinstance(trade.timestamp, datetime) else trade.timestamp,
            trade.symbol,
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.stop_loss,
            trade.take_profit,
            trade.quantity,
            trade.pnl,
            trade.pnl_pct,
            trade.duration_bars,
            trade.exit_reason,
            trade.strategy_used,
            trade.confidence,
            trade.market_regime,
            trade.atr,
            trade.rsi,
            trade.macd_histogram,
            trade.trend_strength,
            trade.volatility,
            1 if trade.was_winner else 0,
            trade.r_multiple,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Remembered trade: {trade.trade_id} - {'WIN' if trade.was_winner else 'LOSS'}")
        
        # Update patterns after each trade
        self._update_patterns()
    
    def recall_all(self, limit: int = 1000) -> List[TradeRecord]:
        """ดึงการเทรดทั้งหมด"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        conn.close()
        
        trades = []
        for _, row in df.iterrows():
            trade = TradeRecord(
                trade_id=row['trade_id'],
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                side=row['side'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                quantity=row['quantity'],
                pnl=row['pnl'],
                pnl_pct=row['pnl_pct'],
                duration_bars=row['duration_bars'],
                exit_reason=row['exit_reason'],
                strategy_used=row['strategy_used'],
                confidence=row['confidence'],
                market_regime=row['market_regime'],
                atr=row['atr'],
                rsi=row['rsi'],
                macd_histogram=row['macd_histogram'],
                trend_strength=row['trend_strength'],
                volatility=row['volatility'],
            )
            trades.append(trade)
        
        return trades
    
    def recall_similar(
        self,
        market_regime: str,
        rsi_range: Tuple[float, float],
        trend_direction: str,
    ) -> List[TradeRecord]:
        """ดึงการเทรดในสถานการณ์คล้ายกัน"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM trades 
            WHERE market_regime = ?
            AND rsi BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT 50
        '''
        
        df = pd.read_sql_query(
            query, conn,
            params=(market_regime, rsi_range[0], rsi_range[1])
        )
        conn.close()
        
        # Convert to TradeRecord list
        trades = []
        for _, row in df.iterrows():
            trade = TradeRecord(
                trade_id=row['trade_id'],
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                side=row['side'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                quantity=row['quantity'],
                pnl=row['pnl'],
                pnl_pct=row['pnl_pct'],
                duration_bars=row['duration_bars'],
                exit_reason=row['exit_reason'],
                strategy_used=row['strategy_used'],
                confidence=row['confidence'],
                market_regime=row['market_regime'],
                atr=row['atr'],
                rsi=row['rsi'],
                macd_histogram=row['macd_histogram'],
                trend_strength=row['trend_strength'],
                volatility=row['volatility'],
            )
            trades.append(trade)
        
        return trades
    
    def _update_patterns(self):
        """อัพเดตรูปแบบที่ค้นพบ"""
        trades = self.recall_all()
        
        if len(trades) < self.min_samples_for_pattern:
            return
        
        # วิเคราะห์ตาม Market Regime
        regimes = {}
        for trade in trades:
            regime = trade.market_regime
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(trade)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for regime, regime_trades in regimes.items():
            if len(regime_trades) < 5:
                continue
            
            win_rate = sum(1 for t in regime_trades if t.was_winner) / len(regime_trades)
            avg_r = np.mean([t.r_multiple for t in regime_trades])
            
            pattern_id = f"regime_{regime}"
            cursor.execute('''
                INSERT OR REPLACE INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                f"Market Regime: {regime}",
                win_rate,
                avg_r,
                len(regime_trades),
                json.dumps({"regime": regime}),
                "TRADE" if win_rate > 0.55 and avg_r > 0.5 else "AVOID",
                datetime.now().isoformat(),
            ))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(self) -> Dict:
        """ดึงสถิติประสิทธิภาพ"""
        trades = self.recall_all()
        
        if not trades:
            return {"message": "No trades recorded"}
        
        winners = [t for t in trades if t.was_winner]
        losers = [t for t in trades if not t.was_winner]
        
        return {
            "total_trades": len(trades),
            "win_rate": len(winners) / len(trades) if trades else 0,
            "average_win": np.mean([t.pnl for t in winners]) if winners else 0,
            "average_loss": np.mean([t.pnl for t in losers]) if losers else 0,
            "average_r_multiple": np.mean([t.r_multiple for t in trades]),
            "best_r": max([t.r_multiple for t in trades]) if trades else 0,
            "worst_r": min([t.r_multiple for t in trades]) if trades else 0,
            "total_pnl": sum(t.pnl for t in trades),
            "avg_confidence_winners": np.mean([t.confidence for t in winners]) if winners else 0,
            "avg_confidence_losers": np.mean([t.confidence for t in losers]) if losers else 0,
        }
    
    def get_strategy_performance(self) -> pd.DataFrame:
        """ดึงประสิทธิภาพแต่ละกลยุทธ์"""
        trades = self.recall_all()
        
        if not trades:
            return pd.DataFrame()
        
        # Group by strategy
        strategy_stats = {}
        for trade in trades:
            strat = trade.strategy_used
            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    "trades": [],
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                    "r_multiples": [],
                }
            
            strategy_stats[strat]["trades"].append(trade)
            strategy_stats[strat]["pnl"] += trade.pnl
            strategy_stats[strat]["r_multiples"].append(trade.r_multiple)
            
            if trade.was_winner:
                strategy_stats[strat]["wins"] += 1
            else:
                strategy_stats[strat]["losses"] += 1
        
        # Create DataFrame
        rows = []
        for strat, stats in strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            rows.append({
                "strategy": strat,
                "trades": total,
                "win_rate": stats["wins"] / total if total > 0 else 0,
                "total_pnl": stats["pnl"],
                "avg_r": np.mean(stats["r_multiples"]),
                "best_r": max(stats["r_multiples"]),
            })
        
        return pd.DataFrame(rows).sort_values("avg_r", ascending=False)
    
    def should_trade(self, market_context: Dict) -> Tuple[bool, str, float]:
        """
        ตัดสินใจว่าควรเทรดหรือไม่ จากประสบการณ์
        
        Returns:
            (should_trade, reason, confidence_adjustment)
        """
        # ดึงเทรดที่คล้ายกัน
        similar = self.recall_similar(
            market_regime=market_context.get("regime", "unknown"),
            rsi_range=(market_context.get("rsi", 50) - 10, market_context.get("rsi", 50) + 10),
            trend_direction=market_context.get("trend", "neutral"),
        )
        
        if len(similar) < 5:
            return True, "ไม่มีประสบการณ์เพียงพอ - เทรดด้วย Default", 0.0
        
        win_rate = sum(1 for t in similar if t.was_winner) / len(similar)
        avg_r = np.mean([t.r_multiple for t in similar])
        
        if win_rate < 0.35:
            return False, f"Win rate ต่ำในสถานการณ์นี้ ({win_rate:.1%})", -0.2
        
        if avg_r < -0.5:
            return False, f"Avg R-Multiple ติดลบ ({avg_r:.2f})", -0.2
        
        if win_rate > 0.6 and avg_r > 1.0:
            return True, f"สถานการณ์ดีมาก ({win_rate:.1%} win, {avg_r:.2f}R)", 0.15
        
        if win_rate > 0.5:
            return True, f"สถานการณ์ดี ({win_rate:.1%} win)", 0.05
        
        return True, "สถานการณ์ปกติ", 0.0


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    # Test
    memory = TradeMemory(db_path="test_memory.db")
    
    # Create sample trade
    trade = TradeRecord(
        trade_id="TEST-001",
        timestamp=datetime.now(),
        symbol="GOLD",
        side="LONG",
        entry_price=2300.0,
        exit_price=2350.0,
        stop_loss=2280.0,
        take_profit=2400.0,
        quantity=1.0,
        pnl=5000.0,
        pnl_pct=0.0217,
        duration_bars=24,
        exit_reason="tp",
        strategy_used="trend_following",
        confidence=0.85,
        market_regime="trending",
        atr=15.0,
        rsi=45.0,
        macd_histogram=0.5,
        trend_strength=0.75,
        volatility=0.012,
    )
    
    memory.remember(trade)
    
    stats = memory.get_performance_stats()
    print("\nPerformance Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Trade Memory Test Complete")
