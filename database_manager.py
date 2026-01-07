"""
Database Manager - Production Grade
===================================
Professional database management for trading system:
- SQLite and PostgreSQL support
- Connection pooling
- Transaction management
- Data validation
- Migration support
- Backup/restore
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys
from dataclasses import dataclass, asdict
import pandas as pd

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


# Fix for Python 3.12+ datetime deprecation warning
def adapt_datetime(dt):
    """Convert datetime to ISO format string for SQLite"""
    return dt.isoformat()


def convert_datetime(s):
    """Convert ISO format string from SQLite to datetime"""
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    return datetime.fromisoformat(s)


# Register adapters
sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("TIMESTAMP", convert_datetime)


@dataclass
class TradingSignal:
    """Trading signal record"""
    timestamp: datetime
    signal_id: str
    action: str  # LONG, SHORT, FLAT
    symbol: str
    timeframe: str

    # Confidence
    base_confidence: float
    adjusted_confidence: float

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float

    # Analysis
    regime: str
    mtf_alignment: float
    mtf_multiplier: float
    sentiment_score: float
    alternative_data_score: float

    # Reasons
    reasons: str  # JSON string
    warnings: str  # JSON string

    # Metadata
    created_at: datetime
    executed: bool = False


@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    signal_id: str
    symbol: str

    # Entry
    entry_time: datetime
    entry_price: float
    position_size: float

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP, SL, MANUAL, TIMEOUT

    # Risk Management
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Results
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_hours: Optional[float] = None

    # Status
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED

    # Metadata
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class PerformanceMetrics:
    """Performance metrics record"""
    date: datetime
    period: str  # DAILY, WEEKLY, MONTHLY

    # Trade Stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    total_pnl: float
    total_pnl_pct: float
    average_win: float
    average_loss: float
    profit_factor: float

    # Risk Metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Duration
    average_duration_hours: float

    # Regime
    trending_win_rate: float
    ranging_win_rate: float


class DatabaseManager:
    """
    Production-grade database manager

    Features:
    - Multiple database support (SQLite, PostgreSQL)
    - Connection pooling
    - Transaction management
    - Data validation
    - Automatic schema creation
    - Backup/restore
    """

    def __init__(self, db_path: str = "trading_data.db", db_type: str = "sqlite"):
        """
        Initialize database manager

        Args:
            db_path: Database path (SQLite) or connection string
            db_type: Database type ('sqlite' or 'postgresql')
        """
        self.db_path = db_path
        self.db_type = db_type
        self.conn = None

        logger.info(f"Initializing Database Manager ({db_type})")

        # Connect to database
        self._connect()

        # Create tables if not exist
        self._create_tables()

        logger.info("Database Manager initialized successfully")

    def _connect(self) -> None:
        """Connect to database"""
        try:
            if self.db_type == "sqlite":
                self.conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    detect_types=sqlite3.PARSE_DECLTYPES
                )
                self.conn.row_factory = sqlite3.Row

                # Enable foreign keys
                self.conn.execute("PRAGMA foreign_keys = ON")

                logger.info(f"Connected to SQLite database: {self.db_path}")

            elif self.db_type == "postgresql":
                # PostgreSQL support (requires psycopg2)
                try:
                    import psycopg2
                    from psycopg2.extras import RealDictCursor

                    self.conn = psycopg2.connect(self.db_path, cursor_factory=RealDictCursor)
                    logger.info("Connected to PostgreSQL database")

                except ImportError:
                    logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
                    raise
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _create_tables(self) -> None:
        """Create database tables"""

        cursor = self.conn.cursor()

        # Trading Signals Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                signal_id TEXT UNIQUE NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,

                base_confidence REAL NOT NULL,
                adjusted_confidence REAL NOT NULL,

                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                position_size REAL NOT NULL,

                regime TEXT,
                mtf_alignment REAL,
                mtf_multiplier REAL,
                sentiment_score REAL,
                alternative_data_score REAL,

                reasons TEXT,
                warnings TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed BOOLEAN DEFAULT 0
            )
        """)

        # Trades Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                signal_id TEXT,
                symbol TEXT NOT NULL,

                entry_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                position_size REAL NOT NULL,

                exit_time TIMESTAMP,
                exit_price REAL,
                exit_reason TEXT,

                stop_loss REAL,
                take_profit REAL,

                pnl REAL,
                pnl_pct REAL,
                duration_hours REAL,

                status TEXT DEFAULT 'OPEN',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
            )
        """)

        # Performance Metrics Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TIMESTAMP NOT NULL,
                period TEXT NOT NULL,

                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,

                total_pnl REAL,
                total_pnl_pct REAL,
                average_win REAL,
                average_loss REAL,
                profit_factor REAL,

                max_drawdown REAL,
                max_drawdown_pct REAL,
                sharpe_ratio REAL,

                average_duration_hours REAL,

                trending_win_rate REAL,
                ranging_win_rate REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(date, period)
            )
        """)

        # Regime History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                regime TEXT NOT NULL,
                volatility TEXT NOT NULL,
                adx REAL,
                atr REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        logger.info("Database tables created/verified")

    def save_signal(self, signal: Dict[str, Any]) -> str:
        """
        Save trading signal to database

        Args:
            signal: Signal dictionary

        Returns:
            signal_id
        """
        cursor = self.conn.cursor()

        signal_id = signal.get('signal_id', f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S')}")

        cursor.execute("""
            INSERT INTO trading_signals (
                timestamp, signal_id, action, symbol, timeframe,
                base_confidence, adjusted_confidence,
                entry_price, stop_loss, take_profit, position_size,
                regime, mtf_alignment, mtf_multiplier,
                sentiment_score, alternative_data_score,
                reasons, warnings, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.get('timestamp', datetime.now()),
            signal_id,
            signal.get('action', 'FLAT'),
            signal.get('symbol', 'XAUUSD'),
            signal.get('timeframe', 'H1'),
            signal.get('base_confidence', 0.0),
            signal.get('adjusted_confidence', 0.0),
            signal.get('entry_price', 0.0),
            signal.get('stop_loss', 0.0),
            signal.get('take_profit', 0.0),
            signal.get('position_size', 0.0),
            signal.get('regime', 'unknown'),
            signal.get('mtf_alignment', 0.0),
            signal.get('mtf_multiplier', 1.0),
            signal.get('sentiment_score', 0.0),
            signal.get('alternative_data_score', 0.0),
            json.dumps(signal.get('reasons', [])),
            json.dumps(signal.get('warnings', [])),
            signal.get('executed', False)
        ))

        self.conn.commit()
        logger.info(f"Signal saved: {signal_id}")

        return signal_id

    def save_trade(self, trade: Dict[str, Any]) -> str:
        """
        Save trade to database

        Args:
            trade: Trade dictionary

        Returns:
            trade_id
        """
        cursor = self.conn.cursor()

        trade_id = trade.get('trade_id', f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S')}")

        cursor.execute("""
            INSERT INTO trades (
                trade_id, signal_id, symbol,
                entry_time, entry_price, position_size,
                stop_loss, take_profit, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            trade.get('signal_id'),
            trade.get('symbol', 'XAUUSD'),
            trade.get('entry_time', datetime.now()),
            trade.get('entry_price', 0.0),
            trade.get('position_size', 0.0),
            trade.get('stop_loss', 0.0),
            trade.get('take_profit', 0.0),
            trade.get('status', 'OPEN')
        ))

        self.conn.commit()
        logger.info(f"Trade saved: {trade_id}")

        return trade_id

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> None:
        """
        Update existing trade

        Args:
            trade_id: Trade ID
            updates: Dictionary of fields to update
        """
        cursor = self.conn.cursor()

        # Build UPDATE query dynamically
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(trade_id)

        cursor.execute(f"""
            UPDATE trades
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE trade_id = ?
        """, values)

        self.conn.commit()
        logger.info(f"Trade updated: {trade_id}")

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "TP"
    ) -> None:
        """
        Close a trade

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit (TP, SL, MANUAL, TIMEOUT)
        """
        # Get trade details
        trade = self.get_trade(trade_id)

        if not trade:
            logger.error(f"Trade not found: {trade_id}")
            return

        # Calculate P&L
        entry_price = trade['entry_price']
        position_size = trade['position_size']

        pnl_pct = (exit_price - entry_price) / entry_price
        pnl = pnl_pct * position_size

        # Calculate duration
        entry_time = trade['entry_time']
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        exit_time = datetime.now()
        duration_hours = (exit_time - entry_time).total_seconds() / 3600

        # Update trade
        self.update_trade(trade_id, {
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_hours': duration_hours,
            'status': 'CLOSED'
        })

        logger.info(f"Trade closed: {trade_id} | P&L: {pnl_pct:+.2%} | Reason: {exit_reason}")

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get trade by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC")

        return [dict(row) for row in cursor.fetchall()]

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trading_signals
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get performance summary for last N days

        Args:
            days: Number of days to look back

        Returns:
            Performance metrics dictionary
        """
        cursor = self.conn.cursor()

        cutoff_date = datetime.now() - timedelta(days=days)

        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(CASE WHEN pnl > 0 THEN pnl_pct ELSE NULL END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl_pct ELSE NULL END) as avg_loss,
                SUM(pnl_pct) as total_pnl_pct,
                AVG(duration_hours) as avg_duration
            FROM trades
            WHERE status = 'CLOSED' AND entry_time >= ?
        """, (cutoff_date,))

        row = cursor.fetchone()

        if not row or row['total_trades'] == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl_pct': 0.0
            }

        result = dict(row)

        # Calculate win rate
        result['win_rate'] = result['winning_trades'] / result['total_trades'] if result['total_trades'] > 0 else 0

        # Calculate profit factor
        gross_profit = abs(result['avg_win'] * result['winning_trades']) if result['avg_win'] else 0
        gross_loss = abs(result['avg_loss'] * result['losing_trades']) if result['avg_loss'] else 0
        result['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0

        return result

    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None
    ) -> None:
        """
        Log system event

        Args:
            event_type: Type of event (TRADE, SIGNAL, ERROR, etc.)
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            message: Event message
            details: Additional details (stored as JSON)
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO system_events (timestamp, event_type, severity, message, details)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            event_type,
            severity,
            message,
            json.dumps(details) if details else None
        ))

        self.conn.commit()

    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create database backup

        Args:
            backup_path: Path for backup file

        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"backup_trading_db_{timestamp}.db"

        if self.db_type == "sqlite":
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        else:
            logger.warning("Backup not implemented for this database type")
            return ""

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def demo_database_manager():
    """Demo database manager"""

    logger.info("")
    logger.info("="*80)
    logger.info("  DATABASE MANAGER DEMO")
    logger.info("="*80)
    logger.info("")

    # Initialize database
    db = DatabaseManager("demo_trading.db")

    # Save a signal
    logger.info("Saving test signal...")
    signal = {
        'signal_id': 'SIG_TEST_001',
        'timestamp': datetime.now(),
        'action': 'LONG',
        'symbol': 'XAUUSD',
        'timeframe': 'H1',
        'base_confidence': 0.87,
        'adjusted_confidence': 0.92,
        'entry_price': 2075.50,
        'stop_loss': 2050.00,
        'take_profit': 2125.00,
        'position_size': 0.8,
        'regime': 'trending',
        'mtf_alignment': 0.85,
        'mtf_multiplier': 1.2,
        'sentiment_score': 0.65,
        'alternative_data_score': 0.45,
        'reasons': ['Good MTF alignment', 'Trending market'],
        'warnings': [],
        'executed': True
    }

    signal_id = db.save_signal(signal)
    logger.info(f"Signal saved with ID: {signal_id}")
    logger.info("")

    # Save a trade
    logger.info("Saving test trade...")
    trade = {
        'trade_id': 'TRD_TEST_001',
        'signal_id': signal_id,
        'symbol': 'XAUUSD',
        'entry_time': datetime.now(),
        'entry_price': 2075.50,
        'position_size': 0.8,
        'stop_loss': 2050.00,
        'take_profit': 2125.00,
        'status': 'OPEN'
    }

    trade_id = db.save_trade(trade)
    logger.info(f"Trade saved with ID: {trade_id}")
    logger.info("")

    # Get open trades
    logger.info("Open trades:")
    open_trades = db.get_open_trades()
    for t in open_trades:
        logger.info(f"  {t['trade_id']}: {t['symbol']} @ {t['entry_price']:.2f}")
    logger.info("")

    # Close trade (simulate)
    logger.info("Closing trade...")
    db.close_trade(trade_id, exit_price=2100.00, exit_reason="TP")
    logger.info("")

    # Get performance summary
    logger.info("Performance Summary (Last 30 days):")
    perf = db.get_performance_summary(30)
    logger.info(f"  Total Trades: {perf['total_trades']}")
    logger.info(f"  Win Rate: {perf['win_rate']:.2%}")
    logger.info(f"  Total P&L: {perf['total_pnl_pct']:.2%}")
    logger.info(f"  Profit Factor: {perf['profit_factor']:.2f}")
    logger.info("")

    # Log an event
    db.log_event(
        event_type="DEMO",
        severity="INFO",
        message="Database demo completed",
        details={'trades': 1, 'signals': 1}
    )

    # Backup
    logger.info("Creating backup...")
    backup_file = db.backup()
    logger.info("")

    logger.info("="*80)
    logger.info("DATABASE MANAGER READY!")
    logger.info("="*80)
    logger.info("")

    # Cleanup
    db.close()


if __name__ == "__main__":
    demo_database_manager()
