"""
Production Gold Trading System - MT5 Integration
================================================
Complete production-grade trading system with MetaTrader 5 integration

Features:
- Real-time data from MT5
- Advanced signal generation
- Risk management & monitoring
- Trade execution via MT5
- Paper trading support
- Performance tracking
- Circuit breakers
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from loguru import logger
import pandas as pd

# Import system components
from config_manager import ConfigManager
from database_manager import DatabaseManager
from mt5_data_provider import MT5DataProvider
from mt5_trade_executor import MT5TradeExecutor
from master_integration_system import MasterIntegrationSystem

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
    colorize=True
)
logger.add(
    "logs/production_mt5_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}"
)


class RiskMonitorMT5:
    """Risk monitoring for MT5 trading"""

    def __init__(self, config: ConfigManager, db: DatabaseManager):
        self.config = config
        self.db = db
        self.risk_config = config.get_risk_config()

    def check_daily_limits(self, executor: MT5TradeExecutor) -> bool:
        """Check if daily loss limits exceeded"""
        account = executor.get_account_info()
        if not account:
            return False

        # Get starting balance for today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Calculate daily P&L
        daily_pnl = account['profit']  # Current open P&L

        # Check daily loss limit
        daily_loss_pct = abs(daily_pnl / account['balance'])
        if daily_loss_pct > self.risk_config.max_daily_loss_pct:
            logger.error(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
            return False

        return True

    def check_position_limits(self, executor: MT5TradeExecutor, symbol: str) -> bool:
        """Check position limits"""
        positions = executor.get_positions(symbol=symbol)

        # Check max open positions
        trading_config = self.config.get_trading_config()
        if len(positions) >= trading_config.max_open_positions:
            logger.warning(f"Max positions reached: {len(positions)}")
            return False

        return True

    def check_drawdown(self, executor: MT5TradeExecutor) -> bool:
        """Check drawdown limits"""
        account = executor.get_account_info()
        if not account:
            return False

        # Calculate drawdown
        equity = account['equity']
        balance = account['balance']

        if balance > 0:
            drawdown = (balance - equity) / balance

            if drawdown > self.risk_config.max_drawdown_pct:
                logger.error(f"Max drawdown exceeded: {drawdown:.2%}")
                return False

        return True

    def validate_trade(self, executor: MT5TradeExecutor, symbol: str, signal) -> bool:
        """Validate trade against all risk checks"""

        # Check daily limits
        if not self.check_daily_limits(executor):
            logger.error("Trade blocked: Daily limits exceeded")
            return False

        # Check position limits
        if not self.check_position_limits(executor, symbol):
            logger.error("Trade blocked: Position limits exceeded")
            return False

        # Check drawdown
        if not self.check_drawdown(executor):
            logger.error("Trade blocked: Drawdown limit exceeded")
            return False

        # Check signal confidence
        trading_config = self.config.get_trading_config()
        if signal.adjusted_confidence < trading_config.min_confidence:
            logger.warning(f"Trade blocked: Low confidence {signal.adjusted_confidence:.2%}")
            return False

        logger.info("[OK] All risk checks passed")
        return True


class PaperTradingSimulator:
    """Simulate trades without real execution"""

    def __init__(self, initial_capital: float = 10000.0):
        self.balance = initial_capital
        self.equity = initial_capital
        self.positions = []
        self.trades_history = []

    def simulate_order(self, symbol: str, action: str, volume: float,
                      price: float, sl: float, tp: float) -> Dict:
        """Simulate order execution"""

        position = {
            'ticket': len(self.trades_history) + 1000,
            'symbol': symbol,
            'type': action,
            'volume': volume,
            'price_open': price,
            'sl': sl,
            'tp': tp,
            'time': datetime.now(),
            'profit': 0.0
        }

        self.positions.append(position)

        logger.info(f"[PAPER] {action} {volume} {symbol} @ {price}")
        logger.info(f"  SL: {sl:.2f} | TP: {tp:.2f}")

        return {
            'success': True,
            'order': position['ticket'],
            'volume': volume,
            'price': price,
            'comment': 'Paper trade'
        }

    def update_positions(self, current_price: float):
        """Update open positions with current price"""
        for pos in self.positions:
            if pos['type'] == 'BUY':
                pos['profit'] = (current_price - pos['price_open']) * pos['volume'] * 100
            else:  # SELL
                pos['profit'] = (pos['price_open'] - current_price) * pos['volume'] * 100

        # Calculate equity
        total_profit = sum(p['profit'] for p in self.positions)
        self.equity = self.balance + total_profit

    def close_position(self, ticket: int, current_price: float) -> bool:
        """Close simulated position"""
        for i, pos in enumerate(self.positions):
            if pos['ticket'] == ticket:
                # Calculate final profit
                if pos['type'] == 'BUY':
                    profit = (current_price - pos['price_open']) * pos['volume'] * 100
                else:
                    profit = (pos['price_open'] - current_price) * pos['volume'] * 100

                # Update balance
                self.balance += profit
                self.equity = self.balance

                # Record trade
                trade = {
                    **pos,
                    'close_price': current_price,
                    'close_time': datetime.now(),
                    'profit': profit
                }
                self.trades_history.append(trade)

                # Remove from positions
                self.positions.pop(i)

                logger.info(f"[PAPER] Closed position {ticket}")
                logger.info(f"  Profit: ${profit:.2f}")
                logger.info(f"  Balance: ${self.balance:.2f}")

                return True

        return False

    def get_account_info(self) -> Dict:
        """Get simulated account info"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'profit': self.equity - self.balance,
            'margin_free': self.balance * 0.9
        }

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions"""
        if symbol:
            return [p for p in self.positions if p['symbol'] == symbol]
        return self.positions


class ProductionSystemMT5:
    """
    Complete production trading system with MT5 integration

    Features:
    - Real-time data from MT5
    - Advanced signal generation
    - Risk management
    - Trade execution
    - Paper trading support
    - Performance monitoring
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        paper_trading: bool = False,
        mt5_login: Optional[int] = None,
        mt5_password: Optional[str] = None,
        mt5_server: Optional[str] = None
    ):
        """Initialize production system"""

        logger.info("="*70)
        logger.info("  PRODUCTION GOLD TRADING SYSTEM - MT5 INTEGRATION")
        logger.info("="*70)

        # Load configuration
        self.config = ConfigManager()
        logger.info("Configuration loaded from config.yaml")

        # Get credentials from environment if not provided
        if not mt5_login:
            mt5_login = os.getenv('MT5_LOGIN')
            mt5_password = os.getenv('MT5_PASSWORD')
            mt5_server = os.getenv('MT5_SERVER')

        # Initialize database
        db_config = self.config.get_database_config()
        self.db = DatabaseManager(db_path=db_config.sqlite_path)
        logger.info("Database initialized")

        # Initialize MT5 data provider
        self.data_provider = MT5DataProvider(
            login=int(mt5_login) if mt5_login else None,
            password=mt5_password,
            server=mt5_server
        )

        # Paper trading mode
        self.paper_trading = paper_trading

        if paper_trading:
            logger.info("[PAPER TRADING MODE] - No real trades will be executed")
            paper_config = self.config.get('paper_trading', {})
            initial_capital = paper_config.get('initial_capital', 10000.0)
            self.paper_simulator = PaperTradingSimulator(initial_capital)
            self.executor = None
        else:
            logger.info("[LIVE TRADING MODE] - Real trades will be executed")
            self.executor = MT5TradeExecutor(config=self.config)
            self.paper_simulator = None

        # Initialize master system
        ml_config = self.config.get('ml', {})
        use_ml = ml_config.get('enabled', False)

        trading_config = self.config.get_trading_config()
        self.master = MasterIntegrationSystem(
            use_ml=use_ml,
            min_confidence=trading_config.min_confidence
        )
        logger.info("Master integration system initialized")

        # Initialize risk monitor
        self.risk_monitor = RiskMonitorMT5(self.config, self.db)
        logger.info("Risk monitor initialized")

        # State
        self.running = False
        self.last_signal_time = None
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

        logger.info("="*70)

    def connect(self) -> bool:
        """Connect to MT5"""
        logger.info("Connecting to MT5...")

        # Connect data provider
        if not self.data_provider.connect():
            logger.error("Failed to connect data provider")
            return False

        logger.info("[OK] Data provider connected")

        # Trade executor uses the same MT5 connection (no separate connect needed)
        if not self.paper_trading:
            logger.info("[OK] Trade executor ready (using shared MT5 connection)")

        logger.info("[OK] MT5 connection established")
        return True

    def disconnect(self):
        """Disconnect from MT5"""
        logger.info("Disconnecting from MT5...")

        self.data_provider.disconnect()

        logger.info("MT5 disconnected")

    def fetch_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch multi-timeframe data from MT5"""

        trading_config = self.config.get_trading_config()
        symbol = trading_config.symbol

        # Get MTF config
        mtf_config = self.config.get('multi_timeframe', {})
        timeframes = list(mtf_config.get('timeframes', {}).keys())

        if not timeframes:
            timeframes = ["W1", "D1", "H4", "H1", "M30", "M15"]

        logger.info(f"Fetching MTF data for {symbol}...")

        mtf_data = self.data_provider.get_multi_timeframe(
            symbol=symbol,
            timeframes=timeframes,
            bars=1000
        )

        if len(mtf_data) == 0:
            logger.error("Failed to fetch MTF data")
            return None

        logger.info(f"[OK] Fetched {len(mtf_data)} timeframes")
        return mtf_data

    def generate_signal(self, mtf_data: Dict[str, pd.DataFrame]):
        """Generate trading signal"""

        logger.info("Generating trading signal...")

        # Get primary timeframe data (H1)
        trading_config = self.config.get_trading_config()
        primary_tf = trading_config.timeframe

        if primary_tf not in mtf_data:
            logger.error(f"Primary timeframe {primary_tf} not in MTF data")
            return None

        df_primary = mtf_data[primary_tf]

        signal = self.master.generate_master_signal(df_primary, mtf_data)

        if signal:
            logger.info("[OK] Signal generated")
            logger.info(f"  Action: {signal.action}")
            logger.info(f"  Confidence: {signal.adjusted_confidence:.2%}")
            logger.info(f"  MTF Multiplier: {signal.mtf_multiplier:.2f}x")
            logger.info(f"  Regime: {signal.regime}")

            # Store in database
            import dataclasses
            signal_dict = dataclasses.asdict(signal)
            self.db.save_signal(signal_dict)

        return signal

    def execute_signal(self, signal) -> bool:
        """Execute trading signal"""

        if signal.action == "HOLD":
            logger.info("Signal: HOLD - No action taken")
            return False

        trading_config = self.config.get_trading_config()
        symbol = trading_config.symbol

        # Get current price
        if self.paper_trading:
            # Use entry price from signal
            price = signal.entry_price
        else:
            prices = self.data_provider.get_current_price(symbol)
            if not prices:
                logger.error("Failed to get current price")
                return False
            bid, ask = prices
            price = ask if signal.action == "BUY" else bid

        # Validate trade
        executor_or_sim = self.paper_simulator if self.paper_trading else self.executor

        if not self.risk_monitor.validate_trade(executor_or_sim, symbol, signal):
            logger.error("Trade validation failed")
            return False

        # Calculate position size
        volume = trading_config.default_position_size

        # Calculate SL/TP
        sl_pct = trading_config.stop_loss_default
        tp_pct = sl_pct * trading_config.risk_reward_min

        if signal.action == "BUY":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:  # SELL
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)

        # Execute order
        logger.info(f"Executing {signal.action} order...")

        if self.paper_trading:
            result = self.paper_simulator.simulate_order(
                symbol=symbol,
                action=signal.action,
                volume=volume,
                price=price,
                sl=sl,
                tp=tp
            )
        else:
            # Map signal action to MT5 order type
            order_type = "BUY" if signal.action == "LONG" else "SELL"
            
            trade_result = self.executor.execute_market_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                stop_loss=sl,
                take_profit=tp
            )
            result = {
                'success': trade_result.success,
                'order': trade_result.order_id,
                'volume': trade_result.volume,
                'price': trade_result.price
            }

        if result and result.get('success'):
            logger.info("[OK] Order executed successfully")

            # Save trade to database
            trade = {
                'signal_id': signal.signal_id,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': signal.action,
                'entry_price': price,
                'volume': volume,
                'stop_loss': sl,
                'take_profit': tp,
                'status': 'OPEN'
            }
            self.db.save_trade(trade)

            return True
        else:
            logger.error("Order execution failed")
            return False

    def run_live(self, check_interval: int = 300):
        """
        Run live trading system

        Args:
            check_interval: Seconds between signal checks (default: 300 = 5 minutes)
        """

        logger.info("="*70)
        logger.info("  STARTING LIVE TRADING")
        logger.info("="*70)
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info("="*70)

        # Connect to MT5
        if not self.connect():
            logger.error("Failed to connect to MT5")
            return

        self.running = True

        try:
            iteration = 0

            while self.running:
                iteration += 1
                logger.info("")
                logger.info(f"{'='*70}")
                logger.info(f"  ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")

                # Check circuit breaker
                if self.circuit_breaker_active:
                    if datetime.now() < self.circuit_breaker_until:
                        remaining = (self.circuit_breaker_until - datetime.now()).total_seconds() / 60
                        logger.warning(f"Circuit breaker active - {remaining:.0f} minutes remaining")
                        time.sleep(60)
                        continue
                    else:
                        logger.info("Circuit breaker deactivated")
                        self.circuit_breaker_active = False

                # Fetch data
                mtf_data = self.fetch_data()

                if not mtf_data:
                    logger.error("Failed to fetch data, retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                # Generate signal
                signal = self.generate_signal(mtf_data)

                if not signal:
                    logger.warning("No signal generated")
                else:
                    # Execute if not HOLD
                    if signal.action != "HOLD":
                        self.execute_signal(signal)

                # Update paper trading positions
                if self.paper_trading and self.paper_simulator:
                    trading_config = self.config.get_trading_config()
                    prices = self.data_provider.get_current_price(trading_config.symbol)
                    if prices:
                        current_price = (prices[0] + prices[1]) / 2
                        self.paper_simulator.update_positions(current_price)

                        account = self.paper_simulator.get_account_info()
                        logger.info(f"Paper Account - Balance: ${account['balance']:.2f} | Equity: ${account['equity']:.2f}")

                # Log account status
                if not self.paper_trading and self.executor:
                    account = self.executor.get_account_info()
                    if account:
                        logger.info(f"Account - Balance: ${account['balance']:.2f} | Equity: ${account['equity']:.2f}")

                # Wait for next iteration
                logger.info(f"Waiting {check_interval} seconds...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("")
            logger.warning("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            logger.info("Stopping system...")
            self.disconnect()
            logger.info("System stopped")

    def activate_circuit_breaker(self, duration_hours: int = 24):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(hours=duration_hours)
        logger.error(f"CIRCUIT BREAKER ACTIVATED - Trading paused for {duration_hours} hours")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Production Gold Trading System - MT5")

    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Run in paper trading mode (simulated trades)'
    )

    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode (real trades)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)'
    )

    parser.add_argument(
        '--login',
        type=int,
        help='MT5 login (or use MT5_LOGIN env variable)'
    )

    parser.add_argument(
        '--password',
        type=str,
        help='MT5 password (or use MT5_PASSWORD env variable)'
    )

    parser.add_argument(
        '--server',
        type=str,
        help='MT5 server (or use MT5_SERVER env variable)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.live and args.paper_trading:
        logger.error("Cannot use both --live and --paper-trading")
        sys.exit(1)

    paper_trading = args.paper_trading or not args.live

    # Initialize system
    system = ProductionSystemMT5(
        paper_trading=paper_trading,
        mt5_login=args.login,
        mt5_password=args.password,
        mt5_server=args.server
    )

    # Run
    system.run_live(check_interval=args.interval)


if __name__ == "__main__":
    main()
