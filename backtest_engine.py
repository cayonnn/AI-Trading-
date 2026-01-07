"""
Advanced Backtesting Engine
===========================
Production-grade backtesting with event-driven simulation.

Features:
- Event-driven architecture
- Realistic execution simulation
- Transaction cost integration
- Walk-forward validation
- Monte Carlo simulation
- Performance analytics

Based on:
- Quantitative trading best practices
- Hedge fund backtesting standards
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
import json
import os

# Local imports
try:
    from transaction_cost_model import TransactionCostModel, TransactionCosts
except ImportError:
    TransactionCostModel = None


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Order:
    """Represents a trading order"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: str = ""
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ORD-{self.timestamp.strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"


@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    transaction_costs: float = 0.0
    net_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    holding_bars: int = 0
    exit_reason: str = ""
    status: PositionStatus = PositionStatus.OPEN


@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    metrics: Dict[str, float]
    monthly_returns: pd.Series
    trade_df: pd.DataFrame
    config: Dict[str, Any] = field(default_factory=dict)
    

class BacktestEngine:
    """
    Production-grade backtesting engine.
    
    Features:
    - Event-driven simulation
    - Transaction cost modeling
    - Realistic execution
    - Comprehensive metrics
    
    Usage:
        engine = BacktestEngine(
            initial_capital=100000,
            symbol="GOLD"
        )
        result = engine.run(data, signals)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        symbol: str = "GOLD",
        position_size: float = 0.1,  # 10% of capital per trade
        max_positions: int = 3,
        use_transaction_costs: bool = True,
        slippage_model: str = "adaptive",
        commission_per_lot: float = 0.0,
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            symbol: Trading symbol
            position_size: Position size as fraction of capital
            max_positions: Maximum concurrent positions
            use_transaction_costs: Include transaction costs
            slippage_model: Type of slippage model
            commission_per_lot: Commission per lot traded
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.symbol = symbol
        self.position_size = position_size
        self.max_positions = max_positions
        self.use_transaction_costs = use_transaction_costs
        
        # Initialize transaction cost model
        if use_transaction_costs and TransactionCostModel:
            self.cost_model = TransactionCostModel(
                symbol=symbol,
                slippage_model=slippage_model,
                commission_override=commission_per_lot,
            )
        else:
            self.cost_model = None
        
        # State
        self.positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.0f} capital")
    
    def reset(self):
        """Reset engine state for new backtest"""
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_history = []
    
    def calculate_position_size(
        self,
        price: float,
        stop_loss: float,
        risk_pct: float = 0.02,
    ) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            price: Entry price
            stop_loss: Stop loss price
            risk_pct: Maximum risk per trade as % of capital
            
        Returns:
            Position size in lots
        """
        risk_amount = self.capital * risk_pct
        price_risk = abs(price - stop_loss)
        
        if price_risk <= 0:
            return 0.1  # Minimum size
        
        # For gold: 1 lot = 100 oz
        lot_size = 100
        size_in_lots = risk_amount / (price_risk * lot_size)
        
        # Apply limits
        max_size = self.capital * self.position_size / (price * lot_size)
        size_in_lots = min(size_in_lots, max_size, 10.0)  # Max 10 lots
        size_in_lots = max(size_in_lots, 0.01)  # Min 0.01 lots
        
        return round(size_in_lots, 2)
    
    def open_position(
        self,
        timestamp: datetime,
        side: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        quantity: Optional[float] = None,
        volatility: float = 0.01,
    ) -> Optional[Trade]:
        """
        Open a new position.
        
        Args:
            timestamp: Entry time
            side: "BUY" or "SELL"
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            quantity: Position size (calculate if None)
            volatility: Current volatility for cost calculation
            
        Returns:
            Trade object or None if rejected
        """
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return None
        
        # Calculate position size
        if quantity is None:
            quantity = self.calculate_position_size(price, stop_loss)
        
        # Calculate transaction costs
        costs = 0.0
        if self.cost_model:
            cost_result = self.cost_model.calculate_costs(
                price=price,
                volume=quantity,
                side=side,
                volatility=volatility,
                include_swap=False,
            )
            costs = cost_result.total_cost
        
        # Create trade
        trade = Trade(
            trade_id=f"T-{timestamp.strftime('%Y%m%d%H%M%S')}-{len(self.closed_trades) + len(self.positions) + 1}",
            symbol=self.symbol,
            side=side,
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            transaction_costs=costs,
            status=PositionStatus.OPEN,
        )
        
        self.positions.append(trade)
        return trade
    
    def close_position(
        self,
        trade: Trade,
        timestamp: datetime,
        price: float,
        reason: str = "signal",
        volatility: float = 0.01,
    ):
        """
        Close an open position.
        
        Args:
            trade: Trade to close
            timestamp: Exit time
            price: Exit price
            reason: Reason for closing
            volatility: Current volatility
        """
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        trade.status = PositionStatus.CLOSED
        
        # Calculate P&L
        lot_size = 100  # Gold lot size
        if trade.side == "BUY":
            trade.pnl = (price - trade.entry_price) * trade.quantity * lot_size
        else:
            trade.pnl = (trade.entry_price - price) * trade.quantity * lot_size
        
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity * lot_size)
        
        # Add exit transaction costs
        if self.cost_model:
            exit_side = "SELL" if trade.side == "BUY" else "BUY"
            exit_costs = self.cost_model.calculate_costs(
                price=price,
                volume=trade.quantity,
                side=exit_side,
                volatility=volatility,
                include_swap=False,
            )
            trade.transaction_costs += exit_costs.total_cost
        
        trade.net_pnl = trade.pnl - trade.transaction_costs
        
        # Update capital
        self.capital += trade.net_pnl
        
        # Move to closed trades
        self.positions.remove(trade)
        self.closed_trades.append(trade)
    
    def check_stops(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ):
        """
        Check stop loss and take profit levels.
        
        Args:
            timestamp: Current bar time
            high: Bar high
            low: Bar low
            close: Bar close
        """
        for trade in self.positions.copy():
            if trade.side == "BUY":
                # Check stop loss
                if low <= trade.stop_loss:
                    self.close_position(trade, timestamp, trade.stop_loss, "stop_loss")
                # Check take profit
                elif high >= trade.take_profit:
                    self.close_position(trade, timestamp, trade.take_profit, "take_profit")
                # Track MFE/MAE
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, high - trade.entry_price)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, trade.entry_price - low)
            else:
                # Short position
                if high >= trade.stop_loss:
                    self.close_position(trade, timestamp, trade.stop_loss, "stop_loss")
                elif low <= trade.take_profit:
                    self.close_position(trade, timestamp, trade.take_profit, "take_profit")
                trade.max_favorable_excursion = max(trade.max_favorable_excursion, trade.entry_price - low)
                trade.max_adverse_excursion = max(trade.max_adverse_excursion, high - trade.entry_price)
            
            trade.holding_bars += 1
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        progress_callback: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation.
        
        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            signals: DataFrame with columns [signal, stop_loss, take_profit, confidence]
                     signal: 1 = BUY, -1 = SELL, 0 = FLAT
            progress_callback: Optional callback for progress updates
            
        Returns:
            BacktestResult with all metrics and trades
        """
        self.reset()
        logger.info(f"Starting backtest with {len(data)} bars")
        
        # Ensure data and signals are aligned
        common_idx = data.index.intersection(signals.index)
        data = data.loc[common_idx]
        signals = signals.loc[common_idx]
        
        total_bars = len(data)
        
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i / total_bars)
            
            # Get signal
            signal_row = signals.loc[timestamp]
            signal = signal_row.get("signal", 0)
            
            # Check stops first
            self.check_stops(timestamp, bar["high"], bar["low"], bar["close"])
            
            # Handle signals
            if signal == 1 and len(self.positions) < self.max_positions:
                # Buy signal
                sl = signal_row.get("stop_loss", bar["close"] * 0.988)
                tp = signal_row.get("take_profit", bar["close"] * 1.024)
                volatility = signal_row.get("volatility", 0.01)
                
                self.open_position(
                    timestamp=timestamp,
                    side="BUY",
                    price=bar["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    volatility=volatility,
                )
            
            elif signal == -1 and len(self.positions) < self.max_positions:
                # Sell signal
                sl = signal_row.get("stop_loss", bar["close"] * 1.012)
                tp = signal_row.get("take_profit", bar["close"] * 0.976)
                volatility = signal_row.get("volatility", 0.01)
                
                self.open_position(
                    timestamp=timestamp,
                    side="SELL",
                    price=bar["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    volatility=volatility,
                )
            
            # Close on opposite signal
            elif signal != 0:
                for trade in self.positions.copy():
                    if (signal == 1 and trade.side == "SELL") or \
                       (signal == -1 and trade.side == "BUY"):
                        self.close_position(trade, timestamp, bar["close"], "signal_reversal")
            
            # Update equity
            unrealized_pnl = self._calculate_unrealized_pnl(bar["close"])
            self.equity_history.append((timestamp, self.capital + unrealized_pnl))
        
        # Close any remaining positions
        if self.positions and len(data) > 0:
            last_bar = data.iloc[-1]
            last_time = data.index[-1]
            for trade in self.positions.copy():
                self.close_position(trade, last_time, last_bar["close"], "end_of_backtest")
        
        # Calculate metrics
        result = self._compile_results()
        
        logger.info(f"Backtest complete: {len(self.closed_trades)} trades")
        logger.info(f"Final equity: ${self.capital:,.2f}")
        
        return result
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for open positions"""
        pnl = 0.0
        lot_size = 100
        
        for trade in self.positions:
            if trade.side == "BUY":
                pnl += (current_price - trade.entry_price) * trade.quantity * lot_size
            else:
                pnl += (trade.entry_price - current_price) * trade.quantity * lot_size
        
        return pnl
    
    def _compile_results(self) -> BacktestResult:
        """Compile backtest results and calculate metrics"""
        # Create equity curve
        if self.equity_history:
            equity_curve = pd.Series(
                [e[1] for e in self.equity_history],
                index=[e[0] for e in self.equity_history]
            )
        else:
            equity_curve = pd.Series([self.initial_capital])
        
        # Calculate drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown_curve = (equity_curve - rolling_max) / rolling_max
        
        # Create trade DataFrame
        if self.closed_trades:
            trade_data = []
            for t in self.closed_trades:
                trade_data.append({
                    "trade_id": t.trade_id,
                    "side": t.side,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "net_pnl": t.net_pnl,
                    "pnl_pct": t.pnl_pct,
                    "transaction_costs": t.transaction_costs,
                    "holding_bars": t.holding_bars,
                    "exit_reason": t.exit_reason,
                    "mfe": t.max_favorable_excursion,
                    "mae": t.max_adverse_excursion,
                })
            trade_df = pd.DataFrame(trade_data)
        else:
            trade_df = pd.DataFrame()
        
        # Calculate monthly returns
        monthly_equity = equity_curve.resample("M").last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(trade_df, equity_curve, drawdown_curve)
        
        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            metrics=metrics,
            monthly_returns=monthly_returns,
            trade_df=trade_df,
            config={
                "initial_capital": self.initial_capital,
                "symbol": self.symbol,
                "position_size": self.position_size,
                "max_positions": self.max_positions,
                "use_transaction_costs": self.use_transaction_costs,
            }
        )
    
    def _calculate_metrics(
        self,
        trade_df: pd.DataFrame,
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        if trade_df.empty:
            return {"total_trades": 0, "message": "No trades"}
        
        # Basic metrics
        metrics["total_trades"] = len(trade_df)
        metrics["winning_trades"] = len(trade_df[trade_df["net_pnl"] > 0])
        metrics["losing_trades"] = len(trade_df[trade_df["net_pnl"] <= 0])
        metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        
        # P&L metrics
        metrics["total_pnl"] = trade_df["pnl"].sum()
        metrics["total_net_pnl"] = trade_df["net_pnl"].sum()
        metrics["total_costs"] = trade_df["transaction_costs"].sum()
        metrics["average_pnl"] = trade_df["net_pnl"].mean()
        
        # Win/Loss analysis
        winners = trade_df[trade_df["net_pnl"] > 0]["net_pnl"]
        losers = trade_df[trade_df["net_pnl"] <= 0]["net_pnl"]
        
        metrics["average_win"] = winners.mean() if len(winners) > 0 else 0
        metrics["average_loss"] = losers.mean() if len(losers) > 0 else 0
        metrics["largest_win"] = winners.max() if len(winners) > 0 else 0
        metrics["largest_loss"] = losers.min() if len(losers) > 0 else 0
        
        # Profit factor
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        metrics["max_drawdown"] = drawdown_curve.min()
        metrics["max_drawdown_pct"] = abs(metrics["max_drawdown"])
        
        # Sharpe ratio (annualized)
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            if returns.std() > 0:
                metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252 * 24)  # Hourly data
            else:
                metrics["sharpe_ratio"] = 0
        else:
            metrics["sharpe_ratio"] = 0
        
        # Sortino ratio
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics["sortino_ratio"] = (returns.mean() / downside_returns.std()) * np.sqrt(252 * 24)
            else:
                metrics["sortino_ratio"] = 0
        else:
            metrics["sortino_ratio"] = 0
        
        # Calmar ratio
        if metrics["max_drawdown_pct"] > 0:
            annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 * 24 / len(equity_curve)) - 1
            metrics["calmar_ratio"] = annual_return / metrics["max_drawdown_pct"]
        else:
            metrics["calmar_ratio"] = 0
        
        # Trade analysis
        metrics["average_holding_bars"] = trade_df["holding_bars"].mean()
        metrics["average_mfe"] = trade_df["mfe"].mean()
        metrics["average_mae"] = trade_df["mae"].mean()
        
        # Exit reason analysis
        metrics["stop_loss_exits"] = len(trade_df[trade_df["exit_reason"] == "stop_loss"])
        metrics["take_profit_exits"] = len(trade_df[trade_df["exit_reason"] == "take_profit"])
        metrics["signal_exits"] = len(trade_df[trade_df["exit_reason"].isin(["signal", "signal_reversal"])])
        
        # Return metrics
        metrics["total_return_pct"] = (self.capital - self.initial_capital) / self.initial_capital
        metrics["final_equity"] = self.capital
        
        # Expectancy
        metrics["expectancy"] = (metrics["win_rate"] * metrics["average_win"]) + \
                                ((1 - metrics["win_rate"]) * metrics["average_loss"])
        
        return metrics


class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing.
    
    Methods:
    - Trade resampling (bootstrap)
    - Parameter sensitivity analysis
    - Confidence interval estimation
    """
    
    def __init__(self, n_simulations: int = 1000, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def bootstrap_trades(
        self,
        trades: List[Trade],
        n_trades_per_sim: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Bootstrap trade resampling simulation.
        
        Args:
            trades: List of historical trades
            n_trades_per_sim: Number of trades per simulation
            
        Returns:
            Dictionary with simulation results
        """
        if not trades:
            return {"error": "No trades provided"}
        
        trade_pnls = [t.net_pnl for t in trades]
        n_trades = n_trades_per_sim or len(trades)
        
        # Results arrays
        final_pnls = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        win_rates = np.zeros(self.n_simulations)
        sharpe_ratios = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            # Random sample with replacement
            sampled_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)
            
            # Calculate metrics
            equity_curve = np.cumsum(sampled_pnls)
            final_pnls[i] = equity_curve[-1]
            
            # Max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / (running_max + 1e-10)
            max_drawdowns[i] = abs(drawdown.min())
            
            # Win rate
            win_rates[i] = np.mean(sampled_pnls > 0)
            
            # Sharpe (simple version)
            if sampled_pnls.std() > 0:
                sharpe_ratios[i] = sampled_pnls.mean() / sampled_pnls.std() * np.sqrt(252)
            else:
                sharpe_ratios[i] = 0
        
        return {
            "final_pnl": final_pnls,
            "max_drawdown": max_drawdowns,
            "win_rate": win_rates,
            "sharpe_ratio": sharpe_ratios,
            "confidence_intervals": self._calculate_confidence_intervals(final_pnls),
            "risk_of_ruin": np.mean(final_pnls < -0.5 * np.abs(final_pnls.mean())),  # 50% loss
        }
    
    def _calculate_confidence_intervals(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate confidence intervals"""
        return {
            "5th_percentile": np.percentile(values, 5),
            "25th_percentile": np.percentile(values, 25),
            "median": np.percentile(values, 50),
            "75th_percentile": np.percentile(values, 75),
            "95th_percentile": np.percentile(values, 95),
            "mean": np.mean(values),
            "std": np.std(values),
        }
    
    def sensitivity_analysis(
        self,
        backtest_func: Callable,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """
        Parameter sensitivity analysis.
        
        Args:
            backtest_func: Function that runs backtest with parameters
            base_params: Base parameters
            param_ranges: Dictionary of parameter names to (min, max) ranges
            n_samples: Number of samples per parameter
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            logger.info(f"Testing sensitivity of {param_name}")
            
            for val in np.linspace(min_val, max_val, n_samples):
                # Create modified params
                params = base_params.copy()
                params[param_name] = val
                
                try:
                    # Run backtest
                    result = backtest_func(**params)
                    
                    results.append({
                        "parameter": param_name,
                        "value": val,
                        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                        "total_return": result.metrics.get("total_return_pct", 0),
                        "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
                        "profit_factor": result.metrics.get("profit_factor", 0),
                        "win_rate": result.metrics.get("win_rate", 0),
                    })
                except Exception as e:
                    logger.warning(f"Failed for {param_name}={val}: {e}")
        
        return pd.DataFrame(results)


class WalkForwardOptimizer:
    """
    Walk-forward optimization and validation.
    
    Prevents overfitting by using rolling out-of-sample testing.
    """
    
    def __init__(
        self,
        train_period: int = 180,  # Training window in bars
        test_period: int = 30,    # Testing window in bars
        step: int = 30,           # Step size between windows
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            train_period: Number of bars for training
            test_period: Number of bars for testing
            step: Step size between windows
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
    
    def generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test windows.
        
        Args:
            data: Full dataset
            
        Returns:
            List of (train_data, test_data) tuples
        """
        windows = []
        total_length = len(data)
        
        start = 0
        while start + self.train_period + self.test_period <= total_length:
            train_end = start + self.train_period
            test_end = train_end + self.test_period
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            windows.append((train_data, test_data))
            start += self.step
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        engine: BacktestEngine,
    ) -> Dict:
        """
        Run walk-forward validation.
        
        Args:
            data: Full OHLCV data
            signals: Full signal data
            engine: BacktestEngine instance
            
        Returns:
            Dictionary with walk-forward results
        """
        windows = self.generate_windows(data)
        
        all_results = []
        aggregated_trades = []
        
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"Running window {i+1}/{len(windows)}")
            
            # Get corresponding signals
            train_signals = signals.loc[train_data.index]
            test_signals = signals.loc[test_data.index]
            
            # Run backtest on test data
            result = engine.run(test_data, test_signals)
            
            all_results.append({
                "window": i + 1,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                "n_trades": result.metrics.get("total_trades", 0),
                "return_pct": result.metrics.get("total_return_pct", 0),
                "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
                "win_rate": result.metrics.get("win_rate", 0),
            })
            
            aggregated_trades.extend(result.trades)
        
        results_df = pd.DataFrame(all_results)
        
        return {
            "window_results": results_df,
            "aggregated_trades": aggregated_trades,
            "summary": {
                "total_windows": len(windows),
                "avg_sharpe": results_df["sharpe_ratio"].mean(),
                "avg_return": results_df["return_pct"].mean(),
                "avg_drawdown": results_df["max_drawdown"].mean(),
                "avg_win_rate": results_df["win_rate"].mean(),
                "consistency": (results_df["return_pct"] > 0).mean(),  # % of profitable windows
            }
        }


def run_backtest_from_csv(
    data_path: str,
    signal_generator: Callable,
    initial_capital: float = 100000,
) -> BacktestResult:
    """
    Convenience function to run backtest from CSV file.
    
    Args:
        data_path: Path to OHLCV CSV file
        signal_generator: Function that generates signals from data
        initial_capital: Starting capital
        
    Returns:
        BacktestResult
    """
    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Generate signals
    signals = signal_generator(df)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(df, signals)
    
    return result


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    logger.info("=" * 60)
    logger.info("Testing Backtest Engine")
    logger.info("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="H")
    
    prices = 2300 + np.cumsum(np.random.randn(500) * 2)
    data = pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(500) * 5,
        "low": prices - np.random.rand(500) * 5,
        "close": prices + np.random.randn(500) * 2,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)
    
    # Create sample signals
    signal_values = np.zeros(500)
    for i in range(20, 500, 50):
        signal_values[i] = 1 if np.random.random() > 0.5 else -1
    
    signals = pd.DataFrame({
        "signal": signal_values,
        "stop_loss": data["close"] * (1 - 0.012 * np.sign(signal_values).replace(0, 1)),
        "take_profit": data["close"] * (1 + 0.024 * np.sign(signal_values).replace(0, 1)),
        "volatility": 0.012,
    }, index=dates)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000, use_transaction_costs=True)
    result = engine.run(data, signals)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:12.4f}")
        else:
            print(f"  {key:25s}: {value}")
    
    # Monte Carlo simulation
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)
    
    mc = MonteCarloSimulator(n_simulations=1000)
    mc_results = mc.bootstrap_trades(result.trades)
    
    print("\nConfidence Intervals for Final P&L:")
    for key, value in mc_results["confidence_intervals"].items():
        print(f"  {key:20s}: ${value:,.2f}")
    
    print(f"\nRisk of Ruin (50% loss): {mc_results['risk_of_ruin']:.2%}")
    
    print("\n✅ Backtest Engine Test Complete")
