"""
Transaction Cost Model
======================
Production-grade transaction cost modeling for realistic backtesting.

Features:
- Slippage modeling (market impact)
- Bid-Ask spread inclusion
- Commission/fee calculation
- Execution quality analysis

Based on:
- Market microstructure research
- Almgren-Chriss impact model
- Hedge fund best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from loguru import logger
from enum import Enum


class ExecutionType(Enum):
    """Trade execution types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TransactionCosts:
    """Container for all transaction costs"""
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    swap_cost: float = 0.0
    total_cost: float = 0.0
    cost_pct: float = 0.0
    
    def __post_init__(self):
        self.total_cost = self.spread_cost + self.slippage_cost + self.commission + self.swap_cost
        

@dataclass
class ExecutionResult:
    """Result of trade execution with costs"""
    intended_price: float
    executed_price: float
    costs: TransactionCosts
    fill_rate: float = 1.0  # Percentage filled
    execution_time_ms: float = 0.0
    slippage_pips: float = 0.0
    

class TransactionCostModel:
    """
    Production-grade transaction cost model.
    
    Includes:
    - Bid-Ask spread modeling
    - Market impact (Almgren-Chriss inspired)
    - Commission calculation
    - Swap/overnight costs
    
    Usage:
        tcm = TransactionCostModel(symbol="GOLD")
        costs = tcm.calculate_costs(
            price=2000.0,
            volume=1.0,
            side="BUY",
            volatility=0.015,
            avg_daily_volume=10000.0
        )
    """
    
    # Default spread configuration (in pips/points)
    DEFAULT_SPREADS = {
        "GOLD": 0.30,      # 30 cents for gold
        "XAUUSD": 0.30,
        "SILVER": 0.03,
        "XAGUSD": 0.03,
        "EURUSD": 0.0001,  # 1 pip
        "GBPUSD": 0.00012,
        "USDJPY": 0.01,
        "SPX500": 0.50,
        "US30": 2.0,
    }
    
    # Commission rates (per lot)
    DEFAULT_COMMISSIONS = {
        "GOLD": 0.0,       # Usually included in spread for CFDs
        "XAUUSD": 0.0,
        "SILVER": 0.0,
        "FOREX": 7.0,      # $7 per lot round trip
    }
    
    def __init__(
        self,
        symbol: str = "GOLD",
        spread_override: Optional[float] = None,
        commission_override: Optional[float] = None,
        slippage_model: str = "adaptive",  # "fixed", "adaptive", "almgren-chriss"
        impact_coefficient: float = 0.1,   # Market impact coefficient
        volatility_impact_scale: float = 0.5,
    ):
        """
        Initialize transaction cost model.
        
        Args:
            symbol: Trading symbol
            spread_override: Override default spread
            commission_override: Override default commission
            slippage_model: Type of slippage model
            impact_coefficient: Market impact coefficient (higher = more slippage)
            volatility_impact_scale: How much volatility affects slippage
        """
        self.symbol = symbol.upper()
        self.slippage_model = slippage_model
        self.impact_coefficient = impact_coefficient
        self.volatility_impact_scale = volatility_impact_scale
        
        # Set spread
        if spread_override is not None:
            self.base_spread = spread_override
        else:
            self.base_spread = self.DEFAULT_SPREADS.get(self.symbol, 0.30)
        
        # Set commission
        if commission_override is not None:
            self.base_commission = commission_override
        else:
            self.base_commission = self.DEFAULT_COMMISSIONS.get(self.symbol, 0.0)
        
        # Statistics tracking
        self.execution_history: List[ExecutionResult] = []
        
        logger.info(f"TransactionCostModel initialized for {symbol}")
        logger.info(f"  Base spread: {self.base_spread}")
        logger.info(f"  Commission: {self.base_commission}")
        logger.info(f"  Slippage model: {slippage_model}")
    
    def calculate_spread_cost(
        self,
        price: float,
        volume: float,
        spread_override: Optional[float] = None
    ) -> float:
        """
        Calculate bid-ask spread cost.
        
        Args:
            price: Current price
            volume: Position size in lots
            spread_override: Override spread for this calculation
            
        Returns:
            Spread cost in currency
        """
        spread = spread_override if spread_override else self.base_spread
        
        # For gold, spread is in dollars per oz
        # 1 lot = 100 oz typically
        lot_size = 100 if "GOLD" in self.symbol or "XAU" in self.symbol else 100000
        
        # Spread cost = spread * volume * lot_size
        cost = spread * volume * lot_size
        
        return cost
    
    def calculate_slippage(
        self,
        price: float,
        volume: float,
        side: str,
        volatility: float = 0.01,
        avg_daily_volume: float = 10000.0,
        urgency: float = 0.5,  # 0 = patient, 1 = urgent
    ) -> Tuple[float, float]:
        """
        Calculate market impact slippage.
        
        Args:
            price: Intended execution price
            volume: Order volume in lots
            side: "BUY" or "SELL"
            volatility: Current volatility (e.g., ATR%)
            avg_daily_volume: Average daily volume
            urgency: Trade urgency (affects market impact)
            
        Returns:
            Tuple of (slippage_amount, slippage_pips)
        """
        if self.slippage_model == "fixed":
            # Fixed slippage model
            slippage_pct = 0.0001  # 0.01% fixed
            slippage = price * slippage_pct
            
        elif self.slippage_model == "adaptive":
            # Adaptive model based on volatility and volume
            # Higher volatility = more slippage
            # Higher relative volume = more slippage
            
            vol_factor = volatility * self.volatility_impact_scale
            
            # Volume impact (relative to ADV)
            lot_size = 100 if "GOLD" in self.symbol else 100000
            order_value = volume * lot_size
            volume_ratio = min(order_value / avg_daily_volume, 1.0) if avg_daily_volume > 0 else 0.1
            
            # Combined slippage
            base_slippage = self.impact_coefficient * 0.0001  # Base slippage
            vol_slippage = vol_factor * 0.0005  # Volatility component
            size_slippage = volume_ratio * 0.0002  # Size impact
            urgency_slippage = urgency * 0.0001  # Urgency impact
            
            slippage_pct = base_slippage + vol_slippage + size_slippage + urgency_slippage
            slippage = price * slippage_pct
            
        elif self.slippage_model == "almgren-chriss":
            # Almgren-Chriss market impact model
            # Permanent impact + Temporary impact
            
            lot_size = 100 if "GOLD" in self.symbol else 100000
            order_size = volume * lot_size
            
            # Permanent impact (linear in volume)
            gamma = self.impact_coefficient * 0.001
            permanent_impact = gamma * order_size / avg_daily_volume
            
            # Temporary impact (square root of volume)
            eta = self.impact_coefficient * 0.01
            temp_impact = eta * np.sqrt(order_size / avg_daily_volume)
            
            # Combine with volatility adjustment
            total_impact = (permanent_impact + temp_impact) * (1 + volatility * 10)
            slippage = price * total_impact
            
        else:
            slippage = 0.0
        
        # Direction matters - buying pushes price up, selling pushes down
        if side.upper() == "BUY":
            slippage = abs(slippage)  # Pay more for buys
        else:
            slippage = -abs(slippage)  # Receive less for sells
        
        # Calculate slippage in pips
        pip_value = 0.01 if "GOLD" in self.symbol else 0.0001
        slippage_pips = abs(slippage) / pip_value
        
        return slippage, slippage_pips
    
    def calculate_commission(
        self,
        volume: float,
        round_trip: bool = True
    ) -> float:
        """
        Calculate commission cost.
        
        Args:
            volume: Order volume in lots
            round_trip: Include both entry and exit commission
            
        Returns:
            Commission cost
        """
        commission = self.base_commission * volume
        if round_trip:
            commission *= 2
        return commission
    
    def calculate_swap_cost(
        self,
        price: float,
        volume: float,
        side: str,
        holding_days: int = 1,
        swap_long: float = -0.0001,  # Daily swap rate for longs
        swap_short: float = -0.00005,  # Daily swap rate for shorts
    ) -> float:
        """
        Calculate overnight swap/rollover cost.
        
        Args:
            price: Position entry price
            volume: Position size in lots
            side: "BUY" or "SELL"
            holding_days: Number of days position is held
            swap_long: Long swap rate (usually negative for gold)
            swap_short: Short swap rate
            
        Returns:
            Swap cost (positive = cost, negative = credit)
        """
        lot_size = 100 if "GOLD" in self.symbol else 100000
        position_value = price * volume * lot_size
        
        swap_rate = swap_long if side.upper() == "BUY" else swap_short
        
        # Wednesday triple swap
        effective_days = holding_days
        # Simplified - add 2 extra days for positions held over Wednesday
        
        swap_cost = position_value * swap_rate * effective_days
        
        return abs(swap_cost)  # Return as positive cost
    
    def calculate_costs(
        self,
        price: float,
        volume: float,
        side: str,
        volatility: float = 0.01,
        avg_daily_volume: float = 10000.0,
        holding_days: int = 1,
        include_swap: bool = True,
        urgency: float = 0.5,
    ) -> TransactionCosts:
        """
        Calculate all transaction costs.
        
        Args:
            price: Entry/exit price
            volume: Position size in lots
            side: "BUY" or "SELL"
            volatility: Current market volatility
            avg_daily_volume: Average daily volume
            holding_days: Expected holding period
            include_swap: Include swap costs
            urgency: Trade urgency (0-1)
            
        Returns:
            TransactionCosts dataclass with all costs
        """
        # Spread cost
        spread_cost = self.calculate_spread_cost(price, volume)
        
        # Slippage
        slippage, slippage_pips = self.calculate_slippage(
            price, volume, side, volatility, avg_daily_volume, urgency
        )
        slippage_cost = abs(slippage) * volume * (100 if "GOLD" in self.symbol else 100000)
        
        # Commission
        commission = self.calculate_commission(volume, round_trip=True)
        
        # Swap
        swap_cost = 0.0
        if include_swap and holding_days > 0:
            swap_cost = self.calculate_swap_cost(price, volume, side, holding_days)
        
        # Create result
        costs = TransactionCosts(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission=commission,
            swap_cost=swap_cost,
        )
        
        # Calculate percentage cost
        lot_size = 100 if "GOLD" in self.symbol else 100000
        position_value = price * volume * lot_size
        costs.cost_pct = costs.total_cost / position_value if position_value > 0 else 0.0
        
        return costs
    
    def simulate_execution(
        self,
        intended_price: float,
        volume: float,
        side: str,
        execution_type: ExecutionType = ExecutionType.MARKET,
        volatility: float = 0.01,
        avg_daily_volume: float = 10000.0,
    ) -> ExecutionResult:
        """
        Simulate trade execution with realistic costs.
        
        Args:
            intended_price: Target execution price
            volume: Order volume
            side: "BUY" or "SELL"
            execution_type: Type of order
            volatility: Current volatility
            avg_daily_volume: ADV for impact calculation
            
        Returns:
            ExecutionResult with all execution details
        """
        # Calculate slippage
        slippage, slippage_pips = self.calculate_slippage(
            intended_price, volume, side, volatility, avg_daily_volume
        )
        
        # Adjust for execution type
        if execution_type == ExecutionType.LIMIT:
            # Limit orders may not fill, reduce slippage but reduce fill rate
            slippage *= 0.3
            fill_rate = 0.7 + np.random.random() * 0.25  # 70-95% fill
        elif execution_type == ExecutionType.MARKET:
            fill_rate = 1.0  # Always fill
        else:
            fill_rate = 0.85 + np.random.random() * 0.15  # 85-100%
        
        # Calculate executed price
        if side.upper() == "BUY":
            # Add half spread + slippage
            executed_price = intended_price + (self.base_spread / 2) + slippage
        else:
            # Subtract half spread + slippage
            executed_price = intended_price - (self.base_spread / 2) - slippage
        
        # Calculate all costs
        costs = self.calculate_costs(
            intended_price, volume, side, volatility, avg_daily_volume
        )
        
        # Simulate execution time
        execution_time_ms = np.random.exponential(50)  # Average 50ms
        
        result = ExecutionResult(
            intended_price=intended_price,
            executed_price=executed_price,
            costs=costs,
            fill_rate=fill_rate,
            execution_time_ms=execution_time_ms,
            slippage_pips=slippage_pips,
        )
        
        # Track history
        self.execution_history.append(result)
        
        return result
    
    def get_execution_statistics(self) -> Dict:
        """Get statistics from execution history."""
        if not self.execution_history:
            return {"message": "No execution history"}
        
        slippages = [e.slippage_pips for e in self.execution_history]
        costs_pct = [e.costs.cost_pct for e in self.execution_history]
        fill_rates = [e.fill_rate for e in self.execution_history]
        exec_times = [e.execution_time_ms for e in self.execution_history]
        
        return {
            "total_executions": len(self.execution_history),
            "avg_slippage_pips": np.mean(slippages),
            "max_slippage_pips": np.max(slippages),
            "avg_cost_pct": np.mean(costs_pct),
            "total_costs": sum(e.costs.total_cost for e in self.execution_history),
            "avg_fill_rate": np.mean(fill_rates),
            "avg_execution_time_ms": np.mean(exec_times),
        }
    
    def apply_costs_to_backtest(
        self,
        trades_df: pd.DataFrame,
        volatility_col: str = "atr_pct",
    ) -> pd.DataFrame:
        """
        Apply transaction costs to backtest results.
        
        Args:
            trades_df: DataFrame with trade results
                Required columns: entry_price, exit_price, volume, side, pnl
            volatility_col: Column with volatility data
            
        Returns:
            DataFrame with adjusted P&L accounting for costs
        """
        df = trades_df.copy()
        
        costs_list = []
        for idx, row in df.iterrows():
            volatility = row.get(volatility_col, 0.01)
            
            # Entry costs
            entry_costs = self.calculate_costs(
                price=row["entry_price"],
                volume=row.get("volume", 1.0),
                side=row["side"],
                volatility=volatility,
                include_swap=False,
            )
            
            # Exit costs (reverse side)
            exit_side = "SELL" if row["side"].upper() == "BUY" else "BUY"
            exit_costs = self.calculate_costs(
                price=row["exit_price"],
                volume=row.get("volume", 1.0),
                side=exit_side,
                volatility=volatility,
                include_swap=False,
            )
            
            total_cost = entry_costs.total_cost + exit_costs.total_cost
            costs_list.append(total_cost)
        
        df["transaction_costs"] = costs_list
        df["pnl_after_costs"] = df["pnl"] - df["transaction_costs"]
        df["cost_pct"] = df["transaction_costs"] / (df["entry_price"] * df.get("volume", 1.0) * 100)
        
        return df


def create_cost_model(
    symbol: str = "GOLD",
    conservative: bool = True
) -> TransactionCostModel:
    """
    Factory function to create appropriate cost model.
    
    Args:
        symbol: Trading symbol
        conservative: Use conservative (higher) cost estimates
        
    Returns:
        Configured TransactionCostModel
    """
    if conservative:
        return TransactionCostModel(
            symbol=symbol,
            slippage_model="adaptive",
            impact_coefficient=0.15,  # Higher impact
            volatility_impact_scale=0.7,
        )
    else:
        return TransactionCostModel(
            symbol=symbol,
            slippage_model="fixed",
            impact_coefficient=0.05,
        )


if __name__ == "__main__":
    # Test transaction cost model
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    logger.info("=" * 60)
    logger.info("Testing Transaction Cost Model")
    logger.info("=" * 60)
    
    # Create model
    tcm = TransactionCostModel(symbol="GOLD", slippage_model="adaptive")
    
    # Test cost calculation
    costs = tcm.calculate_costs(
        price=2300.0,
        volume=1.0,
        side="BUY",
        volatility=0.012,
        avg_daily_volume=50000.0,
        holding_days=3,
    )
    
    print(f"\nCost Breakdown for 1 lot GOLD @ $2300:")
    print(f"  Spread Cost:    ${costs.spread_cost:.2f}")
    print(f"  Slippage Cost:  ${costs.slippage_cost:.2f}")
    print(f"  Commission:     ${costs.commission:.2f}")
    print(f"  Swap Cost:      ${costs.swap_cost:.2f}")
    print(f"  ─────────────────────────")
    print(f"  Total Cost:     ${costs.total_cost:.2f}")
    print(f"  Cost %:         {costs.cost_pct:.4%}")
    
    # Simulate executions
    print("\n" + "=" * 60)
    print("Simulating 10 trade executions...")
    
    for i in range(10):
        result = tcm.simulate_execution(
            intended_price=2300.0 + np.random.randn() * 10,
            volume=np.random.uniform(0.5, 2.0),
            side="BUY" if i % 2 == 0 else "SELL",
            volatility=np.random.uniform(0.008, 0.02),
        )
        print(f"  Trade {i+1}: Intended ${result.intended_price:.2f} → "
              f"Executed ${result.executed_price:.2f} "
              f"(Slip: {result.slippage_pips:.1f} pips, Cost: ${result.costs.total_cost:.2f})")
    
    # Get statistics
    stats = tcm.get_execution_statistics()
    print(f"\nExecution Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✅ Transaction Cost Model Test Complete")
