"""
Execution Quality Analyzer
==========================
Production-grade execution quality analysis.

Features:
- Slippage analysis
- Fill quality metrics
- Best execution monitoring
- Latency tracking
- Market impact measurement

Based on:
- MiFID II best execution requirements
- Institutional trading standards
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


@dataclass
class ExecutionRecord:
    """Record of a single execution"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    intended_price: float
    executed_price: float
    intended_quantity: float
    executed_quantity: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    latency_ms: float = 0.0
    venue: str = "MT5"
    order_type: str = "MARKET"


@dataclass
class ExecutionStats:
    """Aggregated execution statistics"""
    total_orders: int = 0
    total_volume: float = 0.0
    fill_rate: float = 1.0
    avg_slippage_pips: float = 0.0
    avg_slippage_cost: float = 0.0
    positive_slippage_rate: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    spread_capture_rate: float = 0.0
    implementation_shortfall: float = 0.0


class ExecutionQualityAnalyzer:
    """
    Analyze execution quality for trading operations.
    
    Tracks:
    - Slippage (positive and negative)
    - Fill rates
    - Latency
    - Best execution compliance
    """
    
    def __init__(
        self,
        symbol: str = "GOLD",
        pip_value: float = 0.01,
        expected_spread: float = 0.30,
    ):
        """
        Initialize analyzer.
        
        Args:
            symbol: Trading symbol
            pip_value: Value of one pip
            expected_spread: Expected average spread
        """
        self.symbol = symbol
        self.pip_value = pip_value
        self.expected_spread = expected_spread
        
        self.executions: List[ExecutionRecord] = []
        self.daily_stats: Dict[str, ExecutionStats] = {}
        
        logger.info(f"ExecutionQualityAnalyzer initialized for {symbol}")
    
    def record_execution(
        self,
        order_id: str,
        side: str,
        intended_price: float,
        executed_price: float,
        intended_quantity: float,
        executed_quantity: float,
        bid_price: float = 0.0,
        ask_price: float = 0.0,
        latency_ms: float = 0.0,
        timestamp: Optional[datetime] = None,
    ):
        """Record a trade execution for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        record = ExecutionRecord(
            order_id=order_id,
            timestamp=timestamp,
            symbol=self.symbol,
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
            intended_quantity=intended_quantity,
            executed_quantity=executed_quantity,
            bid_price=bid_price,
            ask_price=ask_price,
            latency_ms=latency_ms,
        )
        
        self.executions.append(record)
        logger.debug(f"Recorded execution: {order_id}")
    
    def calculate_slippage(self, record: ExecutionRecord) -> Tuple[float, float]:
        """
        Calculate slippage for an execution.
        
        Returns:
            Tuple of (slippage_pips, slippage_cost)
        """
        if record.side.upper() == "BUY":
            # For buys, positive slippage means paying more
            slippage = record.executed_price - record.intended_price
        else:
            # For sells, positive slippage means receiving less
            slippage = record.intended_price - record.executed_price
        
        slippage_pips = slippage / self.pip_value
        
        # Cost calculation
        lot_size = 100 if "GOLD" in self.symbol else 100000
        slippage_cost = slippage * record.executed_quantity * lot_size
        
        return slippage_pips, slippage_cost
    
    def calculate_spread_capture(self, record: ExecutionRecord) -> float:
        """
        Calculate spread capture percentage.
        
        Measures how well the execution captured the spread.
        100% = executed at best price
        0% = executed at worst price
        """
        if record.bid_price == 0 or record.ask_price == 0:
            return 0.5  # Unknown
        
        spread = record.ask_price - record.bid_price
        if spread <= 0:
            return 1.0
        
        if record.side.upper() == "BUY":
            # For buys, lower is better
            price_improvement = record.ask_price - record.executed_price
        else:
            # For sells, higher is better
            price_improvement = record.executed_price - record.bid_price
        
        capture_rate = (price_improvement / spread) + 0.5  # Centered at 0.5
        return min(max(capture_rate, 0), 1)
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> ExecutionStats:
        """
        Get aggregated execution statistics.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            ExecutionStats with all metrics
        """
        # Filter executions
        filtered = self.executions
        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        
        if not filtered:
            return ExecutionStats()
        
        # Calculate metrics
        slippages = []
        costs = []
        latencies = []
        spread_captures = []
        
        total_intended = 0
        total_executed = 0
        positive_slippage_count = 0
        
        for record in filtered:
            slip_pips, slip_cost = self.calculate_slippage(record)
            slippages.append(slip_pips)
            costs.append(slip_cost)
            latencies.append(record.latency_ms)
            spread_captures.append(self.calculate_spread_capture(record))
            
            total_intended += record.intended_quantity
            total_executed += record.executed_quantity
            
            if slip_pips >= 0:  # Favorable or zero slippage
                positive_slippage_count += 1
        
        # Implementation shortfall
        total_cost = sum(costs)
        total_value = sum(
            r.intended_price * r.intended_quantity * (100 if "GOLD" in self.symbol else 100000)
            for r in filtered
        )
        impl_shortfall = total_cost / total_value if total_value > 0 else 0
        
        return ExecutionStats(
            total_orders=len(filtered),
            total_volume=total_executed,
            fill_rate=total_executed / total_intended if total_intended > 0 else 1.0,
            avg_slippage_pips=np.mean(slippages) if slippages else 0,
            avg_slippage_cost=np.mean(costs) if costs else 0,
            positive_slippage_rate=positive_slippage_count / len(filtered),
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            spread_capture_rate=np.mean(spread_captures) if spread_captures else 0.5,
            implementation_shortfall=impl_shortfall,
        )
    
    def get_daily_breakdown(self) -> pd.DataFrame:
        """Get daily execution statistics"""
        if not self.executions:
            return pd.DataFrame()
        
        # Group by date
        daily_data = {}
        for record in self.executions:
            date_key = record.timestamp.date()
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append(record)
        
        # Calculate stats for each day
        rows = []
        for date, records in sorted(daily_data.items()):
            slippages = [self.calculate_slippage(r)[0] for r in records]
            latencies = [r.latency_ms for r in records]
            
            rows.append({
                "date": date,
                "n_orders": len(records),
                "volume": sum(r.executed_quantity for r in records),
                "avg_slippage": np.mean(slippages),
                "max_slippage": max(slippages) if slippages else 0,
                "min_slippage": min(slippages) if slippages else 0,
                "avg_latency": np.mean(latencies),
                "max_latency": max(latencies) if latencies else 0,
            })
        
        return pd.DataFrame(rows)
    
    def get_venue_comparison(self) -> pd.DataFrame:
        """Compare execution quality across venues"""
        if not self.executions:
            return pd.DataFrame()
        
        venue_data = {}
        for record in self.executions:
            if record.venue not in venue_data:
                venue_data[record.venue] = []
            venue_data[record.venue].append(record)
        
        rows = []
        for venue, records in venue_data.items():
            slippages = [self.calculate_slippage(r)[0] for r in records]
            latencies = [r.latency_ms for r in records]
            
            rows.append({
                "venue": venue,
                "n_orders": len(records),
                "avg_slippage": np.mean(slippages),
                "avg_latency": np.mean(latencies),
                "fill_rate": sum(r.executed_quantity for r in records) / sum(r.intended_quantity for r in records),
            })
        
        return pd.DataFrame(rows)
    
    def generate_report(self) -> str:
        """Generate text report of execution quality"""
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("EXECUTION QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"\nSymbol: {self.symbol}")
        report.append(f"Total Orders: {stats.total_orders}")
        report.append(f"Total Volume: {stats.total_volume:.2f} lots")
        report.append("")
        report.append("FILL QUALITY:")
        report.append(f"  Fill Rate: {stats.fill_rate:.2%}")
        report.append(f"  Spread Capture: {stats.spread_capture_rate:.2%}")
        report.append("")
        report.append("SLIPPAGE ANALYSIS:")
        report.append(f"  Avg Slippage: {stats.avg_slippage_pips:.2f} pips")
        report.append(f"  Avg Slippage Cost: ${stats.avg_slippage_cost:.2f}")
        report.append(f"  Positive Slippage Rate: {stats.positive_slippage_rate:.2%}")
        report.append(f"  Implementation Shortfall: {stats.implementation_shortfall:.4%}")
        report.append("")
        report.append("LATENCY:")
        report.append(f"  Avg Latency: {stats.avg_latency_ms:.1f} ms")
        report.append(f"  Max Latency: {stats.max_latency_ms:.1f} ms")
        report.append("")
        
        # Quality assessment
        quality_score = 0
        if stats.avg_slippage_pips < 2:
            quality_score += 1
        if stats.fill_rate > 0.95:
            quality_score += 1
        if stats.avg_latency_ms < 100:
            quality_score += 1
        if stats.positive_slippage_rate > 0.4:
            quality_score += 1
        
        quality_labels = ["Poor", "Fair", "Good", "Excellent", "Outstanding"]
        report.append(f"OVERALL QUALITY: {quality_labels[quality_score]}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_to_file(self, filepath: str):
        """Save execution data to CSV"""
        if not self.executions:
            logger.warning("No executions to save")
            return
        
        data = []
        for record in self.executions:
            slip_pips, slip_cost = self.calculate_slippage(record)
            data.append({
                "order_id": record.order_id,
                "timestamp": record.timestamp,
                "symbol": record.symbol,
                "side": record.side,
                "intended_price": record.intended_price,
                "executed_price": record.executed_price,
                "intended_qty": record.intended_quantity,
                "executed_qty": record.executed_quantity,
                "slippage_pips": slip_pips,
                "slippage_cost": slip_cost,
                "latency_ms": record.latency_ms,
                "venue": record.venue,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(data)} executions to {filepath}")


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    logger.info("=" * 60)
    logger.info("Testing Execution Quality Analyzer")
    logger.info("=" * 60)
    
    # Create analyzer
    analyzer = ExecutionQualityAnalyzer(symbol="GOLD")
    
    # Simulate some executions
    np.random.seed(42)
    base_price = 2300.0
    
    for i in range(50):
        side = "BUY" if np.random.random() > 0.5 else "SELL"
        intended_price = base_price + np.random.randn() * 10
        
        # Simulate slippage
        slippage = np.random.normal(0.1, 0.15)  # Slight positive slippage on average
        if side == "BUY":
            executed_price = intended_price + slippage
        else:
            executed_price = intended_price - slippage
        
        intended_qty = np.random.uniform(0.5, 2.0)
        executed_qty = intended_qty * np.random.uniform(0.95, 1.0)
        
        analyzer.record_execution(
            order_id=f"ORD-{i:04d}",
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
            intended_quantity=intended_qty,
            executed_quantity=executed_qty,
            bid_price=intended_price - 0.15,
            ask_price=intended_price + 0.15,
            latency_ms=np.random.exponential(40),
            timestamp=datetime.now() - timedelta(hours=50-i),
        )
    
    # Print report
    print(analyzer.generate_report())
    
    # Daily breakdown
    print("\nDaily Breakdown:")
    print(analyzer.get_daily_breakdown().to_string())
    
    print("\n✅ Execution Quality Analyzer Test Complete")
