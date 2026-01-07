"""
Hedge Fund Modules - Comprehensive Test Suite
==============================================
Tests all new hedge fund modules for functionality.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


def test_transaction_cost_model():
    """Test transaction cost model"""
    logger.info("Testing Transaction Cost Model...")
    
    from transaction_cost_model import TransactionCostModel
    
    tcm = TransactionCostModel(symbol="GOLD", slippage_model="adaptive")
    
    # Test cost calculation
    costs = tcm.calculate_costs(
        price=2300.0,
        volume=1.0,
        side="BUY",
        volatility=0.012,
        avg_daily_volume=50000.0,
    )
    
    assert costs.total_cost > 0, "Total cost should be positive"
    assert costs.spread_cost > 0, "Spread cost should be positive"
    
    logger.info(f"  ✅ Costs calculated: ${costs.total_cost:.2f}")
    return True


def test_backtest_engine():
    """Test backtesting engine"""
    logger.info("Testing Backtest Engine...")
    
    from backtest_engine import BacktestEngine, MonteCarloSimulator
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="H")
    prices = 2300 + np.cumsum(np.random.randn(200) * 2)
    
    data = pd.DataFrame({
        "open": prices,
        "high": prices + np.random.rand(200) * 5,
        "low": prices - np.random.rand(200) * 5,
        "close": prices + np.random.randn(200) * 2,
        "volume": np.random.randint(1000, 10000, 200),
    }, index=dates)
    
    # Create signals
    signal_values = np.zeros(200)
    for i in range(20, 200, 40):
        signal_values[i] = 1 if np.random.random() > 0.5 else -1
    
    signals = pd.DataFrame({
        "signal": signal_values,
        "stop_loss": data["close"] * 0.988,
        "take_profit": data["close"] * 1.024,
        "volatility": 0.012,
    }, index=dates)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000, use_transaction_costs=True)
    result = engine.run(data, signals)
    
    assert "total_trades" in result.metrics, "Should have total_trades metric"
    assert len(result.equity_curve) > 0, "Should have equity curve"
    
    logger.info(f"  ✅ Backtest complete: {result.metrics.get('total_trades', 0)} trades")
    
    # Test Monte Carlo
    if result.trades:
        mc = MonteCarloSimulator(n_simulations=100)
        mc_results = mc.bootstrap_trades(result.trades)
        assert "confidence_intervals" in mc_results, "Should have confidence intervals"
        logger.info(f"  ✅ Monte Carlo: {mc_results['confidence_intervals']['median']:.2f}")
    
    return True


def test_multi_asset_portfolio():
    """Test multi-asset portfolio manager"""
    logger.info("Testing Multi-Asset Portfolio...")
    
    from multi_asset_portfolio import MultiAssetPortfolio, AllocationMethod
    
    # Create portfolio
    portfolio = MultiAssetPortfolio(capital=100000)
    portfolio.add_asset("GOLD")
    portfolio.add_asset("SILVER")
    portfolio.add_asset("SPX500")
    
    # Create sample price history
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    
    prices = {
        "GOLD": pd.Series(2300 + np.cumsum(np.random.randn(50) * 10), index=dates),
        "SILVER": pd.Series(28 + np.cumsum(np.random.randn(50) * 0.5), index=dates),
        "SPX500": pd.Series(5000 + np.cumsum(np.random.randn(50) * 30), index=dates),
    }
    
    portfolio.update_prices(prices)
    
    # Create signals
    signals = {
        "GOLD": {"direction": 1, "confidence": 0.85, "price": 2350, "volatility": 0.012},
        "SILVER": {"direction": 1, "confidence": 0.70, "price": 28.5, "volatility": 0.018},
        "SPX500": {"direction": -1, "confidence": 0.60, "price": 5100, "volatility": 0.015},
    }
    
    # Test allocation
    allocation = portfolio.calculate_allocation(signals, AllocationMethod.RISK_PARITY)
    
    assert len(allocation.weights) > 0, "Should have weights"
    assert abs(sum(allocation.weights.values()) - 1.0) < 0.01, "Weights should sum to 1"
    
    logger.info(f"  ✅ Allocation complete: {len(allocation.weights)} assets")
    return True


def test_model_drift_detector():
    """Test model drift detection"""
    logger.info("Testing Model Drift Detector...")
    
    from model_drift_detector import ModelDriftMonitor
    
    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 200),
        "feature2": np.random.normal(5, 2, 200),
    })
    
    baseline_metrics = {
        "accuracy": 0.68,
        "win_rate": 0.62,
    }
    
    # Create monitor
    monitor = ModelDriftMonitor(
        reference_data=reference_data,
        baseline_metrics=baseline_metrics,
        model_name="test_model",
    )
    
    # Add predictions
    for _ in range(100):
        monitor.add_prediction(np.random.random(), np.random.uniform(0.5, 1.0))
    
    # Create current data with drift
    current_data = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1.2, 50),  # Drifted
        "feature2": np.random.normal(5, 2, 50),
    })
    
    # Run check
    report = monitor.run_check(current_data)
    
    assert report.overall_health in ["healthy", "warning", "critical"], "Should have health status"
    
    logger.info(f"  ✅ Drift check: {report.overall_health}")
    return True


def test_execution_quality_analyzer():
    """Test execution quality analyzer"""
    logger.info("Testing Execution Quality Analyzer...")
    
    from execution_quality_analyzer import ExecutionQualityAnalyzer
    
    analyzer = ExecutionQualityAnalyzer(symbol="GOLD")
    
    # Simulate executions
    np.random.seed(42)
    for i in range(20):
        side = "BUY" if i % 2 == 0 else "SELL"
        intended_price = 2300 + np.random.randn() * 10
        slippage = np.random.normal(0.1, 0.15)
        
        if side == "BUY":
            executed_price = intended_price + slippage
        else:
            executed_price = intended_price - slippage
        
        analyzer.record_execution(
            order_id=f"ORD-{i:04d}",
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
            intended_quantity=1.0,
            executed_quantity=1.0,
            latency_ms=np.random.exponential(40),
        )
    
    # Get stats
    stats = analyzer.get_statistics()
    
    assert stats.total_orders == 20, "Should have 20 orders"
    assert stats.avg_slippage_pips is not None, "Should have slippage"
    
    logger.info(f"  ✅ Execution analysis: {stats.total_orders} orders, {stats.avg_slippage_pips:.2f} pips avg slippage")
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("HEDGE FUND MODULES - TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Transaction Cost Model", test_transaction_cost_model),
        ("Backtest Engine", test_backtest_engine),
        ("Multi-Asset Portfolio", test_multi_asset_portfolio),
        ("Model Drift Detector", test_model_drift_detector),
        ("Execution Quality Analyzer", test_execution_quality_analyzer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            logger.error(f"  ❌ {name} failed: {e}")
            results.append((name, "FAIL"))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    
    for name, result in results:
        icon = "✅" if result == "PASS" else "❌"
        logger.info(f"  {icon} {name}: {result}")
    
    logger.info("")
    logger.info(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
