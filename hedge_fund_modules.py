"""
AI Trade - Unified Hedge Fund Module Exports
=============================================
Central import point for all hedge fund modules.

Usage:
    from hedge_fund_modules import (
        TransactionCostModel,
        BacktestEngine,
        MultiAssetPortfolio,
        ModelDriftMonitor,
        ExecutionQualityAnalyzer,
    )
"""

# Transaction Cost Model
from transaction_cost_model import (
    TransactionCostModel,
    TransactionCosts,
    ExecutionResult,
    ExecutionType,
    create_cost_model,
)

# Backtesting Engine
from backtest_engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    Order,
    MonteCarloSimulator,
    WalkForwardOptimizer,
)

# Multi-Asset Portfolio
from multi_asset_portfolio import (
    MultiAssetPortfolio,
    AssetConfig,
    PortfolioAllocation,
    AllocationMethod,
    CrossAssetHedger,
    DEFAULT_ASSETS,
)

# Model Drift Detection
from model_drift_detector import (
    ModelDriftMonitor,
    FeatureDriftDetector,
    PredictionDriftDetector,
    PerformanceMonitor,
    DataQualityMonitor,
    DriftAlert,
    DriftReport,
    AlertLevel,
    DriftType,
    create_drift_monitor,
)

# Execution Quality Analysis
from execution_quality_analyzer import (
    ExecutionQualityAnalyzer,
    ExecutionRecord,
    ExecutionStats,
)

__all__ = [
    # Transaction Costs
    "TransactionCostModel",
    "TransactionCosts",
    "ExecutionResult",
    "ExecutionType",
    "create_cost_model",
    
    # Backtesting
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "Order",
    "MonteCarloSimulator",
    "WalkForwardOptimizer",
    
    # Portfolio
    "MultiAssetPortfolio",
    "AssetConfig",
    "PortfolioAllocation",
    "AllocationMethod",
    "CrossAssetHedger",
    "DEFAULT_ASSETS",
    
    # Drift Detection
    "ModelDriftMonitor",
    "FeatureDriftDetector",
    "PredictionDriftDetector",
    "PerformanceMonitor",
    "DataQualityMonitor",
    "DriftAlert",
    "DriftReport",
    "AlertLevel",
    "DriftType",
    "create_drift_monitor",
    
    # Execution Quality
    "ExecutionQualityAnalyzer",
    "ExecutionRecord",
    "ExecutionStats",
]

# Version info
__version__ = "2.0.0"
__author__ = "AI Trade System"
__description__ = "Production-grade Hedge Fund Trading Modules"
