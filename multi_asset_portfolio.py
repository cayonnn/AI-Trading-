"""
Multi-Asset Portfolio Manager
=============================
Production-grade portfolio management with multiple assets.

Features:
- Multi-asset signal aggregation
- Correlation-based position sizing
- Risk parity allocation
- Portfolio optimization
- Cross-asset hedging

Based on:
- Modern Portfolio Theory
- Risk Parity (Bridgewater-style)
- Correlation-aware sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
import json


@dataclass
class AssetConfig:
    """Configuration for a tradeable asset"""
    symbol: str
    name: str
    category: str  # "commodity", "forex", "index", "crypto"
    lot_size: float = 100.0
    pip_value: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 100.0
    spread: float = 0.30
    margin_requirement: float = 0.05  # 5% margin
    correlation_group: str = ""  # Group for correlation analysis
    enabled: bool = True


@dataclass
class PortfolioPosition:
    """Position in the portfolio"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    weight: float = 0.0  # Portfolio weight


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    weights: Dict[str, float]
    positions: Dict[str, float]  # Position sizes
    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 0.0


class AllocationMethod(Enum):
    """Portfolio allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    INVERSE_VOLATILITY = "inverse_volatility"
    SIGNAL_WEIGHTED = "signal_weighted"


# Default asset universe
DEFAULT_ASSETS = {
    "GOLD": AssetConfig(
        symbol="GOLD",
        name="Gold (XAUUSD)",
        category="commodity",
        lot_size=100,
        pip_value=0.01,
        spread=0.30,
        correlation_group="precious_metals",
    ),
    "SILVER": AssetConfig(
        symbol="SILVER",
        name="Silver (XAGUSD)",
        category="commodity",
        lot_size=5000,
        pip_value=0.001,
        spread=0.03,
        correlation_group="precious_metals",
    ),
    "SPX500": AssetConfig(
        symbol="SPX500",
        name="S&P 500 Index",
        category="index",
        lot_size=1,
        pip_value=0.1,
        spread=0.50,
        correlation_group="equity_indices",
    ),
    "US30": AssetConfig(
        symbol="US30",
        name="Dow Jones 30",
        category="index",
        lot_size=1,
        pip_value=0.1,
        spread=2.0,
        correlation_group="equity_indices",
    ),
    "EURUSD": AssetConfig(
        symbol="EURUSD",
        name="EUR/USD",
        category="forex",
        lot_size=100000,
        pip_value=0.0001,
        spread=0.0001,
        correlation_group="major_forex",
    ),
    "USDJPY": AssetConfig(
        symbol="USDJPY",
        name="USD/JPY",
        category="forex",
        lot_size=100000,
        pip_value=0.01,
        spread=0.01,
        correlation_group="major_forex",
    ),
    "BTCUSD": AssetConfig(
        symbol="BTCUSD",
        name="Bitcoin",
        category="crypto",
        lot_size=1,
        pip_value=0.01,
        spread=50.0,
        correlation_group="crypto",
    ),
}


class MultiAssetPortfolio:
    """
    Multi-asset portfolio manager.
    
    Features:
    - Multi-asset tracking
    - Correlation-aware sizing
    - Risk parity allocation
    - Portfolio rebalancing
    
    Usage:
        portfolio = MultiAssetPortfolio(capital=100000)
        portfolio.add_asset("GOLD")
        portfolio.add_asset("SILVER")
        allocation = portfolio.calculate_allocation(signals, method="risk_parity")
    """
    
    def __init__(
        self,
        capital: float = 100000.0,
        max_portfolio_risk: float = 0.15,  # 15% max portfolio risk
        max_asset_weight: float = 0.40,     # 40% max single asset
        max_correlation_exposure: float = 0.70,
        risk_free_rate: float = 0.05,      # 5% risk-free rate
    ):
        """
        Initialize portfolio manager.
        
        Args:
            capital: Total portfolio capital
            max_portfolio_risk: Maximum portfolio volatility
            max_asset_weight: Maximum weight for single asset
            max_correlation_exposure: Maximum correlation exposure
            risk_free_rate: Annual risk-free rate
        """
        self.capital = capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_asset_weight = max_asset_weight
        self.max_correlation_exposure = max_correlation_exposure
        self.risk_free_rate = risk_free_rate
        
        self.assets: Dict[str, AssetConfig] = {}
        self.positions: Dict[str, PortfolioPosition] = {}
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        logger.info(f"MultiAssetPortfolio initialized with ${capital:,.0f}")
    
    def add_asset(
        self,
        symbol: str,
        config: Optional[AssetConfig] = None,
    ):
        """
        Add asset to portfolio universe.
        
        Args:
            symbol: Asset symbol
            config: Asset configuration (uses default if None)
        """
        if config:
            self.assets[symbol] = config
        elif symbol in DEFAULT_ASSETS:
            self.assets[symbol] = DEFAULT_ASSETS[symbol]
        else:
            logger.warning(f"Unknown asset {symbol}, creating default config")
            self.assets[symbol] = AssetConfig(
                symbol=symbol,
                name=symbol,
                category="other",
            )
        
        logger.info(f"Added asset: {symbol}")
    
    def update_prices(
        self,
        prices: Dict[str, pd.Series],
    ):
        """
        Update price history for correlation calculation.
        
        Args:
            prices: Dictionary of symbol -> price series
        """
        for symbol, price_series in prices.items():
            if symbol in self.assets:
                self.price_history[symbol] = price_series
                self.returns_history[symbol] = price_series.pct_change().dropna()
        
        self._update_correlation_matrix()
    
    def _update_correlation_matrix(self):
        """Update correlation matrix from returns"""
        if len(self.returns_history) < 2:
            return
        
        # Align all return series
        returns_df = pd.DataFrame(self.returns_history)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            logger.warning("Insufficient data for correlation (need 30+ bars)")
            return
        
        self.correlation_matrix = returns_df.corr()
        logger.debug(f"Updated correlation matrix for {len(self.assets)} assets")
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two assets"""
        if self.correlation_matrix is None:
            return 0.0
        
        if symbol1 in self.correlation_matrix.columns and \
           symbol2 in self.correlation_matrix.columns:
            return self.correlation_matrix.loc[symbol1, symbol2]
        
        return 0.0
    
    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate annualized volatility for an asset"""
        if symbol not in self.returns_history:
            return 0.15  # Default 15%
        
        returns = self.returns_history[symbol]
        if len(returns) < window:
            return returns.std() * np.sqrt(252 * 24) if len(returns) > 0 else 0.15
        
        # Use recent window
        recent_vol = returns.tail(window).std() * np.sqrt(252 * 24)
        return recent_vol
    
    def calculate_allocation(
        self,
        signals: Dict[str, Dict],
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioAllocation:
        """
        Calculate optimal portfolio allocation.
        
        Args:
            signals: Dictionary of symbol -> signal dict with keys:
                     - direction: 1 (long), -1 (short), 0 (flat)
                     - confidence: 0.0 - 1.0
                     - volatility: expected volatility
            method: Allocation method
            custom_weights: Custom weights (for SIGNAL_WEIGHTED)
            
        Returns:
            PortfolioAllocation with optimal weights and sizes
        """
        active_symbols = [s for s, sig in signals.items() if sig.get("direction", 0) != 0]
        
        if not active_symbols:
            return PortfolioAllocation(weights={}, positions={})
        
        # Calculate weights based on method
        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(active_symbols)
        
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(active_symbols, signals)
        
        elif method == AllocationMethod.INVERSE_VOLATILITY:
            weights = self._inverse_volatility_allocation(active_symbols)
        
        elif method == AllocationMethod.SIGNAL_WEIGHTED:
            weights = self._signal_weighted_allocation(active_symbols, signals)
        
        else:
            weights = self._equal_weight_allocation(active_symbols)
        
        # Apply correlation adjustments
        weights = self._apply_correlation_adjustment(weights, signals)
        
        # Normalize and apply limits
        weights = self._normalize_weights(weights)
        
        # Calculate position sizes
        positions = self._calculate_position_sizes(weights, signals)
        
        # Calculate portfolio metrics
        expected_return = self._calculate_expected_return(weights, signals)
        expected_risk = self._calculate_expected_risk(weights)
        sharpe = (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0
        
        return PortfolioAllocation(
            weights=weights,
            positions=positions,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe,
            diversification_ratio=self._calculate_diversification_ratio(weights),
        )
    
    def _equal_weight_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight allocation"""
        n = len(symbols)
        return {s: 1.0 / n for s in symbols}
    
    def _risk_parity_allocation(
        self,
        symbols: List[str],
        signals: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Risk parity allocation.
        
        Allocates such that each asset contributes equally to portfolio risk.
        """
        # Get volatilities
        volatilities = {}
        for symbol in symbols:
            vol = signals.get(symbol, {}).get("volatility", None)
            if vol is None:
                vol = self.calculate_volatility(symbol)
            volatilities[symbol] = max(vol, 0.01)  # Minimum 1%
        
        # Inverse volatility weights
        inv_vol = {s: 1.0 / v for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        weights = {s: v / total_inv_vol for s, v in inv_vol.items()}
        
        return weights
    
    def _inverse_volatility_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Inverse volatility weighting"""
        volatilities = {s: self.calculate_volatility(s) for s in symbols}
        inv_vol = {s: 1.0 / max(v, 0.01) for s, v in volatilities.items()}
        total = sum(inv_vol.values())
        return {s: v / total for s, v in inv_vol.items()}
    
    def _signal_weighted_allocation(
        self,
        symbols: List[str],
        signals: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Weight by signal confidence"""
        confidences = {}
        for symbol in symbols:
            conf = signals.get(symbol, {}).get("confidence", 0.5)
            confidences[symbol] = max(conf, 0.1)  # Minimum 10%
        
        total_conf = sum(confidences.values())
        return {s: c / total_conf for s, c in confidences.items()}
    
    def _apply_correlation_adjustment(
        self,
        weights: Dict[str, float],
        signals: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Adjust weights based on correlation.
        
        Reduces weight for highly correlated positions in same direction.
        """
        if self.correlation_matrix is None or len(weights) < 2:
            return weights
        
        adjusted_weights = weights.copy()
        symbols = list(weights.keys())
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self.get_correlation(sym1, sym2)
                
                # If highly correlated and same direction, reduce both
                dir1 = signals.get(sym1, {}).get("direction", 0)
                dir2 = signals.get(sym2, {}).get("direction", 0)
                
                if abs(corr) > self.max_correlation_exposure and dir1 * dir2 > 0:
                    # Both same direction and highly correlated
                    reduction_factor = 1 - (abs(corr) - self.max_correlation_exposure)
                    reduction_factor = max(reduction_factor, 0.5)  # At least 50% weight
                    
                    # Reduce the smaller weight position
                    if adjusted_weights[sym1] < adjusted_weights[sym2]:
                        adjusted_weights[sym1] *= reduction_factor
                    else:
                        adjusted_weights[sym2] *= reduction_factor
                    
                    logger.debug(f"Correlation adjustment: {sym1}/{sym2} corr={corr:.2f}")
        
        return adjusted_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights and apply limits"""
        if not weights:
            return weights
        
        # Apply max weight limit
        for symbol in weights:
            if weights[symbol] > self.max_asset_weight:
                weights[symbol] = self.max_asset_weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        
        return weights
    
    def _calculate_position_sizes(
        self,
        weights: Dict[str, float],
        signals: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Calculate actual position sizes in lots"""
        positions = {}
        
        for symbol, weight in weights.items():
            if symbol not in self.assets:
                continue
            
            asset = self.assets[symbol]
            signal = signals.get(symbol, {})
            
            # Calculate position value
            position_value = self.capital * weight
            
            # Get price from signal or use default
            price = signal.get("price", 2000.0)  # Default for testing
            
            # Calculate lots
            lot_value = price * asset.lot_size
            lots = position_value / lot_value if lot_value > 0 else 0
            
            # Apply limits
            lots = max(asset.min_lot, min(lots, asset.max_lot))
            
            # Direction
            direction = signal.get("direction", 1)
            positions[symbol] = lots * direction
        
        return positions
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        signals: Dict[str, Dict],
    ) -> float:
        """Calculate expected portfolio return"""
        expected_return = 0.0
        
        for symbol, weight in weights.items():
            # Use historical mean return or signal expectation
            if symbol in self.returns_history:
                mean_return = self.returns_history[symbol].mean() * 252 * 24
            else:
                mean_return = 0.05  # Default 5% annual
            
            # Adjust for confidence
            confidence = signals.get(symbol, {}).get("confidence", 0.5)
            expected_return += weight * mean_return * confidence
        
        return expected_return
    
    def _calculate_expected_risk(self, weights: Dict[str, float]) -> float:
        """Calculate expected portfolio volatility"""
        if not weights or self.correlation_matrix is None:
            return 0.15  # Default
        
        symbols = list(weights.keys())
        n = len(symbols)
        
        # Build weight vector and covariance matrix
        w = np.array([weights.get(s, 0) for s in symbols])
        
        # Build covariance matrix
        cov_matrix = np.zeros((n, n))
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if sym1 in self.returns_history and sym2 in self.returns_history:
                    vol1 = self.calculate_volatility(sym1)
                    vol2 = self.calculate_volatility(sym2)
                    corr = self.get_correlation(sym1, sym2)
                    cov_matrix[i, j] = vol1 * vol2 * corr
                else:
                    cov_matrix[i, j] = 0.0 if i != j else 0.15**2
        
        # Portfolio variance
        portfolio_var = w @ cov_matrix @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        return portfolio_vol
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """
        Calculate diversification ratio.
        
        DR = weighted average volatility / portfolio volatility
        Higher is better (more diversification benefit).
        """
        if not weights:
            return 1.0
        
        # Weighted average volatility
        weighted_vol = sum(
            weights.get(s, 0) * self.calculate_volatility(s)
            for s in weights
        )
        
        # Portfolio volatility
        portfolio_vol = self._calculate_expected_risk(weights)
        
        if portfolio_vol > 0:
            return weighted_vol / portfolio_vol
        return 1.0
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_value = self.capital
        total_margin = 0.0
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.assets:
                asset = self.assets[symbol]
                pos_value = abs(position.quantity) * position.current_price * asset.lot_size
                positions_value += pos_value
                total_margin += pos_value * asset.margin_requirement
        
        return {
            "capital": self.capital,
            "positions_value": positions_value,
            "margin_used": total_margin,
            "free_margin": self.capital - total_margin,
            "margin_level": (self.capital / total_margin * 100) if total_margin > 0 else float('inf'),
            "n_positions": len(self.positions),
            "n_assets": len(self.assets),
        }


class CrossAssetHedger:
    """
    Cross-asset hedging strategies.
    
    Implements:
    - Gold-DXY inverse correlation hedging
    - Gold-Equity correlation hedging
    - VIX-based protection
    """
    
    # Known correlations (approximate, should be updated dynamically)
    KNOWN_CORRELATIONS = {
        ("GOLD", "DXY"): -0.80,      # Gold inversely correlated with USD
        ("GOLD", "SPX500"): -0.20,   # Slight inverse
        ("GOLD", "VIX"): 0.40,       # Gold rises with fear
        ("GOLD", "SILVER"): 0.90,    # Highly correlated
        ("SPX500", "US30"): 0.95,    # Indices highly correlated
        ("SPX500", "VIX"): -0.75,    # VIX inverse to stocks
    }
    
    def __init__(self, portfolio: MultiAssetPortfolio):
        """
        Initialize hedger.
        
        Args:
            portfolio: MultiAssetPortfolio instance
        """
        self.portfolio = portfolio
    
    def calculate_hedge_ratio(
        self,
        asset: str,
        hedge_asset: str,
    ) -> float:
        """
        Calculate optimal hedge ratio.
        
        Args:
            asset: Asset to hedge
            hedge_asset: Asset to use for hedging
            
        Returns:
            Hedge ratio (units of hedge per unit of asset)
        """
        # Check for known correlation
        pair = (asset, hedge_asset)
        reverse_pair = (hedge_asset, asset)
        
        if pair in self.KNOWN_CORRELATIONS:
            corr = self.KNOWN_CORRELATIONS[pair]
        elif reverse_pair in self.KNOWN_CORRELATIONS:
            corr = self.KNOWN_CORRELATIONS[reverse_pair]
        else:
            corr = self.portfolio.get_correlation(asset, hedge_asset)
        
        # Calculate hedge ratio based on volatility ratio and correlation
        vol_asset = self.portfolio.calculate_volatility(asset)
        vol_hedge = self.portfolio.calculate_volatility(hedge_asset)
        
        if vol_hedge > 0 and abs(corr) > 0.3:  # Only hedge if meaningful correlation
            # Beta-style hedge ratio
            hedge_ratio = (vol_asset / vol_hedge) * corr
            return hedge_ratio
        
        return 0.0
    
    def suggest_hedges(
        self,
        positions: Dict[str, float],
        max_hedge_cost: float = 0.005,  # Max 0.5% of position
    ) -> Dict[str, Dict]:
        """
        Suggest hedging positions.
        
        Args:
            positions: Current positions (symbol -> lots)
            max_hedge_cost: Maximum acceptable hedge cost
            
        Returns:
            Dictionary of suggested hedges
        """
        hedges = {}
        
        for symbol, lots in positions.items():
            if lots == 0:
                continue
            
            # Find suitable hedge assets
            suitable_hedges = []
            
            for hedge_symbol in self.portfolio.assets:
                if hedge_symbol == symbol:
                    continue
                
                ratio = self.calculate_hedge_ratio(symbol, hedge_symbol)
                
                if abs(ratio) > 0.2:  # Meaningful hedge
                    suitable_hedges.append({
                        "symbol": hedge_symbol,
                        "ratio": ratio,
                        "hedge_lots": -lots * ratio,  # Negative = opposite direction
                    })
            
            # Sort by hedge effectiveness (absolute ratio)
            suitable_hedges.sort(key=lambda x: abs(x["ratio"]), reverse=True)
            
            if suitable_hedges:
                hedges[symbol] = {
                    "position": lots,
                    "suggested_hedges": suitable_hedges[:3],  # Top 3 hedges
                }
        
        return hedges


def create_portfolio_from_config(config: Dict) -> MultiAssetPortfolio:
    """
    Factory function to create portfolio from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MultiAssetPortfolio
    """
    portfolio = MultiAssetPortfolio(
        capital=config.get("capital", 100000),
        max_portfolio_risk=config.get("max_risk", 0.15),
        max_asset_weight=config.get("max_weight", 0.40),
    )
    
    # Add configured assets
    for symbol in config.get("assets", ["GOLD"]):
        portfolio.add_asset(symbol)
    
    return portfolio


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    logger.info("=" * 60)
    logger.info("Testing Multi-Asset Portfolio Manager")
    logger.info("=" * 60)
    
    # Create portfolio
    portfolio = MultiAssetPortfolio(capital=100000)
    portfolio.add_asset("GOLD")
    portfolio.add_asset("SILVER")
    portfolio.add_asset("SPX500")
    
    # Create sample price history
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    
    prices = {
        "GOLD": pd.Series(2300 + np.cumsum(np.random.randn(100) * 10), index=dates),
        "SILVER": pd.Series(28 + np.cumsum(np.random.randn(100) * 0.5), index=dates),
        "SPX500": pd.Series(5000 + np.cumsum(np.random.randn(100) * 30), index=dates),
    }
    
    portfolio.update_prices(prices)
    
    # Show correlation matrix
    print("\nCorrelation Matrix:")
    print(portfolio.correlation_matrix)
    
    # Create sample signals
    signals = {
        "GOLD": {"direction": 1, "confidence": 0.85, "price": 2350, "volatility": 0.012},
        "SILVER": {"direction": 1, "confidence": 0.70, "price": 28.5, "volatility": 0.018},
        "SPX500": {"direction": -1, "confidence": 0.60, "price": 5100, "volatility": 0.015},
    }
    
    # Calculate allocations with different methods
    print("\n" + "=" * 60)
    print("ALLOCATION COMPARISON")
    print("=" * 60)
    
    for method in AllocationMethod:
        allocation = portfolio.calculate_allocation(signals, method)
        print(f"\n{method.value.upper()}:")
        print(f"  Weights: {allocation.weights}")
        print(f"  Expected Return: {allocation.expected_return:.2%}")
        print(f"  Expected Risk: {allocation.expected_risk:.2%}")
        print(f"  Sharpe Ratio: {allocation.sharpe_ratio:.2f}")
        print(f"  Diversification: {allocation.diversification_ratio:.2f}")
    
    # Test hedging
    print("\n" + "=" * 60)
    print("HEDGING SUGGESTIONS")
    print("=" * 60)
    
    hedger = CrossAssetHedger(portfolio)
    
    positions = {"GOLD": 2.0, "SPX500": -0.5}
    hedges = hedger.suggest_hedges(positions)
    
    for symbol, suggestion in hedges.items():
        print(f"\n{symbol} position: {suggestion['position']:.2f} lots")
        for hedge in suggestion.get("suggested_hedges", []):
            print(f"  → Hedge with {hedge['symbol']}: {hedge['hedge_lots']:.2f} lots (ratio: {hedge['ratio']:.2f})")
    
    print("\n✅ Multi-Asset Portfolio Test Complete")
