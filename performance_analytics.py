"""
Performance Analytics Tool - Production Grade
============================================
Advanced analytics and reporting for trading system performance

Features:
- Comprehensive performance metrics
- Signal quality analysis
- Regime performance breakdown
- Risk analysis
- Trade analysis
- HTML report generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys
from database_manager import DatabaseManager
from config_manager import ConfigManager

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


class PerformanceAnalytics:
    """
    Production-grade performance analytics

    Capabilities:
    - Performance metrics calculation
    - Signal quality analysis
    - Risk metrics
    - Trade analysis
    - Report generation
    """

    def __init__(self, db_path: str = "trading_data.db"):
        """Initialize analytics"""
        self.db = DatabaseManager(db_path=db_path)
        self.config = ConfigManager()
        logger.info("Performance Analytics initialized")

    def calculate_performance_metrics(
        self,
        days: int = 30,
        min_trades: int = 10
    ) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            days: Number of days to analyze
            min_trades: Minimum trades required for analysis

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Calculating performance metrics for last {days} days...")

        # Get recent trades
        cutoff = datetime.now() - timedelta(days=days)
        trades = self.db.get_recent_trades(limit=1000)

        if not trades:
            logger.warning("No trades found")
            return {}

        # Filter by date
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df[trades_df['timestamp'] >= cutoff]

        if len(trades_df) < min_trades:
            logger.warning(f"Insufficient trades: {len(trades_df)} < {min_trades}")
            return {}

        # Calculate metrics
        metrics = {}

        # Basic stats
        metrics['total_trades'] = len(trades_df)
        metrics['period_days'] = days

        # Win/Loss
        if 'profit' in trades_df.columns:
            winning_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]

            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

            # Profit metrics
            gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0

            metrics['gross_profit'] = gross_profit
            metrics['gross_loss'] = gross_loss
            metrics['net_profit'] = gross_profit - gross_loss

            # Profit factor
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0

            # Average win/loss
            metrics['avg_win'] = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0

            # Risk/Reward
            if metrics['avg_loss'] != 0:
                metrics['avg_risk_reward'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            else:
                metrics['avg_risk_reward'] = 0

            # Largest win/loss
            metrics['largest_win'] = winning_trades['profit'].max() if len(winning_trades) > 0 else 0
            metrics['largest_loss'] = losing_trades['profit'].min() if len(losing_trades) > 0 else 0

            # Consecutive wins/losses
            metrics['max_consecutive_wins'] = self._max_consecutive(trades_df['profit'] > 0)
            metrics['max_consecutive_losses'] = self._max_consecutive(trades_df['profit'] < 0)

            # Returns analysis
            if len(trades_df) > 1:
                returns = trades_df['profit'].values

                # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
                if returns.std() > 0:
                    metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0

                # Max Drawdown
                cumulative = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                metrics['max_drawdown'] = abs(drawdown.min()) if len(drawdown) > 0 else 0
                metrics['max_drawdown_pct'] = abs(drawdown.min() / running_max.max()) if running_max.max() > 0 else 0

        logger.info(f"Performance metrics calculated: {len(metrics)} metrics")
        return metrics

    def analyze_signal_quality(self, days: int = 30) -> Dict:
        """
        Analyze signal quality and distribution

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of signal quality metrics
        """
        logger.info(f"Analyzing signal quality for last {days} days...")

        # Get recent signals
        cutoff = datetime.now() - timedelta(days=days)
        signals = self.db.get_recent_signals(limit=1000)

        if not signals:
            logger.warning("No signals found")
            return {}

        # Filter by date
        signals_df = pd.DataFrame(signals)
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df = signals_df[signals_df['timestamp'] >= cutoff]

        if len(signals_df) == 0:
            return {}

        analysis = {}

        # Signal counts
        analysis['total_signals'] = len(signals_df)
        analysis['signals_per_day'] = len(signals_df) / days

        # Action distribution
        if 'action' in signals_df.columns:
            action_counts = signals_df['action'].value_counts()
            analysis['long_signals'] = action_counts.get('LONG', 0)
            analysis['short_signals'] = action_counts.get('SHORT', 0)
            analysis['flat_signals'] = action_counts.get('FLAT', 0)

        # Confidence analysis
        if 'adjusted_confidence' in signals_df.columns:
            analysis['avg_confidence'] = signals_df['adjusted_confidence'].mean()
            analysis['min_confidence'] = signals_df['adjusted_confidence'].min()
            analysis['max_confidence'] = signals_df['adjusted_confidence'].max()
            analysis['std_confidence'] = signals_df['adjusted_confidence'].std()

            # High confidence signals (>90%)
            high_conf = signals_df[signals_df['adjusted_confidence'] > 0.90]
            analysis['high_confidence_signals'] = len(high_conf)
            analysis['high_confidence_pct'] = len(high_conf) / len(signals_df)

        # Execution analysis
        if 'executed' in signals_df.columns:
            executed = signals_df[signals_df['executed'] == 1]
            analysis['executed_signals'] = len(executed)
            analysis['execution_rate'] = len(executed) / len(signals_df) if len(signals_df) > 0 else 0

        # Regime analysis
        if 'regime' in signals_df.columns:
            regime_counts = signals_df['regime'].value_counts()
            analysis['trending_signals'] = regime_counts.get('trending', 0)
            analysis['ranging_signals'] = regime_counts.get('ranging', 0)

        logger.info(f"Signal quality analyzed: {len(analysis)} metrics")
        return analysis

    def analyze_by_regime(self, days: int = 30) -> Dict:
        """
        Analyze performance by market regime

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of regime-specific performance
        """
        logger.info(f"Analyzing performance by regime for last {days} days...")

        cutoff = datetime.now() - timedelta(days=days)

        # Get signals with regime info
        signals = self.db.get_recent_signals(limit=1000)
        if not signals:
            return {}

        signals_df = pd.DataFrame(signals)
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        signals_df = signals_df[signals_df['timestamp'] >= cutoff]

        if 'regime' not in signals_df.columns:
            return {}

        analysis = {}

        for regime in ['trending', 'ranging']:
            regime_signals = signals_df[signals_df['regime'] == regime]

            if len(regime_signals) > 0:
                regime_data = {
                    'total_signals': len(regime_signals),
                    'avg_confidence': regime_signals['adjusted_confidence'].mean() if 'adjusted_confidence' in regime_signals.columns else 0,
                    'execution_rate': regime_signals['executed'].mean() if 'executed' in regime_signals.columns else 0
                }

                analysis[regime] = regime_data

        logger.info(f"Regime analysis complete: {len(analysis)} regimes")
        return analysis

    def analyze_risk_metrics(self, days: int = 30) -> Dict:
        """
        Analyze risk metrics

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of risk metrics
        """
        logger.info(f"Analyzing risk metrics for last {days} days...")

        cutoff = datetime.now() - timedelta(days=days)
        trades = self.db.get_recent_trades(limit=1000)

        if not trades:
            return {}

        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df[trades_df['timestamp'] >= cutoff]

        if len(trades_df) == 0 or 'profit' not in trades_df.columns:
            return {}

        risk_metrics = {}

        # Value at Risk (VaR) - 95% confidence
        returns = trades_df['profit'].values
        risk_metrics['var_95'] = np.percentile(returns, 5)

        # Conditional VaR (Expected Shortfall)
        var_threshold = risk_metrics['var_95']
        tail_losses = returns[returns <= var_threshold]
        risk_metrics['cvar_95'] = tail_losses.mean() if len(tail_losses) > 0 else 0

        # Volatility
        risk_metrics['volatility'] = returns.std()
        risk_metrics['annualized_volatility'] = returns.std() * np.sqrt(252)

        # Downside deviation
        downside_returns = returns[returns < 0]
        risk_metrics['downside_deviation'] = downside_returns.std() if len(downside_returns) > 0 else 0

        # Sortino Ratio
        if risk_metrics['downside_deviation'] > 0:
            risk_metrics['sortino_ratio'] = (returns.mean() / risk_metrics['downside_deviation']) * np.sqrt(252)
        else:
            risk_metrics['sortino_ratio'] = 0

        # Risk/Reward distribution
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) > 0:
            risk_metrics['win_loss_ratio'] = abs(wins.mean() / losses.mean()) if len(wins) > 0 else 0
        else:
            risk_metrics['win_loss_ratio'] = 0

        logger.info(f"Risk metrics calculated: {len(risk_metrics)} metrics")
        return risk_metrics

    def generate_summary_report(self, days: int = 30) -> str:
        """
        Generate comprehensive text summary report

        Args:
            days: Number of days to analyze

        Returns:
            Formatted text report
        """
        logger.info(f"Generating summary report for last {days} days...")

        # Gather all analytics
        performance = self.calculate_performance_metrics(days)
        signal_quality = self.analyze_signal_quality(days)
        regime_analysis = self.analyze_by_regime(days)
        risk_metrics = self.analyze_risk_metrics(days)

        # Generate report
        report = []
        report.append("="*80)
        report.append("TRADING SYSTEM PERFORMANCE REPORT")
        report.append("="*80)
        report.append(f"Period: Last {days} days")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")

        # Performance Metrics
        if performance:
            report.append("PERFORMANCE METRICS")
            report.append("-"*80)
            report.append(f"Total Trades: {performance.get('total_trades', 0)}")
            report.append(f"Win Rate: {performance.get('win_rate', 0):.2%}")
            report.append(f"Winning Trades: {performance.get('winning_trades', 0)}")
            report.append(f"Losing Trades: {performance.get('losing_trades', 0)}")
            report.append("")
            report.append(f"Gross Profit: ${performance.get('gross_profit', 0):.2f}")
            report.append(f"Gross Loss: ${performance.get('gross_loss', 0):.2f}")
            report.append(f"Net Profit: ${performance.get('net_profit', 0):.2f}")
            report.append(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
            report.append("")
            report.append(f"Average Win: ${performance.get('avg_win', 0):.2f}")
            report.append(f"Average Loss: ${performance.get('avg_loss', 0):.2f}")
            report.append(f"Avg Risk/Reward: 1:{performance.get('avg_risk_reward', 0):.2f}")
            report.append("")
            report.append(f"Largest Win: ${performance.get('largest_win', 0):.2f}")
            report.append(f"Largest Loss: ${performance.get('largest_loss', 0):.2f}")
            report.append(f"Max Consecutive Wins: {performance.get('max_consecutive_wins', 0)}")
            report.append(f"Max Consecutive Losses: {performance.get('max_consecutive_losses', 0)}")
            report.append("")
            report.append(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
            report.append(f"Max Drawdown: ${performance.get('max_drawdown', 0):.2f} ({performance.get('max_drawdown_pct', 0):.2%})")
            report.append("")

        # Signal Quality
        if signal_quality:
            report.append("SIGNAL QUALITY")
            report.append("-"*80)
            report.append(f"Total Signals: {signal_quality.get('total_signals', 0)}")
            report.append(f"Signals per Day: {signal_quality.get('signals_per_day', 0):.1f}")
            report.append("")
            report.append(f"LONG Signals: {signal_quality.get('long_signals', 0)}")
            report.append(f"SHORT Signals: {signal_quality.get('short_signals', 0)}")
            report.append(f"FLAT Signals: {signal_quality.get('flat_signals', 0)}")
            report.append("")
            report.append(f"Average Confidence: {signal_quality.get('avg_confidence', 0):.2%}")
            report.append(f"Min Confidence: {signal_quality.get('min_confidence', 0):.2%}")
            report.append(f"Max Confidence: {signal_quality.get('max_confidence', 0):.2%}")
            report.append(f"High Confidence (>90%): {signal_quality.get('high_confidence_signals', 0)} ({signal_quality.get('high_confidence_pct', 0):.1%})")
            report.append("")
            report.append(f"Executed Signals: {signal_quality.get('executed_signals', 0)}")
            report.append(f"Execution Rate: {signal_quality.get('execution_rate', 0):.2%}")
            report.append("")

        # Regime Analysis
        if regime_analysis:
            report.append("REGIME ANALYSIS")
            report.append("-"*80)
            for regime, data in regime_analysis.items():
                report.append(f"{regime.upper()}:")
                report.append(f"  Signals: {data.get('total_signals', 0)}")
                report.append(f"  Avg Confidence: {data.get('avg_confidence', 0):.2%}")
                report.append(f"  Execution Rate: {data.get('execution_rate', 0):.2%}")
                report.append("")

        # Risk Metrics
        if risk_metrics:
            report.append("RISK METRICS")
            report.append("-"*80)
            report.append(f"Volatility: ${risk_metrics.get('volatility', 0):.2f}")
            report.append(f"Annualized Volatility: ${risk_metrics.get('annualized_volatility', 0):.2f}")
            report.append(f"Downside Deviation: ${risk_metrics.get('downside_deviation', 0):.2f}")
            report.append(f"Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
            report.append("")
            report.append(f"VaR (95%): ${risk_metrics.get('var_95', 0):.2f}")
            report.append(f"CVaR (95%): ${risk_metrics.get('cvar_95', 0):.2f}")
            report.append(f"Win/Loss Ratio: {risk_metrics.get('win_loss_ratio', 0):.2f}")
            report.append("")

        # Targets vs Actual
        config_targets = self.config.get('performance', {}).get('benchmarks', {})
        if performance and config_targets:
            report.append("TARGETS VS ACTUAL")
            report.append("-"*80)

            win_rate_target = config_targets.get('win_rate_target', 0.6)
            win_rate_actual = performance.get('win_rate', 0)
            status = "✅ PASS" if win_rate_actual >= win_rate_target else "❌ FAIL"
            report.append(f"Win Rate: {win_rate_actual:.2%} (Target: {win_rate_target:.0%}) {status}")

            pf_target = config_targets.get('profit_factor_target', 2.0)
            pf_actual = performance.get('profit_factor', 0)
            status = "✅ PASS" if pf_actual >= pf_target else "❌ FAIL"
            report.append(f"Profit Factor: {pf_actual:.2f} (Target: {pf_target:.1f}) {status}")

            sharpe_target = config_targets.get('sharpe_ratio_target', 2.0)
            sharpe_actual = performance.get('sharpe_ratio', 0)
            status = "✅ PASS" if sharpe_actual >= sharpe_target else "❌ FAIL"
            report.append(f"Sharpe Ratio: {sharpe_actual:.2f} (Target: {sharpe_target:.1f}) {status}")

            dd_limit = config_targets.get('max_drawdown_limit', -0.15)
            dd_actual = -performance.get('max_drawdown_pct', 0)
            status = "✅ PASS" if dd_actual >= dd_limit else "❌ FAIL"
            report.append(f"Max Drawdown: {dd_actual:.2%} (Limit: {dd_limit:.0%}) {status}")
            report.append("")

        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        report_text = "\n".join(report)
        logger.info("Summary report generated")

        return report_text

    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        if len(series) == 0:
            return 0

        max_count = 0
        current_count = 0

        for value in series:
            if value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def save_report_to_file(self, days: int = 30, filename: Optional[str] = None):
        """
        Save report to file

        Args:
            days: Number of days to analyze
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.generate_summary_report(days)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report saved to: {filename}")
        return filename


def main():
    """Main function - Demo/CLI"""
    logger.info("")
    logger.info("="*80)
    logger.info("  PERFORMANCE ANALYTICS TOOL")
    logger.info("="*80)
    logger.info("")

    # Initialize analytics
    analytics = PerformanceAnalytics()

    # Generate report for last 30 days
    report = analytics.generate_summary_report(days=30)

    # Print to console
    print(report)

    # Save to file
    filename = analytics.save_report_to_file(days=30)
    logger.info(f"Report also saved to: {filename}")

    logger.info("")
    logger.info("="*80)
    logger.info("ANALYTICS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
