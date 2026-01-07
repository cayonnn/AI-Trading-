"""
Performance Monitoring Dashboard
=================================
Real-time HTML dashboard for trading system performance

Features:
- Live performance metrics
- Interactive charts
- System health status
- Signal quality analysis
- Auto-refresh capability
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json

from database_manager import DatabaseManager
from config_manager import ConfigManager
from performance_analytics import PerformanceAnalytics
from system_health_monitor import SystemHealthMonitor


class PerformanceDashboard:
    """Generate HTML dashboard for performance monitoring"""

    def __init__(self):
        self.db = DatabaseManager()
        self.config = ConfigManager()
        self.analytics = PerformanceAnalytics()
        self.health_monitor = SystemHealthMonitor()

    def generate_html_dashboard(self, filename: str = "dashboard.html") -> str:
        """Generate complete HTML dashboard"""

        print("Generating performance dashboard...")

        # Collect all data
        perf_metrics = self.analytics.calculate_performance_metrics(days=30)
        signal_quality = self.analytics.analyze_signal_quality(days=30)
        risk_metrics = self.analytics.analyze_risk_metrics(days=30)
        regime_analysis = self.analytics.analyze_by_regime(days=30)
        system_resources = self.health_monitor.check_system_resources()
        mt5_status = self.health_monitor.check_mt5_connection()
        db_health = self.health_monitor.check_database_health()

        # Generate HTML
        html = self._generate_html(
            perf_metrics,
            signal_quality,
            risk_metrics,
            regime_analysis,
            system_resources,
            mt5_status,
            db_health
        )

        # Save to file
        dashboard_path = Path("dashboard") / filename
        dashboard_path.parent.mkdir(exist_ok=True)

        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Dashboard saved to: {dashboard_path}")
        return str(dashboard_path)

    def _generate_html(
        self,
        perf_metrics: Dict,
        signal_quality: Dict,
        risk_metrics: Dict,
        regime_analysis: Dict,
        system_resources: Dict,
        mt5_status: Dict,
        db_health: Dict
    ) -> str:
        """Generate HTML content"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>Gold Trading System - Performance Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }}

        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}

        .header .last-update {{
            color: #999;
            font-size: 0.9em;
            margin-top: 10px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .card-header {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}

        .metric:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            color: #666;
            font-size: 1em;
        }}

        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}

        .metric-value.good {{
            color: #10b981;
        }}

        .metric-value.warning {{
            color: #f59e0b;
        }}

        .metric-value.critical {{
            color: #ef4444;
        }}

        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            text-transform: uppercase;
        }}

        .status-badge.ok {{
            background: #d1fae5;
            color: #065f46;
        }}

        .status-badge.warning {{
            background: #fef3c7;
            color: #92400e;
        }}

        .status-badge.critical {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}

        .big-number {{
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }}

        .big-number.good {{
            color: #10b981;
        }}

        .big-number.warning {{
            color: #f59e0b;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th {{
            background: #f9fafb;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}

        .alert.info {{
            background: #dbeafe;
            color: #1e40af;
            border-left: 4px solid #3b82f6;
        }}

        .alert.success {{
            background: #d1fae5;
            color: #065f46;
            border-left: 4px solid #10b981;
        }}

        .alert.warning {{
            background: #fef3c7;
            color: #92400e;
            border-left: 4px solid #f59e0b;
        }}

        .footer {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            color: #666;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🏆 Gold Trading System</h1>
            <div class="subtitle">Real-Time Performance Dashboard</div>
            <div class="last-update">Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="last-update">Auto-refresh every 60 seconds</div>
        </div>

        <!-- System Health Alert -->
        {self._generate_health_alert(system_resources, mt5_status, db_health)}

        <!-- Performance Metrics Grid -->
        <div class="grid">
            <!-- Win Rate Card -->
            <div class="card">
                <div class="card-header">Win Rate</div>
                <div class="big-number {'good' if perf_metrics.get('win_rate', 0) > 0.6 else 'warning'}">
                    {perf_metrics.get('win_rate', 0):.1%}
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {perf_metrics.get('win_rate', 0) * 100}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value">{perf_metrics.get('total_trades', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Wins</span>
                    <span class="metric-value good">{perf_metrics.get('winning_trades', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Losses</span>
                    <span class="metric-value critical">{perf_metrics.get('losing_trades', 0)}</span>
                </div>
            </div>

            <!-- Profit Factor Card -->
            <div class="card">
                <div class="card-header">Profit Factor</div>
                <div class="big-number {'good' if perf_metrics.get('profit_factor', 0) > 2.0 else 'warning'}">
                    {perf_metrics.get('profit_factor', 0):.2f}
                </div>
                <div class="metric">
                    <span class="metric-label">Gross Profit</span>
                    <span class="metric-value good">${perf_metrics.get('gross_profit', 0):,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Gross Loss</span>
                    <span class="metric-value critical">${perf_metrics.get('gross_loss', 0):,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Net Profit</span>
                    <span class="metric-value">${perf_metrics.get('net_profit', 0):,.2f}</span>
                </div>
            </div>

            <!-- Sharpe Ratio Card -->
            <div class="card">
                <div class="card-header">Risk Metrics</div>
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value {'good' if perf_metrics.get('sharpe_ratio', 0) > 2.0 else 'warning'}">
                        {perf_metrics.get('sharpe_ratio', 0):.2f}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sortino Ratio</span>
                    <span class="metric-value">
                        {risk_metrics.get('sortino_ratio', 0):.2f}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value critical">
                        {perf_metrics.get('max_drawdown_pct', 0):.2%}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">VaR (95%)</span>
                    <span class="metric-value">
                        ${risk_metrics.get('var_95', 0):,.2f}
                    </span>
                </div>
            </div>

            <!-- Signal Quality Card -->
            <div class="card">
                <div class="card-header">Signal Quality</div>
                <div class="metric">
                    <span class="metric-label">Total Signals</span>
                    <span class="metric-value">{signal_quality.get('total_signals', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence</span>
                    <span class="metric-value {'good' if signal_quality.get('avg_confidence', 0) > 0.8 else 'warning'}">
                        {signal_quality.get('avg_confidence', 0):.1%}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Execution Rate</span>
                    <span class="metric-value">
                        {signal_quality.get('execution_rate', 0):.1%}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Signals/Day</span>
                    <span class="metric-value">
                        {signal_quality.get('signals_per_day', 0):.1f}
                    </span>
                </div>
            </div>
        </div>

        <!-- System Resources -->
        <div class="card">
            <div class="card-header">System Resources</div>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                <div>
                    <div class="metric-label" style="margin-bottom: 10px;">CPU Usage</div>
                    <div class="big-number {'good' if system_resources.get('cpu_percent', 0) < 80 else 'warning'}" style="font-size: 2em;">
                        {system_resources.get('cpu_percent', 0):.1f}%
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {system_resources.get('cpu_percent', 0)}%"></div>
                    </div>
                </div>
                <div>
                    <div class="metric-label" style="margin-bottom: 10px;">Memory Usage</div>
                    <div class="big-number {'good' if system_resources.get('memory_percent', 0) < 80 else 'warning'}" style="font-size: 2em;">
                        {system_resources.get('memory_percent', 0):.1f}%
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {system_resources.get('memory_percent', 0)}%"></div>
                    </div>
                </div>
                <div>
                    <div class="metric-label" style="margin-bottom: 10px;">Disk Usage</div>
                    <div class="big-number {'good' if system_resources.get('disk_percent', 0) < 80 else 'warning'}" style="font-size: 2em;">
                        {system_resources.get('disk_percent', 0):.1f}%
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {system_resources.get('disk_percent', 0)}%"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Regime Performance -->
        <div class="card">
            <div class="card-header">Performance by Market Regime</div>
            <table>
                <thead>
                    <tr>
                        <th>Regime</th>
                        <th>Signals</th>
                        <th>Win Rate</th>
                        <th>Avg Confidence</th>
                        <th>Best Performing</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_regime_rows(regime_analysis)}
                </tbody>
            </table>
        </div>

        <!-- MT5 Status -->
        <div class="card">
            <div class="card-header">MT5 Connection Status</div>
            <div class="metric">
                <span class="metric-label">Status</span>
                <span class="status-badge {'ok' if mt5_status.get('connected') else 'critical'}">
                    {mt5_status.get('status', 'UNKNOWN')}
                </span>
            </div>
            {self._generate_mt5_details(mt5_status)}
        </div>

        <!-- Footer -->
        <div class="footer">
            <p><strong>Production-Grade Gold Trading System v1.0.0</strong></p>
            <p>Dashboard auto-refreshes every 60 seconds</p>
            <p>Last Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_health_alert(self, system_resources: Dict, mt5_status: Dict, db_health: Dict) -> str:
        """Generate system health alert"""

        issues = []

        # Check system resources
        if system_resources.get('cpu_status') != 'OK':
            issues.append(f"High CPU usage: {system_resources.get('cpu_percent', 0):.1f}%")
        if system_resources.get('memory_status') != 'OK':
            issues.append(f"High memory usage: {system_resources.get('memory_percent', 0):.1f}%")
        if system_resources.get('disk_status') != 'OK':
            issues.append(f"Low disk space: {system_resources.get('disk_free_gb', 0):.1f} GB free")

        # Check MT5
        if not mt5_status.get('connected'):
            issues.append("MT5 not connected")

        # Check database
        if db_health.get('status') != 'OK':
            issues.append("Database issue detected")

        if len(issues) == 0:
            return '''
            <div class="alert success">
                <strong>✓ System Healthy</strong> - All systems operational
            </div>
            '''
        elif len(issues) < 3:
            issues_html = "<br>".join([f"• {issue}" for issue in issues])
            return f'''
            <div class="alert warning">
                <strong>⚠ Warning</strong> - Minor issues detected:<br>{issues_html}
            </div>
            '''
        else:
            issues_html = "<br>".join([f"• {issue}" for issue in issues])
            return f'''
            <div class="alert critical">
                <strong>✗ Critical</strong> - Multiple issues detected:<br>{issues_html}
            </div>
            '''

    def _generate_regime_rows(self, regime_analysis: Dict) -> str:
        """Generate table rows for regime analysis"""

        regimes = regime_analysis.get('regimes', {})
        if not regimes:
            return "<tr><td colspan='5'>No regime data available</td></tr>"

        rows = []
        for regime, data in regimes.items():
            row = f"""
                <tr>
                    <td><strong>{regime.replace('_', ' ').title()}</strong></td>
                    <td>{data.get('signal_count', 0)}</td>
                    <td>{data.get('win_rate', 0):.1%}</td>
                    <td>{data.get('avg_confidence', 0):.1%}</td>
                    <td>{'✓' if data.get('win_rate', 0) > 0.6 else '✗'}</td>
                </tr>
            """
            rows.append(row)

        return "\n".join(rows)

    def _generate_mt5_details(self, mt5_status: Dict) -> str:
        """Generate MT5 connection details"""

        if not mt5_status.get('connected'):
            return ""

        return f"""
            <div class="metric">
                <span class="metric-label">Terminal</span>
                <span class="metric-value">{mt5_status.get('terminal_name', 'Unknown')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Build</span>
                <span class="metric-value">{mt5_status.get('build', 'Unknown')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">GOLD Bid</span>
                <span class="metric-value">${mt5_status.get('current_bid', 0):,.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">GOLD Ask</span>
                <span class="metric-value">${mt5_status.get('current_ask', 0):,.2f}</span>
            </div>
        """


def main():
    """Main function"""
    print()
    print("=" * 80)
    print("  PERFORMANCE DASHBOARD GENERATOR")
    print("=" * 80)
    print()

    try:
        dashboard = PerformanceDashboard()
        filepath = dashboard.generate_html_dashboard()

        print()
        print("=" * 80)
        print("  DASHBOARD GENERATED!")
        print("=" * 80)
        print()
        print(f"Dashboard saved to: {filepath}")
        print()
        print("Open the dashboard in your web browser:")
        print(f"  file:///{os.path.abspath(filepath)}")
        print()
        print("Dashboard will auto-refresh every 60 seconds")
        print()

    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
