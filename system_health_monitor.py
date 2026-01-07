"""
System Health Monitor - Production Grade
========================================
Comprehensive system health monitoring and diagnostics

Features:
- System status checking
- Database health
- MT5 connection status
- Performance metrics
- Error rate monitoring
- Disk space monitoring
- Automated health checks
"""

import os
import sys
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from pathlib import Path
import MetaTrader5 as mt5

from database_manager import DatabaseManager
from config_manager import ConfigManager
from mt5_data_provider import MT5DataProvider

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


class SystemHealthMonitor:
    """
    Production-grade system health monitoring

    Monitors:
    - System resources (CPU, RAM, Disk)
    - MT5 connection status
    - Database health
    - Recent errors
    - Performance degradation
    """

    def __init__(self):
        """Initialize health monitor"""
        self.config = ConfigManager()
        self.db = DatabaseManager()
        logger.info("System Health Monitor initialized")

    def check_system_resources(self) -> Dict:
        """Check system resources"""
        logger.info("Checking system resources...")

        resources = {}

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        resources['cpu_percent'] = cpu_percent
        resources['cpu_count'] = psutil.cpu_count()
        resources['cpu_status'] = 'OK' if cpu_percent < 80 else 'WARNING' if cpu_percent < 95 else 'CRITICAL'

        # Memory
        memory = psutil.virtual_memory()
        resources['memory_total_gb'] = memory.total / (1024**3)
        resources['memory_available_gb'] = memory.available / (1024**3)
        resources['memory_percent'] = memory.percent
        resources['memory_status'] = 'OK' if memory.percent < 80 else 'WARNING' if memory.percent < 95 else 'CRITICAL'

        # Disk
        disk = psutil.disk_usage('.')
        resources['disk_total_gb'] = disk.total / (1024**3)
        resources['disk_free_gb'] = disk.free / (1024**3)
        resources['disk_percent'] = disk.percent
        resources['disk_status'] = 'OK' if disk.percent < 80 else 'WARNING' if disk.percent < 95 else 'CRITICAL'

        logger.info(f"System resources checked: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%, Disk {disk.percent:.1f}%")
        return resources

    def check_mt5_connection(self) -> Dict:
        """Check MT5 connection status"""
        logger.info("Checking MT5 connection...")

        mt5_status = {}

        try:
            # Try to initialize
            provider = MT5DataProvider()

            if provider.connect():
                mt5_status['connected'] = True
                mt5_status['status'] = 'OK'

                # Get terminal info
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    mt5_status['terminal_name'] = terminal_info.name
                    mt5_status['build'] = terminal_info.build
                    mt5_status['connected_to_server'] = terminal_info.connected

                # Test data fetching
                try:
                    symbol_info = provider.get_symbol_info("GOLD")
                    if symbol_info:
                        mt5_status['symbol_available'] = True
                        mt5_status['current_bid'] = symbol_info['bid']
                        mt5_status['current_ask'] = symbol_info['ask']
                    else:
                        mt5_status['symbol_available'] = False
                except:
                    mt5_status['symbol_available'] = False

                provider.disconnect()
            else:
                mt5_status['connected'] = False
                mt5_status['status'] = 'FAILED'

        except Exception as e:
            mt5_status['connected'] = False
            mt5_status['status'] = 'ERROR'
            mt5_status['error'] = str(e)

        logger.info(f"MT5 connection status: {mt5_status.get('status', 'UNKNOWN')}")
        return mt5_status

    def check_database_health(self) -> Dict:
        """Check database health"""
        logger.info("Checking database health...")

        db_health = {}

        try:
            # Check database file exists
            db_config = self.config.get_database_config()
            db_path = Path(db_config.sqlite_path or "trading_data.db")

            db_health['exists'] = db_path.exists()

            if db_path.exists():
                # File size
                size_mb = db_path.stat().st_size / (1024**2)
                db_health['size_mb'] = size_mb

                # Check tables
                try:
                    # Try to get recent signals
                    signals = self.db.get_recent_signals(limit=1)
                    db_health['signals_table'] = 'OK'
                except:
                    db_health['signals_table'] = 'ERROR'

                try:
                    # Try to get recent trades
                    trades = self.db.get_recent_trades(limit=1)
                    db_health['trades_table'] = 'OK'
                except:
                    db_health['trades_table'] = 'ERROR'

                db_health['status'] = 'OK'
            else:
                db_health['status'] = 'MISSING'

        except Exception as e:
            db_health['status'] = 'ERROR'
            db_health['error'] = str(e)

        logger.info(f"Database health: {db_health.get('status', 'UNKNOWN')}")
        return db_health

    def check_recent_errors(self, hours: int = 24) -> Dict:
        """Check for recent errors in logs"""
        logger.info(f"Checking logs for errors in last {hours} hours...")

        error_analysis = {
            'period_hours': hours,
            'errors_found': 0,
            'recent_errors': []
        }

        try:
            # Get log directory
            log_dir = Path("logs")

            if not log_dir.exists():
                error_analysis['status'] = 'NO_LOGS'
                return error_analysis

            # Get recent log files
            cutoff_time = datetime.now() - timedelta(hours=hours)

            for log_file in log_dir.glob("production_mt5_*.log"):
                try:
                    # Check if file is recent enough
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff_time:
                        continue

                    # Read file and count errors
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if 'ERROR' in line or 'CRITICAL' in line:
                                error_analysis['errors_found'] += 1
                                if len(error_analysis['recent_errors']) < 10:
                                    error_analysis['recent_errors'].append(line.strip())

                except Exception as e:
                    logger.warning(f"Could not read log file {log_file}: {e}")

            error_analysis['status'] = 'CRITICAL' if error_analysis['errors_found'] > 10 else 'WARNING' if error_analysis['errors_found'] > 0 else 'OK'

        except Exception as e:
            error_analysis['status'] = 'ERROR'
            error_analysis['error'] = str(e)

        logger.info(f"Recent errors: {error_analysis['errors_found']}")
        return error_analysis

    def check_performance_degradation(self, days: int = 7) -> Dict:
        """Check for performance degradation"""
        logger.info(f"Checking for performance degradation over last {days} days...")

        perf_check = {}

        try:
            # Get signals from last N days
            cutoff = datetime.now() - timedelta(days=days)
            signals = self.db.get_recent_signals(limit=1000)

            if not signals:
                perf_check['status'] = 'NO_DATA'
                return perf_check

            # Filter by date
            import pandas as pd
            signals_df = pd.DataFrame(signals)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            signals_df = signals_df[signals_df['timestamp'] >= cutoff]

            if len(signals_df) > 0:
                # Calculate metrics
                perf_check['total_signals'] = len(signals_df)

                if 'adjusted_confidence' in signals_df.columns:
                    perf_check['avg_confidence'] = signals_df['adjusted_confidence'].mean()

                    # Check if confidence is declining
                    half_point = len(signals_df) // 2
                    first_half = signals_df.iloc[:half_point]['adjusted_confidence'].mean()
                    second_half = signals_df.iloc[half_point:]['adjusted_confidence'].mean()

                    perf_check['confidence_trend'] = 'IMPROVING' if second_half > first_half else 'DECLINING'
                    perf_check['confidence_change'] = second_half - first_half

                # Check execution rate
                if 'executed' in signals_df.columns:
                    execution_rate = signals_df['executed'].mean()
                    perf_check['execution_rate'] = execution_rate

                perf_check['status'] = 'OK'
            else:
                perf_check['status'] = 'NO_DATA'

        except Exception as e:
            perf_check['status'] = 'ERROR'
            perf_check['error'] = str(e)

        logger.info(f"Performance check: {perf_check.get('status', 'UNKNOWN')}")
        return perf_check

    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        logger.info("Generating comprehensive health report...")

        # Gather all health checks
        system_resources = self.check_system_resources()
        mt5_status = self.check_mt5_connection()
        db_health = self.check_database_health()
        recent_errors = self.check_recent_errors(hours=24)
        performance = self.check_performance_degradation(days=7)

        # Generate report
        report = []
        report.append("="*80)
        report.append("SYSTEM HEALTH REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")

        # Overall Status
        overall_status = "OK"
        issues = []

        # System Resources
        report.append("SYSTEM RESOURCES")
        report.append("-"*80)
        report.append(f"CPU Usage: {system_resources.get('cpu_percent', 0):.1f}% [{system_resources.get('cpu_status', 'UNKNOWN')}]")
        report.append(f"  Cores: {system_resources.get('cpu_count', 0)}")
        report.append(f"Memory Usage: {system_resources.get('memory_percent', 0):.1f}% [{system_resources.get('memory_status', 'UNKNOWN')}]")
        report.append(f"  Total: {system_resources.get('memory_total_gb', 0):.1f} GB")
        report.append(f"  Available: {system_resources.get('memory_available_gb', 0):.1f} GB")
        report.append(f"Disk Usage: {system_resources.get('disk_percent', 0):.1f}% [{system_resources.get('disk_status', 'UNKNOWN')}]")
        report.append(f"  Total: {system_resources.get('disk_total_gb', 0):.1f} GB")
        report.append(f"  Free: {system_resources.get('disk_free_gb', 0):.1f} GB")
        report.append("")

        if system_resources.get('cpu_status') != 'OK':
            issues.append(f"High CPU usage: {system_resources.get('cpu_percent', 0):.1f}%")
        if system_resources.get('memory_status') != 'OK':
            issues.append(f"High memory usage: {system_resources.get('memory_percent', 0):.1f}%")
        if system_resources.get('disk_status') != 'OK':
            issues.append(f"Low disk space: {system_resources.get('disk_free_gb', 0):.1f} GB free")

        # MT5 Connection
        report.append("MT5 CONNECTION")
        report.append("-"*80)
        mt5_connected = mt5_status.get('connected', False)
        report.append(f"Status: {mt5_status.get('status', 'UNKNOWN')}")

        if mt5_connected:
            report.append(f"Terminal: {mt5_status.get('terminal_name', 'Unknown')}")
            report.append(f"Build: {mt5_status.get('build', 'Unknown')}")
            report.append(f"Connected to Server: {mt5_status.get('connected_to_server', False)}")

            if mt5_status.get('symbol_available'):
                report.append(f"GOLD Symbol: Available")
                report.append(f"  Bid: {mt5_status.get('current_bid', 0):.2f}")
                report.append(f"  Ask: {mt5_status.get('current_ask', 0):.2f}")
            else:
                report.append(f"GOLD Symbol: NOT AVAILABLE")
                issues.append("GOLD symbol not available")
        else:
            report.append("WARNING: MT5 Not Connected!")
            issues.append("MT5 not connected")

        report.append("")

        # Database
        report.append("DATABASE")
        report.append("-"*80)
        report.append(f"Status: {db_health.get('status', 'UNKNOWN')}")

        if db_health.get('exists'):
            report.append(f"Size: {db_health.get('size_mb', 0):.2f} MB")
            report.append(f"Signals Table: {db_health.get('signals_table', 'UNKNOWN')}")
            report.append(f"Trades Table: {db_health.get('trades_table', 'UNKNOWN')}")

            if db_health.get('signals_table') != 'OK':
                issues.append("Signals table error")
            if db_health.get('trades_table') != 'OK':
                issues.append("Trades table error")
        else:
            report.append("WARNING: Database file not found!")
            issues.append("Database not found")

        report.append("")

        # Recent Errors
        report.append("RECENT ERRORS (24 hours)")
        report.append("-"*80)
        error_count = recent_errors.get('errors_found', 0)
        report.append(f"Total Errors: {error_count}")

        if error_count > 0:
            report.append(f"Status: {recent_errors.get('status', 'UNKNOWN')}")
            report.append("Recent Error Examples:")
            for err in recent_errors.get('recent_errors', [])[:5]:
                report.append(f"  {err}")

            if error_count > 10:
                issues.append(f"High error count: {error_count} errors")
                overall_status = "WARNING"
        else:
            report.append("Status: OK - No errors found")

        report.append("")

        # Performance
        report.append("PERFORMANCE (7 days)")
        report.append("-"*80)

        if performance.get('status') == 'OK':
            report.append(f"Total Signals: {performance.get('total_signals', 0)}")
            report.append(f"Average Confidence: {performance.get('avg_confidence', 0):.2%}")
            report.append(f"Confidence Trend: {performance.get('confidence_trend', 'UNKNOWN')}")
            report.append(f"Confidence Change: {performance.get('confidence_change', 0):+.2%}")
            report.append(f"Execution Rate: {performance.get('execution_rate', 0):.2%}")

            if performance.get('confidence_trend') == 'DECLINING':
                issues.append("Signal confidence declining")
        else:
            report.append(f"Status: {performance.get('status', 'UNKNOWN')}")

        report.append("")

        # Overall Status
        report.append("="*80)
        report.append("OVERALL STATUS")
        report.append("="*80)

        if len(issues) == 0:
            report.append("✅ SYSTEM HEALTHY - All checks passed")
            overall_status = "OK"
        else:
            report.append(f"⚠️ {len(issues)} ISSUE(S) FOUND:")
            for issue in issues:
                report.append(f"  - {issue}")
            overall_status = "WARNING" if len(issues) < 3 else "CRITICAL"

        report.append(f"Overall Status: {overall_status}")
        report.append("="*80)

        report_text = "\n".join(report)
        logger.info("Health report generated")

        return report_text

    def save_health_report(self, filename: Optional[str] = None):
        """Save health report to file"""
        if filename is None:
            filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.generate_health_report()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Health report saved to: {filename}")
        return filename


def main():
    """Main function - CLI interface"""
    logger.info("")
    logger.info("="*80)
    logger.info("  SYSTEM HEALTH MONITOR")
    logger.info("="*80)
    logger.info("")

    # Initialize monitor
    monitor = SystemHealthMonitor()

    # Generate and display report
    report = monitor.generate_health_report()
    print(report)

    # Save to file
    filename = monitor.save_health_report()
    logger.info(f"Report also saved to: {filename}")

    logger.info("")
    logger.info("="*80)
    logger.info("HEALTH CHECK COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
