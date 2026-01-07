"""
Project Cleanup Script
======================
Removes unused files and cleans up the project structure
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import json

class ProjectCleaner:
    """Clean up unused files from the project"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.removed_files = []
        self.removed_dirs = []
        self.kept_files = []

    def get_files_to_remove(self) -> dict:
        """Get categorized list of files to remove"""

        files_to_remove = {
            "duplicate_docs": [
                "mt5_connector.py.old",
                "README_PRODUCTION.md",
                "COMPLETE_SYSTEM_SUMMARY.md",
                "QUICK_REFERENCE.md",
                "PRODUCTION_DEPLOYMENT_GUIDE.md",
                "SYSTEM_ARCHITECTURE.md",
            ],

            "analysis_debug_files": [
                "analyze_drawdown.py",
                "analyze_losing_trades.py",
                "analyze_production_signals.py",
                "analyze_quality.py",
                "analyze_short_problem.py",
                "analyze_signal_filtering.py",
                "analyze_trades_quality.py",
                "debug_signals.py",
                "DEBUGGING_SUMMARY.md",
            ],

            "old_results_files": [
                "elite_long_only_results.csv",
                "elite_optimized_v2_results.csv",
                "elite_signals.csv",
                "elite_ultra_high_winrate_results.csv",
                "ensemble_comparison.csv",
                "ml_model_comparison.csv",
                "parameter_optimization_results.csv",
                "quarterly_performance.csv",
                "yearly_performance.csv",
                "production_signals.csv",
            ],

            "old_documentation": [
                "ANALYSIS_AND_IMPROVEMENTS.md",
                "ELITE_LONG_ONLY_ROADMAP.md",
                "ELITE_LONG_ONLY_STRATEGY.md",
                "FINAL_OPTIMIZATION_REPORT.md",
                "FINAL_PRODUCTION_SUMMARY.md",
                "IMPROVEMENTS_COMPLETED.md",
                "OPTIMIZATION_RESULTS.md",
                "OPTIMIZATION_SUMMARY.md",
                "PAPER_TRADING_IMPLEMENTATION.md",
                "VERSION_3_RESULTS.md",
                "CHANGELOG.md",
                "DEPLOY_NOW.md",
                "mt5_integration_guide.md",
            ],

            "old_system_files": [
                "advanced_ml_system.py",
                "alternative_data_integrator.py",
                "elite_strategy_v3_improved.py",
                "elite_trading_bot.py",
                "elite_long_only_backtest.py",
                "elite_optimized_v2.py",
                "ensemble_strategy_system.py",
                "improve_winrate.py",
                "ml_improvements.py",
                "main.py",
                "master_integration_system.py",
                "production_system.py",
                "realtime_regime_detector.py",
                "risk_manager.py",
                "run_master_system.bat",
                "run_paper_trading.py",
                "run_paper_trading_test.py",
                "sentiment_analysis_system.py",
                "signal_generator.py",
                "test_all_components.py",
                "test_deployment.py",
                "train.py",
                "walk_forward_validation.py",
                "add_regime_detection.py",
                "multi_timeframe_analyzer.py",
                "mt5_trade_executor.py",
            ],

            "temp_data_files": [
                "nul",
                "best_ml_model.json",
                "best_parameters_v3.json",
                "bot_config.json",
                "master_signal_output.json",
                "mtf_analysis_example.json",
                "optimization_params_v2.json",
                "regime_adaptive_params.json",
                "sentiment_analysis_example.json",
                "validation_results.json",
            ],

            "test_databases": [
                "demo_trading.db",
                "test_integration.db",
                "test_smoke.db",
                "backup_trading_db_20251230_232352.db",
            ],
        }

        dirs_to_remove = [
            "__pycache__",
            "catboost_info",
            "archive",
            "analysis",
            "backtesting",
            "strategies",
            "trading",
        ]

        return files_to_remove, dirs_to_remove

    def create_backup_list(self):
        """Create a list of files that will be removed"""
        files_to_remove, dirs_to_remove = self.get_files_to_remove()

        backup_info = {
            "timestamp": "2025-12-31",
            "files_to_remove": files_to_remove,
            "dirs_to_remove": dirs_to_remove,
            "total_files": sum(len(v) for v in files_to_remove.values()),
            "total_dirs": len(dirs_to_remove)
        }

        backup_file = self.project_root / "cleanup_backup_list.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=2)

        print(f"Backup list created: {backup_file}")
        return backup_info

    def remove_files(self, dry_run: bool = True):
        """Remove unused files"""
        files_to_remove, dirs_to_remove = self.get_files_to_remove()

        print("=" * 80)
        print("PROJECT CLEANUP")
        print("=" * 80)
        print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'LIVE (files will be deleted)'}")
        print()

        # Remove files by category
        total_removed = 0
        for category, files in files_to_remove.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 80)

            for file in files:
                file_path = self.project_root / file
                if file_path.exists():
                    if not dry_run:
                        try:
                            file_path.unlink()
                            self.removed_files.append(str(file))
                            print(f"  [OK] Removed: {file}")
                        except Exception as e:
                            print(f"  [ERR] Error removing {file}: {e}")
                    else:
                        print(f"  [->] Would remove: {file}")
                    total_removed += 1
                else:
                    print(f"  [-] Not found: {file}")

        # Remove directories
        print(f"\nDIRECTORIES:")
        print("-" * 80)
        for dir_name in dirs_to_remove:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                if not dry_run:
                    try:
                        shutil.rmtree(dir_path)
                        self.removed_dirs.append(dir_name)
                        print(f"  [OK] Removed directory: {dir_name}")
                    except Exception as e:
                        print(f"  [ERR] Error removing {dir_name}: {e}")
                else:
                    print(f"  [->] Would remove directory: {dir_name}")
            else:
                print(f"  [-] Not found: {dir_name}")

        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Files to remove: {total_removed}")
        print(f"Directories to remove: {len(dirs_to_remove)}")

        if not dry_run:
            print(f"Files actually removed: {len(self.removed_files)}")
            print(f"Directories actually removed: {len(self.removed_dirs)}")

        return self.removed_files, self.removed_dirs

    def list_remaining_files(self):
        """List core files that should remain"""
        core_files = {
            "Core System": [
                "production_system_mt5.py",
                "mt5_data_provider.py",
                "trade_executor.py",
                "config_manager.py",
                "database_manager.py",
            ],

            "Analytics & Monitoring": [
                "performance_analytics.py",
                "system_health_monitor.py",
            ],

            "Testing": [
                "test_mt5_connection.py",
                "test_mt5_integration.py",
            ],

            "Scripts": [
                "start_paper_trading.bat",
                "run_tests.bat",
                "view_performance.bat",
                "check_system_health.bat",
            ],

            "Documentation": [
                "README.md",
                "DEPLOYMENT_GUIDE.md",
                "SYSTEM_SUMMARY.md",
                "QUICK_START.md",
            ],

            "Configuration": [
                "config.yaml",
                "requirements.txt",
            ],

            "Data": [
                "trading_data.db",
                "logs/",
                "models/",
                "data/",
                "config/",
                "dashboard/",
            ]
        }

        print()
        print("=" * 80)
        print("CORE FILES (TO KEEP)")
        print("=" * 80)

        for category, files in core_files.items():
            print(f"\n{category}:")
            print("-" * 80)
            for file in files:
                file_path = self.project_root / file
                exists = "[OK]" if file_path.exists() else "[XX]"
                print(f"  {exists} {file}")


def main():
    """Main cleanup function"""
    print()
    print("=" * 80)
    print("  GOLD TRADING SYSTEM - PROJECT CLEANUP")
    print("=" * 80)
    print()

    cleaner = ProjectCleaner(".")

    # Create backup list
    backup_info = cleaner.create_backup_list()
    print(f"\nWill remove {backup_info['total_files']} files and {backup_info['total_dirs']} directories")

    # Dry run first
    print("\nRunning DRY RUN first...")
    cleaner.remove_files(dry_run=True)

    # Show core files
    cleaner.list_remaining_files()

    # Confirm deletion
    print()
    print("=" * 80)
    response = input("\nDo you want to proceed with actual deletion? (yes/no): ")

    if response.lower() == 'yes':
        print("\nProceeding with cleanup...")
        cleaner = ProjectCleaner(".")  # Reset
        cleaner.remove_files(dry_run=False)

        print()
        print("=" * 80)
        print("  CLEANUP COMPLETE!")
        print("=" * 80)
        print()
        print("Project is now clean and ready for production.")
    else:
        print("\nCleanup cancelled. No files were removed.")


if __name__ == "__main__":
    main()
